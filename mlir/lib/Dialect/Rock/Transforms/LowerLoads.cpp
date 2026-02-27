//===- LowerLoads.cpp - Lower rock.load_marker ops to blockwise loads -----===//
//
// Copyright 2026 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This pass runs AFTER GridwiseGemmToBlockwise and InsertOutputFusionLoads.
// It converts rock.load_marker ops into actual rock.blockwise_load ops.
//
// LoadMarkerOps are introduced by:
// 1. ToBlockwise passes (via loadTile) for GEMM/Attention input loads
// 2. InsertOutputFusionLoads for output fusion extra inputs
//
// The LoadMarkerOp's source is the tensor (possibly through
// a fusion chain with rock.transform ops) that traces back to a func block
// argument. The tiling transforms are carried as metadata in extraViews.
// This pass applies those views and traces back through the source chain
// (transforms, fusion ops) to func block arguments, creating a
// BlockwiseLoadOp for each block argument encountered.
//
// Example (no fusion):
//   Before:
//     %t1 = rock.transform %arg0 by <pre_transform>
//     %tile = rock.load_marker %t1 views [<tiling>] [%kIter, %g, %m, %n]
//               : tensor<...> -> tensor<tile>
//
//   After:
//     %combined = rock.transform %arg0 by <pre_transform + tiling>
//     %tile = rock.blockwise_load %combined[%kIter, %g, %m, %n]
//
// Fusion example:
//   Before:
//     %t1 = rock.transform %arg0 by <pre1>
//     %t2 = rock.transform %arg1 by <pre2>
//     %fused = arith.addf %t1, %t2
//     %tile = rock.load_marker %fused views [<tiling>] [indices]
//               : tensor<...> -> tensor<tile>
//
//   After:
//     %combined1 = rock.transform %arg0 by <pre1 + tiling>
//     %loaded1 = rock.blockwise_load %combined1[indices]
//     %combined2 = rock.transform %arg1 by <pre2 + tiling>
//     %loaded2 = rock.blockwise_load %combined2[indices]
//     %fused_tile = arith.addf %loaded1, %loaded2
//
// Output fusion example (LoadMarkerOp + UntileOp):
//   Before:
//     %bias_t = rock.transform %arg_bias by <transforms>
//     %tile = rock.load_marker %bias_t views [<output_tiling>] [%g, %m, %n]
//               : tensor<full> -> tensor<tile>
//     %full = rock.untile %tile : tensor<tile> -> tensor<full>
//
//   After:
//     %combined = rock.transform %arg_bias by <transforms + output_tiling>
//     %tile = rock.blockwise_load %combined[%g, %m, %n]
//     %full = rock.untile %tile : tensor<tile> -> tensor<full>
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-loads"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Recursively reconstruct a value for blockwise loading.
/// Traces through transforms, fusion ops, and func block arguments.
static FailureOr<Value> reconstructForBlockwiseLoad(
    OpBuilder &builder, Location loc, Value originalVal,
    ArrayRef<TransformMapAttr> postTransforms, // Transforms in [last_to_apply,
                                               // ..., first_to_apply] order
    ValueRange blockIndices, // Indices for blockwise_load_tile
    Type tileType,           // Result type of blockwise_load_tile
    IRMapping &valueMapping, // Maps original values to loaded values
    func::FuncOp funcOp      // The kernel function (for block arg validation)
) {
  // Check if we've already processed this value
  if (valueMapping.contains(originalVal)) {
    return valueMapping.lookup(originalVal);
  }

  Operation *defOp = originalVal.getDefiningOp();

  // Case 1: Block argument of the kernel function — create a BlockwiseLoadOp.
  // The recursion has traced through transforms and accumulated them in
  // postTransforms; apply them all to the block arg and emit the load.
  if (!defOp) {
    auto blockArg = cast<BlockArgument>(originalVal);
    if (blockArg.getOwner() != &funcOp.front())
      return failure();

    Value source = originalVal;
    if (!postTransforms.empty()) {
      SmallVector<Attribute> transforms(postTransforms.begin(),
                                        postTransforms.end());
      ArrayAttr transformsAttr = builder.getArrayAttr(transforms);
      source = rock::transform(builder, source, transformsAttr);
    }

    auto loadTileOp =
        BlockwiseLoadOp::create(builder, loc, tileType, source, blockIndices);
    valueMapping.map(originalVal, loadTileOp.getResult());
    return loadTileOp.getResult();
  }

  // Case 2: rock.transform - accumulate and recurse
  // We walk backwards: load_marker -> tiling -> pad -> ... -> block arg.
  // rock::transform expects [last_to_apply, ..., first_to_apply].
  // So we append each transform we encounter (tiling first, pad second)
  // giving [tiling, pad], which applies as pad then tiling -- correct.
  if (auto transformOp = dyn_cast<TransformOp>(defOp)) {
    SmallVector<TransformMapAttr> newPostTransforms;
    newPostTransforms.append(postTransforms.begin(), postTransforms.end());
    newPostTransforms.push_back(transformOp.getTransform());

    FailureOr<Value> result = reconstructForBlockwiseLoad(
        builder, loc, transformOp.getInput(), newPostTransforms, blockIndices,
        tileType, valueMapping, funcOp);
    if (succeeded(result))
      valueMapping.map(originalVal, result.value());
    return result;
  }

  // Case 3: Fusion op (arith or math dialect)
  if (rock::isFusionOp(defOp)) {
    // Reconstruct each operand with its own tile type. For type-changing ops
    // like arith.truncf (f32->f16), the operand's element type differs from
    // the result's, so the tile type must match the operand's element type.
    auto tileShape = cast<RankedTensorType>(tileType).getShape();
    IRMapping fusionMapping;
    for (Value operand : defOp->getOperands()) {
      auto operandElemType =
          cast<RankedTensorType>(operand.getType()).getElementType();
      auto operandTileType = RankedTensorType::get(tileShape, operandElemType);
      FailureOr<Value> reconstructed = reconstructForBlockwiseLoad(
          builder, loc, operand, postTransforms, blockIndices, operandTileType,
          valueMapping, funcOp);
      if (failed(reconstructed))
        return failure();
      fusionMapping.map(operand, reconstructed.value());
    }

    // Clone the fusion op with reconstructed operands
    Operation *cloned = builder.clone(*defOp, fusionMapping);
    cloned->getResult(0).setType(tileType);
    valueMapping.map(originalVal, cloned->getResult(0));
    return cloned->getResult(0);
  }

  // Case 4: arith.constant - must be a splat; reshape to tile type
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!attr || !attr.isSplat()) {
      defOp->emitOpError("non-splat constant in fusion chain not supported");
      return failure();
    }
    auto tileRankedType = cast<RankedTensorType>(tileType);
    auto newAttr =
        DenseElementsAttr::get(tileRankedType, attr.getSplatValue<Attribute>());
    auto newConst =
        arith::ConstantOp::create(builder, loc, tileRankedType, newAttr);
    valueMapping.map(originalVal, newConst.getResult());
    return newConst.getResult();
  }

  // Unknown op - shouldn't happen for valid input
  return failure();
}

struct RockLowerLoadsPass
    : public rock::impl::RockLowerLoadsPassBase<RockLowerLoadsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockLowerLoadsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    return;
  }

  // Collect all load_marker ops
  SmallVector<LoadMarkerOp> loadMarkersToProcess;
  funcOp.walk(
      [&](LoadMarkerOp markerOp) { loadMarkersToProcess.push_back(markerOp); });

  LLVM_DEBUG(llvm::dbgs() << "Found " << loadMarkersToProcess.size()
                          << " load_marker ops to process\n");

  // Process each load_marker
  for (LoadMarkerOp markerOp : loadMarkersToProcess) {
    OpBuilder builder(markerOp);
    Location loc = markerOp.getLoc();

    Value source = markerOp.getSource();
    ValueRange indices = markerOp.getExtraIndices();

    // The result type of the LoadMarkerOp is the tile type.
    auto tileType = cast<RankedTensorType>(markerOp.getResult().getType());

    // The extraViews are the tiling transforms that must be applied.
    // The source is the untransformed tensor, so we seed
    // reconstructForBlockwiseLoad with the marker's views as the initial
    // post-transforms.
    auto initialTransforms = llvm::to_vector(
        markerOp.getExtraViews().getAsRange<TransformMapAttr>());

    LLVM_DEBUG(llvm::dbgs() << "Processing: " << markerOp << "\n");

    IRMapping valueMapping;
    FailureOr<Value> maybeTileResult =
        reconstructForBlockwiseLoad(builder, loc, source, initialTransforms,
                                    indices, tileType, valueMapping, funcOp);
    if (failed(maybeTileResult)) {
      markerOp->emitError("Couldn't lower to BlockwiseLoad");
      return signalPassFailure();
    }
    Value tileResult = maybeTileResult.value();

    markerOp.getResult().replaceAllUsesWith(tileResult);
    markerOp.erase();

    LLVM_DEBUG(llvm::dbgs() << "  Replaced with: " << tileResult << "\n");
  }
}
