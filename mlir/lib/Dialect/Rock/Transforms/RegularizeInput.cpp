//===- RegularizeInput.cpp - Push fusions past load_markers ---------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// This pass runs AFTER InsertOutputFusionLoads and BEFORE RockLowerLoads.
// It ensures that every rock.load_marker's source is a pure transform chain
// (no fusion ops in between), by distributing load_markers past any fusions.
//
// For each load_marker whose source traces through fusion ops:
// 1. Walk backwards collecting transforms between the marker and fusions
// 2. For each leaf of the fusion DAG (transform chain to block arg):
//    apply the collected post-fusion transforms, isolate the chain,
//    create a new load_marker
// 3. For splat constants: create a tile-shaped constant (no load_marker)
// 4. Clone fusion ops after the load_markers, operating on tile types
// 5. Replace the original load_marker with the cloned fusion result
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
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEINPUTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-regularize-input"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Apply a list of TransformMapAttrs on top of `source`, creating
/// rock.transform ops.
static Value applyTransforms(OpBuilder &builder, Value source,
                             ArrayRef<TransformMapAttr> transforms) {
  SmallVector<Attribute> attrs(transforms.begin(), transforms.end());
  ArrayAttr transformsAttr = builder.getArrayAttr(attrs);
  return rock::transform(builder, source, transformsAttr);
}

/// Recursively resolve a value in the fusion DAG, returning a tile-typed
/// replacement. Each leaf (block arg or constant) gets its own load_marker;
/// fusion ops are cloned to operate on tile types. Transforms are accumulated
/// during the walk and applied at the leaves.
static FailureOr<Value> distributeLoadMarker(
    OpBuilder &builder, Location loc, Value originalVal,
    ArrayRef<TransformMapAttr> postTransforms, ArrayAttr extraViews,
    ValueRange extraIndices, Type tileType, CacheModifier cache,
    llvm::DenseMap<std::pair<Value, ArrayAttr>, Value> &valueMapping,
    func::FuncOp funcOp) {
  ArrayAttr postTransformsAttr = builder.getArrayAttr(
      SmallVector<Attribute>(postTransforms.begin(), postTransforms.end()));
  std::pair<Value, ArrayAttr> cacheKey{originalVal, postTransformsAttr};
  auto cached = valueMapping.find(cacheKey);
  if (cached != valueMapping.end())
    return cached->second;

  Operation *defOp = originalVal.getDefiningOp();

  // Leaf: block argument — apply postTransforms, create new load_marker.
  if (!defOp) {
    auto blockArg = cast<BlockArgument>(originalVal);
    if (blockArg.getOwner() != &funcOp.front())
      return failure();

    Value source = applyTransforms(builder, originalVal, postTransforms);
    auto newMarker = LoadMarkerOp::create(builder, loc, tileType, source,
                                          extraViews, extraIndices, cache);
    valueMapping.insert({cacheKey, newMarker.getResult()});
    return newMarker.getResult();
  }

  // rock.transform — accumulate and recurse.
  if (auto transformOp = dyn_cast<TransformOp>(defOp)) {
    SmallVector<TransformMapAttr> newPostTransforms;
    newPostTransforms.append(postTransforms.begin(), postTransforms.end());
    newPostTransforms.push_back(transformOp.getTransform());

    FailureOr<Value> result = distributeLoadMarker(
        builder, loc, transformOp.getInput(), newPostTransforms, extraViews,
        extraIndices, tileType, cache, valueMapping, funcOp);
    if (succeeded(result))
      valueMapping.insert({cacheKey, result.value()});
    return result;
  }

  // Fusion op — recurse on each operand, clone with tile types.
  if (rock::isFusionOp(defOp)) {
    auto tileShape = cast<RankedTensorType>(tileType).getShape();
    IRMapping fusionMapping;
    for (Value operand : defOp->getOperands()) {
      auto operandElemType =
          cast<RankedTensorType>(operand.getType()).getElementType();
      auto operandTileType = RankedTensorType::get(tileShape, operandElemType);
      FailureOr<Value> resolved = distributeLoadMarker(
          builder, loc, operand, postTransforms, extraViews, extraIndices,
          operandTileType, cache, valueMapping, funcOp);
      if (failed(resolved))
        return failure();
      fusionMapping.map(operand, resolved.value());
    }

    Operation *cloned = builder.clone(*defOp, fusionMapping);
    cloned->getResult(0).setType(tileType);
    valueMapping.insert({cacheKey, cloned->getResult(0)});
    return cloned->getResult(0);
  }

  // Leaf: splat arith.constant — reshape to tile type.
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
    valueMapping.insert({cacheKey, newConst.getResult()});
    return newConst.getResult();
  }

  return failure();
}

struct RockRegularizeInputPass
    : public rock::impl::RockRegularizeInputPassBase<RockRegularizeInputPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockRegularizeInputPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  SmallVector<LoadMarkerOp> loadMarkersToProcess;
  funcOp.walk(
      [&](LoadMarkerOp markerOp) { loadMarkersToProcess.push_back(markerOp); });

  LLVM_DEBUG(llvm::dbgs() << "RegularizeInput: found "
                          << loadMarkersToProcess.size()
                          << " load_marker ops\n");

  for (LoadMarkerOp markerOp : loadMarkersToProcess) {
    LLVM_DEBUG(llvm::dbgs() << "  Processing: " << markerOp << "\n");

    OpBuilder builder(markerOp);
    Location loc = markerOp.getLoc();
    auto tileType = cast<RankedTensorType>(markerOp.getResult().getType());

    llvm::DenseMap<std::pair<Value, ArrayAttr>, Value> valueMapping;
    FailureOr<Value> replacement = distributeLoadMarker(
        builder, loc, markerOp.getSource(), /*postTransforms=*/{},
        markerOp.getExtraViews(), markerOp.getExtraIndices(), tileType,
        markerOp.getCacheModifier(), valueMapping, funcOp);

    if (failed(replacement)) {
      markerOp->emitError("Failed to distribute load_marker past fusions");
      return signalPassFailure();
    }

    markerOp.getResult().replaceAllUsesWith(replacement.value());
    markerOp.erase();
  }
}
