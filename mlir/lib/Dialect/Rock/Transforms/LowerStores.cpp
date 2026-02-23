//===- LowerStores.cpp - Lower rock.store ops to blockwise_store_tile -----===//
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
// This pass runs AFTER RockLowerLoads and converts rock.store to
// rock.blockwise_store_tile.
//
// At this point, the IR has:
// - TileOp wrapping tile results with output transforms (from ToBlockwise)
// - UntileOp converting tiles to full tensors for fusion ops
// - Fusion ops (arith.addf, etc.) operating on full tensor types
// - rock.store storing the fused result
//
// This pass:
// 1. Traces back from rock.store through fusion ops to find StoreMarkerOp
// 2. Gets the tile values and transforms from StoreMarkerOp
// 3. Clones fusion ops to operate on tiles
// 4. Combines StoreMarkerOp transforms with store destination transforms
// 5. Creates BlockwiseStoreOp with the combined transform chain
// 6. Cleans up UntileOp, StoreMarkerOp and dead ops
//
// Example:
//   Before:
//     %gemm_tile = scf.for ... -> tensor<16x16xf32>
//     %marked = rock.tile %gemm_tile views [<transforms>] [%g, %m, %n]
//     %gemm_full = rock.untile %marked : tile -> full
//     %fused = arith.addf %gemm_full, %bias_full : tensor<full>
//     %dest_tr = rock.transform %arg2 by <dest_transforms>
//     %out = rock.store %fused to %dest_tr by set
//
//   After:
//     %gemm_tile = scf.for ... -> tensor<16x16xf32>
//     %fused_tile = arith.addf %gemm_tile, %bias_tile : tensor<16x16xf32>
//     %combined_dest = rock.transform %arg2 by <dest_transforms +
//     tile_transforms> %out = rock.blockwise_store_tile %fused_tile ->
//     %combined_dest[%g, %m, %n] by set
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
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERSTORESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-stores"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Check if an operation is a fusion op (arith or math dialect).
static bool isFusionOp(Operation *op) {
  return isa<arith::ArithDialect, math::MathDialect>(op->getDialect());
}

/// Information extracted from a StoreMarkerOp found in the store chain.
struct StoreMarkerInfo {
  StoreMarkerOp storeMarkerOp;
  ArrayAttr extraViews;
  SmallVector<Value> extraIndices;
};

/// Recursively search for a StoreMarkerOp through UntileOp and fusion ops.
/// Returns the StoreMarkerOp if found, or nullptr otherwise.
static StoreMarkerOp findStoreMarkerOp(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return nullptr;

  // Check if this is a StoreMarkerOp
  if (auto storeMarkerOp = dyn_cast<StoreMarkerOp>(defOp))
    return storeMarkerOp;

  // UntileOp: look at its source
  if (auto untileOp = dyn_cast<UntileOp>(defOp))
    return findStoreMarkerOp(untileOp.getSource());

  // Fusion op: recursively search operands
  if (isFusionOp(defOp)) {
    for (Value operand : defOp->getOperands()) {
      if (StoreMarkerOp found = findStoreMarkerOp(operand))
        return found;
    }
  }

  return nullptr;
}

/// Recursively convert a full-tensor value to its tile equivalent.
/// Handles UntileOp, StoreMarkerOp and fusion ops.
/// Also collects StoreMarkerOp info if found.
static Value convertToTile(OpBuilder &builder, Location loc, Value fullVal,
                           Type tileType, IRMapping &fullToTileMapping,
                           StoreMarkerInfo *outStoreMarkerInfo = nullptr) {
  // Check if already converted
  if (fullToTileMapping.contains(fullVal))
    return fullToTileMapping.lookup(fullVal);
  
  Operation *defOp = fullVal.getDefiningOp();
  if (!defOp) {
    // Block argument - shouldn't happen
    return fullVal;
  }

  // UntileOp: recurse into its source
  if (auto untileOp = dyn_cast<UntileOp>(defOp)) {
    Value tile = convertToTile(builder, loc, untileOp.getSource(), tileType,
                               fullToTileMapping, outStoreMarkerInfo);
    fullToTileMapping.map(fullVal, tile);
    return tile;
  }

  // StoreMarkerOp: extract info and return the tile source
  if (auto storeMarkerOp = dyn_cast<StoreMarkerOp>(defOp)) {
    if (outStoreMarkerInfo && !outStoreMarkerInfo->storeMarkerOp) {
      outStoreMarkerInfo->storeMarkerOp = storeMarkerOp;
      outStoreMarkerInfo->extraViews = storeMarkerOp.getExtraViews();
      outStoreMarkerInfo->extraIndices =
          llvm::to_vector(storeMarkerOp.getExtraIndices());
    }
    Value tile = storeMarkerOp.getSource();
    fullToTileMapping.map(fullVal, tile);
    return tile;
  }

  // Fusion op: recursively convert operands and clone with tile types
  if (isFusionOp(defOp) && defOp->getNumResults() == 1) {
    IRMapping fusionMapping;
    for (Value operand : defOp->getOperands()) {
      Value tileOperand = convertToTile(builder, loc, operand, tileType,
                                        fullToTileMapping, outStoreMarkerInfo);
      fusionMapping.map(operand, tileOperand);
    }
    
    // Clone the fusion op with tile operands
    Operation *clonedOp = builder.clone(*defOp, fusionMapping);
    clonedOp->getResult(0).setType(tileType);
    
    fullToTileMapping.map(fullVal, clonedOp->getResult(0));
    return clonedOp->getResult(0);
  }
  
  // Other ops - return as-is (shouldn't happen for well-formed IR)
  return fullVal;
}

/// Get tile shape from a StoreMarkerOp source type.
static ArrayRef<int64_t>
getShapeFromStoreMarkerOp(StoreMarkerOp storeMarkerOp) {
  auto sourceType = cast<RankedTensorType>(storeMarkerOp.getSource().getType());
  ArrayRef<int64_t> shape = sourceType.getShape();
  return shape;
}

struct RockLowerStoresPass
    : public rock::impl::RockLowerStoresPassBase<RockLowerStoresPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockLowerStoresPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
      !funcOp->hasAttr(rock::KernelLegacyAttr::getMnemonic())) {
    return;
  }

  // Collect all rock.store ops
  SmallVector<StoreOp> storeOps;
  funcOp.walk([&](StoreOp storeOp) {
    storeOps.push_back(storeOp);
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << storeOps.size()
                          << " rock.store ops to process\n");

  for (StoreOp storeOp : storeOps) {
    OpBuilder builder(storeOp);
    Location loc = storeOp.getLoc();
    
    Value storeSource = storeOp.getSource();  // The fused result (full tensor)
    Value storeDest = storeOp.getDest();      // The destination (transformed arg)

    // Find the StoreMarkerOp in the store chain to get transforms and indices
    StoreMarkerOp storeMarkerOp = findStoreMarkerOp(storeSource);
    if (!storeMarkerOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "No StoreMarkerOp found for store: " << storeOp << "\n");
      continue;
    }

    auto storeMarkerShape = getShapeFromStoreMarkerOp(storeMarkerOp);

    // Get the output type for determining tile type
    auto outputType = cast<RankedTensorType>(storeSource.getType());

    // Determine tile type
    auto tileType =
        RankedTensorType::get(storeMarkerShape, outputType.getElementType());

    // Convert the store source from full tensor to tile, collecting
    // StoreMarkerOp info
    StoreMarkerInfo storeMarkerInfo;
    IRMapping fullToTileMapping;
    Value fusedTile = convertToTile(builder, loc, storeSource, tileType,
                                    fullToTileMapping, &storeMarkerInfo);

    if (!storeMarkerInfo.storeMarkerOp) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to extract StoreMarkerOp info\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Converted store source to tile: " << fusedTile
                            << "\n");

    // Combine transforms: StoreMarkerOp's extraViews are applied on top of
    // any existing transforms on storeDest.
    // Use untransform to get existing transforms, then combine with
    // StoreMarkerOp's
    SmallVector<TransformMapAttr> destTransforms;
    auto [destRoot, _] = rock::untransform(storeDest, destTransforms);

    // Build combined transforms: extraViews (from StoreMarkerOp) +
    // destTransforms
    SmallVector<Attribute> combinedTransforms;
    combinedTransforms.append(storeMarkerInfo.extraViews.begin(),
                              storeMarkerInfo.extraViews.end());
    combinedTransforms.append(destTransforms.begin(), destTransforms.end());

    // Apply combined transforms to the root destination
    Value combinedDest = destRoot;
    if (!combinedTransforms.empty()) {
      ArrayAttr transformsAttr = builder.getArrayAttr(combinedTransforms);
      combinedDest = rock::transform(builder, destRoot, transformsAttr);
    }

    // Create BlockwiseStoreOp with combined transforms
    auto bstOp = BlockwiseStoreOp::create(
        builder, loc, storeOp.getResult().getType(), fusedTile, combinedDest,
        storeMarkerInfo.extraIndices, storeOp.getStoreMethod());

    LLVM_DEBUG(llvm::dbgs() << "Created BlockwiseStoreOp: " << bstOp << "\n");

    // Replace rock.store with BlockwiseStoreOp result
    storeOp.getResult().replaceAllUsesWith(bstOp.getResult());
    storeOp.erase();
  }

  // Clean up dead ops (UntileOp, StoreMarkerOp, fusion ops, etc.)
  bool changed = true;
  while (changed) {
    changed = false;

    funcOp.walk([&](UntileOp untileOp) {
      if (untileOp.getResult().use_empty()) {
        untileOp.erase();
        changed = true;
      }
    });

    funcOp.walk([&](StoreMarkerOp storeMarkerOp) {
      if (storeMarkerOp.getResult().use_empty()) {
        storeMarkerOp.erase();
        changed = true;
      }
    });

    funcOp.walk([&](BlockwiseLoadOp loadTileOp) {
      if (loadTileOp.getResult().use_empty()) {
        loadTileOp.erase();
        changed = true;
      }
    });

    funcOp.walk([&](TransformOp transformOp) {
      if (transformOp.getOutput().use_empty()) {
        transformOp.erase();
        changed = true;
      }
    });

    funcOp.walk([&](Operation *op) {
      if (isFusionOp(op) &&
          op->getNumResults() == 1 &&
          op->getResult(0).use_empty()) {
        op->erase();
        changed = true;
      }
    });
  }
}
