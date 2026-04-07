//===- InsertOutputFusionLoads.cpp - Insert BlockwiseLoadOp for output fusions
//===//
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
// This pass runs AFTER GridwiseGemmToBlockwise and BEFORE RockLowerLoads.
// It creates rock.load_marker + rock.untile for extra fusion inputs in output
// fusions.
//
// Extra fusion inputs are operands of fusion ops (arith/math) that are NOT
// part of the GEMM-result chain — e.g., a bias tensor added to the GEMM
// output. These come from func block arguments through rock.transform chains.
//
// Example:
//   Before (after GridwiseGemmToBlockwise):
//     %result = scf.for ... {
//       %loadedA = rock.blockwise_load %a[indices]
//       ...
//     }
//     %fusionRoot = rock.store_marker %result views [...] [%g, %m, %n]
//                     : tensor<64x64xf16> -> tensor<1x128x128xf16>
//     %bias_t = rock.transform %bias_arg
//     %fused = arith.addf %fusionRoot, %bias_t
//     %out = rock.store %fused to %dest
//
//   After:
//     %result = scf.for ... { ... }
//     %fusionRoot = rock.store_marker %result views [...] [%g, %m, %n]
//                     : tensor<64x64xf16> -> tensor<1x128x128xf16>
//     %bias_t = rock.transform %bias_arg
//     %bias_tile = rock.load_marker %bias_t views [<output_tiling>] [%g, %m,
//     %n] %bias_full = rock.untile %bias_tile : tile -> full %fused =
//     arith.addf %fusionRoot, %bias_full %out = rock.store %fused to %dest
//
// The chain block_argument -> transform(s) -> rock.load_marker -> rock.untile
// allows LowerLoads to trace back to the block argument and combine the
// pre-existing transforms with the output tiling transforms.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKINSERTOUTPUTFUSIONLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-insert-output-fusion-loads"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockInsertOutputFusionLoadsPass
    : public rock::impl::RockInsertOutputFusionLoadsPassBase<
          RockInsertOutputFusionLoadsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockInsertOutputFusionLoadsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // Step 1: Find the fusion root StoreMarkerOp. This is the one created
  // by ToBlockwise — it has non-empty extraViews (output
  // transforms) and extraIndices (grid coordinates).

  SmallVector<StoreMarkerOp> storeMarkers;
  funcOp.walk([&](StoreMarkerOp op) { storeMarkers.push_back(op); });

  if (storeMarkers.empty()) {
    funcOp->emitError("No StoreMarkerOp found");
    return signalPassFailure();
  }

  for (StoreMarkerOp storeMarker : storeMarkers) {
    // Extract output transforms and grid coordinates directly from the GEMM's
    // StoreMarkerOp — no need to scrape them from BlockwiseLoadOps inside the
    // loop.
    ArrayAttr outputViews = storeMarker.getExtraViews();
    SmallVector<Value> gridCoords(storeMarker.getExtraIndices());

    LLVM_DEBUG(llvm::dbgs() << "StoreMarkerOp: " << storeMarker << "\n");

    // Step 2: Collect extra fusion inputs (operands of fusion ops that are NOT
    // in the root-result chain).
    auto fusionInputMap =
        rock::collectFusionExtraInputs(storeMarker.getResult());

    if (fusionInputMap.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No extra fusion inputs found\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Found " << fusionInputMap.size()
                            << " unique extra fusion inputs\n");

    // Step 3: For each unique extra input, create a LoadMarkerOp:
    //   original_value -> rock.transform(outputViews) -> rock.load_marker
    //                  -> (full tensor result, used by fusion ops)
    // LowerLoads will later convert this into an actual BlockwiseLoadOp
    // (plus UntileOp to bridge back to the full tensor type).
    // Then update the map so replaceFusionExtraInputs can wire them in.
    OpBuilder builder(funcOp.getContext());
    Location loc = storeMarker.getLoc();

    for (auto &[originalVal, mappedVal] : fusionInputMap) {
      // Find the latest-defined operation among all values we'll reference
      // (originalVal, storeMarker, gridCoords) to maintain SSA dominance.
      // The extra input may be defined before or after the GEMM store marker.
      Operation *latestOp = storeMarker.getOperation();
      if (auto *defOp = originalVal.getDefiningOp())
        if (latestOp->isBeforeInBlock(defOp))
          latestOp = defOp;
      for (Value coord : gridCoords)
        if (auto *defOp = coord.getDefiningOp())
          if (latestOp->isBeforeInBlock(defOp))
            latestOp = defOp;
      builder.setInsertionPointAfter(latestOp);

      // Apply the same output transforms used by the GEMM tile.
      Value wrappedSource = transform(builder, originalVal, outputViews);

      // Determine tile type (last 2 dimensions of the transformed shape).
      auto sourceType = cast<RankedTensorType>(wrappedSource.getType());
      auto wrappedShape = sourceType.getShape();
      int64_t numElements = wrappedShape.size() - gridCoords.size();
      assert(numElements > 0);
      auto tileType = RankedTensorType::get(wrappedShape.take_back(numElements),
                                            sourceType.getElementType());

      // Create LoadMarkerOp with the GEMM's grid coordinates.
      auto markerOp = LoadMarkerOp::create(builder, loc, tileType, originalVal,
                                           outputViews, gridCoords);

      // Create UntileOp to map tile back to the original full tensor type.
      // LowerStores will strip these when converting back to tile operations.
      auto untileOp = UntileOp::create(builder, loc, originalVal.getType(),
                                       markerOp.getResult());

      // Update the map: original -> UntileOp result.
      mappedVal = untileOp.getResult();

      LLVM_DEBUG(llvm::dbgs() << "Created LoadMarkerOp for extra fusion input: "
                              << markerOp << "\n"
                              << "  with UntileOp: " << untileOp << "\n");
    }

    // Step 4: Replace extra input operands in fusion ops with the new values.
    rock::replaceFusionExtraInputs(storeMarker.getResult(), fusionInputMap);
  }
}
