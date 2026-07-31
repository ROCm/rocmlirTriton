//===- DecomposeNonPow2K.cpp - decompose non-pow2 blockwise K tiles -----===//
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
// See the rock-decompose-nonpow2-k description in Passes.td for what this pass
// does and why it sits where it does in the pipeline. The notes below cover
// only what that description leaves out.
//
// The decomposition has to happen at the blockwise layer, not alongside the M
// and N tiles in DecomposeNonPow2Tiles, because K segments cannot be expressed
// on gridwise ops at all: doing so would mean two gridwise_gemms joined by an
// elementwise op, which the output-fusion passes cannot represent since they
// assume a single fusion root per store chain.
//
// The per-segment operands are rebuilt rather than analysed. rock.load_marker
// retains everything the rock::loadTile call that made it was given, so the
// pass reads that back (getLoadTileRecipe), slices the source down to the
// segment (sliceBlockedDims, shared with DecomposeNonPow2Tiles) and calls
// loadTile again on the slice with the segment's K.
//
// Matching on rock.blockwise_gemm rather than on a gemm-specific op means the
// same rewrite covers attention's first GEMM, whose K loop is built with the
// same loadTile / blockwise_gemm helpers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKDECOMPOSENONPOW2KPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-decompose-nonpow2-k"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockDecomposeNonPow2KPass
    : public rock::impl::RockDecomposeNonPow2KPassBase<
          RockDecomposeNonPow2KPass> {
  void runOnOperation() override;
};

/// The arguments a rock::loadTile call was made with, plus the K iteration
/// count that loadTile derived from them, as recovered from the
/// rock.load_marker it left behind.
struct LoadTileRecipe {
  Value source;
  Value kIter;
  layout::GridCoordinates gridCoords;
  SmallVector<int64_t, 3> bidGridLengths;
  int64_t kPerBlock;
  int64_t dPerBlock;
  int64_t kIterations;
  CacheModifier cache;
};

} // end anonymous namespace

static bool isSubByte(Type t) {
  return t.isIntOrFloat() && t.getIntOrFloatBitWidth() < 8;
}

/// Recover the arguments of the rock::loadTile call that produced `marker`, so
/// that the load can be re-issued over a view of the same source.
static FailureOr<LoadTileRecipe> getLoadTileRecipe(LoadMarkerOp marker,
                                                   bool isKFirst) {
  // loadTile leaves behind exactly one view, the one getLoadRegsAsTileViews
  // builds over ["k_loop", "g_block", "m_block", "n_block", <the two tile
  // dims>], and indexes it with [kIter, g_block, m_block, n_block].
  ArrayAttr views = marker.getExtraViews();
  ValueRange indices = marker.getExtraIndices();
  if (views.size() != 1 || indices.size() != 4)
    return failure();
  auto tiling = dyn_cast<TransformMapAttr>(views[0]);
  if (!tiling)
    return failure();
  // The verifier already ties the view's bounds to the marker: the upper ones
  // to the tile shape and the index count, the lower ones to the source. So the
  // bounds are the only thing left to read, and requiring six upper and three
  // lower ones is what pins the view down to the tiling of a [G, K, D] /
  // [G, D, K] matrix above.
  ArrayRef<int64_t> upper = tiling.getUpperBounds();
  if (upper.size() != 6 || tiling.getLowerBounds().size() != 3)
    return failure();

  LoadTileRecipe recipe;
  recipe.source = marker.getSource();
  recipe.kIter = indices[0];
  recipe.gridCoords =
      layout::GridCoordinates{indices[1], indices[2], indices[3]};
  recipe.kIterations = upper[0];
  recipe.bidGridLengths.assign(upper.begin() + 1, upper.begin() + 4);
  recipe.kPerBlock = isKFirst ? upper[4] : upper[5];
  recipe.dPerBlock = isKFirst ? upper[5] : upper[4];
  recipe.cache = marker.getCacheModifier();
  return recipe;
}

/// View of `recipe`'s operand keeping only the `seg` slice of every K tile.
static Value sliceKSegment(OpBuilder &b, Location loc,
                           const LoadTileRecipe &recipe, Pow2Segment seg,
                           bool isKFirst) {
  return sliceBlockedDims(b, loc, recipe.source,
                          /*sliceDims=*/{isKFirst ? 1u : 2u},
                          /*blocks=*/{recipe.kIterations},
                          /*tiles=*/{recipe.kPerBlock}, {seg});
}

//===----------------------------------------------------------------------===//
// blockwise_gemm decomposition
//===----------------------------------------------------------------------===//

static LogicalResult decomposeBlockwiseGemm(BlockwiseGemmOp gemm,
                                            ArrayRef<Pow2Segment> kSegs) {
  Location loc = gemm.getLoc();
  Type elemTypeA =
      cast<RankedTensorType>(gemm.getMatrixA().getType()).getElementType();
  Type elemTypeB =
      cast<RankedTensorType>(gemm.getMatrixB().getType()).getElementType();

  if (gemm.getMatrixScaleA() || gemm.getMatrixScaleB())
    return gemm.emitOpError("non-power-of-two K tile is not supported for "
                            "scaled gemm (should have been rejected by affix)");

  auto markerA = gemm.getMatrixA().getDefiningOp<LoadMarkerOp>();
  auto markerB = gemm.getMatrixB().getDefiningOp<LoadMarkerOp>();
  if (!markerA || !markerB) {
    rock::markAsNotApplicable(gemm);
    return gemm.emitOpError("non-power-of-two K tile requires both operands to "
                            "be loaded by rock.load_marker");
  }

  FailureOr<LoadTileRecipe> maybeA =
      getLoadTileRecipe(markerA, /*isKFirst=*/false);
  FailureOr<LoadTileRecipe> maybeB =
      getLoadTileRecipe(markerB, /*isKFirst=*/true);
  if (failed(maybeA) || failed(maybeB))
    return gemm.emitOpError(
        "could not recover the operand tiling of a non-power-of-two K tile "
        "(rock.load_marker does not have the shape rock::loadTile produces)");

  LoadTileRecipe recipeA = std::move(*maybeA);
  LoadTileRecipe recipeB = std::move(*maybeB);

  if (recipeA.kPerBlock != recipeB.kPerBlock)
    return gemm.emitOpError("operands of a non-power-of-two K tile must share "
                            "their K tile, but got ")
           << recipeA.kPerBlock << " and " << recipeB.kPerBlock;

  // Decomposing packs each segment on its own, which LegalizeFloatTypes
  // cannot do for sub-byte operands. The fused load types matter as much as
  // the tile types: an operand may be f16 here yet be dequantized from an i4
  // kernel argument, in which case the same packing runs.
  FailureOr<Type> loadTypeA = getInputFusionElementType(recipeA.source);
  FailureOr<Type> loadTypeB = getInputFusionElementType(recipeB.source);
  if (failed(loadTypeA) || failed(loadTypeB))
    return gemm.emitOpError("could not determine the underlying operand data "
                            "types of a non-power-of-two K tile");
  if (isSubByte(elemTypeA) || isSubByte(elemTypeB) || isSubByte(*loadTypeA) ||
      isSubByte(*loadTypeB)) {
    rock::markAsNotApplicable(gemm);
    return gemm.emitOpError(
        "non-power-of-two K tile is not supported for sub-byte operands");
  }

  // This is due to a bug in Triton, reported here:
  // https://github.com/ROCm/triton/issues/958
  constexpr int64_t minIntKSegment = 4;
  if (isa<IntegerType>(elemTypeA) && isa<IntegerType>(elemTypeB) &&
      llvm::any_of(kSegs, [](const Pow2Segment &seg) {
        return seg.length < minIntKSegment;
      })) {
    rock::markAsNotApplicable(gemm);
    return gemm.emitOpError("non-power-of-two K tile is not supported for "
                            "integer operands when it decomposes into K "
                            "segments narrower than ")
           << minIntKSegment;
  }

  LLVM_DEBUG(llvm::dbgs() << "decomposing K tile " << recipeA.kPerBlock
                          << " into " << kSegs.size() << " segments\n");

  // Materialize every segment's operand view up front, anchored at its source
  // so the views end up where the un-sliced view already sat, i.e. outside the
  // K loop. Successive insertions on one builder keep their relative order, so
  // one builder per operand emits that operand's views in segment order.
  OpBuilder sliceA(gemm.getContext());
  OpBuilder sliceB(gemm.getContext());
  sliceA.setInsertionPointAfterValue(recipeA.source);
  sliceB.setInsertionPointAfterValue(recipeB.source);
  SmallVector<Value> aViews, bViews;
  for (Pow2Segment seg : kSegs) {
    aViews.push_back(sliceKSegment(sliceA, loc, recipeA, seg,
                                   /*isKFirst=*/false));
    bViews.push_back(sliceKSegment(sliceB, loc, recipeB, seg,
                                   /*isKFirst=*/true));
  }

  // Emit each segment's loads and gemm where the original blockwise_gemm sat.
  // An operand whose load was hoisted out of the K loop keeps its own position
  // instead, so that it stays hoisted: attention prefetches Q outside the loop
  // when the head dimension fits in a single tile.
  auto anchorFor = [&](LoadMarkerOp marker) -> Operation * {
    return marker->getBlock() == gemm->getBlock() ? gemm.getOperation()
                                                  : marker.getOperation();
  };
  OpBuilder bA(anchorFor(markerA));
  OpBuilder bB(anchorFor(markerB));
  OpBuilder bGemm(gemm);

  Value acc = gemm.getMatrixC();
  for (auto [segIdx, seg] : llvm::enumerate(kSegs)) {
    Value loadedB = rock::loadTile(bB, loc, bViews[segIdx], recipeB.kIter, "n",
                                   recipeB.gridCoords, seg.length,
                                   recipeB.dPerBlock, /*isKFirst=*/true,
                                   recipeB.bidGridLengths, recipeB.cache);
    Value loadedA = rock::loadTile(bA, loc, aViews[segIdx], recipeA.kIter, "m",
                                   recipeA.gridCoords, seg.length,
                                   recipeA.dPerBlock, /*isKFirst=*/false,
                                   recipeA.bidGridLengths, recipeA.cache);
    acc = BlockwiseGemmOp::create(
        bGemm, loc, loadedA, loadedB, acc, /*matrixScaleA=*/nullptr,
        /*matrixScaleB=*/nullptr, gemm.getQuantBlockSizeAttr(),
        gemm.getMatrixAOrigElemTypeAttr(), gemm.getMatrixBOrigElemTypeAttr(),
        gemm.getMatrixAKPackAttr(), gemm.getMatrixBKPackAttr());
  }

  gemm.getResult().replaceAllUsesWith(acc);
  gemm.erase();
  return success();
}

void RockDecomposeNonPow2KPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic())) {
    LLVM_DEBUG(llvm::dbgs()
               << "skipping " << func.getSymName() << ": not a rock.kernel\n");
    return;
  }

  // Anchor on the K tile of the blockwise gemm itself, which is exactly the
  // tile the Triton layouts cannot represent when it is not a power of two.
  SmallVector<BlockwiseGemmOp> targets;
  func.walk([&](BlockwiseGemmOp gemm) {
    if (!llvm::isPowerOf2_64(gemm.getKDim()))
      targets.push_back(gemm);
  });

  for (BlockwiseGemmOp gemm : targets) {
    SmallVector<Pow2Segment> kSegs = decomposePow2(gemm.getKDim());
    if (failed(decomposeBlockwiseGemm(gemm, kSegs)))
      return signalPassFailure();
  }
}
