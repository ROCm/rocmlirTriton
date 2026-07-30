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
// The K tile size kPerBlock may legally be non-power-of-two, which
// lets the tuner pick a tile that divides K evenly (48 divides a K of 576,
// where the next power of two down, 32, does not). The Triton layouts produced
// later by RockToTTIR need power-of-two shapes, so such a tile has to be
// decomposed before then.
//
// DecomposeNonPow2Tiles does the equivalent job for the M and N tiles, but it
// cannot do K at the same layer. M and N segments are independent output cells,
// so they can become separate gridwise_gemms; K segments all reduce into one
// accumulator, and expressing that on gridwise ops would mean two
// gridwise_gemms combined by an elementwise op, which the output-fusion passes
// cannot represent (they assume a single fusion root per store chain).
//
// So this pass runs one step later, after Gridwise{Gemm,Attn}ToBlockwise, where
// both the K loop and the accumulator exist. For a rock.blockwise_gemm whose
// K tile is not a power of two it splits that tile into power-of-two segments
// (48 -> {32, 16}) and emits one blockwise_gemm per segment, threading the
// accumulator through them so every segment of a K iteration reduces into the
// same tile:
//
//   %acc1 = rock.blockwise_gemm(%aSeg0, %bSeg0, %acc)   // K = 32
//   %acc2 = rock.blockwise_gemm(%aSeg1, %bSeg1, %acc1)  // K = 16
//
// The per-segment operands are rebuilt rather than analysed. rock.load_marker
// retains the original un-tiled source along with the tiling views, the loop
// and grid indices and the cache modifier, so the pass slices the source to the
// segment (sliceBlockedDims, shared with DecomposeNonPow2Tiles) and calls
// rock::loadTile again with the segment's K. Everything else loadTile needs, in
// particular the K iteration count and the M/N block count, it re-derives from
// the sliced source shape.
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

/// Everything rock::loadTile needs in order to re-issue an operand load,
/// recovered from the rock.load_marker that Gridwise{Gemm,Attn}ToBlockwise left
/// behind.
struct LoadRecipe {
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

//===----------------------------------------------------------------------===//
// Recovering the load parameters
//===----------------------------------------------------------------------===//

/// Recover the arguments of the rock::loadTile call that produced `marker`.
/// `isKFirst` tells whether the source is laid out [G, K, D] (the B operand) or
/// [G, D, K] (the A operand), which decides how the tile and source shapes are
/// read. Fails if `marker` does not have the shape loadTile produces.
static FailureOr<LoadRecipe> getLoadRecipe(LoadMarkerOp marker, bool isKFirst) {
  // getLoadRegsAsTileViews emits exactly one map, whose upper bounds are
  // [k_loop, g_block, m_block, n_block, <the two tile dims>], and loadTile
  // indexes it with [kIter, g_block, m_block, n_block].
  ArrayAttr views = marker.getExtraViews();
  ValueRange indices = marker.getExtraIndices();
  if (views.size() != 1 || indices.size() != 4)
    return failure();
  auto map = dyn_cast<TransformMapAttr>(views[0]);
  if (!map)
    return failure();
  ArrayRef<int64_t> upper = map.getUpperBounds();
  if (upper.size() != 6)
    return failure();

  auto tileType = dyn_cast<RankedTensorType>(marker.getResult().getType());
  auto sourceType = dyn_cast<RankedTensorType>(marker.getSource().getType());
  if (!tileType || tileType.getRank() != 2 || !sourceType ||
      sourceType.getRank() != 3)
    return failure();
  ArrayRef<int64_t> tile = tileType.getShape();
  ArrayRef<int64_t> source = sourceType.getShape();

  LoadRecipe recipe;
  recipe.source = marker.getSource();
  recipe.kIter = indices[0];
  recipe.gridCoords =
      layout::GridCoordinates{indices[1], indices[2], indices[3]};
  recipe.bidGridLengths.assign(upper.begin() + 1, upper.begin() + 4);
  recipe.kPerBlock = isKFirst ? tile[0] : tile[1];
  recipe.dPerBlock = isKFirst ? tile[1] : tile[0];
  recipe.cache = marker.getCacheModifier();

  int64_t kGlobal = isKFirst ? source[1] : source[2];
  if (recipe.kPerBlock <= 0 || kGlobal % recipe.kPerBlock != 0)
    return failure();
  recipe.kIterations = kGlobal / recipe.kPerBlock;
  // The recovered iteration count must agree with the one baked into the view,
  // otherwise the marker was not built by loadTile the way we assume.
  if (recipe.kIterations != upper[0])
    return failure();
  return recipe;
}

/// A builder that inserts immediately after `v`'s definition, or at the top of
/// its block when `v` is a block argument. Successive insertions keep their
/// relative order, so one such builder per operand emits the segment views in
/// segment order.
static OpBuilder builderAfter(Value v) {
  if (Operation *def = v.getDefiningOp())
    return OpBuilder(def->getBlock(), std::next(def->getIterator()));
  Block *owner = cast<BlockArgument>(v).getOwner();
  return OpBuilder(owner, owner->begin());
}

/// View of `recipe`'s operand keeping only the `seg` slice of every K tile.
static Value sliceKSegment(OpBuilder &b, Location loc, const LoadRecipe &recipe,
                           Pow2Segment seg, bool isKFirst) {
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

  FailureOr<LoadRecipe> maybeA = getLoadRecipe(markerA, /*isKFirst=*/false);
  FailureOr<LoadRecipe> maybeB = getLoadRecipe(markerB, /*isKFirst=*/true);
  if (failed(maybeA) || failed(maybeB))
    return gemm.emitOpError(
        "could not recover the operand tiling of a non-power-of-two K tile "
        "(rock.load_marker does not have the shape rock::loadTile produces)");

  LoadRecipe recipeA = std::move(*maybeA);
  LoadRecipe recipeB = std::move(*maybeB);
  assert(recipeA.kPerBlock == recipeB.kPerBlock &&
         "blockwise_gemm operands must share their K tile");

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
  // K loop.
  OpBuilder sliceA = builderAfter(recipeA.source);
  OpBuilder sliceB = builderAfter(recipeB.source);
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
    if (!llvm::isPowerOf2_64(gemm.getKPerBlock()))
      targets.push_back(gemm);
  });

  for (BlockwiseGemmOp gemm : targets) {
    SmallVector<Pow2Segment> kSegs = decomposePow2(gemm.getKPerBlock());
    if (failed(decomposeBlockwiseGemm(gemm, kSegs)))
      return signalPassFailure();
  }
}
