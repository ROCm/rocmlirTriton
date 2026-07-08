//===- DecomposeNonPow2Tiles.cpp - split non-pow2 gridwise tiles --------===//
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
// The per-block tile sizes live in the GEMM tuning parameters (mPerBlock /
// nPerBlock) and may legally be non-power-of-two. For example, to cover a GEMM
// with M = 77 the tuning logic may pick an mPerBlock of 80 (= 64 + 16), which
// is not a power of two. The Triton layouts produced later by RockToTTIR,
// however, require power-of-two tensor shapes, so these non-power-of-two tiles
// must be split before then.
//
// This pass runs at the *gridwise* layer: after rock.gemm has been lowered to
// rock.gridwise_gemm (so transposes/padding are resolved, the operands are
// canonical 3-D [G, M, K] / [G, K, N] views, and the func `grid_size` is set),
// but *before* the gridwise op is expanded into blockwise loads/loops/stores.
// At this layer there are no loops, no blockwise ops, no Triton ops and no
// softmax internals, so the only thing the pass has to do is replace one
// non-pow2 gridwise_gemm with a grid of pow2-tile gridwise_gemms.
//
// For a gridwise_gemm with mPerBlock=M0, nPerBlock=N0 the M0/N0 tiles are split
// into power-of-two segments (decomposePow2: e.g. 80 -> {64,16}). For each
// (mSeg, nSeg) cell we emit a sub-gridwise_gemm whose:
//
//   * A operand is a view of A with M restructured per block and sliced to the
//     M-segment (so the sub-op's M' = mBlocks * mSeg.length); shared across N;
//   * B operand is the analogous N-slice of B; shared across M;
//   * params have mPerBlock = mSeg.length, nPerBlock = nSeg.length (pow2);
//   * output / output-fusion / store side is replicated per cell, with every
//     extra fusion input sliced the same way and one rock.store per cell into
//     the sliced output view.
//
// The slice restructures the dimension being split, D is M (for an M-segment)
// or N (for an N-segment), with D = blocks * tile into (block, iter), slices
// the iter sub-dim to the segment, and re-merges, so block `b` of the sub-op's
// dimension maps onto rows `b*tile + seg.offset + i` of the original. Because
// every sub-op keeps the same mBlocks/nBlocks/G, GridwiseGemmToBlockwise
// derives an identical grid layout and identical K-loop bounds for all of them,
// and the func `grid_size` is left untouched.
//
// Input fusion needs no handling: it lives on the A/B views and is tiled per
// block downstream, so once the sub-op's tile is pow2 the input-fusion tiles
// are pow2 too.
//
// NOTE: the per-cell sub-ops each lower to their own K-loop; merging those
// sibling loops back into one (via scf sibling-loop fusion + CSE) is a separate
// follow-up pass.
//
// rock.gridwise_attention is handled analogously by processGridwiseAttention:
// its two output axes -- seqLenQ (params0.mPerBlock, the grid M tile) and
// headDimV (the result N / gemm1N) -- are the decomposable axes (a head-dim
// column slice is an independent attention over sliced V/output columns sharing
// the same softmax(QK)), while seqLenK (gemm0 nPerBlock, the softmax reduction)
// and the kPerBlocks stay power-of-two. Each (mSeg, nSeg) cell becomes a
// sub-gridwise_attention over sliced queries (M) / values (head dim) / pre-
// softmax inputs (M) / output / LSE, with the preSoftmax region cloned in.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKDECOMPOSENONPOW2TILESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-decompose-nonpow2-tiles"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockDecomposeNonPow2TilesPass
    : public rock::impl::RockDecomposeNonPow2TilesPassBase<
          RockDecomposeNonPow2TilesPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// OutputSplitter: per-cell replication of the output-fusion DAG
//===----------------------------------------------------------------------===//

namespace {

/// Materializes, per (m,n) cell, the value of any tile-shaped output value
/// reachable from the gridwise_gemm result through output fusion. The gemm
/// result grid is seeded; fusion ops are cloned per cell, splat constants are
/// re-splatted, and any other leaf (an extra fusion input, a full [G,M,N] view)
/// is block-sliced like the output. Cells are laid out row-major
/// `m * nSegs + n`.
class OutputSplitter {
public:
  // `outRank` is the rank of the values being split: 3 for the [G, M, N] main
  // output, 2 for the M-only [G, M] LSE (in which case `nSegs` must be a single
  // trivial segment so there is one cell per M segment and no N slicing).
  OutputSplitter(OpBuilder &b, Location loc, int64_t g, int64_t mBlocks,
                 int64_t nBlocks, int64_t mPerBlock, int64_t nPerBlock,
                 ArrayRef<Pow2Segment> mSegs, ArrayRef<Pow2Segment> nSegs,
                 unsigned outRank = 3)
      : b(b), loc(loc), g(g), mBlocks(mBlocks), nBlocks(nBlocks),
        mPerBlock(mPerBlock), nPerBlock(nPerBlock), mSegs(mSegs), nSegs(nSegs),
        outRank(outRank) {}

  void seed(Value v, SmallVector<Value> grid) { memo[v] = std::move(grid); }

  int64_t numCells() const {
    return static_cast<int64_t>(mSegs.size() * nSegs.size());
  }

  /// Cell shape for cell `cell`: [G, mBlocks*mSeg.len] and, when rank 3, a
  /// trailing [nBlocks*nSeg.len].
  SmallVector<int64_t> cellShape(int64_t cell) const {
    int64_t i = cell / static_cast<int64_t>(nSegs.size());
    int64_t j = cell % static_cast<int64_t>(nSegs.size());
    SmallVector<int64_t> shape = {g, mBlocks * mSegs[i].length};
    if (outRank == 3)
      shape.push_back(nBlocks * nSegs[j].length);
    return shape;
  }

  /// Block-slice the full [G,M,N] value `v` for cell `cell`.
  Value sliceCell(Value v, int64_t cell) {
    int64_t i = cell / static_cast<int64_t>(nSegs.size());
    int64_t j = cell % static_cast<int64_t>(nSegs.size());
    SmallVector<unsigned> dims;
    SmallVector<int64_t> blocks, tiles;
    SmallVector<Pow2Segment> segs;
    if (mSegs[i].length != mPerBlock) {
      dims.push_back(1);
      blocks.push_back(mBlocks);
      tiles.push_back(mPerBlock);
      segs.push_back(mSegs[i]);
    }
    if (nSegs[j].length != nPerBlock) {
      dims.push_back(2);
      blocks.push_back(nBlocks);
      tiles.push_back(nPerBlock);
      segs.push_back(nSegs[j]);
    }
    if (dims.empty())
      return v;
    return sliceBlockedDims(b, loc, v, dims, blocks, tiles, segs);
  }

  FailureOr<SmallVector<Value>> split(Value v) {
    if (auto it = memo.find(v); it != memo.end())
      return it->second;

    auto type = dyn_cast<RankedTensorType>(v.getType());
    if (!type)
      return failure();
    Operation *def = v.getDefiningOp();

    SmallVector<Value> grid;
    if (auto cst = dyn_cast_or_null<arith::ConstantOp>(def)) {
      if (failed(splitConstant(cst, type, grid)))
        return failure();
    } else if (def && isFusionOp(def)) {
      if (failed(splitFusion(def, grid)))
        return failure();
    } else {
      // Leaf: an extra fusion input (a full [G,M,N] or [G,M] view).
      // Block-slice it.
      if (type.getRank() != static_cast<int64_t>(outRank))
        return failure();
      for (int64_t cell = 0; cell < numCells(); ++cell)
        grid.push_back(sliceCell(v, cell));
    }
    memo[v] = grid;
    return memo[v];
  }

private:
  LogicalResult splitConstant(arith::ConstantOp cst, RankedTensorType type,
                              SmallVector<Value> &grid) {
    auto splat = dyn_cast<SplatElementsAttr>(cst.getValue());
    if (!splat)
      return failure();
    Attribute elem = splat.getSplatValue<Attribute>();
    for (int64_t cell = 0; cell < numCells(); ++cell) {
      auto subTy =
          RankedTensorType::get(cellShape(cell), type.getElementType());
      grid.push_back(arith::ConstantOp::create(
          b, loc, subTy, SplatElementsAttr::get(subTy, elem)));
    }
    return success();
  }

  LogicalResult splitFusion(Operation *op, SmallVector<Value> &grid) {
    assert(op->getNumResults() == 1 &&
           "fusion ops with multiple results are not supported");
    SmallVector<SmallVector<Value>> operandGrids;
    for (Value operand : op->getOperands()) {
      FailureOr<SmallVector<Value>> operandGrid = split(operand);
      if (failed(operandGrid))
        return failure();
      operandGrids.push_back(*operandGrid);
    }
    for (const auto &operandGrid : operandGrids)
      if (static_cast<int64_t>(operandGrid.size()) != numCells())
        return failure();

    for (int64_t cell = 0; cell < numCells(); ++cell) {
      IRMapping m;
      for (auto [oi, operand] : llvm::enumerate(op->getOperands()))
        m.map(operand, operandGrids[oi][cell]);
      Operation *cloned = b.clone(*op, m);
      for (OpResult res : cloned->getResults()) {
        auto rt = cast<RankedTensorType>(res.getType());
        res.setType(
            RankedTensorType::get(cellShape(cell), rt.getElementType()));
      }
      grid.push_back(cloned->getResult(0));
    }
    return success();
  }

  OpBuilder &b;
  Location loc;
  int64_t g, mBlocks, nBlocks, mPerBlock, nPerBlock;
  ArrayRef<Pow2Segment> mSegs, nSegs;
  unsigned outRank;
  DenseMap<Value, SmallVector<Value>> memo;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Store replication shared by gridwise_gemm / gridwise_attention splitting
//===----------------------------------------------------------------------===//

/// Returns `stores` in program order (by walk order of the enclosing function)
/// so explicit resultAlias chains thread through the decomposed sub-stores
/// correctly.
static SmallVector<Operation *>
orderStoresInProgramOrder(Operation *anchor, const SetVector<StoreOp> &stores) {
  func::FuncOp func = anchor->getParentOfType<func::FuncOp>();
  DenseMap<Operation *, unsigned> opOrder;
  unsigned nextOpOrder = 0;
  func.walk([&](Operation *op) { opOrder.try_emplace(op, nextOpOrder++); });

  SmallVector<Operation *> ordered;
  ordered.reserve(stores.size());
  for (StoreOp st : stores)
    ordered.push_back(st.getOperation());
  llvm::sort(ordered, [&](Operation *lhs, Operation *rhs) {
    return opOrder.lookup(lhs) < opOrder.lookup(rhs);
  });
  return ordered;
}

/// Replicates each store in `orderedStores` across every cell of `splitter`:
/// the store source is split into a per-cell grid, the destination views are
/// re-rooted at the original destination and sliced per cell, and the
/// resultAlias is threaded through the emitted sub-store chain. `orderedStores`
/// must already be in program order (see orderStoresInProgramOrder). `anchor`
/// and `what` are used only for diagnostics; `what` is an optional infix (e.g.
/// "LSE ") spliced into the "failed to split <what>store source" error.
static LogicalResult replicateStoresPerCell(OpBuilder &b, Location loc,
                                            OutputSplitter &splitter,
                                            ArrayRef<Operation *> orderedStores,
                                            Operation *anchor, StringRef what) {
  for (Operation *storeOp : orderedStores) {
    auto store = cast<StoreOp>(storeOp);
    b.setInsertionPoint(store);
    FailureOr<SmallVector<Value>> srcGrid = splitter.split(store.getSource());
    if (failed(srcGrid))
      return anchor->emitError("rock-decompose-nonpow2-tiles: failed to split ")
             << what << "store source";

    auto [destRoot, destMaps, _] = rock::untransform(b, store.getDest());
    Type storeResultType = store.getResult().getType();
    StoreMethodAttr method = store.getStoreMethodAttr();

    // Keep destination views rooted at the original destination. Only the
    // result alias advances through the sub-store chain.
    Value currentResultAlias =
        store.getResultAlias() ? store.getResultAlias() : destRoot;
    for (int64_t cell = 0; cell < splitter.numCells(); ++cell) {
      Value view = rock::transform(b, destRoot, destMaps);
      Value destCell = splitter.sliceCell(view, cell);
      auto st = StoreOp::create(b, loc, storeResultType, (*srcGrid)[cell],
                                destCell, currentResultAlias, method);
      currentResultAlias = st.getResult();
    }
    store.getResult().replaceAllUsesWith(currentResultAlias);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// gridwise_gemm splitting
//===----------------------------------------------------------------------===//

static LogicalResult processGridwiseGemm(GridwiseGemmOp gemm) {
  Location loc = gemm.getLoc();

  // Block-scaled gemms are not handled yet (scales would need to be split along
  // M/N like A/B).
  if (gemm.getScaleA() || gemm.getScaleB())
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: scaled gridwise_gemm not supported");

  GemmParamsAttr params = gemm.getParams();
  int64_t mPerBlock = params.getMPerBlock();
  int64_t nPerBlock = params.getNPerBlock();

  // This pass only peels the M and N tiles; the contraction (K) dimension is
  // left untouched here. A non-power-of-two kPerBlock rides along on the
  // sub-gemms unchanged and is peeled into power-of-two K segments downstream
  // by rock-gridwise-gemm-to-blockwise, so it needs no special handling here.
  SmallVector<Pow2Segment> mSegs = decomposePow2(mPerBlock);
  SmallVector<Pow2Segment> nSegs = decomposePow2(nPerBlock);

  Value a = gemm.getA();                                      // [G, M, K]
  Value bMat = gemm.getB();                                   // [G, K, N]
  auto cType = cast<RankedTensorType>(gemm.getC().getType()); // [G, M, N]
  if (cType.getRank() != 3)
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: expected 3-D gridwise_gemm output");
  int64_t G = cType.getShape()[0];
  int64_t M = cType.getShape()[1];
  int64_t N = cType.getShape()[2];
  assert(M % mPerBlock == 0 &&
         "gemm M dimension must be a multiple of mPerBlock");
  assert(N % nPerBlock == 0 &&
         "gemm N dimension must be a multiple of nPerBlock");
  int64_t mBlocks = M / mPerBlock;
  int64_t nBlocks = N / nPerBlock;
  Type cElemType = cType.getElementType();

  MLIRContext *ctx = gemm.getContext();
  OpBuilder b(gemm);

  // Build one sub-gridwise_gemm per (m,n) cell (row-major m*nCol + n).
  auto subParams = [&](int64_t mLen, int64_t nLen) {
    return GemmParamsAttr::get(
        ctx, mLen, nLen, params.getKPerBlock(), params.getKpack(),
        params.getNumCTAs(), params.getNumWaves(),
        params.getMatrixInstrNonkdim(), params.getSplitKFactor(),
        params.getNumStages(), params.getWavesPerEU(),
        params.getGridGroupSize(), params.getUseAsyncCopy(),
        params.getUseBlockPingpong(), params.getUseInThreadTranspose(),
        params.getUseBufferOps(), params.getUseBufferAtomics(),
        params.getUseReductionLayout(), params.getUseOptimizeEpilogue());
  };

  SmallVector<Value> resultGrid;
  for (auto [i, mSeg] : llvm::enumerate(mSegs)) {
    Value aCell =
        sliceBlockedDims(b, loc, a, {1}, {mBlocks}, {mPerBlock}, {mSeg});
    for (auto [j, nSeg] : llvm::enumerate(nSegs)) {
      Value bCell =
          sliceBlockedDims(b, loc, bMat, {2}, {nBlocks}, {nPerBlock}, {nSeg});
      auto cCellType = RankedTensorType::get(
          {G, mBlocks * mSeg.length, nBlocks * nSeg.length}, cElemType);
      auto sub = GridwiseGemmOp::create(
          b, loc, cCellType, aCell, bCell, /*scaleA=*/Value(),
          /*scaleB=*/Value(), gemm.getQuantBlockSizeAttr(),
          subParams(mSeg.length, nSeg.length));
      resultGrid.push_back(sub.getResult());
    }
  }

  // Output side: trace the gemm result to its stores through the output-fusion
  // DAG, then replicate the fusion + stores per cell.
  FailureOr<OutputsAndFusionInputs> maybeViews =
      traceOutputsAndFusionInputs(gemm.getResult());
  if (failed(maybeViews))
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: cannot trace gridwise_gemm output to "
        "rock.store");
  SetVector<StoreOp> &stores = maybeViews->stores;
  if (stores.empty())
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: no rock.store for gridwise_gemm result");

  OutputSplitter splitter(b, loc, G, mBlocks, nBlocks, mPerBlock, nPerBlock,
                          mSegs, nSegs);
  splitter.seed(gemm.getResult(), resultGrid);

  // Process stores in program order so explicit resultAlias chains thread
  // through the decomposed sub-stores correctly.
  return replicateStoresPerCell(b, loc, splitter,
                                orderStoresInProgramOrder(gemm, stores), gemm,
                                /*what=*/"");
}

//===----------------------------------------------------------------------===//
// gridwise_attention splitting
//===----------------------------------------------------------------------===//

/// Copy `base` overriding only mPerBlock / nPerBlock.
static GemmParamsAttr withMN(GemmParamsAttr base, int64_t mPerBlock,
                             int64_t nPerBlock) {
  return GemmParamsAttr::get(
      base.getContext(), mPerBlock, nPerBlock, base.getKPerBlock(),
      base.getKpack(), base.getNumCTAs(), base.getNumWaves(),
      base.getMatrixInstrNonkdim(), base.getSplitKFactor(), base.getNumStages(),
      base.getWavesPerEU(), base.getGridGroupSize(), base.getUseAsyncCopy(),
      base.getUseBlockPingpong(), base.getUseInThreadTranspose(),
      base.getUseBufferOps(), base.getUseBufferAtomics(),
      base.getUseReductionLayout(), base.getUseOptimizeEpilogue());
}

// Split a gridwise_attention along its two output axes: seqLenQ (= gemm0/gemm1
// M, params0.mPerBlock) and headDimV (= gemm1 N, the result's N dimension).
// Both axes share the same softmax(QK), so a head-dim column slice is an
// independent attention over a V-column / output-column slice; a seqLenQ tile
// slice is the usual grid M split. The softmax reduction (seqLenK = gemm0 N)
// and the contraction dims (kPerBlock) are not splittable and must stay pow2.
static LogicalResult processGridwiseAttention(GridwiseAttentionOp attn) {
  Location loc = attn.getLoc();
  OpBuilder b(attn);

  GemmParamsAttr params0 = attn.getParams0();
  GemmParamsAttr params1 = attn.getParams1();

  if (!llvm::isPowerOf2_64(params0.getNPerBlock()))
    return attn.emitError(
        "rock-decompose-nonpow2-tiles: non-power-of-two gemm0 "
        "nPerBlock (seqLenK) is not supported for attention");
  if (!llvm::isPowerOf2_64(params0.getKPerBlock()) ||
      !llvm::isPowerOf2_64(params1.getKPerBlock()))
    return attn.emitError("rock-decompose-nonpow2-tiles: non-power-of-two "
                          "kPerBlock is not supported");

  int64_t mPerBlock = params0.getMPerBlock();

  // params1's M and N need no pow2 precondition: the split normalizes both.
  // The M tile is shared between the two gemms (both are built from
  // MPerBlockG0), and withMN overwrites p0/p1's M to the pow2 mSeg.length
  // below, so params1.mPerBlock is only ever used via this shared value.
  assert(params1.getMPerBlock() == mPerBlock &&
         "gemm0/gemm1 must share the seqLenQ (M) tile");
  // params1.nPerBlock is the head-dim tile (nPerBlockG1, or the full gemm1N
  // when untiled). It is clamped to min(nPerBlockG1, nSeg.length) per cell;
  // nSeg is pow2 and nPerBlockG1 is either pow2 (tiled) or the whole head dim
  // (untiled, hence >= every nSeg.length), so the clamp always yields a pow2
  // sub-tile.

  auto qType = cast<RankedTensorType>(attn.getQueries().getType());
  auto outType = cast<RankedTensorType>(attn->getResult(0).getType());
  if (qType.getRank() != 3 || outType.getRank() != 3)
    return attn.emitError("rock-decompose-nonpow2-tiles: expected 3-D "
                          "gridwise_attention operands");

  int64_t gemm0M = qType.getShape()[1]; // padded seqLenQ
  int64_t gOut = outType.getShape()[0];
  int64_t gemm1M = outType.getShape()[1]; // padded seqLenQ (== gemm0M)
  int64_t gemm1N = outType.getShape()[2]; // padded headDimV

  SmallVector<Pow2Segment> mSegs = decomposePow2(mPerBlock);
  SmallVector<Pow2Segment> nSegs = decomposePow2(gemm1N);
  if (mSegs.size() == 1 && nSegs.size() == 1)
    return success();

  assert(gemm0M % mPerBlock == 0 && "seqLenQ must be a multiple of mPerBlock");
  assert(gemm1M == gemm0M && "gemm0 and gemm1 M must match");
  int64_t mBlocks = gemm0M / mPerBlock;

  // Forward prePadG0M unchanged; each M segment also records how it maps back
  // onto the original M (mBlocks * mPerBlock, sliced at mSeg.offset) so
  // lowering can replay the slice + pad for the first-gemm mask.
  IntegerAttr prePadG0NAttr = attn.getPrePadG0NAttr();
  IntegerAttr prePadMAttr = attn.getPrePadG0MAttr();

  Value lse = attn.getLse();
  bool hasLse = static_cast<bool>(lse);
  RankedTensorType lseType;
  if (hasLse)
    lseType = cast<RankedTensorType>(lse.getType());

  Value queries = attn.getQueries();
  Value values = attn.getValues();

  auto sliceM = [&](Value v, const Pow2Segment &mSeg) -> Value {
    if (mSeg.length == mPerBlock)
      return v;
    return sliceBlockedDims(b, loc, v, {1}, {mBlocks}, {mPerBlock}, {mSeg});
  };
  auto sliceN = [&](Value v, const Pow2Segment &nSeg) -> Value {
    if (nSeg.length == gemm1N)
      return v;
    return sliceBlockedDims(b, loc, v, {2}, {1}, {gemm1N}, {nSeg});
  };

  // Build one sub-gridwise_attention per (m,n) cell (row-major m*nCol + n). The
  // LSE is head-dim independent, so it is produced only by the n==0 cell of
  // each M-segment row.
  SmallVector<Value> resultGrid;
  SmallVector<Value> lsePerMSeg;
  for (auto [i, mSeg] : llvm::enumerate(mSegs)) {
    Value qCell = sliceM(queries, mSeg);
    // Forward the original prePadG0M unchanged. Whenever this segment narrows
    // mPerBlock (a genuine M slice), record the original tile and slice offset
    // so the lowering (decomposeTileView in GridwiseAttnToBlockwise) can replay
    // the (mBlocks, mPerBlock) restructure + slice and map this sub-op's sliced
    // M tile back onto the original full M. This is needed not only to replay
    // the pad mask, but also so any first-gemm realignment (pad + GQA undo +
    // splitKV) sees the full M extent rather than a partial slice.
    bool mIsSliced = mSeg.length != mPerBlock;
    IntegerAttr mOrigPerBlockAttr =
        mIsSliced ? b.getIndexAttr(mPerBlock) : nullptr;
    IntegerAttr mSliceOffsetAttr =
        mIsSliced ? b.getIndexAttr(mSeg.offset) : nullptr;
    // preSoftmax elementwise inputs (e.g. attention scale/bias) live in the
    // full, unpadded, pre-GQA [G, seqLenQ, seqLenK] logical space and are
    // always forwarded whole: the lowering realigns each sub-op's first-gemm
    // tile back to that same full space (decomposeTileView + unpadTileView +
    // undoGQATransforms) before applying the region, so it indexes the full
    // input directly. Slicing them here by the padded / GQA-folded
    // (mBlocks * mPerBlock) M tiling would both mis-shape them and, in general,
    // not even divide the unpadded seqLenQ.
    SmallVector<Value> preInputsCell(attn.getPreSoftmaxElemWiseInputs().begin(),
                                     attn.getPreSoftmaxElemWiseInputs().end());
    GemmParamsAttr p0 = withMN(params0, mSeg.length, params0.getNPerBlock());
    for (auto [j, nSeg] : llvm::enumerate(nSegs)) {
      Value vCell = sliceN(values, nSeg);
      int64_t nPerBlock1 =
          std::min<int64_t>(params1.getNPerBlock(), nSeg.length);
      GemmParamsAttr p1 = withMN(params1, mSeg.length, nPerBlock1);
      auto cellOutType = RankedTensorType::get(
          {gOut, mBlocks * mSeg.length, nSeg.length}, outType.getElementType());
      // LSE is head-dim (N) independent, so emit it once per M segment (j ==
      // 0).
      bool wantLse = hasLse && j == 0;
      RankedTensorType cellLseType;
      if (wantLse)
        cellLseType = RankedTensorType::get(
            {lseType.getShape()[0], mBlocks * mSeg.length},
            lseType.getElementType());
      auto sub = GridwiseAttentionOp::create(
          b, loc, cellOutType, cellLseType, qCell, attn.getKeys(), vCell,
          preInputsCell, attn.getCurrentSeqLen(), attn.getPrefixOffset(),
          attn.getCausalAttr(), attn.getSplitKVAttr(),
          attn.getSlidingWindowSizeAttr(), attn.getDisableQBypassLDSAttr(),
          prePadMAttr, prePadG0NAttr, mOrigPerBlockAttr, mSliceOffsetAttr,
          attn.getNumRepeatsGQAAttr(), attn.getSoftmaxTypeAttr(), p0, p1,
          b.getBoolAttr(attn.getEnableSoftmax()),
          b.getBoolAttr(attn.getPreSoftmaxHasSplitKVTransforms()));
      IRMapping regionMap;
      attn.getPreSoftmaxBody().cloneInto(&sub.getPreSoftmaxBody(), regionMap);
      resultGrid.push_back(sub->getResult(0));
      if (wantLse)
        lsePerMSeg.push_back(sub.getLse());
    }
  }

  // Main output: trace to stores, replicate fusion + stores per cell.
  FailureOr<OutputsAndFusionInputs> maybeViews =
      traceOutputsAndFusionInputs(attn->getResult(0));
  if (failed(maybeViews))
    return attn.emitError("rock-decompose-nonpow2-tiles: cannot trace "
                          "gridwise_attention output to rock.store");
  SetVector<StoreOp> &stores = maybeViews->stores;
  if (stores.empty())
    return attn.emitError("rock-decompose-nonpow2-tiles: no rock.store for "
                          "gridwise_attention result");

  OutputSplitter splitter(b, loc, gOut, mBlocks, /*nBlocks=*/1, mPerBlock,
                          /*nPerBlock=*/gemm1N, mSegs, nSegs);
  splitter.seed(attn->getResult(0), resultGrid);

  if (failed(replicateStoresPerCell(b, loc, splitter,
                                    orderStoresInProgramOrder(attn, stores),
                                    attn, /*what=*/"")))
    return failure();

  // LSE output: M-only (head-dim independent), so it is split with a rank-2
  // OutputSplitter carrying a single trivial N segment (one cell per M
  // segment). This replicates any LSE output fusion per M segment, just like
  // the main output above.
  if (hasLse) {
    FailureOr<OutputsAndFusionInputs> maybeLse =
        traceOutputsAndFusionInputs(lse);
    if (failed(maybeLse))
      return attn.emitError(
          "rock-decompose-nonpow2-tiles: cannot trace gridwise_attention LSE");

    SmallVector<Pow2Segment> lseNSegs = {{/*offset=*/0, /*length=*/1}};
    OutputSplitter lseSplitter(b, loc, lseType.getShape()[0], mBlocks,
                               /*nBlocks=*/1, mPerBlock, /*nPerBlock=*/1, mSegs,
                               lseNSegs, /*outRank=*/2);
    lseSplitter.seed(lse, lsePerMSeg);

    if (failed(replicateStoresPerCell(
            b, loc, lseSplitter,
            orderStoresInProgramOrder(attn, maybeLse->stores), attn,
            /*what=*/"LSE ")))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

void RockDecomposeNonPow2TilesPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // Anchor on gridwise_gemms whose blockwise tile (mPerBlock/nPerBlock) has a
  // non-power-of-two dimension. The surrounding output fusion and stores are
  // split as part of processing each gemm; input fusion rides along on the
  // sliced A/B views and is tiled (pow2) downstream.
  SmallVector<GridwiseGemmOp> targets;
  func.walk([&](GridwiseGemmOp gemm) {
    GemmParamsAttr params = gemm.getParams();
    if (!llvm::isPowerOf2_64(params.getMPerBlock()) ||
        !llvm::isPowerOf2_64(params.getNPerBlock()))
      targets.push_back(gemm);
  });

  for (GridwiseGemmOp gemm : targets)
    if (failed(processGridwiseGemm(gemm)))
      return signalPassFailure();

  // Anchor on gridwise_attentions whose seqLenQ tile (params0.mPerBlock) or
  // head dim (result N = gemm1N) is non-power-of-two; both output axes are
  // split into pow2 sub-attentions. seqLenK (gemm0 nPerBlock) is the softmax
  // reduction and stays pow2.
  SmallVector<GridwiseAttentionOp> attnTargets;
  func.walk([&](GridwiseAttentionOp attn) {
    GemmParamsAttr params0 = attn.getParams0();
    int64_t gemm1N =
        cast<RankedTensorType>(attn->getResult(0).getType()).getShape()[2];
    if (!llvm::isPowerOf2_64(params0.getMPerBlock()) ||
        !llvm::isPowerOf2_64(gemm1N))
      attnTargets.push_back(attn);
  });

  for (GridwiseAttentionOp attn : attnTargets)
    if (failed(processGridwiseAttention(attn)))
      return signalPassFailure();
}
