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
// Rock tensors may carry non-power-of-two blockwise tiles, but the Triton
// layouts produced later by RockToTTIR require power-of-two tensor shapes. The
// per-block tile sizes live in the GEMM tuning parameters (mPerBlock /
// nPerBlock); when one of them is not a power of two (e.g. 80, the alignment of
// a real extent of 77 up to a multiple of the MMA granularity), the blockwise
// tiles produced downstream are non-power-of-two.
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
// The slice restructures dim D (= blocks * tile) into (block, iter), slices the
// iter sub-dim to the segment, and re-merges, so block `b` of the sub-op's
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
// follow-up pass. Attention (rock.gridwise_attention) is also handled
// separately.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
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

/// A power-of-two segment of a tile dimension: [offset, offset + length).
struct Segment {
  int64_t offset;
  int64_t length;
};

struct RockDecomposeNonPow2TilesPass
    : public rock::impl::RockDecomposeNonPow2TilesPassBase<
          RockDecomposeNonPow2TilesPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Partition helpers
//===----------------------------------------------------------------------===//

/// Decompose `n` into a minimal sequence of power-of-two segments (largest
/// first) that exactly cover [0, n). A power-of-two `n` yields a single
/// segment, so the split is a no-op for that dimension.
static SmallVector<Segment> decomposePow2(int64_t n) {
  SmallVector<Segment> segs;
  int64_t off = 0;
  for (int64_t rem = n; rem > 0;) {
    int64_t seg = static_cast<int64_t>(llvm::bit_floor<uint64_t>(rem));
    segs.push_back({off, seg});
    off += seg;
    rem -= seg;
  }
  return segs;
}

/// Build a view of the rank-3 gridwise operand `view` in which each dimension
/// listed in `sliceDims` (which has size `blocks[k] * tiles[k]`) is restructured
/// into (block, iter) = (blocks[k], tiles[k]), the iter sub-dim is sliced to
/// `segs[k]`, and the two are re-merged. The resulting dimension has size
/// `blocks[k] * segs[k].length`, and block `bk` of it maps onto original
/// indices `bk*tiles[k] + segs[k].offset + i`. Dimensions not in `sliceDims`
/// pass through unchanged.
static Value sliceBlockedDims(OpBuilder &b, Location loc, Value view,
                              ArrayRef<unsigned> sliceDims,
                              ArrayRef<int64_t> blocks, ArrayRef<int64_t> tiles,
                              ArrayRef<Segment> segs) {
  auto type = cast<RankedTensorType>(view.getType());
  ArrayRef<int64_t> shape = type.getShape();
  unsigned rank = shape.size();

  auto sliceIdx = [&](unsigned d) -> int {
    for (auto [k, dd] : llvm::enumerate(sliceDims))
      if (dd == d)
        return static_cast<int>(k);
    return -1;
  };

  SmallVector<std::string> baseStore, blkStore, itStore;
  for (unsigned i = 0; i < rank; ++i) {
    baseStore.push_back(("d" + Twine(i)).str());
    blkStore.push_back(("d" + Twine(i) + "b").str());
    itStore.push_back(("d" + Twine(i) + "i").str());
  }
  SmallVector<StringRef> baseNames(baseStore.begin(), baseStore.end());

  // Layer 1: restructure each sliced dim into (block, iter).
  BottomUpTMBuilder l1(b, baseNames, shape, loc);
  {
    unsigned up = 0;
    for (unsigned i = 0; i < rank; ++i) {
      int k = sliceIdx(i);
      if (k >= 0) {
        l1.unmerge({StringRef(blkStore[i]), StringRef(itStore[i])},
                   {up, up + 1}, StringRef(baseStore[i]),
                   {blocks[k], tiles[k]});
        up += 2;
      } else {
        l1.passThrough({StringRef(baseStore[i])}, {up}, {StringRef(baseStore[i])});
        up += 1;
      }
    }
  }
  TransformMapAttr a1 = l1.get();

  // Layer 2: slice only the iter sub-dims that are actually narrowed; every
  // other dim (including iter sub-dims whose segment already covers the whole
  // tile) passes through unchanged, so the Slice transform lists just the dims
  // it really shrinks instead of all of them.
  BottomUpTMBuilder l2 = BottomUpTMBuilder::above(l1, a1);
  SmallVector<StringRef> names2;
  l2.getStartNames(names2);

  llvm::SmallDenseSet<StringRef> slicedNames;
  SmallVector<StringRef> sliceNames;
  SmallVector<int64_t> begins, ends;
  for (unsigned i = 0; i < rank; ++i) {
    int k = sliceIdx(i);
    if (k < 0 || (segs[k].offset == 0 && segs[k].length == tiles[k]))
      continue;
    StringRef nm(itStore[i]);
    slicedNames.insert(nm);
    sliceNames.push_back(nm);
    begins.push_back(segs[k].offset);
    ends.push_back(segs[k].offset + segs[k].length);
  }

  SmallVector<StringRef> passNames;
  for (StringRef nm : names2)
    if (!slicedNames.contains(nm))
      passNames.push_back(nm);
  if (!passNames.empty())
    l2.passThrough(passNames);
  if (!sliceNames.empty())
    l2.slice(sliceNames, sliceNames, begins, ends);
  TransformMapAttr a2 = l2.get();

  // Layer 3: re-merge (block, iter) back into the original dimension.
  BottomUpTMBuilder l3 = BottomUpTMBuilder::above(l2, a2);
  {
    unsigned up = 0;
    for (unsigned i = 0; i < rank; ++i) {
      int k = sliceIdx(i);
      if (k >= 0) {
        l3.merge(StringRef(baseStore[i]), up,
                 {StringRef(blkStore[i]), StringRef(itStore[i])});
      } else {
        l3.passThrough({StringRef(baseStore[i])}, {up}, {StringRef(baseStore[i])});
      }
      up += 1;
    }
  }
  TransformMapAttr a3 = l3.get();

  return rock::transform(b, view, b.getArrayAttr({a3, a2, a1}));
}

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
  OutputSplitter(OpBuilder &b, Location loc, int64_t g, int64_t mBlocks,
                 int64_t nBlocks, int64_t mPerBlock, int64_t nPerBlock,
                 ArrayRef<Segment> mSegs, ArrayRef<Segment> nSegs)
      : b(b), loc(loc), g(g), mBlocks(mBlocks), nBlocks(nBlocks),
        mPerBlock(mPerBlock), nPerBlock(nPerBlock), mSegs(mSegs), nSegs(nSegs) {}

  void seed(Value v, SmallVector<Value> grid) { memo[v] = std::move(grid); }

  int64_t numCells() const {
    return static_cast<int64_t>(mSegs.size() * nSegs.size());
  }

  /// Cell shape [G, mBlocks*mSeg.len, nBlocks*nSeg.len] for cell `cell`.
  SmallVector<int64_t> cellShape(int64_t cell, Type elemType) const {
    int64_t i = cell / static_cast<int64_t>(nSegs.size());
    int64_t j = cell % static_cast<int64_t>(nSegs.size());
    return {g, mBlocks * mSegs[i].length, nBlocks * nSegs[j].length};
  }

  /// Block-slice the full [G,M,N] value `v` for cell `cell`.
  Value sliceCell(Value v, int64_t cell) {
    int64_t i = cell / static_cast<int64_t>(nSegs.size());
    int64_t j = cell % static_cast<int64_t>(nSegs.size());
    SmallVector<unsigned> dims;
    SmallVector<int64_t> blocks, tiles;
    SmallVector<Segment> segs;
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
      // Leaf: an extra fusion input (a full [G,M,N] view). Block-slice it.
      if (type.getRank() != 3)
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
          RankedTensorType::get(cellShape(cell, type.getElementType()),
                                type.getElementType());
      grid.push_back(arith::ConstantOp::create(
          b, loc, subTy, SplatElementsAttr::get(subTy, elem)));
    }
    return success();
  }

  LogicalResult splitFusion(Operation *op, SmallVector<Value> &grid) {
    SmallVector<SmallVector<Value>> operandGrids;
    for (Value operand : op->getOperands()) {
      FailureOr<SmallVector<Value>> gg = split(operand);
      if (failed(gg))
        return failure();
      operandGrids.push_back(*gg);
    }
    for (const auto &gg : operandGrids)
      if (static_cast<int64_t>(gg.size()) != numCells())
        return failure();

    for (int64_t cell = 0; cell < numCells(); ++cell) {
      IRMapping m;
      for (auto [oi, operand] : llvm::enumerate(op->getOperands()))
        m.map(operand, operandGrids[oi][cell]);
      Operation *cloned = b.clone(*op, m);
      for (OpResult res : cloned->getResults()) {
        auto rt = cast<RankedTensorType>(res.getType());
        res.setType(
            RankedTensorType::get(cellShape(cell, rt.getElementType()),
                                  rt.getElementType()));
      }
      grid.push_back(cloned->getResult(0));
    }
    return success();
  }

  OpBuilder &b;
  Location loc;
  int64_t g, mBlocks, nBlocks, mPerBlock, nPerBlock;
  ArrayRef<Segment> mSegs, nSegs;
  DenseMap<Value, SmallVector<Value>> memo;
};

} // end anonymous namespace

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

  // This pass only peels the M and N tiles; it cannot split the contraction
  // dimension. A non-power-of-two kPerBlock would therefore still yield a
  // non-power-of-two K tile downstream, which the Triton layouts cannot
  // represent, so reject it explicitly rather than failing later.
  if (!llvm::isPowerOf2_64(params.getKPerBlock()))
    return gemm.emitError("rock-decompose-nonpow2-tiles: non-power-of-two "
                          "kPerBlock is not supported");

  SmallVector<Segment> mSegs = decomposePow2(mPerBlock);
  SmallVector<Segment> nSegs = decomposePow2(nPerBlock);

  Value a = gemm.getA();   // [G, M, K]
  Value bMat = gemm.getB(); // [G, K, N]
  auto cType = cast<RankedTensorType>(gemm.getC().getType()); // [G, M, N]
  if (cType.getRank() != 3)
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: expected 3-D gridwise_gemm output");
  int64_t G = cType.getShape()[0];
  int64_t M = cType.getShape()[1];
  int64_t N = cType.getShape()[2];
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
        params.getScheduleHint());
  };

  SmallVector<Value> resultGrid;
  for (auto [i, mSeg] : llvm::enumerate(mSegs)) {
    Value aCell = a;
    if (mSeg.length != mPerBlock)
      aCell = sliceBlockedDims(b, loc, a, {1}, {mBlocks}, {mPerBlock}, {mSeg});
    for (auto [j, nSeg] : llvm::enumerate(nSegs)) {
      Value bCell = bMat;
      if (nSeg.length != nPerBlock)
        bCell =
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

  // Process stores in program order so chained stores (store N+1 writes through
  // a view of store N's result) thread correctly.
  func::FuncOp func = gemm->getParentOfType<func::FuncOp>();
  SmallVector<Operation *> orderedStores;
  for (StoreOp st : stores)
    orderedStores.push_back(st.getOperation());
  DenseMap<Operation *, unsigned> opOrder;
  unsigned nextOpOrder = 0;
  func.walk([&](Operation *op) { opOrder.try_emplace(op, nextOpOrder++); });
  llvm::sort(orderedStores, [&](Operation *lhs, Operation *rhs) {
    return opOrder.lookup(lhs) < opOrder.lookup(rhs);
  });

  for (Operation *storeOp : orderedStores) {
    auto store = cast<StoreOp>(storeOp);
    b.setInsertionPoint(store);
    FailureOr<SmallVector<Value>> srcGrid = splitter.split(store.getSource());
    if (failed(srcGrid))
      return gemm.emitError(
          "rock-decompose-nonpow2-tiles: failed to split store source");

    auto [destRoot, destMaps, _] = rock::untransform(b, store.getDest());
    Type storeResultType = store.getResult().getType();
    StoreMethodAttr method = store.getStoreMethodAttr();

    // Thread the destination through each sub-store so the final tensor value
    // represents all disjoint tile writes.
    Value currentDestRoot = destRoot;
    for (int64_t cell = 0; cell < splitter.numCells(); ++cell) {
      Value view = rock::transform(b, currentDestRoot, destMaps);
      Value destCell = splitter.sliceCell(view, cell);
      auto st = StoreOp::create(b, loc, storeResultType, (*srcGrid)[cell],
                                destCell, method);
      currentDestRoot = st.getResult();
    }
    store.getResult().replaceAllUsesWith(currentDestRoot);
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
}
