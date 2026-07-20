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
// follow-up pass. Attention (rock.gridwise_attention) is also handled
// separately.
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
  OutputSplitter(OpBuilder &b, Location loc, int64_t g, int64_t mBlocks,
                 int64_t nBlocks, int64_t mPerBlock, int64_t nPerBlock,
                 ArrayRef<Pow2Segment> mSegs, ArrayRef<Pow2Segment> nSegs)
      : b(b), loc(loc), g(g), mBlocks(mBlocks), nBlocks(nBlocks),
        mPerBlock(mPerBlock), nPerBlock(nPerBlock), mSegs(mSegs), nSegs(nSegs) {
  }

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
      auto subTy = RankedTensorType::get(cellShape(cell, type.getElementType()),
                                         type.getElementType());
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
        res.setType(RankedTensorType::get(cellShape(cell, rt.getElementType()),
                                          rt.getElementType()));
      }
      grid.push_back(cloned->getResult(0));
    }
    return success();
  }

  OpBuilder &b;
  Location loc;
  int64_t g, mBlocks, nBlocks, mPerBlock, nPerBlock;
  ArrayRef<Pow2Segment> mSegs, nSegs;
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
        params.getUseReductionLayout());
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

  // Process stores in program order so explicit resultAlias chains thread
  // through the decomposed sub-stores correctly.
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
