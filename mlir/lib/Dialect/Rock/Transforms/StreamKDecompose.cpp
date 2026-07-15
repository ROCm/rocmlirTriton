//===- StreamKDecompose.cpp - hybrid stream-K via gridwise splitting -----===//
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
// Hybrid stream-K, implemented purely at the gridwise layer by reusing the
// "several rock.gridwise_gemm in one kernel" pattern proven by
// rock-decompose-nonpow2-tiles.
//
// A batch-1/skinny GEMM whose tile grid does not evenly fill the machine wastes
// the ragged tail wave. Given a persistent launch of P = num_cu workgroups, we
// split one gridwise_gemm covering gridFull = mBlocks*nBlocks*G output tiles
// into data-parallel waves plus a split-K remainder.
//
// Each sub-gemm must be a rectangular rock.gridwise_gemm, so we cannot slice an
// arbitrary flat tile range. Instead we partition along a single dimension and
// span the other two fully, keeping every wave (and the leftover slab)
// rectangular. The partition dimension is chosen by priority N -> M -> G: for
// dimension D with `others` = product of the other two block counts, a wave
// spans `span = P / others` blocks of D, so it has span*others == P tiles.
//
//   * W = floor(blocks(D) / span) *data-parallel wave* sub-gemms. Each slices
//     the operand(s) touching D (A for M/G, B for N/G; the other operand is
//     shared) and the output along D, has exactly P grid tiles, stored `set`.
//   * one *remainder* sub-gemm over the leftover blocks(D) % span blocks. Its K
//     is split by splitK = P / remainderTiles, folded into G exactly like
//     split-K, so it also has P grid tiles; accumulated with `atomic_add` into
//     a zero-prefilled output.
//
// Because every sub-gemm resolves to P grid tiles, the func grid_size is P and
// each workgroup runs all sub-gemms in sequence: its data-parallel tile in each
// wave, then its K-slice of a remainder tile. This is the "data-parallel +
// stream-K remainder" (two-tile) hybrid.
//
// The persistent grid size P is controlled by the `streamKMultiple` tuning
// parameter: 0 (the default) disables the pass, and any value >= 1 launches
// streamKMultiple * num_cu workgroups.
//
// The output side mirrors rock-decompose-nonpow2-tiles: the gemm result is
// traced through the output-fusion DAG to every rock.store, the fusion ops are
// cloned per cell (each data-parallel wave plus the split-K remainder), and one
// rock.store is emitted per cell. Data-parallel waves keep the original store
// method; the split-K remainder always accumulates via atomic_add into a
// zero-prefilled output. Extra fusion inputs are sliced (and, for the
// remainder, split-K folded) exactly like the output. Scaled gemms are not yet
// supported.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSTREAMKDECOMPOSEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-stream-k-decompose"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockStreamKDecomposePass
    : public rock::impl::RockStreamKDecomposePassBase<
          RockStreamKDecomposePass> {
  void runOnOperation() override;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// View helpers
//===----------------------------------------------------------------------===//

/// Return a view of the rank-3 tensor `view` ([G, M, N] or [G, M, K]) that
/// selects whole blocks [blockStart, blockStart+blockCount) along dimension
/// `dim` of a rank-3 gridwise operand/output. `dim` is restructured into
/// (block, iter) = (numBlocks, perBlock), the block sub-dim is sliced to the
/// requested range, and the two are re-merged, so the result's `dim` has size
/// blockCount*perBlock and block `b` maps onto original block `blockStart+b`.
/// For the G dimension use perBlock = 1.
static Value sliceBlockRange(OpBuilder &b, Location loc, Value view,
                             unsigned dim, int64_t numBlocks, int64_t perBlock,
                             int64_t blockStart, int64_t blockCount) {
  auto type = cast<RankedTensorType>(view.getType());
  ArrayRef<int64_t> shape = type.getShape();
  unsigned rank = shape.size();
  assert(rank == 3 && dim < rank && "expected rank-3 gridwise operand/output");

  SmallVector<std::string> dimNameStore(rank);
  for (unsigned i = 0; i < rank; ++i)
    dimNameStore[i] = ("d" + Twine(i)).str();
  SmallVector<StringRef> dimNames(dimNameStore.begin(), dimNameStore.end());
  std::string blockNameStore = dimNameStore[dim] + "b";
  std::string iterNameStore = dimNameStore[dim] + "i";
  StringRef blockName(blockNameStore), iterName(iterNameStore);

  // Layer 1: split dim -> (blockName, iterName).
  BottomUpTMBuilder splitBuilder(b, dimNames, shape, loc);
  unsigned upperDim = 0;
  for (unsigned i = 0; i < rank; ++i) {
    if (i == dim) {
      splitBuilder.unmerge({blockName, iterName}, {upperDim, upperDim + 1},
                           dimNames[i], {numBlocks, perBlock});
      upperDim += 2;
    } else {
      splitBuilder.passThrough({dimNames[i]}, {upperDim}, {dimNames[i]});
      upperDim += 1;
    }
  }
  TransformMapAttr splitMap = splitBuilder.get();

  // Layer 2: slice the block sub-dim to the wave's block range; pass the rest
  // through.
  BottomUpTMBuilder sliceBuilder =
      BottomUpTMBuilder::above(splitBuilder, splitMap);
  SmallVector<StringRef> passNames;
  for (unsigned i = 0; i < rank; ++i)
    passNames.push_back(i == dim ? iterName : dimNames[i]);
  sliceBuilder.passThrough(passNames);
  sliceBuilder.slice({blockName}, {blockName}, {blockStart},
                     {blockStart + blockCount});
  TransformMapAttr sliceMap = sliceBuilder.get();

  // Layer 3: merge (blockName, iterName) -> dim.
  BottomUpTMBuilder mergeBuilder =
      BottomUpTMBuilder::above(sliceBuilder, sliceMap);
  upperDim = 0;
  for (unsigned i = 0; i < rank; ++i) {
    if (i == dim)
      mergeBuilder.merge(dimNames[i], upperDim, {blockName, iterName});
    else
      mergeBuilder.passThrough({dimNames[i]}, {upperDim}, {dimNames[i]});
    upperDim += 1;
  }
  TransformMapAttr mergeMap = mergeBuilder.get();

  return rock::transform(b, view,
                         b.getArrayAttr({mergeMap, sliceMap, splitMap}));
}

//===----------------------------------------------------------------------===//
// OutputSplitter: per-cell replication of the output-fusion DAG
//===----------------------------------------------------------------------===//

namespace {

/// Describes one output cell of the stream-K decomposition: a data-parallel
/// wave (a block range along the partition dim) or the split-K remainder (the
/// trailing block range, additionally split along K by `splitK`).
struct StreamKCell {
  int64_t blockStart; // first block along the partition dim
  int64_t blockCount; // blocks along the partition dim
  bool isRemainder;   // remainder cell -> split-K folded + atomic_add
};

/// Materializes, per cell, the value of any tile-shaped output value reachable
/// from the gridwise_gemm result through output fusion. The gemm result grid is
/// seeded; fusion ops are cloned per cell, splat constants are re-splatted, and
/// any other leaf (an extra fusion input, a full [G, M, N] view) is sliced like
/// the output (block-sliced along the partition dim, and split-K folded for the
/// remainder cell). Mirrors the OutputSplitter in rock-decompose-nonpow2-tiles.
class StreamKOutputSplitter {
public:
  StreamKOutputSplitter(OpBuilder &b, Location loc, unsigned partDim,
                        int64_t numBlocks, int64_t perBlock, int64_t g,
                        int64_t m, int64_t n, int64_t splitK,
                        ArrayRef<StreamKCell> cells)
      : b(b), loc(loc), partDim(partDim), numBlocks(numBlocks),
        perBlock(perBlock), g(g), m(m), n(n), splitK(splitK), cells(cells) {}

  void seed(Value v, SmallVector<Value> grid) { memo[v] = std::move(grid); }

  int64_t numCells() const { return static_cast<int64_t>(cells.size()); }

  /// Cell shape [G(*splitK), M, N] with the partition dim narrowed to the
  /// cell's block range.
  SmallVector<int64_t> cellShape(int64_t cell) const {
    SmallVector<int64_t> shape = {g, m, n};
    shape[partDim] = cells[cell].blockCount * perBlock;
    if (cells[cell].isRemainder)
      shape[0] *= splitK;
    return shape;
  }

  /// Slice the full [G, M, N] value `v` for cell `cell`.
  Value sliceCell(Value v, int64_t cell) {
    const StreamKCell &c = cells[cell];
    Value sliced = sliceBlockRange(b, loc, v, partDim, numBlocks, perBlock,
                                   c.blockStart, c.blockCount);
    if (c.isRemainder)
      sliced = rock::splitKFoldOutputView(b, loc, sliced, splitK);
    return sliced;
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
      // Leaf: an extra fusion input (a full [G, M, N] view). Slice it.
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
          RankedTensorType::get(cellShape(cell), type.getElementType());
      grid.push_back(arith::ConstantOp::create(
          b, loc, subTy, SplatElementsAttr::get(subTy, elem)));
    }
    return success();
  }

  LogicalResult splitFusion(Operation *op, SmallVector<Value> &grid) {
    if (op->getNumResults() != 1)
      return failure();
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
      IRMapping map;
      for (auto [oi, operand] : llvm::enumerate(op->getOperands()))
        map.map(operand, operandGrids[oi][cell]);
      Operation *cloned = b.clone(*op, map);
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
  unsigned partDim;
  int64_t numBlocks, perBlock, g, m, n, splitK;
  ArrayRef<StreamKCell> cells;
  DenseMap<Value, SmallVector<Value>> memo;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// gridwise_gemm splitting
//===----------------------------------------------------------------------===//

static LogicalResult processGridwiseGemm(GridwiseGemmOp gemm) {
  Location loc = gemm.getLoc();

  // Scaled gemms are not handled yet: the scales would need to be sliced and
  // split-K folded along with A/B.
  if (gemm.getScaleA() || gemm.getScaleB()) {
    LLVM_DEBUG(llvm::dbgs()
               << "stream-K: scaled gemm not supported -> leaving unchanged\n");
    return success(); // scaled gemm: leave unchanged.
  }

  GemmParamsAttr params = gemm.getParams();

  // streamKMultiple gates the pass: < 1 (0 = disabled, the default) leaves the
  // gemm unchanged, and >= 1 launches that multiple of num_cu.
  int64_t streamKMultiple = params.getStreamKMultiple();
  if (streamKMultiple < 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "stream-K: disabled (streamKMultiple=" << streamKMultiple
               << ") -> leaving gemm unchanged\n");
    return success(); // stream-K disabled via perfConfig.
  }

  // The remainder wave re-splits K, which cannot compose with an already
  // split-K gemm. If stream-K is explicitly requested on one, fail rather than
  // silently ignoring the request.
  if (params.getSplitKFactor() != 1)
    return gemm.emitError("rock-stream-k-decompose: streamKMultiple is "
                          "incompatible with splitKFactor > 1");

  int64_t mPerBlock = params.getMPerBlock();
  int64_t nPerBlock = params.getNPerBlock();
  int64_t kPerBlock = params.getKPerBlock();

  Value a = gemm.getA();                                      // [G, M, K]
  Value bMat = gemm.getB();                                   // [G, K, N]
  auto cType = cast<RankedTensorType>(gemm.getC().getType()); // [G, M, N]
  int64_t G = cType.getShape()[0];
  int64_t M = cType.getShape()[1];
  int64_t N = cType.getShape()[2];
  int64_t K = cast<RankedTensorType>(a.getType()).getShape()[2];
  // GemmToGridwise pads M/N/K up to multiples of the per-block tiles before
  // emitting the gridwise_gemm, so these are invariants at this layer.
  assert(M % mPerBlock == 0 && N % nPerBlock == 0 && K % kPerBlock == 0 &&
         "gridwise gemm dims must be multiples of the per-block tiles");
  int64_t mBlocks = M / mPerBlock;
  int64_t nBlocks = N / nPerBlock;

  // The requested persistent grid size = streamKMultiple * num_cu, which
  // approximates launching `streamKMultiple` resident blocks per CU (cf. Triton
  // total_programs_streamk). The actual grid size P is searched for below and
  // may be slightly smaller than this target.
  int64_t numCU = rock::getNumCUValue(gemm);
  assert(numCU > 0 && "num_cu must be positive");
  const int64_t targetP = streamKMultiple * numCU;

  int64_t gridFull = mBlocks * nBlocks * G;
  // GemmToGridwise set the func grid_size to exactly (M/mPerBlock) *
  // (N/nPerBlock) * G, i.e. gridFull.
  assert(rock::getGridSize(gemm).value().getInt() == gridFull &&
         "gridFull must match the func grid_size set by GemmToGridwise");

  // Choose the partition dimension by priority N -> M -> G. A wave spans the
  // other two dimensions fully and `span` blocks along the partition dim, so
  // wave tiles = span * others == P and both the waves and the trailing
  // remainder slab stay rectangular (a single gridwise_gemm each).
  struct Candidate {
    unsigned dim;      // output/tile dim: 0=G, 1=M, 2=N
    int64_t numBlocks; // blocks along dim
    int64_t perBlock;  // elems per block (1 for G)
    int64_t others;    // product of the other two block counts
  };
  const Candidate cands[3] = {/*N*/ {2, nBlocks, nPerBlock, G * mBlocks},
                              /*M*/ {1, mBlocks, mPerBlock, G * nBlocks},
                              /*G*/ {0, G, 1, mBlocks * nBlocks}};

  // Search for the largest persistent grid size P at or just below the
  // requested targetP that yields a clean rectangular decomposition. The
  // invariants (P % others == 0, rb | s, P % remTiles == 0) are pure
  // integer-divisibility on P, so the requested targetP often misses a solution
  // that a slightly smaller P satisfies. Shrinking P trades a little occupancy
  // for enabling stream-K, which is usually a net win. We only allow reducing P
  // by a small fraction of a CU-wave (~20% of num_cu) so the effective
  // occupancy stays close to what streamKMultiple requested rather than
  // collapsing toward a single block per CU. This is purely a host-side tuning
  // choice with no correctness impact.
  const Candidate *chosen = nullptr;
  int64_t P = 0, span = 0, numWaves = 0, remBlocks = 0, splitK = 0;
  const int64_t minP = targetP - numCU / 5;
  for (int64_t candP = targetP; candP >= minP && !chosen; --candP) {
    for (const Candidate &c : cands) {
      if (c.others == 0 || candP % c.others != 0)
        continue;
      int64_t s = candP / c.others; // blocks per wave along dim
      if (s <= 0 || s > c.numBlocks)
        continue;
      int64_t rb = c.numBlocks % s; // trailing remainder blocks
      if (rb == 0)
        continue; // no ragged tail -> no stream-K benefit
      int64_t remTiles = rb * c.others;
      // The remainder's K is padded up to a multiple of sk*kPerBlock
      // (zero-fill), so K need not divide evenly; only the tile-refill
      // (candP % remTiles) matters.
      if (candP % remTiles != 0)
        continue;
      chosen = &c;
      P = candP;
      span = s;
      numWaves = c.numBlocks / s;
      remBlocks = rb;
      splitK = candP / remTiles; // split-K factor that refills P
      break;
    }
  }
  if (!chosen) {
    // No rectangular decomposition refills any P in [numCU, targetP] exactly
    // for any partition dim; leave the gemm unchanged.
    LLVM_DEBUG(llvm::dbgs()
               << "stream-K: no rectangular decomposition for P in [" << minP
               << ", " << targetP << "] gridFull=" << gridFull
               << " -> leaving gemm unchanged\n");
    return success();
  }

  // Output side: trace the gemm result to all its stores through the
  // output-fusion DAG, then replicate the fusion + stores per cell.
  FailureOr<OutputsAndFusionInputs> maybeViews =
      traceOutputsAndFusionInputs(gemm.getResult());
  if (failed(maybeViews) || maybeViews->stores.empty())
    return gemm.emitError("rock-stream-k-decompose: could not trace the gemm "
                          "result to its output stores");
  // The split-K remainder accumulates partial products via atomic_add into a
  // zero-prefilled buffer, which is only sound when the base store overwrites
  // (Set) or already accumulates (AtomicAdd). atomic_max cannot be
  // reconstructed from partial sums, so fail rather than silently miscompiling.
  for (StoreOp st : maybeViews->stores)
    if (st.getStoreMethod() == StoreMethod::AtomicMax)
      return st.emitError("rock-stream-k-decompose: atomic_max output is "
                          "incompatible with the split-K remainder");

  LLVM_DEBUG(llvm::dbgs() << "stream-K: P=" << P << " gridFull=" << gridFull
                          << " partDim=" << chosen->dim << " span=" << span
                          << " numWaves=" << numWaves << " remBlocks="
                          << remBlocks << " splitK=" << splitK << "\n");

  MLIRContext *ctx = gemm.getContext();
  func::FuncOp func = gemm->getParentOfType<func::FuncOp>();
  OpBuilder b(gemm);
  Type cElemType = cType.getElementType();

  // The sub-gemms must not recursively stream-K decompose, so force
  // streamKMultiple = 0 (disabled) on them.
  auto waveParams = GemmParamsAttr::get(
      ctx, mPerBlock, nPerBlock, kPerBlock, params.getKpack(),
      params.getNumCTAs(), params.getNumWaves(), params.getMatrixInstrNonkdim(),
      /*splitKFactor=*/1, params.getNumStages(), params.getWavesPerEU(),
      params.getGridGroupSize(), /*streamKMultiple=*/0,
      params.getUseAsyncCopy(), params.getUseBlockPingpong(),
      params.getUseInThreadTranspose(), params.getUseBufferOps(),
      params.getUseBufferAtomics(), params.getUseReductionLayout());

  // Slice A / B along the chosen partition dim. A ([G,M,K]) is sliced for a G-
  // or M-partition (shared for N); B ([G,K,N]) for a G- or N-partition (shared
  // for M).
  unsigned pd = chosen->dim;
  auto sliceA = [&](Value v, int64_t start, int64_t count) -> Value {
    if (pd == 2)
      return v; // N-partition: A shared
    return sliceBlockRange(b, loc, v, pd, chosen->numBlocks,
                           pd == 0 ? 1 : mPerBlock, start, count);
  };
  auto sliceB = [&](Value v, int64_t start, int64_t count) -> Value {
    if (pd == 1)
      return v; // M-partition: B shared
    return sliceBlockRange(b, loc, v, pd, chosen->numBlocks,
                           pd == 0 ? 1 : nPerBlock, start, count);
  };
  auto makeType = [&](int64_t count, int64_t gMul) {
    SmallVector<int64_t, 3> s = {G, M, N};
    s[pd] = count * chosen->perBlock;
    s[0] *= gMul; // split-K folds into G
    return RankedTensorType::get(s, cElemType);
  };

  // Build the per-cell sub-gridwise_gemm result grid: one data-parallel wave
  // per block span, plus the split-K remainder as the last cell.
  SmallVector<StreamKCell> cells;
  SmallVector<Value> resultGrid;
  for (int64_t w = 0; w < numWaves; ++w) {
    int64_t start = w * span;
    Value aCell = sliceA(a, start, span);
    Value bCell = sliceB(bMat, start, span);
    auto sub = GridwiseGemmOp::create(b, loc, makeType(span, 1), aCell, bCell,
                                      /*scaleA=*/Value(), /*scaleB=*/Value(),
                                      gemm.getQuantBlockSizeAttr(), waveParams);
    resultGrid.push_back(sub.getResult());
    cells.push_back({start, span, /*isRemainder=*/false});
  }
  {
    int64_t start = numWaves * span;
    // Pad K (zero-fill) up to a multiple of splitK*kPerBlock so it folds evenly
    // into the split-K remainder; padded reads contribute 0 to the reduction.
    int64_t kPad = llvm::alignTo(K, splitK * kPerBlock);
    int64_t kRightPad = kPad - K;
    Value aSlice = sliceA(a, start, remBlocks);
    Value bSlice = sliceB(bMat, start, remBlocks);
    if (kRightPad != 0) {
      aSlice = padMatrix(aSlice, b, loc, "gemmM", 0, "gemmK", kRightPad);
      bSlice = padMatrix(bSlice, b, loc, "gemmK", kRightPad, "gemmN", 0);
    }
    Value aRem = rock::splitKFoldOperand(b, loc, aSlice, splitK,
                                         /*kDim=*/2, /*nonKDim=*/1, "gemmM");
    Value bRem = rock::splitKFoldOperand(b, loc, bSlice, splitK,
                                         /*kDim=*/1, /*nonKDim=*/2, "gemmN");
    auto sub = GridwiseGemmOp::create(
        b, loc, makeType(remBlocks, splitK), aRem, bRem, /*scaleA=*/Value(),
        /*scaleB=*/Value(), gemm.getQuantBlockSizeAttr(), waveParams);
    resultGrid.push_back(sub.getResult());
    cells.push_back({start, remBlocks, /*isRemainder=*/true});
  }

  StreamKOutputSplitter splitter(b, loc, pd, chosen->numBlocks,
                                 chosen->perBlock, G, M, N, splitK, cells);
  splitter.seed(gemm.getResult(), resultGrid);

  // Process stores in program order so explicit resultAlias chains thread
  // through the decomposed sub-stores correctly.
  SmallVector<Operation *> orderedStores;
  for (StoreOp st : maybeViews->stores)
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
          "rock-stream-k-decompose: failed to split store source");

    // Keep destination views rooted at the original destination; only the
    // result alias advances through the sub-store chain.
    auto [destRoot, destMaps, _] = rock::untransform(b, store.getDest());
    Type storeResultType = store.getResult().getType();
    StoreMethod origMethod = store.getStoreMethod();
    Value currentResultAlias =
        store.getResultAlias() ? store.getResultAlias() : destRoot;
    for (int64_t cell = 0; cell < splitter.numCells(); ++cell) {
      Value view = rock::transform(b, destRoot, destMaps);
      Value destCell = splitter.sliceCell(view, cell);
      // Data-parallel waves keep the original store method; the split-K
      // remainder accumulates via atomic_add into a zero-prefilled output.
      bool isRemainder = cells[cell].isRemainder;
      StoreMethod method = isRemainder ? StoreMethod::AtomicAdd : origMethod;
      auto st = StoreOp::create(b, loc, storeResultType, (*srcGrid)[cell],
                                destCell, currentResultAlias, method);
      if (isRemainder &&
          failed(setStoreMethodAndPrefill(b, st, StoreMethod::AtomicAdd)))
        return failure();
      currentResultAlias = st.getResult();
    }
    store.getResult().replaceAllUsesWith(currentResultAlias);
  }

  // The original gridwise_gemm, fusion ops and stores are now dead (their store
  // results are rewired); DCE / -canonicalize removes them. The whole func now
  // launches P workgroups.
  func->setAttr(rock::GridSizeAttr::getMnemonic(), b.getI32IntegerAttr(P));
  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

void RockStreamKDecomposePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  SmallVector<GridwiseGemmOp> targets;
  func.walk([&](GridwiseGemmOp gemm) { targets.push_back(gemm); });

  for (GridwiseGemmOp gemm : targets)
    if (failed(processGridwiseGemm(gemm)))
      return signalPassFailure();
}
