//===- DecomposeNonPow2Tiles.cpp - split non-pow2 blockwise tiles --------===//
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
// Rock tensors may carry non-power-of-two dimensions, but the Triton layouts
// produced later by RockToTTIR require power-of-two tensor shapes. When the
// chosen mPerBlock/nPerBlock is not a power of two (e.g. 80, the alignment of a
// real extent of 77 up to a multiple of the MMA granularity) the blockwise
// tiles around the GEMM are non-power-of-two and must be broken up.
//
// The pass *starts from the blockwise_loads*. A blockwise_load that produces a
// non-power-of-two tile is the seed: its tile is split into the canonical grid
// of power-of-two sub-tiles, and that grid is propagated forward through the
// program:
//
//   * a blockwise_load  -> one sub-load per cell, slicing the source view (the
//                          non-power-of-two tile is never materialized);
//   * an elementwise     -> the op is cloned per cell over its operands' grids
//     fusion op             (input fusion such as `C = A + B` and output fusion
//                            such as `truncf`/bias-add are the same case);
//   * a splat constant   -> re-splat per cell;
//   * the blockwise_gemm -> the M-grid of A and the N-grid of B combine as an
//                            outer product into the (m,n) grid of C;
//   * a blockwise_store  -> one sub-store per cell into the sliced destination.
//
// The split of a value is a pure function of its type (`partitionOf`), and
// elementwise ops are shape-preserving, so operand grids always line up by
// construction; no role classification or unification is needed.
//
// The one structural step is the contraction. The GEMM lives inside the K-loop
// with the accumulator as the single loop-carried value, and K is never split.
// The loop is rebuilt with one accumulator per (m,n) cell so each sub-GEMM has
// its own pow2 accumulator; A is split along M (shared across N), B along N
// (shared across M). The grid layout is row-major `m * nSegs + n` everywhere so
// the GEMM grid, every output-fusion grid, and the stores all index alike.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

/// Per-dimension power-of-two segmentation of a tile shape.
using Partition = SmallVector<SmallVector<Segment>>;

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

/// The canonical sub-tiling of `type`: each non-power-of-two dim is split into
/// power-of-two segments, every other dim stays whole.
static Partition partitionOf(RankedTensorType type) {
  Partition part;
  for (int64_t dim : type.getShape())
    part.push_back(llvm::isPowerOf2_64(dim) ? SmallVector<Segment>{{0, dim}}
                                            : decomposePow2(dim));
  return part;
}

static bool isWholePartition(const Partition &part) {
  return llvm::all_of(
      part, [](const SmallVector<Segment> &d) { return d.size() == 1; });
}

/// Number of sub-tile cells = product of per-dim segment counts.
static int64_t numCells(const Partition &part) {
  int64_t n = 1;
  for (const auto &dim : part)
    n *= static_cast<int64_t>(dim.size());
  return n;
}

/// Mixed-radix decode of a flat cell index into a per-dim segment index, with
/// dim 0 most significant (so a 2-D [M,N] cell is `m * nSegs + n`).
static SmallVector<int64_t> cellCoords(int64_t cell, const Partition &part) {
  SmallVector<int64_t> coords(part.size());
  for (int d = static_cast<int>(part.size()) - 1; d >= 0; --d) {
    coords[d] = cell % static_cast<int64_t>(part[d].size());
    cell /= static_cast<int64_t>(part[d].size());
  }
  return coords;
}

/// Build a view of `view` in which the dimensions in `dims` are restricted to
/// the sub-ranges [offs[k], offs[k] + lens[k]); other dimensions pass through.
/// Implemented as one `slice` over all dimensions (full [0, size) for the
/// non-sliced ones), which preserves dimension order.
static Value sliceViewDims(OpBuilder &b, Location loc, Value view,
                           ArrayRef<unsigned> dims, ArrayRef<int64_t> offs,
                           ArrayRef<int64_t> lens) {
  auto viewType = cast<RankedTensorType>(view.getType());
  ArrayRef<int64_t> shape = viewType.getShape();
  unsigned rank = shape.size();

  SmallVector<std::string> nameStore;
  for (unsigned i = 0; i < rank; ++i)
    nameStore.push_back(("d" + Twine(i)).str());
  SmallVector<StringRef> names(nameStore.begin(), nameStore.end());

  SmallVector<int64_t> begins(rank, 0);
  SmallVector<int64_t> ends(shape.begin(), shape.end());
  for (auto [k, d] : llvm::enumerate(dims)) {
    begins[d] = offs[k];
    ends[d] = offs[k] + lens[k];
  }

  BottomUpTMBuilder tm(b, names, shape, loc);
  tm.slice(names, names, begins, ends);
  return TransformOp::create(b, loc, view, tm.get());
}

/// Clone into the current insertion point the defining-op chain of `v` that
/// lies inside `oldBody`, reusing values defined outside `oldBody` and any
/// mappings already present in `map` (e.g. the old K-loop IV -> the new IV).
/// Returns the equivalent value at the new location.
static Value cloneChainInto(OpBuilder &b, Value v, IRMapping &map,
                            Block *oldBody) {
  if (Value mapped = map.lookupOrNull(v))
    return mapped;
  Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != oldBody)
    return v;
  for (Value operand : def->getOperands())
    cloneChainInto(b, operand, map, oldBody);
  b.clone(*def, map);
  return map.lookup(v);
}

//===----------------------------------------------------------------------===//
// TileSplitter: forward sub-tile materialization
//===----------------------------------------------------------------------===//

namespace {

/// Materializes the full per-cell sub-tile grid of tile-shaped values,
/// memoized. A grid is laid out in `cellCoords` order (dim 0 major). The same
/// primitive serves the input side (loads emitted inside the rebuilt loop, with
/// the old K-loop IV remapped via the seeded `cloneMap`) and the output side
/// (where the source views live outside the loop and carry no IV, so cloning is
/// a no-op).
class TileSplitter {
public:
  TileSplitter(OpBuilder &b, Location loc, IRMapping &cloneMap, Block *oldBody)
      : b(b), loc(loc), cloneMap(cloneMap), oldBody(oldBody) {}

  /// Pre-populate the grid of `v` (used to bind the old loop result to the
  /// rebuilt loop's per-cell results before splitting the output-fusion DAG).
  void seed(Value v, SmallVector<Value> grid) { memo[v] = std::move(grid); }

  /// Returns the full grid of sub-tiles for `v` (one Value per cell), or
  /// failure on an unsupported producer.
  FailureOr<SmallVector<Value>> split(Value v) {
    if (auto it = memo.find(v); it != memo.end())
      return it->second;

    auto type = dyn_cast<RankedTensorType>(v.getType());
    if (!type)
      return failure();
    Operation *def = v.getDefiningOp();
    Partition part = partitionOf(type);

    SmallVector<Value> grid;
    if (auto load = dyn_cast_or_null<BlockwiseLoadOp>(def)) {
      if (failed(splitLoad(load, part, grid)))
        return failure();
    } else if (auto cst = dyn_cast_or_null<arith::ConstantOp>(def)) {
      if (failed(splitConstant(cst, type, part, grid)))
        return failure();
    } else if (def && isFusionOp(def)) {
      if (failed(splitFusion(def, part, grid)))
        return failure();
    } else {
      return failure();
    }
    memo[v] = grid;
    return memo[v];
  }

private:
  LogicalResult splitLoad(BlockwiseLoadOp load, const Partition &part,
                          SmallVector<Value> &grid) {
    auto tileType = cast<RankedTensorType>(load.getResult().getType());
    ArrayRef<int64_t> tileShape = tileType.getShape();
    Value src = cloneChainInto(b, load.getSource(), cloneMap, oldBody);
    SmallVector<Value> idx;
    for (Value i : load.getSourceIndices())
      idx.push_back(cloneChainInto(b, i, cloneMap, oldBody));
    // For a blockwise_load, tile dim d maps to source view dim (numIndices +
    // d).
    unsigned numIdx =
        cast<RankedTensorType>(src.getType()).getRank() - tileShape.size();

    int64_t cells = numCells(part);
    for (int64_t cell = 0; cell < cells; ++cell) {
      SmallVector<int64_t> coords = cellCoords(cell, part);
      SmallVector<unsigned> sliceDims;
      SmallVector<int64_t> offs, lens;
      for (unsigned d = 0; d < tileShape.size(); ++d) {
        Segment seg = part[d][coords[d]];
        if (seg.length != tileShape[d]) {
          sliceDims.push_back(numIdx + d);
          offs.push_back(seg.offset);
          lens.push_back(seg.length);
        }
      }
      Value sliced = sliceDims.empty()
                         ? src
                         : sliceViewDims(b, loc, src, sliceDims, offs, lens);
      grid.push_back(BlockwiseLoadOp::create(b, loc, sliced, idx));
    }
    return success();
  }

  LogicalResult splitConstant(arith::ConstantOp cst, RankedTensorType type,
                              const Partition &part, SmallVector<Value> &grid) {
    auto splat = dyn_cast<SplatElementsAttr>(cst.getValue());
    if (!splat)
      return failure();
    Attribute elem = splat.getSplatValue<Attribute>();
    int64_t cells = numCells(part);
    for (int64_t cell = 0; cell < cells; ++cell) {
      SmallVector<int64_t> coords = cellCoords(cell, part);
      SmallVector<int64_t> subShape;
      for (unsigned d = 0; d < part.size(); ++d)
        subShape.push_back(part[d][coords[d]].length);
      auto subTy = RankedTensorType::get(subShape, type.getElementType());
      grid.push_back(arith::ConstantOp::create(
          b, loc, subTy, SplatElementsAttr::get(subTy, elem)));
    }
    return success();
  }

  LogicalResult splitFusion(Operation *op, const Partition &part,
                            SmallVector<Value> &grid) {
    SmallVector<SmallVector<Value>> operandGrids;
    for (Value operand : op->getOperands()) {
      FailureOr<SmallVector<Value>> g = split(operand);
      if (failed(g))
        return failure();
      operandGrids.push_back(*g);
    }
    int64_t cells = numCells(part);
    for (const auto &g : operandGrids)
      if (static_cast<int64_t>(g.size()) != cells)
        return failure();

    for (int64_t cell = 0; cell < cells; ++cell) {
      SmallVector<int64_t> coords = cellCoords(cell, part);
      SmallVector<int64_t> subShape;
      for (unsigned d = 0; d < part.size(); ++d)
        subShape.push_back(part[d][coords[d]].length);

      IRMapping m;
      for (auto [oi, operand] : llvm::enumerate(op->getOperands()))
        m.map(operand, operandGrids[oi][cell]);
      Operation *cloned = b.clone(*op, m);
      for (OpResult res : cloned->getResults()) {
        auto rt = cast<RankedTensorType>(res.getType());
        res.setType(RankedTensorType::get(subShape, rt.getElementType()));
      }
      grid.push_back(cloned->getResult(0));
    }
    return success();
  }

  OpBuilder &b;
  Location loc;
  IRMapping &cloneMap;
  Block *oldBody;
  DenseMap<Value, SmallVector<Value>> memo;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// GEMM splitting
//===----------------------------------------------------------------------===//

/// Split a single non-power-of-two `blockwise_gemm` (and its surrounding
/// input/output fusion) into a grid of power-of-two sub-tile GEMMs. Handles
/// both the K-loop form (matrixC is the loop's single iter_arg, result yielded)
/// and the loopless form left when a single-K-step loop was folded away
/// (matrixC is a plain init value consumed directly).
static LogicalResult processGemm(BlockwiseGemmOp gemm) {
  Location loc = gemm.getLoc();

  // Scaled GEMMs are not handled yet (scales would need to be split along M/N
  // like A/B; the optional attrs below are already threaded through).
  if (gemm.getMatrixScaleA() || gemm.getMatrixScaleB())
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: scaled blockwise_gemm not supported");

  auto cType = cast<RankedTensorType>(gemm.getMatrixC().getType());
  auto aType = cast<RankedTensorType>(gemm.getMatrixA().getType());
  auto bType = cast<RankedTensorType>(gemm.getMatrixB().getType());
  if (cType.getRank() != 2 || aType.getRank() != 2 || bType.getRank() != 2)
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: expected 2-D GEMM operands");
  Partition cPart = partitionOf(cType);
  const SmallVector<Segment> &mSegs = cPart[0];
  const SmallVector<Segment> &nSegs = cPart[1];
  assert(!isWholePartition(cPart) &&
         "This must not happen, we only call processGemm() if there's at least "
         "a non-power-of-two dimension");

  int64_t nCol = nSegs.size();

  // The contraction dim is never split.
  if (!llvm::isPowerOf2_64(aType.getShape()[1]) ||
      !llvm::isPowerOf2_64(bType.getShape()[0]))
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: non-power-of-two K is not supported");

  // Does the GEMM accumulate in a K-loop? If so matrixC is the loop's single
  // iter_arg and the result is yielded; otherwise (single K-step, the loop was
  // folded away) matrixC is a plain init value consumed directly.
  scf::ForOp loop = gemm->getParentOfType<scf::ForOp>();
  if (loop) {
    if (loop.getNumRegionIterArgs() != 1 ||
        gemm.getMatrixC() != loop.getRegionIterArg(0))
      return gemm.emitError("rock-decompose-nonpow2-tiles: GEMM is inside a "
                            "loop that is not its accumulator K-loop");
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    if (yield.getNumOperands() != 1 || yield.getOperand(0) != gemm.getResult())
      return gemm.emitError(
          "rock-decompose-nonpow2-tiles: loop does not yield the GEMM result");
  }

  // `accResult` is the value the output fusion/stores consume; `initValue` is
  // the accumulator init we split into one init per (m,n) cell.
  Value accResult = loop ? loop.getResult(0) : gemm.getResult();
  Value initValue = loop ? loop.getInitArgs()[0] : gemm.getMatrixC();

  // Collect the blockwise_store sinks reachable from accResult through the
  // output-fusion DAG. The now-dead fusion ops, old loop, and original loads
  // are all Pure, so DCE reclaims them; only the (side-effecting) stores need
  // erasing.
  SetVector<Operation *> stores;
  DenseSet<Operation *> visitedFusion;
  SmallVector<Value> worklist{accResult};
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (auto st = dyn_cast<BlockwiseStoreOp>(user)) {
        if (st.getSource() != v)
          return gemm.emitError("rock-decompose-nonpow2-tiles: GEMM result "
                                "reaches a store as a non-source operand");
        stores.insert(st);
        continue;
      }
      if (isFusionOp(user)) {
        if (visitedFusion.insert(user).second)
          for (Value res : user->getResults())
            worklist.push_back(res);
        continue;
      }
      return gemm.emitError(
          "rock-decompose-nonpow2-tiles: unsupported use of GEMM result");
    }
  }
  if (stores.empty())
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: no blockwise_store for GEMM result");

  // Build the sub-tiles just before the loop (looped) or the GEMM (loopless).
  // In the looped form we also remap the old K-loop IV and clone in-loop source
  // views into the new body (oldBody = loop body); in the loopless form the
  // source views already dominate, so nothing is cloned (oldBody = null).
  Operation *anchor = loop ? loop.getOperation() : gemm.getOperation();
  Block *oldBody = loop ? loop.getBody() : nullptr;
  OpBuilder b(anchor);
  IRMapping cloneMap;
  TileSplitter splitter(b, loc, cloneMap, oldBody);

  // One accumulator init per (m,n) cell (row-major m * nCol + n), by splitting
  // the original init value.
  FailureOr<SmallVector<Value>> initGrid = splitter.split(initValue);
  if (failed(initGrid))
    return gemm.emitError(
        "rock-decompose-nonpow2-tiles: failed to split accumulator init");

  // Emit one sub-GEMM per (m,n) cell: A split along M (shared across N), B
  // along N (shared across M), preserving all optional GEMM attrs.
  auto makeSubGemm = [&](Value a, Value bMat, Value acc) {
    return BlockwiseGemmOp::create(
        b, loc, a, bMat, acc, /*matrixScaleA=*/Value(),
        /*matrixScaleB=*/Value(), gemm.getQuantBlockSizeAttr(),
        gemm.getMatrixAOrigElemTypeAttr(), gemm.getMatrixBOrigElemTypeAttr(),
        gemm.getMatrixAKPackAttr(), gemm.getMatrixBKPackAttr());
  };

  SmallVector<Value> resultGrid;
  if (loop) {
    auto newLoop =
        scf::ForOp::create(b, loc, loop.getLowerBound(), loop.getUpperBound(),
                           loop.getStep(), *initGrid);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(newLoop.getBody());
      cloneMap.map(loop.getInductionVar(), newLoop.getInductionVar());

      FailureOr<SmallVector<Value>> aGrid = splitter.split(gemm.getMatrixA());
      FailureOr<SmallVector<Value>> bGrid = splitter.split(gemm.getMatrixB());
      if (failed(aGrid) || failed(bGrid))
        return gemm.emitError("rock-decompose-nonpow2-tiles: failed to split "
                              "GEMM input operands");

      SmallVector<Value> yields;
      for (int64_t i = 0; i < (int64_t)mSegs.size(); ++i)
        for (int64_t j = 0; j < (int64_t)nSegs.size(); ++j)
          yields.push_back(makeSubGemm((*aGrid)[i], (*bGrid)[j],
                                       newLoop.getRegionIterArg(i * nCol + j)));
      scf::YieldOp::create(b, loc, yields);
    }
    resultGrid.assign(newLoop.getResults().begin(), newLoop.getResults().end());
  } else {
    FailureOr<SmallVector<Value>> aGrid = splitter.split(gemm.getMatrixA());
    FailureOr<SmallVector<Value>> bGrid = splitter.split(gemm.getMatrixB());
    if (failed(aGrid) || failed(bGrid))
      return gemm.emitError("rock-decompose-nonpow2-tiles: failed to split "
                            "GEMM input operands");
    for (int64_t i = 0; i < (int64_t)mSegs.size(); ++i)
      for (int64_t j = 0; j < (int64_t)nSegs.size(); ++j)
        resultGrid.push_back(
            makeSubGemm((*aGrid)[i], (*bGrid)[j], (*initGrid)[i * nCol + j]));
  }

  // Output side: bind accResult's grid to the per-cell results, then split each
  // store's source and emit one sub-store per cell into the sliced destination.
  // New ops go at each original store, not at the terminator: in chained-store
  // cases the store result feeds later views, so the replacement must dominate
  // those existing uses.
  splitter.seed(accResult, resultGrid);

  SmallVector<Operation *> orderedStores(stores.begin(), stores.end());
  llvm::sort(orderedStores, [](Operation *lhs, Operation *rhs) {
    return lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs);
  });

  for (Operation *storeOp : orderedStores) {
    b.setInsertionPoint(storeOp);
    auto store = cast<BlockwiseStoreOp>(storeOp);
    FailureOr<SmallVector<Value>> srcGrid = splitter.split(store.getSource());
    if (failed(srcGrid))
      return gemm.emitError(
          "rock-decompose-nonpow2-tiles: failed to split store source");

    // Defensive invariant: every store-source sub-tile must dominate this
    // store's replacement point. This guards against a value shared between the
    // in-loop GEMM-input split and the post-loop output-fusion split, whose
    // cached in-loop sub-tiles would not dominate here. Recomputed per store
    // since each split() may have inserted new ops.
    DominanceInfo domInfo(gemm->getParentOfType<func::FuncOp>());
    for (Value sub : *srcGrid)
      if (!domInfo.dominates(sub, storeOp))
        return gemm.emitError(
            "rock-decompose-nonpow2-tiles: store-source sub-tile does not "
            "dominate the store insertion point (a value is shared between "
            "in-loop GEMM-input and post-loop output-fusion splitting)");

    auto [destRoot, destMaps, _] = rock::untransform(b, store.getDest());
    Type storeResultType = store.getResult().getType();
    SmallVector<Value> extraIdx(store.getExtraIndices().begin(),
                                store.getExtraIndices().end());
    StoreMethod method = store.getStoreMethod();

    // Thread the destination through each sub-store so the final tensor value
    // represents all disjoint tile writes, and every pure store result is live.
    Value currentDestRoot = destRoot;
    for (auto [i, mSeg] : llvm::enumerate(mSegs)) {
      for (auto [j, nSeg] : llvm::enumerate(nSegs)) {
        Value view = rock::transform(b, currentDestRoot, destMaps);
        unsigned viewRank = cast<RankedTensorType>(view.getType()).getRank();
        Value destSeg = sliceViewDims(
            b, loc, view, {viewRank - 2, viewRank - 1},
            {mSeg.offset, nSeg.offset}, {mSeg.length, nSeg.length});
        auto st = BlockwiseStoreOp::create(b, loc, storeResultType,
                                           (*srcGrid)[i * nCol + j], destSeg,
                                           extraIdx, method);
        currentDestRoot = st.getResult();
      }
    }
    store.getResult().replaceAllUsesWith(currentDestRoot);
  }

  // Erase the original stores (writers, so DCE won't). Everything else now dead
  // is Pure and reclaimed by the post-pass DCE.
  for (Operation *storeOp : stores)
    storeOp->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

static bool hasNonPow2Dim(Value v) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  return type && llvm::any_of(type.getShape(), [](int64_t d) {
           return !llvm::isPowerOf2_64(d);
         });
}

void RockDecomposeNonPow2TilesPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // Anchor on the GEMM: a blockwise_gemm whose accumulator (result) tile has a
  // non-power-of-two M or N must be decomposed. This covers both the K-loop
  // form and the loopless form left when a single-K-step loop was folded away.
  // The surrounding input/output fusion (loads, elementwise ops, stores) is
  // split as part of processing each GEMM, so loads need no separate entry.
  SmallVector<BlockwiseGemmOp> targets;
  func.walk([&](BlockwiseGemmOp gemm) {
    if (hasNonPow2Dim(gemm.getResult()))
      targets.push_back(gemm);
  });

  for (BlockwiseGemmOp gemm : targets)
    if (failed(processGemm(gemm)))
      return signalPassFailure();
}
