//===- LowerReduceToBlockwise.cpp - Tile a standalone rock.reduce -------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Lowers the `rock.reduce` of a standalone reduce kernel into a tiled
// `rock.blockwise_reduce`, which `rock-to-ttir` turns into a `tt.reduce`.
//
// The tile shape recorded in the op's `rock.reduce_params` is the entire
// schedule; see `docs/reduce-kernel-plan.md` for why there is nothing else to
// decide here.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"

#include "ReduceUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERREDUCETOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockLowerReduceToBlockwisePass
    : public rock::impl::RockLowerReduceToBlockwisePassBase<
          RockLowerReduceToBlockwisePass> {
  using rock::impl::RockLowerReduceToBlockwisePassBase<
      RockLowerReduceToBlockwisePass>::RockLowerReduceToBlockwisePassBase;
  void runOnOperation() override;

private:
  LogicalResult lowerReduce(func::FuncOp func, ReduceOp reduceOp);
};

/// Names `d0 .. d(rank-1)`, kept alive by the caller for the builders that
/// take `StringRef`s.
struct DimNames {
  SmallVector<SmallString<4>> storage;
  SmallVector<StringRef> names;

  explicit DimNames(int64_t rank) : storage(rank) {
    names.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      storage[i] = ("d" + Twine(i)).str();
      names.push_back(storage[i]);
    }
  }
};

} // namespace

/// Place `b` so that a view built on `v` dominates every use of `v`. A view of
/// a block argument goes after `anchor`, the `tt.get_program_id` the tile
/// markers need, since inserting at the top of the block would put the view
/// before it.
static void setInsertionPointForViewOf(OpBuilder &b, Value v,
                                       Operation *anchor) {
  if (Operation *defOp = v.getDefiningOp())
    b.setInsertionPointAfter(defOp);
  else
    b.setInsertionPointAfter(anchor);
}

/// View `v`, of rank >= 1, as a 2-D `rows x cols` tensor by merging every
/// leading dimension into one. A rank-1 input gains a unit row dimension
/// instead, since there is nothing to merge.
static Value viewAs2D(OpBuilder &b, Location loc, Value v, Operation *anchor) {
  auto vType = cast<RankedTensorType>(v.getType());
  int64_t rank = vType.getRank();

  OpBuilder::InsertionGuard guard(b);
  setInsertionPointForViewOf(b, v, anchor);

  DimNames dims(rank);
  BottomUpTMBuilder builder(b, dims.names, vType.getShape(), loc);
  if (rank == 1) {
    builder.addDim("row", 0, 1);
    builder.passThrough({"col"}, {1}, {dims.names[0]});
  } else if (rank == 2) {
    // Already 2-D. A `Merge` of the single leading dimension would be a
    // degenerate one, which the pointer-arith expansion does not accept.
    builder.passThrough({"row", "col"}, {0, 1}, {dims.names[0], dims.names[1]});
  } else {
    builder.merge("row", 0, ArrayRef<StringRef>(dims.names).drop_back());
    builder.passThrough({"col"}, {1}, {dims.names[rank - 1]});
  }
  return TransformOp::create(b, loc, v, builder.get());
}

/// Pad a 2-D `v` out to `paddedRows x paddedCols`. The padded lanes read as
/// the identity of the reduction (`PreserveMaskedLoadSemantics` fills a masked
/// load feeding a sum with zero) and are masked off on the store side, so the
/// padding costs correctness nothing.
static Value padTo(OpBuilder &b, Location loc, Value v, int64_t paddedRows,
                   int64_t paddedCols, Operation *anchor) {
  auto vType = cast<RankedTensorType>(v.getType());
  int64_t rows = vType.getDimSize(0), cols = vType.getDimSize(1);
  if (rows == paddedRows && cols == paddedCols)
    return v;

  OpBuilder::InsertionGuard guard(b);
  setInsertionPointForViewOf(b, v, anchor);

  BottomUpTMBuilder builder(b, {"row", "col"}, vType.getShape(), loc);
  builder.pad({"row", "col"}, {0, paddedRows - rows, 0, paddedCols - cols});
  return TransformOp::create(b, loc, v, builder.get());
}

/// View a rank-`r` store destination as the flat list of outputs it is, then
/// pad it out to the rows the grid covers.
static Value viewDestAsPaddedRows(OpBuilder &b, Location loc, Value dest,
                                  int64_t paddedRows, Operation *anchor) {
  auto destType = cast<RankedTensorType>(dest.getType());
  int64_t rank = destType.getRank();

  OpBuilder::InsertionGuard guard(b);
  setInsertionPointForViewOf(b, dest, anchor);

  Value flat = dest;
  if (rank > 1) {
    DimNames dims(rank);
    BottomUpTMBuilder merger(b, dims.names, destType.getShape(), loc);
    merger.merge("row", 0, dims.names);
    flat = TransformOp::create(b, loc, dest, merger.get());
  }

  int64_t rows = cast<RankedTensorType>(flat.getType()).getDimSize(0);
  if (rows == paddedRows)
    return flat;
  BottomUpTMBuilder padder(b, {"row"}, {rows}, loc);
  padder.pad("row_padded", "row", 0, paddedRows - rows);
  return TransformOp::create(b, loc, flat, padder.get());
}

/// The view from a workgroup's tile coordinates to the padded 2-D input.
static ArrayAttr buildInputTileView(OpBuilder &b, Location loc,
                                    int64_t gridSize, int64_t rowsPerBlock,
                                    int64_t colsPerBlock) {
  TopDownTMBuilder builder(b, {"bid", "row_iter", "col_iter"},
                           {gridSize, rowsPerBlock, colsPerBlock}, loc);
  builder.unmerge("row", 0, {"bid", "row_iter"}, {gridSize, rowsPerBlock});
  builder.passThrough({"col"}, {1}, {"col_iter"});
  return b.getArrayAttr({builder.get()});
}

/// The view from a workgroup's output coordinates to the padded output rows.
static ArrayAttr buildOutputTileView(OpBuilder &b, Location loc,
                                     int64_t gridSize, int64_t rowsPerBlock) {
  TopDownTMBuilder builder(b, {"bid", "row_iter"}, {gridSize, rowsPerBlock},
                           loc);
  builder.unmerge("row", 0, {"bid", "row_iter"}, {gridSize, rowsPerBlock});
  return b.getArrayAttr({builder.get()});
}

/// Walk back from `root` through the fusion ops feeding the reduction and
/// collect the top of each input's transform chain. These are the values that
/// become `rock.load_marker`s.
static LogicalResult collectReduceInputs(Value root, Operation *errorOp,
                                         llvm::SetVector<Value> &leaves) {
  SmallVector<Value> worklist{root};
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    Operation *def = current.getDefiningOp();
    // A block argument, a view of one, or a constant is a leaf: the load
    // lowering traces the rest of the chain itself.
    if (!def || isa<TransformOp>(def) || isa<arith::ConstantOp>(def)) {
      leaves.insert(current);
      continue;
    }
    if (isFusionOp(def)) {
      worklist.append(def->getOperands().begin(), def->getOperands().end());
      continue;
    }
    return errorOp->emitError(
               "unsupported op feeding the reduction; expected only "
               "elementwise ops and views, got: ")
           << def->getName();
  }
  return success();
}

/// Reject an input whose last axis is not the contiguous one. The reduce axis
/// being last says nothing about the memory layout, and `tritongpu-coalesce`
/// derives the blocked layout from the pointer's contiguity, not from our axis
/// numbering: were the last axis strided, the tile shape chosen here would be
/// exactly backwards.
///
/// Kernel arguments are flat by the time they get here, so the composed chain
/// maps the view's coordinates to a single buffer offset, and "contiguous" is
/// simply that stepping the last coordinate by one steps the offset by one.
static LogicalResult checkLastAxisIsContiguous(Value leaf, Operation *errorOp) {
  SmallVector<TransformMapAttr> transforms;
  auto [underlying, _] = untransform(leaf, transforms);
  if (!isa<BlockArgument>(underlying) && !matchPattern(underlying, m_Constant()))
    return errorOp->emitError(
        "reduction input does not trace back to a kernel argument");
  // An untransformed input is the buffer itself, so it is contiguous.
  if (transforms.empty())
    return success();

  auto leafType = cast<RankedTensorType>(leaf.getType());
  int64_t rank = leafType.getRank();
  // A unit last axis has no second element to be contiguous with.
  if (leafType.getDimSize(rank - 1) < 2)
    return success();

  AffineMap composed = composeTransforms(transforms);
  if (composed.getNumResults() != 1)
    return errorOp->emitError(
               "expected the reduction input to view a flat kernel argument, "
               "but its transform chain ends in ")
           << composed.getNumResults() << " dimensions";

  MLIRContext *ctx = errorOp->getContext();
  auto offsetAt = [&](int64_t lastCoord) -> std::optional<int64_t> {
    SmallVector<Attribute> coords(rank,
                                  IntegerAttr::get(IndexType::get(ctx), 0));
    coords[rank - 1] = IntegerAttr::get(IndexType::get(ctx), lastCoord);
    SmallVector<Attribute> results;
    if (failed(composed.constantFold(coords, results)) || results.size() != 1)
      return std::nullopt;
    auto intAttr = dyn_cast<IntegerAttr>(results[0]);
    if (!intAttr)
      return std::nullopt;
    return intAttr.getInt();
  };

  std::optional<int64_t> first = offsetAt(0), second = offsetAt(1);
  if (!first || !second)
    return errorOp->emitError(
        "could not establish whether the reduction input is contiguous along "
        "the reduced axis");
  if (*second - *first != 1)
    return errorOp->emitError("reduction input is strided by ")
           << (*second - *first)
           << " along the reduced axis; the tile shape this pass picks "
              "assumes that axis is the contiguous one";
  return success();
}

/// Row-major index of `coords` in a tensor of `shape`.
static int64_t rowMajorIndex(ArrayRef<int64_t> coords, ArrayRef<int64_t> shape) {
  int64_t index = 0;
  for (auto [coord, extent] : llvm::zip(coords, shape))
    index = index * extent + coord;
  return index;
}

/// The coordinates of row-major index `index` in a tensor of `shape`.
static SmallVector<int64_t> rowMajorCoords(int64_t index,
                                           ArrayRef<int64_t> shape) {
  SmallVector<int64_t> coords(shape.size());
  for (int64_t dim = shape.size() - 1; dim >= 0; --dim) {
    coords[dim] = index % shape[dim];
    index /= shape[dim];
  }
  return coords;
}

/// Check that `views`, the chain carrying the reduction result to its store,
/// only reshapes it: element `i` in row-major order has to stay element `i`.
///
/// This pass drops those views, because `rock-lower-stores` wants the store
/// source to be the marker itself, and re-expresses both sides of the store in
/// one flat space instead. That is only sound for an order-preserving
/// reshape. The maps are affine, so agreeing on zero and on each power-of-two
/// index settles it for every index.
static LogicalResult checkViewsAreRowMajorReshapes(
    ArrayRef<TransformOp> views, ArrayRef<int64_t> resultShape,
    ArrayRef<int64_t> storedShape, Operation *errorOp) {
  if (views.empty())
    return success();

  SmallVector<TransformMapAttr> maps;
  // `views` runs from the reduction towards the store, and each map's upper
  // bounds are its own result, so composing from the store backwards gives a
  // map from stored coordinates to result coordinates.
  for (TransformOp view : llvm::reverse(views))
    maps.push_back(view.getTransform());
  AffineMap composed = composeTransforms(maps);

  MLIRContext *ctx = errorOp->getContext();
  int64_t numElements = 1;
  for (int64_t extent : storedShape)
    numElements *= extent;

  auto mapsTo = [&](int64_t index) -> std::optional<int64_t> {
    SmallVector<Attribute> operands;
    for (int64_t coord : rowMajorCoords(index, storedShape))
      operands.push_back(IntegerAttr::get(IndexType::get(ctx), coord));
    SmallVector<Attribute> results;
    if (failed(composed.constantFold(operands, results)))
      return std::nullopt;
    SmallVector<int64_t> coords;
    for (Attribute result : results) {
      auto intAttr = dyn_cast<IntegerAttr>(result);
      if (!intAttr)
        return std::nullopt;
      coords.push_back(intAttr.getInt());
    }
    if (coords.size() != resultShape.size())
      return std::nullopt;
    return rowMajorIndex(coords, resultShape);
  };

  for (int64_t index = 0; index < numElements;
       index = index == 0 ? 1 : index * 2) {
    std::optional<int64_t> mapped = mapsTo(index);
    if (!mapped)
      return errorOp->emitError(
          "could not tell whether the views between the reduction and its "
          "rock.store merely reshape the result");
    if (*mapped != index)
      return errorOp->emitError(
                 "the views between the reduction and its rock.store reorder "
                 "the result: element ")
             << index << " of the store source is element " << *mapped
             << " of the reduction result";
  }
  return success();
}

/// The `rock.store` the reduction result feeds, plus the views on the way
/// there. Views are expected: `rock-flatten-tosa-func-args` reshapes the
/// kernel's results, so a reduction is usually followed by one. An elementwise
/// op is post-reduce fusion, which this pipeline does not support.
static FailureOr<StoreOp> findTerminalStore(Value reduced, Operation *errorOp,
                                            SmallVectorImpl<TransformOp> &views) {
  StoreOp found;
  Value current = reduced;
  while (true) {
    TransformOp nextView;
    for (Operation *user : current.getUsers()) {
      if (auto storeOp = dyn_cast<StoreOp>(user)) {
        if (found && found != storeOp)
          return errorOp->emitError(
              "reduction result feeds more than one rock.store");
        found = storeOp;
        continue;
      }
      if (auto viewOp = dyn_cast<TransformOp>(user)) {
        if (nextView)
          return errorOp->emitError(
              "the reduction result is viewed in more than one way before "
              "being stored");
        nextView = viewOp;
        continue;
      }
      return errorOp->emitError(
                 "post-reduce fusion is not supported; the reduction result "
                 "must reach a rock.store through views only, but it is used "
                 "by: ")
             << user->getName();
    }
    if (!nextView)
      break;
    views.push_back(nextView);
    current = nextView.getResult();
  }
  if (!found)
    return errorOp->emitError("reduction result does not reach a rock.store");
  if (found.getSource() != current)
    return errorOp->emitError(
        "expected the last view of the reduction result to be the source of "
        "its rock.store");
  return found;
}

LogicalResult
RockLowerReduceToBlockwisePass::lowerReduce(func::FuncOp func,
                                            ReduceOp reduceOp) {
  if (reduceOp.getReduceMethod() != ReduceMethod::Sum)
    return reduceOp.emitError(
        "only sum reductions are supported by the reduce pipeline");

  auto inType = cast<RankedTensorType>(reduceOp.getIn().getType());
  if (!inType.hasStaticShape())
    return reduceOp.emitError("reduction input must have a static shape");

  int64_t rank = inType.getRank();
  int64_t axis = reduceOp.getAxis().getSExtValue();
  if (axis != rank - 1)
    return reduceOp.emitError("only last-axis reductions are supported, got "
                              "axis ")
           << axis << " of a rank-" << rank << " input";

  int64_t numCols = inType.getDimSize(rank - 1);
  int64_t numRows = inType.getNumElements() / numCols;

  SmallVector<TransformOp> resultViews;
  FailureOr<StoreOp> maybeStore =
      findTerminalStore(reduceOp.getResult(), reduceOp, resultViews);
  if (failed(maybeStore))
    return failure();
  StoreOp storeOp = *maybeStore;
  if (failed(checkViewsAreRowMajorReshapes(
          resultViews, cast<RankedTensorType>(reduceOp.getType()).getShape(),
          cast<RankedTensorType>(storeOp.getSource().getType()).getShape(),
          reduceOp)))
    return failure();

  llvm::SetVector<Value> leaves;
  if (failed(collectReduceInputs(reduceOp.getIn(), reduceOp, leaves)))
    return failure();
  for (Value leaf : leaves) {
    auto leafType = cast<RankedTensorType>(leaf.getType());
    if (leafType.getShape() != inType.getShape())
      return reduceOp.emitError(
                 "every reduction input must have the shape of the reduced "
                 "tensor, got ")
             << leafType << " for " << inType;
    if (failed(checkLastAxisIsContiguous(leaf, reduceOp)))
      return failure();
  }

  // Copied out of the pass options because an `Option<int64_t>` does not
  // stream into a diagnostic.
  int64_t waves = numWaves;
  int64_t maxCols = maxTileCols;

  // Validate a schedule the IR already carries rather than overriding it: a
  // silently ignored `params` would hide a wrong tile behind a passing test.
  int64_t waveSize = rock::getWaveSize(arch);
  ReduceParamsAttr params = reduceOp.getParamsAttr();
  if (params) {
    if (params.getNumWaves() != waves)
      return reduceOp.emitError("rock.reduce_params has numWaves ")
             << params.getNumWaves() << " but the pass was given " << waves
             << "; the two must agree, since numWaves also sets the launch "
                "block size";
    if (params.getRowsPerBlock() >
        static_cast<int64_t>(llvm::bit_ceil(static_cast<uint64_t>(numRows))))
      return reduceOp.emitError("rock.reduce_params has rowsPerBlock ")
             << params.getRowsPerBlock() << ", more rows than the "
             << numRows << " the reduction has";
    if (params.getColsPerBlock() < numCols)
      return reduceOp.emitError("rock.reduce_params has colsPerBlock ")
             << params.getColsPerBlock() << ", which is short of the "
             << numCols
             << "-element reduction; looping over the reduced axis is not "
                "implemented yet";
  } else {
    if (static_cast<int64_t>(llvm::bit_ceil(static_cast<uint64_t>(numCols))) >
        maxCols)
      return reduceOp.emitError("reduction over ")
             << numCols << " elements needs a tile wider than the " << maxCols
             << "-element limit; looping over the reduced axis is not "
                "implemented yet";
    params = chooseReduceParams(reduceOp, numRows, numCols, waveSize, waves,
                                maxCols);
    reduceOp.setParamsAttr(params);
  }

  int64_t rowsPerBlock = params.getRowsPerBlock();
  int64_t colsPerBlock = params.getColsPerBlock();
  int64_t gridSize = llvm::divideCeil(numRows, rowsPerBlock);
  int64_t paddedRows = gridSize * rowsPerBlock;

  OpBuilder b(func.getContext());
  Location loc = reduceOp.getLoc();

  func->setAttr(rock::GridSizeAttr::getMnemonic(),
                b.getI32IntegerAttr(gridSize));

  b.setInsertionPointToStart(&func.getBody().front());
  Value bid = triton::GetProgramIdOp::create(b, loc, triton::ProgramIDDim::X);
  Operation *bidOp = bid.getDefiningOp();

  // Inputs: view as 2-D, pad to the grid, then mark the tile each workgroup
  // loads. `propagateOutputType` retypes the elementwise ops in between onto
  // the tile.
  ArrayAttr inputView =
      buildInputTileView(b, loc, gridSize, rowsPerBlock, colsPerBlock);
  Value newReduceInput;
  for (Value leaf : leaves) {
    auto leafType = cast<RankedTensorType>(leaf.getType());
    Value padded = padTo(b, loc, viewAs2D(b, loc, leaf, bidOp), paddedRows,
                         colsPerBlock, bidOp);

    OpBuilder::InsertionGuard guard(b);
    setInsertionPointForViewOf(b, padded, bidOp);
    auto tileType = RankedTensorType::get({rowsPerBlock, colsPerBlock},
                                          leafType.getElementType());
    auto loadMarker =
        LoadMarkerOp::create(b, loc, tileType, padded, inputView,
                             ValueRange{bid}, rock::CacheModifier::CS);
    if (leaf == reduceOp.getIn())
      newReduceInput = loadMarker.getResult();
    propagateOutputType(leaf, loadMarker.getResult());
  }
  // A reduction straight off an input has no elementwise op for
  // `propagateOutputType` to walk, so rewire it here.
  if (newReduceInput)
    reduceOp.getInMutable().assign(newReduceInput);

  // The reduction itself. `rock.blockwise_reduce` drops the reduced axis,
  // which is what the flattened output view below expects.
  b.setInsertionPoint(reduceOp);
  Value tileSum = BlockwiseReduceOp::create(
      b, loc, reduceOp.getIn(), b.getIndexAttr(1),
      b.getAttr<ReduceMethodAttr>(ReduceMethod::Sum));

  // Output: both sides of the store move into one flat, padded row space, the
  // same shape the elementwise pipeline settles on. The views that used to
  // reshape the result are dropped, which the row-major check above licensed.
  auto resultType = cast<RankedTensorType>(reduceOp.getResult().getType());
  auto paddedResultType =
      RankedTensorType::get({paddedRows}, resultType.getElementType());
  auto storeMarker = StoreMarkerOp::create(
      b, loc, paddedResultType, tileSum,
      buildOutputTileView(b, loc, gridSize, rowsPerBlock), ValueRange{bid});
  storeOp.getSourceMutable().assign(storeMarker.getResult());
  storeOp.getDestMutable().assign(
      viewDestAsPaddedRows(b, loc, storeOp.getDest(), paddedRows, bidOp));

  for (TransformOp view : llvm::reverse(resultViews))
    view.erase();
  reduceOp.erase();
  return success();
}

void RockLowerReduceToBlockwisePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!isStandaloneReduceKernel(func))
    return;

  if (arch.empty()) {
    func.emitError("rock-lower-reduce-to-blockwise needs an arch to know the "
                   "wave size");
    return signalPassFailure();
  }

  SmallVector<ReduceOp> reduceOps;
  func.walk([&](ReduceOp op) { reduceOps.push_back(op); });
  if (reduceOps.empty())
    return;
  if (reduceOps.size() != 1) {
    func.emitError("a reduce kernel must hold exactly one rock.reduce, found ")
        << reduceOps.size();
    return signalPassFailure();
  }

  if (failed(lowerReduce(func, reduceOps.front())))
    return signalPassFailure();
}
