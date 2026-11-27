//===- LowerNonGemmToBlockwise.cpp - Tile a standalone nongemm kernel ---===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Tiles a standalone nongemm kernel: a graph of views and elementwise ops,
// optionally ending in a single `rock.reduce`. Each workgroup's share of the
// work becomes a `rock.load_marker` per leaf input and a `rock.store_marker`
// on the way out, with a `rock.blockwise_reduce` in between when there is a
// reduction; `rock-to-ttir` turns those into `tt.load`, `tt.store` and
// `tt.reduce`.
//
// Both kinds of kernel are tiled over the same canonical `M x N` view:
//
//   with a reduce:  M rows of the reduced tensor by the reduced extent N
//   without one:    1 x (total elements)
//
// and the same schedule, `rock.nongemm_params`, which is the whole of it:
// `tritongpu-coalesce` derives the blocked layout of the load from the tile,
// and that layout is what decides whether a `tt.reduce` combines in
// registers, across the lanes of a wave, or through LDS.
//
// The one structural difference is where the index of a chunk of N comes
// from. A reduction has to accumulate its chunks in one workgroup to keep a
// single writer per output row, so chunks become an `scf.for`. A map has no
// such tie, so its chunks go to the grid and there is no loop.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#include "NonGemmUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERNONGEMMTOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockLowerNonGemmToBlockwisePass
    : public rock::impl::RockLowerNonGemmToBlockwisePassBase<
          RockLowerNonGemmToBlockwisePass> {
  using rock::impl::RockLowerNonGemmToBlockwisePassBase<
      RockLowerNonGemmToBlockwisePass>::RockLowerNonGemmToBlockwisePassBase;
  void runOnOperation() override;

private:
  /// The schedule to tile with, over a canonical `numRows x numCols` view.
  /// Fills `rock.nongemm_params` in when the function does not carry one, and
  /// validates rather than overrides one that it does.
  FailureOr<NonGemmParamsAttr> resolveParams(func::FuncOp func,
                                             Operation *errorOp, bool hasReduce,
                                             Type elemType, int64_t numRows,
                                             int64_t numCols);
  /// Tile a kernel that ends in `reduceOp`, gridding over rows and walking the
  /// reduced axis in a loop.
  LogicalResult lowerReduce(func::FuncOp func, ReduceOp reduceOp,
                            StoreOp storeOp);
  /// Tile a kernel that is a pure map, gridding over the flattened elements.
  LogicalResult lowerMap(func::FuncOp func, StoreOp storeOp);

  /// Lanes per wave, from the `arch` option or from the kernel's `rock.arch`.
  int64_t waveSize = 0;
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

/// Merge `v` to 1-D then pad it out to `paddedSize`, the flat space a map
/// kernel's grid covers. The padded lanes are masked off on both sides, so
/// what they read does not matter.
static Value flattenAndPad(OpBuilder &b, Location loc, Value v,
                           int64_t paddedSize, Operation *anchor) {
  auto vType = cast<RankedTensorType>(v.getType());

  OpBuilder::InsertionGuard guard(b);
  setInsertionPointForViewOf(b, v, anchor);

  Value flat = v;
  if (vType.getRank() > 1) {
    DimNames dims(vType.getRank());
    BottomUpTMBuilder merger(b, dims.names, vType.getShape(), loc);
    merger.merge("flat", 0, dims.names);
    flat = TransformOp::create(b, loc, v, merger.get());
  }

  int64_t numElems = cast<RankedTensorType>(flat.getType()).getDimSize(0);
  if (numElems == paddedSize)
    return flat;
  BottomUpTMBuilder padder(b, {"flat"}, {numElems}, loc);
  padder.pad("padded", "flat", 0, paddedSize - numElems);
  return TransformOp::create(b, loc, flat, padder.get());
}

/// The view from a workgroup's tile coordinates to the padded flat space of a
/// map kernel. There is no loop here: a map's chunks are independent, so the
/// chunk index is the workgroup id itself.
static ArrayAttr buildFlatTileView(OpBuilder &b, Location loc, int64_t gridSize,
                                   int64_t tileSize) {
  TopDownTMBuilder builder(b, {"bid", "tile_elem"}, {gridSize, tileSize}, loc);
  builder.unmerge("elem", 0, {"bid", "tile_elem"}, {gridSize, tileSize});
  return b.getArrayAttr({builder.get()});
}

/// The view from a workgroup's tile coordinates to the padded 2-D input.
/// When the reduction is cut into `numIters` chunks, the iteration number
/// joins the workgroup id as a second index, supplied by the loop.
static ArrayAttr buildInputTileView(OpBuilder &b, Location loc,
                                    int64_t gridSize, int64_t numIters,
                                    int64_t rowsPerBlock,
                                    int64_t colsPerBlock) {
  SmallVector<StringRef> upperNames{"bid"};
  SmallVector<int64_t> upperBounds{gridSize};
  if (numIters > 1) {
    upperNames.push_back("iter");
    upperBounds.push_back(numIters);
  }
  upperNames.append({"row_iter", "col_iter"});
  upperBounds.append({rowsPerBlock, colsPerBlock});

  TopDownTMBuilder builder(b, upperNames, upperBounds, loc);
  builder.unmerge("row", 0, {"bid", "row_iter"}, {gridSize, rowsPerBlock});
  if (numIters > 1)
    builder.unmerge("col", 1, {"iter", "col_iter"}, {numIters, colsPerBlock});
  else
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

/// Walk back from `root` through the fusion ops feeding it and collect the
/// top of each input's transform chain, which is what becomes a
/// `rock.load_marker`, along with the fusion ops themselves, which the loop
/// form has to move into the loop body.
static LogicalResult collectInputs(Value root, Operation *errorOp,
                                   llvm::SetVector<Value> &leaves,
                                   llvm::SetVector<Operation *> &chain) {
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
      chain.insert(def);
      worklist.append(def->getOperands().begin(), def->getOperands().end());
      continue;
    }
    return errorOp->emitError(
               "unsupported op in a nongemm kernel; expected only elementwise "
               "ops and views, got: ")
           << def->getName();
  }
  // The walk runs backwards from the root, so put the ops back into
  // program order before anything moves them.
  SmallVector<Operation *> ordered(chain.begin(), chain.end());
  llvm::sort(ordered, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  chain.clear();
  chain.insert(ordered.begin(), ordered.end());
  return success();
}

/// Reject an input whose last axis is neither the contiguous one nor a
/// broadcast. The reduce axis being last says nothing about the memory
/// layout, and `tritongpu-coalesce` derives the blocked layout from the
/// pointer's contiguity, not from our axis numbering: were the last axis
/// strided, the tile shape chosen here would be exactly backwards.
///
/// Kernel arguments are flat by the time they get here, so the composed chain
/// maps the view's coordinates to a single buffer offset, and "contiguous" is
/// simply that stepping the last coordinate by one steps the offset by one. A
/// step of zero is a broadcast along the reduced axis, which is uniform
/// rather than strided and so is allowed as well.
static LogicalResult checkLastAxisIsContiguous(Value leaf, Operation *errorOp) {
  SmallVector<TransformMapAttr> transforms;
  auto [underlying, _] = untransform(leaf, transforms);
  if (!isa<BlockArgument>(underlying) &&
      !matchPattern(underlying, m_Constant()))
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
  int64_t stride = *second - *first;
  // Stride 0 is a `Broadcast` along the reduced axis, as in the `x - mean`
  // the mean itself was reduced out of. Every lane of a row then reads the
  // same element, which is uniform rather than strided and says nothing about
  // the tile: the shape is decided by whichever inputs do vary along the
  // axis.
  if (stride != 1 && stride != 0)
    return errorOp->emitError("reduction input is strided by ")
           << stride
           << " along the reduced axis; the tile shape this pass picks "
              "assumes that axis is the contiguous one";
  return success();
}

/// Check that `view`, one of the links carrying a reduction result to its
/// store, only reshapes it row-major and never moves an element.
///
/// Those views get dropped, because `rock-lower-stores` wants the store
/// source to be the marker itself, and both sides of the store are
/// re-expressed in one flat space instead. Dropping a view is only sound when
/// the view does nothing: element `i` in row-major order has to stay element
/// `i`.
///
/// The views that normally show up here are not user ops. `rock.reduce` keeps
/// the reduced axis at extent one and `rock-flatten-tosa-func-args` makes the
/// kernel's result rank 1, so the frontend leaves a `Merge` behind to join the
/// two. What this rules out is a view that reorders, such as the
/// `PassThrough` a `tosa.transpose` after the reduction would become: the
/// types would still line up after dropping it, and every output would be
/// written to the wrong address.
///
/// `Merge` and `Unmerge` are row-major by definition, so the test is
/// structural: nothing but those and `PassThrough` may appear, and reading
/// the transforms in order of their upper dimensions must walk the lower
/// dimensions in order too. A permutation shows up as `PassThrough` whose
/// lower dimensions are out of order, padding or broadcasting as a transform
/// type that is not on the list.
static LogicalResult checkViewIsRowMajorReshape(TransformOp view,
                                                Operation *errorOp) {
  SmallVector<std::pair<uint32_t, SmallVector<uint32_t>>> byUpperDim;
  for (TransformAttr op : view.getTransform().getOps()) {
    switch (op.getType()) {
    case TransformType::PassThrough:
    case TransformType::Merge:
    case TransformType::Unmerge:
      break;
    default:
      return errorOp->emitError(
                 "the views between the reduction and its rock.store must "
                 "only reshape the result, but one of them is a ")
             << stringifyEnum(op.getType());
    }
    // A `PassThrough` carries its dimensions pairwise, so each pair stands on
    // its own; `Merge` and `Unmerge` relate one dimension on one side to a run
    // of them on the other.
    if (op.getType() == TransformType::PassThrough) {
      for (auto [upper, lower] :
           llvm::zip(op.getUpperDims(), op.getLowerDims()))
        byUpperDim.emplace_back(upper, SmallVector<uint32_t>{lower});
    } else {
      byUpperDim.emplace_back(*llvm::min_element(op.getUpperDims()),
                              SmallVector<uint32_t>(op.getLowerDims()));
    }
  }

  llvm::sort(byUpperDim, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  uint32_t expected = 0;
  for (auto &[upperDim, lowerDims] : byUpperDim) {
    for (uint32_t lowerDim : lowerDims) {
      if (lowerDim != expected)
        return errorOp->emitError(
                   "the views between the reduction and its rock.store "
                   "reorder the result: dimension ")
               << lowerDim << " of " << view.getInput().getType()
               << " is read out of order";
      ++expected;
    }
  }
  return success();
}

/// Trace a reduction result to the `rock.store` it feeds, collecting the
/// views on the way there and vetting each one as it is crossed.
///
/// Views are expected: `rock-flatten-tosa-func-args` reshapes the kernel's
/// results, so a reduction is usually followed by one. They are collected
/// because this pass drops them, and vetted here for the same reason -- the
/// walk is the only place that knows which views are on the path, so asking
/// as each is crossed both answers at once and names the first one that is
/// not inert. An elementwise op on the way is post-reduce fusion, which this
/// pipeline does not support.
static FailureOr<StoreOp>
traceReduceToStore(Value reduced, Operation *errorOp,
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
    if (failed(checkViewIsRowMajorReshape(nextView, errorOp)))
      return failure();
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

/// The element type the tile is sized against: the widest of everything the
/// kernel touches. A thread wants `128 / elementBits` elements for a
/// full-width load, and sizing to the widest type keeps the register
/// footprint of the tile within what that budget intends, at the cost of
/// leaving a narrower operand's load short of full width.
static Type widestElementType(ArrayRef<Value> values) {
  Type widest;
  for (Value v : values) {
    Type elemType = getElementTypeOrSelf(v.getType());
    if (!widest ||
        elemType.getIntOrFloatBitWidth() > widest.getIntOrFloatBitWidth())
      widest = elemType;
  }
  return widest;
}

FailureOr<NonGemmParamsAttr> RockLowerNonGemmToBlockwisePass::resolveParams(
    func::FuncOp func, Operation *errorOp, bool hasReduce, Type elemType,
    int64_t numRows, int64_t numCols) {
  // Copied out of the pass options because an `Option<int64_t>` does not
  // stream into a diagnostic.
  int64_t maxCols = maxTileCols;

  // Validate a schedule the IR already carries rather than overriding it: a
  // silently ignored `params` would hide a wrong tile behind a passing test.
  // It rides on the function, not on the reduce: a nongemm kernel without a
  // reduce needs the same schedule and has no op to hang it on.
  auto params = func->getAttrOfType<NonGemmParamsAttr>(
      rock::NonGemmParamsAttr::getFuncAttrName());
  if (!params) {
    params = chooseNonGemmParams(func.getContext(), elemType, hasReduce,
                                 numRows, numCols, waveSize, maxCols);
    func->setAttr(rock::NonGemmParamsAttr::getFuncAttrName(), params);
    return params;
  }

  // A tile only means anything against a thread count, and the thread count
  // is not ours to pick: it follows the `numWarps` the Triton pipeline runs
  // with, which this pass cannot set.
  if (params.getNumWaves() != kNonGemmNumWaves)
    return errorOp->emitError("rock.nongemm_params has numWaves ")
           << params.getNumWaves() << ", but a nongemm kernel is launched "
           << kNonGemmNumWaves
           << " waves to a workgroup; a tile chosen for any other thread "
              "count would not be the one that runs";
  // Same reasoning for numStages: nothing carries it to the Triton pipeline
  // yet, so it keeps the `TritonOptions::numStages` default.
  if (params.getNumStages() != kNonGemmNumStages)
    return errorOp->emitError("rock.nongemm_params has numStages ")
           << params.getNumStages()
           << ", but nothing propagates it to the Triton pipeline yet, so "
              "the loop is scheduled with "
           << kNonGemmNumStages << " stages whatever this says";
  // Rows exist to make a cross-lane combine cheap. A map has none, so rows
  // would only cap the vector width each thread loads with.
  if (!hasReduce && params.getRowsPerBlock() != 1)
    return errorOp->emitError("rock.nongemm_params has rowsPerBlock ")
           << params.getRowsPerBlock()
           << ", but a nongemm kernel with no reduce is tiled over its "
              "flattened elements, where rows buy nothing and only cap the "
              "vector width";
  if (params.getRowsPerBlock() >
      static_cast<int64_t>(llvm::bit_ceil(static_cast<uint64_t>(numRows))))
    return errorOp->emitError("rock.nongemm_params has rowsPerBlock ")
           << params.getRowsPerBlock() << ", more rows than the " << numRows
           << " the reduction has";
  if (params.getColsPerBlock() > maxCols)
    return errorOp->emitError("rock.nongemm_params has colsPerBlock ")
           << params.getColsPerBlock() << ", wider than the " << maxCols
           << "-element limit on what one workgroup holds in registers";
  return params;
}

LogicalResult RockLowerNonGemmToBlockwisePass::lowerReduce(
    func::FuncOp func, ReduceOp reduceOp, StoreOp expectedStore) {
  if (reduceOp.getReduceMethod() != ReduceMethod::Sum)
    return reduceOp.emitError(
        "only sum reductions are supported by the nongemm pipeline");

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
      traceReduceToStore(reduceOp.getResult(), reduceOp, resultViews);
  if (failed(maybeStore))
    return failure();
  StoreOp storeOp = *maybeStore;
  // The kernel's only store has to be the one the reduction feeds; anything
  // else would be a second output, which this pipeline does not tile.
  if (storeOp != expectedStore)
    return reduceOp.emitError(
        "the reduction does not feed the kernel's rock.store");
  llvm::SetVector<Value> leaves;
  llvm::SetVector<Operation *> chainOps;
  if (failed(collectInputs(reduceOp.getIn(), reduceOp, leaves, chainOps)))
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

  SmallVector<Value> sizedBy(leaves.begin(), leaves.end());
  sizedBy.push_back(reduceOp.getIn());
  sizedBy.push_back(storeOp.getSource());
  FailureOr<NonGemmParamsAttr> maybeParams =
      resolveParams(func, reduceOp, /*hasReduce=*/true,
                    widestElementType(sizedBy), numRows, numCols);
  if (failed(maybeParams))
    return failure();
  NonGemmParamsAttr params = *maybeParams;

  int64_t rowsPerBlock = params.getRowsPerBlock();
  int64_t colsPerBlock = params.getColsPerBlock();
  // A reduction wider than one tile is summed a tile at a time in a loop, so
  // the register footprint stays put however long the reduction is.
  int64_t numIters = llvm::divideCeil(numCols, colsPerBlock);
  int64_t gridSize = llvm::divideCeil(numRows, rowsPerBlock);
  int64_t paddedRows = gridSize * rowsPerBlock;
  int64_t paddedCols = numIters * colsPerBlock;

  OpBuilder b(func.getContext());
  Location loc = reduceOp.getLoc();

  func->setAttr(rock::GridSizeAttr::getMnemonic(),
                b.getI32IntegerAttr(gridSize));

  b.setInsertionPointToStart(&func.getBody().front());
  Value bid = triton::GetProgramIdOp::create(b, loc, triton::ProgramIDDim::X);
  Operation *bidOp = bid.getDefiningOp();

  // Inputs: view as 2-D and pad out to what the grid and the loop cover.
  ArrayAttr inputView = buildInputTileView(b, loc, gridSize, numIters,
                                           rowsPerBlock, colsPerBlock);
  SmallVector<Value> paddedLeaves;
  for (Value leaf : leaves)
    paddedLeaves.push_back(padTo(b, loc, viewAs2D(b, loc, leaf, bidOp),
                                 paddedRows, paddedCols, bidOp));

  auto tileTypeOf = [&](Value v) {
    return RankedTensorType::get(
        {rowsPerBlock, colsPerBlock},
        cast<RankedTensorType>(v.getType()).getElementType());
  };

  // Mark the tile each workgroup loads and retype the elementwise ops in
  // between onto it. A reduction straight off an input has no elementwise op
  // for `propagateOutputType` to walk, so the caller is told which marker
  // replaced the reduction's own operand.
  //
  // Without a loop the elementwise ops stay where they are, so each marker is
  // placed next to the view it reads in order to dominate them. Inside a loop
  // the markers come first and the elementwise ops move down to follow them.
  auto markTiles = [&](ValueRange extraIndices,
                       bool anchorAtEachLeaf) -> Value {
    Value directInput;
    for (auto [leaf, padded] : llvm::zip(leaves, paddedLeaves)) {
      OpBuilder::InsertionGuard guard(b);
      if (anchorAtEachLeaf)
        setInsertionPointForViewOf(b, padded, bidOp);
      auto loadMarker =
          LoadMarkerOp::create(b, loc, tileTypeOf(leaf), padded, inputView,
                               extraIndices, rock::CacheModifier::CS);
      if (leaf == reduceOp.getIn())
        directInput = loadMarker.getResult();
      propagateOutputType(leaf, loadMarker.getResult());
    }
    return directInput ? directInput : reduceOp.getIn();
  };

  Value tile;
  if (numIters == 1) {
    tile = markTiles(ValueRange{bid}, /*anchorAtEachLeaf=*/true);
  } else {
    auto tileType = tileTypeOf(reduceOp.getIn());
    Type elemType = tileType.getElementType();

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(reduceOp);
    // Each lane keeps a running sum in registers and the whole workgroup
    // combines once, after the loop, so there is still only one cross-lane
    // reduction however long the reduction is.
    Value zero =
        arith::ConstantOp::create(b, loc, tileType, b.getZeroAttr(tileType));
    Value lower = arith::ConstantIntOp::create(b, loc, b.getI32Type(), 0);
    Value upper =
        arith::ConstantIntOp::create(b, loc, b.getI32Type(), numIters);
    Value step = arith::ConstantIntOp::create(b, loc, b.getI32Type(), 1);
    auto loop =
        scf::ForOp::create(b, loc, lower, upper, step, ValueRange{zero});

    b.setInsertionPointToStart(loop.getBody());
    Value chunk = markTiles(ValueRange{bid, loop.getInductionVar()},
                            /*anchorAtEachLeaf=*/false);
    // The elementwise ops still sit where they were, ahead of the loop; the
    // tile they now read is only defined inside it.
    for (Operation *op : chainOps)
      op->moveBefore(b.getInsertionBlock(), b.getInsertionPoint());

    Value acc = loop.getRegionIterArg(0);
    Value sum = isa<FloatType>(elemType)
                    ? arith::AddFOp::create(b, loc, acc, chunk).getResult()
                    : arith::AddIOp::create(b, loc, acc, chunk).getResult();
    scf::YieldOp::create(b, loc, ValueRange{sum});
    tile = loop.getResult(0);
  }

  // The reduction itself. `rock.blockwise_reduce` drops the reduced axis,
  // which is what the flattened output view below expects.
  b.setInsertionPoint(reduceOp);
  Value tileSum =
      BlockwiseReduceOp::create(b, loc, tile, b.getIndexAttr(1),
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

LogicalResult RockLowerNonGemmToBlockwisePass::lowerMap(func::FuncOp func,
                                                        StoreOp storeOp) {
  Value root = storeOp.getSource();
  auto rootType = dyn_cast<RankedTensorType>(root.getType());
  if (!rootType || !rootType.hasStaticShape())
    return storeOp.emitError(
        "the stored value of a nongemm kernel must be a statically shaped "
        "tensor");
  // `rock-regularize-nongemm` pushes every view down to the leaves, so what
  // is left above the store is the elementwise graph itself. A view here
  // means the kernel is a plain copy, which has no tile to choose and no
  // fusion op to hang a marker on.
  Operation *rootOp = root.getDefiningOp();
  if (!rootOp || !isFusionOp(rootOp))
    return storeOp.emitError(
        "a nongemm kernel with no reduce must compute its stored value with "
        "at least one elementwise op");

  llvm::SetVector<Value> leaves;
  llvm::SetVector<Operation *> chainOps;
  if (failed(collectInputs(root, storeOp, leaves, chainOps)))
    return failure();
  for (Value leaf : leaves) {
    auto leafType = cast<RankedTensorType>(leaf.getType());
    if (leafType.getShape() != rootType.getShape())
      return storeOp.emitError("every input of a nongemm kernel must have the "
                               "shape of its result, got ")
             << leafType << " for " << rootType;
  }

  // A map has no reduced axis, so its canonical view is one row holding every
  // element: the whole tile budget goes to the columns, and `rowsPerBlock`
  // falls out of the schedule as 1.
  int64_t numElements = rootType.getNumElements();
  SmallVector<Value> sizedBy(leaves.begin(), leaves.end());
  sizedBy.push_back(root);
  FailureOr<NonGemmParamsAttr> maybeParams =
      resolveParams(func, storeOp, /*hasReduce=*/false,
                    widestElementType(sizedBy), /*numRows=*/1, numElements);
  if (failed(maybeParams))
    return failure();
  int64_t tileSize = maybeParams->getColsPerBlock();

  // Chunks of a map are independent, so they go to the grid rather than to a
  // loop: there is no accumulator to carry between them.
  int64_t gridSize = llvm::divideCeil(numElements, tileSize);
  int64_t paddedSize = gridSize * tileSize;

  OpBuilder b(func.getContext());
  Location loc = storeOp.getLoc();
  func->setAttr(rock::GridSizeAttr::getMnemonic(),
                b.getI32IntegerAttr(gridSize));

  b.setInsertionPointToStart(&func.getBody().front());
  Value bid = triton::GetProgramIdOp::create(b, loc, triton::ProgramIDDim::X);
  Operation *bidOp = bid.getDefiningOp();

  ArrayAttr tileView = buildFlatTileView(b, loc, gridSize, tileSize);

  for (Value leaf : leaves) {
    Value padded = flattenAndPad(b, loc, leaf, paddedSize, bidOp);
    auto tileType = RankedTensorType::get(
        {tileSize}, cast<RankedTensorType>(leaf.getType()).getElementType());

    OpBuilder::InsertionGuard guard(b);
    setInsertionPointForViewOf(b, padded, bidOp);
    auto loadMarker =
        LoadMarkerOp::create(b, loc, tileType, padded, tileView,
                             ValueRange{bid}, rock::CacheModifier::CS);
    propagateOutputType(leaf, loadMarker.getResult());
  }

  b.setInsertionPoint(storeOp);
  auto paddedResultType =
      RankedTensorType::get({paddedSize}, rootType.getElementType());
  auto storeMarker = StoreMarkerOp::create(
      b, loc, paddedResultType, storeOp.getSource(), tileView, ValueRange{bid});
  storeOp.getSourceMutable().assign(storeMarker.getResult());
  storeOp.getDestMutable().assign(
      flattenAndPad(b, loc, storeOp.getDest(), paddedSize, bidOp));
  return success();
}

void RockLowerNonGemmToBlockwisePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!isStandaloneNonGemmKernel(func))
    return;

  SmallVector<ReduceOp> reduceOps;
  func.walk([&](ReduceOp op) { reduceOps.push_back(op); });
  if (reduceOps.size() > 1) {
    func.emitError("a nongemm kernel must hold at most one rock.reduce, found ")
        << reduceOps.size();
    return signalPassFailure();
  }

  // One output per kernel: a second one would want a grid of its own, since
  // nothing says two results have the same shape.
  SmallVector<StoreOp> stores;
  func.walk([&](StoreOp op) { stores.push_back(op); });
  if (stores.size() != 1) {
    func.emitError("a nongemm kernel must hold exactly one rock.store, found ")
        << stores.size();
    return signalPassFailure();
  }

  // The option wins when it is given, which is how the pipelines pass the
  // arch down; otherwise the kernel's own `rock.arch` answers, which is what
  // a lit test running this pass alone relies on.
  StringRef archName = arch;
  if (archName.empty()) {
    FailureOr<StringAttr> funcArch = rock::getArchOnFunc(func);
    if (succeeded(funcArch))
      archName = funcArch->getValue();
  }
  if (archName.empty()) {
    func.emitError("rock-lower-nongemm-to-blockwise needs an arch to know the "
                   "wave size, from its option or from rock.arch");
    return signalPassFailure();
  }
  waveSize = rock::getWaveSize(archName);

  LogicalResult result =
      reduceOps.empty() ? lowerMap(func, stores.front())
                        : lowerReduce(func, reduceOps.front(), stores.front());
  if (failed(result))
    return signalPassFailure();
}
