//===- NarrowRedundantLoads.cpp - Drop redundant tt.load lanes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites a `tt.load` whose address is invariant along one or more tensor
// dimensions into a load of the index-0 slice along those dimensions plus a
// `tt.broadcast` back to the original shape.
//
// The rewrite runs on TTIR, before layouts exist, because the cost of the
// redundancy is paid during layout assignment: the redundant copies force a
// layout conversion or a pipeliner stage that routes the whole tile through
// shared memory, and a tile holding one element per thread cannot vectorize
// those accesses.
//
// Invariance is read off Triton's alignment analysis rather than recomputed
// here. The rewrite itself only re-materializes the address computation at a
// narrower shape, which is value-preserving because taking a slice of a tensor
// is; anything it does not know how to re-materialize is left alone.
//
// Masked loads of a broadcast operand need a second path: result constancy
// folds in the mask, so a repeating address with a varying mask looks
// non-constant. Those loads narrow from address constancy and drop the mask.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKNARROWREDUNDANTLOADSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-narrow-redundant-loads"

using namespace mlir;
namespace tt = mlir::triton;

namespace {

/// How the rewrite handles the original load's mask.
enum class MaskPolicy {
  /// Mask is constant along narrowed dims; slice it with the address.
  Narrow,
  /// Mask varies along narrowed dims; load unmasked, re-apply after broadcast.
  ReapplyAfterBroadcast,
};

/// A load worth rewriting, together with the shape it should be narrowed to.
struct NarrowingCandidate {
  tt::LoadOp load;
  SmallVector<int64_t> narrowShape;
  MaskPolicy maskPolicy;
};

/// Re-materializes values at a reduced shape, which amounts to taking their
/// slice at index 0 along every dimension the caller narrowed. Callers must
/// only use the result where the value is known to be invariant along those
/// dimensions, since any other slice would have done equally well.
///
/// Every rewrite is emitted at a single insertion point, which is sound because
/// the values being re-materialized all dominate it.
class SliceMaterializer {
public:
  SliceMaterializer(OpBuilder &builder) : builder(builder) {}

  /// Returns `v` restricted to `shape`, or failure if some operation in its
  /// definition cannot be re-materialized at a narrower shape.
  FailureOr<Value> slice(Value v, ArrayRef<int64_t> shape);

  /// Discards everything re-materialized so far. A narrowing that gives up
  /// partway leaves behind an address computation nothing reads, so undo it
  /// rather than relying on a later pass to notice.
  void rollback() {
    for (Operation *op : llvm::reverse(created))
      op->erase();
    created.clear();
    cache.clear();
  }

private:
  /// Narrows `type` to `shape`, keeping the element type and encoding.
  static RankedTensorType narrowType(RankedTensorType type,
                                     ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, type.getElementType(),
                                 type.getEncoding());
  }

  /// Clones a single-result elementwise operation with narrowed operands. The
  /// caller has already checked that `op` is safe to clone.
  FailureOr<Value> sliceElementwise(Operation *op, ArrayRef<int64_t> shape);

  /// Materializes a scalar holding a splat attribute's element.
  Value materializeSplatScalar(Location loc, SplatElementsAttr splat) {
    return track(arith::ConstantOp::create(
        builder, loc, cast<TypedAttr>(splat.getSplatValue<Attribute>())));
  }

  /// Notes `op` as re-materialized, so `rollback` can undo it, and returns the
  /// result it contributes to the narrowed address.
  Value track(Operation *op) {
    created.push_back(op);
    return op->getResult(0);
  }

  OpBuilder &builder;
  DenseMap<Value, Value> cache;
  SmallVector<Operation *> created;
};

/// Returns true if `op` computes its result elementwise from operands of the
/// same shape, so cloning it at a narrower shape computes the same slice.
/// Constants are excluded because their value attribute is shaped, and so has
/// to be rebuilt rather than carried over.
bool isNarrowableElementwise(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumRegions() != 0)
    return false;
  if (isa<arith::ConstantOp>(op) || !isMemoryEffectFree(op))
    return false;
  if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
    return false;

  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return false;
  // A scalar operand contributes the same value to every lane, so only the
  // tensor operands have to line up with the result.
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    return !type || type.getShape() == resultType.getShape();
  });
}

/// Returns true if `v` holds the same element in every lane, which lets a
/// narrowed load reuse it unchanged.
bool isSplatLike(Value v) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return false;
  if (isa<tt::SplatOp>(defOp))
    return true;
  auto constant = dyn_cast<arith::ConstantOp>(defOp);
  return constant && isa<SplatElementsAttr>(constant.getValue());
}

FailureOr<Value> SliceMaterializer::slice(Value v, ArrayRef<int64_t> shape) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  // A scalar already holds a single value, and a value that is already the
  // requested shape needs nothing done to it.
  if (!type || type.getShape() == shape)
    return v;

  RankedTensorType resultType = narrowType(type, shape);
  if (Value cached = cache.lookup(v); cached && cached.getType() == resultType)
    return cached;

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return failure();

  Location loc = defOp->getLoc();
  FailureOr<Value> result = failure();

  if (auto splat = dyn_cast<tt::SplatOp>(defOp)) {
    result =
        track(tt::SplatOp::create(builder, loc, resultType, splat.getSrc()));
  } else if (auto constant = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto splatAttr = dyn_cast<SplatElementsAttr>(constant.getValue())) {
      Value scalar = materializeSplatScalar(loc, splatAttr);
      result = track(tt::SplatOp::create(builder, loc, resultType, scalar));
    }
  } else if (auto range = dyn_cast<tt::MakeRangeOp>(defOp)) {
    // Narrowing a range keeps only its first element, so the range collapses
    // to its start. Reaching here at all means the caller proved that which
    // element of the range is used does not matter.
    if (shape.size() == 1 && shape[0] == 1) {
      Value start = track(arith::ConstantOp::create(
          builder, loc, builder.getI32IntegerAttr(range.getStart())));
      result = track(tt::SplatOp::create(builder, loc, resultType, start));
    }
  } else if (auto broadcast = dyn_cast<tt::BroadcastOp>(defOp)) {
    // The source is already unit-sized along the dimensions it broadcasts, so
    // narrowing it to the same shape leaves those dimensions alone.
    auto srcType = cast<RankedTensorType>(broadcast.getSrc().getType());
    SmallVector<int64_t> srcShape;
    for (auto [srcDim, dim] : llvm::zip(srcType.getShape(), shape))
      srcShape.push_back(std::min(srcDim, dim));
    FailureOr<Value> src = slice(broadcast.getSrc(), srcShape);
    if (succeeded(src)) {
      result =
          src->getType() == resultType
              ? *src
              : track(tt::BroadcastOp::create(builder, loc, resultType, *src));
    }
  } else if (auto expand = dyn_cast<tt::ExpandDimsOp>(defOp)) {
    // The expanded dimension is unit-sized in the result, hence unit-sized in
    // any narrowing of it, so it can be dropped and re-inserted unchanged.
    int64_t axis = expand.getAxis();
    SmallVector<int64_t> srcShape(shape);
    srcShape.erase(srcShape.begin() + axis);
    FailureOr<Value> src = slice(expand.getSrc(), srcShape);
    if (succeeded(src))
      result =
          track(tt::ExpandDimsOp::create(builder, loc, resultType, *src, axis));
  } else if (auto addPtr = dyn_cast<tt::AddPtrOp>(defOp)) {
    FailureOr<Value> ptr = slice(addPtr.getPtr(), shape);
    FailureOr<Value> offset = slice(addPtr.getOffset(), shape);
    if (succeeded(ptr) && succeeded(offset))
      result =
          track(tt::AddPtrOp::create(builder, loc, resultType, *ptr, *offset));
  } else if (isNarrowableElementwise(defOp)) {
    result = sliceElementwise(defOp, shape);
  }

  if (succeeded(result))
    cache[v] = *result;
  return result;
}

FailureOr<Value> SliceMaterializer::sliceElementwise(Operation *op,
                                                     ArrayRef<int64_t> shape) {
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    FailureOr<Value> narrowed = slice(operand, shape);
    if (failed(narrowed))
      return failure();
    operands.push_back(*narrowed);
  }

  auto resultType =
      narrowType(cast<RankedTensorType>(op->getResult(0).getType()), shape);
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(resultType);
  state.addAttributes(op->getAttrs());
  return track(builder.create(state));
}

/// Returns the shape `load` can be narrowed to, or nothing if it reads
/// something different in every lane along every dimension.
///
/// A dimension qualifies when the analysis proves the loaded values repeat
/// across its whole extent. That constancy already folds in both the address
/// and the mask; `other` is checked separately because it does not.
std::optional<SmallVector<int64_t>>
getNarrowShape(tt::LoadOp load, tt::ModuleAxisInfoAnalysis &axisInfo) {
  auto type = dyn_cast<RankedTensorType>(load.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  if (load.getOther() && !isSplatLike(load.getOther()))
    return std::nullopt;

  tt::AxisInfo *info = axisInfo.getAxisInfo(load.getResult());
  if (!info || info->getRank() != type.getRank())
    return std::nullopt;

  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<int64_t> narrowShape(shape);
  bool narrowed = false;
  for (auto [dim, extent] : llvm::enumerate(shape)) {
    if (extent > 1 && info->getConstancy(dim) == extent) {
      narrowShape[dim] = 1;
      narrowed = true;
    }
  }
  return narrowed ? std::optional(narrowShape) : std::nullopt;
}

/// True if `v` is constant along every dimension that `narrowShape` keeps.
bool isConstantOutsideNarrowedDims(Value v, ArrayRef<int64_t> shape,
                                   ArrayRef<int64_t> narrowShape,
                                   tt::ModuleAxisInfoAnalysis &axisInfo) {
  tt::AxisInfo *info = axisInfo.getAxisInfo(v);
  if (!info || info->getRank() != static_cast<int64_t>(shape.size()))
    return false;
  for (auto [dim, extent] : llvm::enumerate(shape)) {
    if (narrowShape[dim] == 1)
      continue;
    if (extent > 1 && info->getConstancy(dim) != extent)
      return false;
  }
  return true;
}

/// Narrowing shape when the address is constant along some dims even though
/// the result is not (the mask varies per lane). Drops the mask: it cannot
/// ride on a load that no longer has those dims. Safe if the mask is constant
/// on surviving dims, so the narrowed load never touches extra addresses.
/// `narrowLoad` re-applies the mask to restore `other`.
std::optional<SmallVector<int64_t>>
getBroadcastNarrowShape(tt::LoadOp load,
                        tt::ModuleAxisInfoAnalysis &axisInfo) {
  auto type = dyn_cast<RankedTensorType>(load.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  // Re-applying the mask needs a uniform `other`, or none.
  if (load.getOther() && !isSplatLike(load.getOther()))
    return std::nullopt;
  // Without a mask the plain constancy path above already applies.
  if (!load.getMask())
    return std::nullopt;

  tt::AxisInfo *info = axisInfo.getAxisInfo(load.getPtr());
  if (!info || info->getRank() != type.getRank())
    return std::nullopt;

  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<int64_t> narrowShape(shape);
  bool narrowed = false;
  for (auto [dim, extent] : llvm::enumerate(shape)) {
    if (extent > 1 && info->getConstancy(dim) == extent) {
      narrowShape[dim] = 1;
      narrowed = true;
    }
  }
  if (!narrowed)
    return std::nullopt;

  if (!isConstantOutsideNarrowedDims(load.getMask(), shape, narrowShape,
                                     axisInfo))
    return std::nullopt;
  return narrowShape;
}

LogicalResult narrowLoad(const NarrowingCandidate &candidate) {
  tt::LoadOp load = candidate.load;
  IRRewriter rewriter(load);
  SliceMaterializer materializer(rewriter);

  // Varying mask: load unmasked, re-apply after broadcast. Drop `other` too;
  // tt.load requires mask and other together.
  bool reapplyMask = candidate.maskPolicy == MaskPolicy::ReapplyAfterBroadcast;

  // Narrow every operand before building the load, so that giving up on one of
  // them leaves the original load in place.
  SmallVector<Value> operands;
  for (Value operand : {load.getPtr(), load.getMask(), load.getOther()}) {
    if (!operand || (reapplyMask && operand != load.getPtr())) {
      operands.push_back(nullptr);
      continue;
    }
    FailureOr<Value> narrowed =
        materializer.slice(operand, candidate.narrowShape);
    if (failed(narrowed)) {
      materializer.rollback();
      return failure();
    }
    operands.push_back(*narrowed);
  }

  auto type = cast<RankedTensorType>(load.getType());
  auto narrowedType = RankedTensorType::get(
      candidate.narrowShape, type.getElementType(), type.getEncoding());
  Value narrowedLoad =
      tt::LoadOp::create(rewriter, load.getLoc(), narrowedType, operands[0],
                         operands[1], operands[2], load.getCacheAttr(),
                         load.getEvictAttr(), load.getIsVolatileAttr());
  Value result =
      tt::BroadcastOp::create(rewriter, load.getLoc(), type, narrowedLoad);

  // Restore masked-out lanes. Skip if there was no `other`: they were undefined.
  if (reapplyMask && load.getOther())
    result = arith::SelectOp::create(rewriter, load.getLoc(), load.getMask(),
                                     result, load.getOther());

  rewriter.replaceOp(load, result);
  return success();
}

struct RockNarrowRedundantLoadsPass
    : public rock::impl::RockNarrowRedundantLoadsPassBase<
          RockNarrowRedundantLoadsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    tt::ModuleAxisInfoAnalysis axisInfo(module);

    // Collect first: rewriting the loads invalidates the analysis.
    SmallVector<NarrowingCandidate> candidates;
    module.walk([&](tt::LoadOp load) {
      // Prefer the exact slice when it applies.
      if (std::optional<SmallVector<int64_t>> narrowShape =
              getNarrowShape(load, axisInfo)) {
        candidates.push_back({load, std::move(*narrowShape),
                              MaskPolicy::Narrow});
        return;
      }
      if (std::optional<SmallVector<int64_t>> narrowShape =
              getBroadcastNarrowShape(load, axisInfo))
        candidates.push_back({load, std::move(*narrowShape),
                              MaskPolicy::ReapplyAfterBroadcast});
    });

    for (const NarrowingCandidate &candidate : candidates) {
      // Logged before the rewrite because a narrowed load is erased by it.
      LLVM_DEBUG(llvm::dbgs() << "considering " << candidate.load << "\n");
      // A load whose address cannot be re-materialized narrower stays as it
      // was, which costs the redundant reads but is always correct.
      if (failed(narrowLoad(candidate)))
        LLVM_DEBUG(llvm::dbgs() << "declined: address not re-materializable\n");
    }
  }
};

} // end anonymous namespace
