//===- PointerArithExpand.cpp - shared transform->arith helpers ----------===//
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
//===----------------------------------------------------------------------===//

#include "PointerArithExpand.h"

#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::rock;

//===----------------------------------------------------------------------===//
// Range / broadcasting helpers.
//===----------------------------------------------------------------------===//

Value mlir::rock::makeRange(OpBuilder &b, Location loc, int32_t start,
                            int32_t end, int64_t numDims, int64_t nonUnitDim,
                            Type elemType) {
  assert(numDims > 0);
  assert(nonUnitDim >= 0 && nonUnitDim < numDims);

  SmallVector<int64_t> unitDimIndices;
  for (int64_t i = 0; i < numDims; ++i)
    if (nonUnitDim != i)
      unitDimIndices.push_back(i);

  auto tensorType1D = RankedTensorType::get({end - start}, b.getI32Type());
  Value rangeTensor =
      triton::MakeRangeOp::create(b, loc, tensorType1D, start, end);

  // Restore the original rank by inserting unit dimensions.
  Value expandedTensor = rangeTensor;
  for (int64_t unitDimIdx : unitDimIndices) {
    auto currentType = cast<RankedTensorType>(expandedTensor.getType());
    SmallVector<int64_t> newShape(currentType.getShape().begin(),
                                  currentType.getShape().end());
    assert(unitDimIdx <= static_cast<int64_t>(newShape.size()) &&
           "unit dim insertion index out of bounds");
    newShape.insert(newShape.begin() + unitDimIdx, 1);
    auto expandedType = RankedTensorType::get(newShape, b.getI32Type());
    expandedTensor = triton::ExpandDimsOp::create(b, loc, expandedType,
                                                  expandedTensor, unitDimIdx);
  }

  // Widen the i32 range to the required index width if needed.
  if (elemType != b.getI32Type()) {
    auto currentType = cast<RankedTensorType>(expandedTensor.getType());
    auto widenedType = RankedTensorType::get(currentType.getShape(), elemType);
    expandedTensor =
        arith::ExtSIOp::create(b, loc, widenedType, expandedTensor);
  }
  return expandedTensor;
}

namespace {
// Broadcast two tensors to a common shape (unit dims expand to the peer size).
static std::pair<Value, Value>
broadcastTensors(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  auto tensorLhsType = cast<RankedTensorType>(lhs.getType());
  auto tensorRhsType = cast<RankedTensorType>(rhs.getType());
  auto lhsShape = tensorLhsType.getShape();
  auto rhsShape = tensorRhsType.getShape();
  auto rank = lhsShape.size();

  bool needsBroadcast = false;
  SmallVector<int64_t> resultShape(rank);
  for (size_t i = 0; i < rank; ++i) {
    if (lhsShape[i] == 1 && rhsShape[i] != 1) {
      resultShape[i] = rhsShape[i];
      needsBroadcast = true;
    } else if (rhsShape[i] == 1 && lhsShape[i] != 1) {
      resultShape[i] = lhsShape[i];
      needsBroadcast = true;
    } else if (lhsShape[i] == rhsShape[i]) {
      resultShape[i] = lhsShape[i];
    } else {
      resultShape[i] = std::max(lhsShape[i], rhsShape[i]);
    }
  }

  if (!needsBroadcast)
    return {lhs, rhs};

  auto resultType =
      RankedTensorType::get(resultShape, tensorLhsType.getElementType());
  Value broadcastedLhs = lhs;
  if (!llvm::equal(lhsShape, resultShape))
    broadcastedLhs = triton::BroadcastOp::create(builder, loc, resultType, lhs);

  Value broadcastedRhs = rhs;
  if (!llvm::equal(rhsShape, resultShape)) {
    auto rhsResultType =
        RankedTensorType::get(resultShape, tensorRhsType.getElementType());
    broadcastedRhs =
        triton::BroadcastOp::create(builder, loc, rhsResultType, rhs);
  }
  return {broadcastedLhs, broadcastedRhs};
}

static Value broadcastScalarToTensor(OpBuilder &builder, Location loc,
                                     Value scalar, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto resultType =
      RankedTensorType::get(tensorType.getShape(), scalar.getType());
  return triton::SplatOp::create(builder, loc, resultType, scalar);
}

static SmallVector<Value>
ensureCompatibleShapes(OpBuilder &builder, Location loc, ValueRange values) {
  if (values.size() < 2)
    return SmallVector<Value>(values);

  SmallVector<Value> results(values);
  // Two passes to propagate broadcasts across all operands.
  for (int pass = 0; pass < 2; pass++) {
    Value lhs = results[0];
    for (size_t i = 1; i < results.size(); i++) {
      Value rhs = results[i];
      auto lhsTensorType = dyn_cast<RankedTensorType>(lhs.getType());
      auto rhsTensorType = dyn_cast<RankedTensorType>(rhs.getType());

      if (!lhsTensorType && !rhsTensorType) {
        // both scalars
      } else if (lhsTensorType && !rhsTensorType) {
        rhs = broadcastScalarToTensor(builder, loc, rhs, lhs);
      } else if (!lhsTensorType && rhsTensorType) {
        lhs = broadcastScalarToTensor(builder, loc, lhs, rhs);
      } else {
        std::tie(lhs, rhs) = broadcastTensors(builder, loc, lhs, rhs);
      }
      results[i - 1] = lhs;
      results[i] = rhs;
      lhs = rhs;
    }
  }
  return results;
}

// Broadcast two (scalar or tensor) values to a common shape, inserting the
// needed triton.splat / triton.broadcast ops, and return the rewired pair.
static std::pair<Value, Value>
ensureCompatible(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  auto results = ensureCompatibleShapes(builder, loc, {lhs, rhs});
  return {results[0], results[1]};
}

} // namespace

Value mlir::rock::broadcastToShape(OpBuilder &b, Location loc, Value v,
                                   ArrayRef<int64_t> shape) {
  auto tt = dyn_cast<RankedTensorType>(v.getType());
  Type elem = tt ? tt.getElementType() : v.getType();
  auto target = RankedTensorType::get(shape, elem);
  if (!tt)
    return triton::SplatOp::create(b, loc, target, v);
  if (tt.getShape() == shape)
    return v;
  // Make sure its broadcastable.
  ArrayRef<int64_t> srcShape = tt.getShape();
  if (srcShape.size() != shape.size())
    return {};
  for (auto [srcDim, dstDim] : llvm::zip_equal(srcShape, shape))
    if (srcDim != dstDim && srcDim != 1)
      return {};
  return triton::BroadcastOp::create(b, loc, target, v);
}

//===----------------------------------------------------------------------===//
// Affine map expansion.
//===----------------------------------------------------------------------===//

namespace {
/// Adapted from mlir/lib/Dialect/Affine/Utils/Utils.cpp (upstream MLIR).
/// The upstream version operates on scalar Values; this version adds
/// ensureCompatible() calls to handle tensor operands that may need
/// broadcasting before each arith operation.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  AffineApplyExpander(OpBuilder &builder, Location loc, ValueRange dimValues,
                      ValueRange symbolValues, Type indexType)
      : builder(builder), loc(loc), dimValues(dimValues),
        symbolValues(symbolValues), indexType(indexType) {}

  template <typename OpTy>
  Value buildBinaryExpr(AffineBinaryOpExpr expr,
                        arith::IntegerOverflowFlags overflowFlags =
                            arith::IntegerOverflowFlags::none) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto [l, r] = ensureCompatible(builder, loc, lhs, rhs);
    return OpTy::create(builder, loc, l, r, overflowFlags);
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<arith::AddIOp>(expr);
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<arith::MulIOp>(expr,
                                          arith::IntegerOverflowFlags::nsw);
  }

  /// Euclidean modulo operation: negative RHS is not allowed.
  ///
  /// We lower this to a plain `arith.remui` (divisor is verified positive)
  /// rather than the signed remainder wrapped in a negative-value correction
  /// `select`. Everything here is element-wise on the coordinate tensors: each
  /// tensor element is one computed coordinate with its own mask bit (a single
  /// thread/loop iteration contributes many such elements, masked
  /// independently). For every element whose result is actually observed, the
  /// dividend is non-negative, so euclidean `mod` coincides with the unsigned
  /// remainder:
  ///   - The transforms that emit `mod`/`floordiv` (`Merge`, `Broadcast`) only
  ///     ever consume iteration coordinates that trace back to `make_range` /
  ///     block-thread ids, i.e. non-negative values. (`Merge`'s own affine-map
  ///     construction already assumes this; see assembleMapFor.)
  ///   - Negative coordinates *can* arise, but only as outputs of `Pad`/`Embed`
  ///     for elements that land in the padding/halo region. Those elements are
  ///     bounds-checked at the point they are produced, validity is
  ///     AND-accumulated (monotonic, so a masked element can never be
  ///     un-masked), and the eventual `tt.load` uses that mask with `other =
  ///     0`. So a "wrong" unsigned result on such an element is never
  ///     dereferenced nor observed.
  ///
  /// Besides being cheaper, this keeps the result analyzable by Triton's
  /// AxisInfoAnalysis. The previous `srem; cmpi slt 0; select` form collapsed
  /// pointer contiguity to 1 (defeating global-load vectorization), because the
  /// `select` takes the gcd with the comparison's constancy, which is unknown
  /// (AxisInfo cannot prove the operand is non-negative across the tile).
  Value visitModExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "modulo by non-positive value is not supported");
        return nullptr;
      }
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    auto [l, r] = ensureCompatible(builder, loc, lhs, rhs);
    // Unsigned: observed elements are non-negative (== euclidean mod); negative
    // elements are masked and never loaded, so the result is irrelevant.
    return arith::RemUIOp::create(builder, loc, l, r);
  }

  /// Floor division operation (rounds towards negative infinity).
  ///
  /// As with `visitModExpr`, the divisor is positive and the dividend is
  /// non-negative for every observed element, so floor division coincides with
  /// unsigned division. We emit `arith.divui` directly rather than the signed
  /// quotient guarded by a negative-value correction `select`. This is cheaper
  /// and, crucially, keeps pointer contiguity visible to Triton's
  /// AxisInfoAnalysis so that loads can be vectorized (see `visitModExpr` for
  /// why negative-coordinate elements are masked and therefore irrelevant).
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "division by non-positive value is not supported");
        return nullptr;
      }
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    auto [l, r] = ensureCompatible(builder, loc, lhs, rhs);
    return arith::DivUIOp::create(builder, loc, l, r);
  }

  /// Ceiling division operation (rounds towards positive infinity).
  ///
  /// As with `visitModExpr`, the divisor is positive and the dividend is
  /// non-negative for every observed element, so `ceildiv(a, b) == (a + b - 1)
  /// udiv b`. This avoids the signed negative-value correction `select`,
  /// keeping the result analyzable by Triton's AxisInfoAnalysis (see the
  /// comment on `visitModExpr`).
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if (rhsConst.getValue() <= 0) {
        emitError(loc, "division by non-positive value is not supported");
        return nullptr;
      }
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value oneCst = arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(indexType, 1));
    auto [r, o] = ensureCompatible(builder, loc, rhs, oneCst);
    Value divisorMinusOne = arith::SubIOp::create(builder, loc, r, o);
    auto [l, dm1] = ensureCompatible(builder, loc, lhs, divisorMinusOne);
    Value numerator = arith::AddIOp::create(builder, loc, l, dm1);
    auto [num, r2] = ensureCompatible(builder, loc, numerator, rhs);
    return arith::DivUIOp::create(builder, loc, num, r2);
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    return arith::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(indexType, expr.getValue()));
  }

  Value visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  OpBuilder &builder;
  Location loc;
  ValueRange dimValues;
  ValueRange symbolValues;
  // Element type (i32 or i64) used for constants introduced while expanding the
  // affine expression. Must match the width of the incoming coordinate values.
  Type indexType;
};

static Value expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                              ValueRange dimValues, ValueRange symbolValues,
                              Type indexType) {
  return AffineApplyExpander(builder, loc, dimValues, symbolValues, indexType)
      .visit(expr);
}
} // namespace

FailureOr<SmallVector<Value>> mlir::rock::expandAffineMap(OpBuilder &builder,
                                                          Location loc,
                                                          AffineMap affineMap,
                                                          ValueRange operands,
                                                          Type indexType) {
  auto numDims = affineMap.getNumDims();
  // rocMLIR currently uses static strides/shapes, so symbols are always 0.
  assert(affineMap.getNumSymbols() == 0 &&
         "dynamic shapes (affine symbols) not yet supported");
  if (operands.size() != numDims)
    return failure();

  auto expanded = llvm::to_vector(
      llvm::map_range(affineMap.getResults(), [&builder, loc, operands,
                                               indexType](AffineExpr expr) {
        return expandAffineExpr(builder, loc, expr, operands, {}, indexType);
      }));
  if (llvm::all_of(expanded, [](Value v) { return v; }))
    return expanded;
  return failure();
}

//===----------------------------------------------------------------------===//
// Validity.
//===----------------------------------------------------------------------===//

// Emit the validity (bounds) checks contributed by a single validity-impacting
// `map`, given the just-computed lower coordinates `outputs`. Returns an i1
// value (scalar or tensor) that is true where every check passes.
static Value updateValidityAfter(OpBuilder &b, Location loc,
                                 TransformMapAttr map, ValueRange outputs,
                                 Type indexType) {
  Value isValid = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  ArrayRef<int64_t> lowerBounds = map.getLowerBounds();

  // unsigned < catches both negatives (as all negatives are > the bound)
  // and being too large on the right.
  auto addLowerDimUltClamp = [&](uint32_t lowerDim) {
    int64_t bound = lowerBounds[lowerDim];
    Value boundConst =
        arith::ConstantOp::create(b, loc, b.getIntegerAttr(indexType, bound));
    Value output = outputs[lowerDim];
    auto [o, bc] = ensureCompatible(b, loc, output, boundConst);
    Value inBounds =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, o, bc);
    auto [ib, iv] = ensureCompatible(b, loc, inBounds, isValid);
    isValid = arith::AndIOp::create(b, loc, ib, iv);
  };

  for (TransformAttr op : map.getOps()) {
    TransformType type = op.getType();
    ArrayRef<uint32_t> lowerDims = op.getLowerDims();
    ArrayRef<int64_t> params = op.getParams();
    if (type == TransformType::Pad) {
      for (const auto &pair : llvm::enumerate(lowerDims)) {
        size_t leftParam = 2 * pair.index();
        size_t rightParam = leftParam + 1;
        uint32_t lowerDim = pair.value();
        if (params[leftParam] == 0 && params[rightParam] == 0)
          continue;
        addLowerDimUltClamp(lowerDim);
      }
    }
    if (type == TransformType::Embed) {
      if (!embedCanBeInvalid(map, op))
        continue;
      addLowerDimUltClamp(op.getLowerDims()[0]);
    }
  }
  return isValid;
}

//===----------------------------------------------------------------------===//
// Chain expansion core.
//===----------------------------------------------------------------------===//

FailureOr<OffsetAndMask> mlir::rock::expandCoordsToOffsetAndMask(
    OpBuilder &b, Location loc, ArrayRef<TransformMapAttr> transforms,
    ValueRange startCoords, ArrayRef<int64_t> outShape, Type indexType,
    bool computeOffset) {
  using AffineResults = SmallVector<Value>;

  // Break the chain into segments that each end at a validity-impacting map,
  // composing the intervening maps into a single affine map. The trailing
  // segment (the offset-only maps below the last validity check) is only
  // needed when computing the offset; the mask-only path skips it.
  SmallVector<std::pair<AffineMap, TransformMapAttr>> composedMaps;
  SmallVector<TransformMapAttr> toCompose;
  for (TransformMapAttr t : transforms) {
    toCompose.push_back(t);
    if (mapImpactsValidity(t)) {
      composedMaps.emplace_back(composeTransforms(toCompose), t);
      toCompose.clear();
    }
  }
  if (computeOffset)
    composedMaps.emplace_back(composeTransforms(toCompose), nullptr);

  AffineResults computed(startCoords);
  Value isValid = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
  for (const auto &[composedMap, transform] : composedMaps) {
    if (!composedMap) // empty trailing segment
      continue;
    FailureOr<AffineResults> transformed =
        expandAffineMap(b, loc, composedMap, computed, indexType);
    if (failed(transformed))
      return failure();
    computed.assign(*transformed);
    if (transform) {
      Value validityUpdate =
          updateValidityAfter(b, loc, transform, computed, indexType);
      auto [vu, iv] = ensureCompatible(b, loc, validityUpdate, isValid);
      isValid = arith::AndIOp::create(b, loc, vu, iv);
    }
  }

  OffsetAndMask result;
  result.mask = broadcastToShape(b, loc, isValid, outShape);
  if (!result.mask)
    return emitError(loc) << "cannot broadcast validity mask of type "
                          << isValid.getType() << " to the output tile shape";
  if (computeOffset) {
    if (computed.size() != 1)
      return failure();
    result.offset = broadcastToShape(b, loc, computed[0], outShape);
    if (!result.offset)
      return emitError(loc)
             << "cannot broadcast offset of type " << computed[0].getType()
             << " to the output tile shape";
  }
  return result;
}
