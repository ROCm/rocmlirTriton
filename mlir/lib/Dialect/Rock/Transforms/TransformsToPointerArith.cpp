//===- TransformsToPointerArith.cpp - Expand transform maps to arith ------===//
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
//
// This pass lowers TransformsToPtrOp by expanding transform map chains into
// arithmetic operations that compute pointer offsets and validity masks.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTRANSFORMSTOPOINTERARITHPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-transforms-to-pointer-arith"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockTransformsToPointerArithPass
    : public rock::impl::RockTransformsToPointerArithPassBase<
          RockTransformsToPointerArithPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

namespace {

// Helper function to create a tensor range using tt.make_range with proper
// shape. Creates a tensor where the non-unit dimension contains values [start,
// end).
static Value makeRange(OpBuilder &b, Location loc, int32_t start, int32_t end,
                       int64_t numDims, int64_t nonUnitDim) {
  SmallVector<int64_t> unitDimIndices;
  assert(numDims > 0);
  assert(nonUnitDim >= 0 && nonUnitDim < numDims);

  for (int64_t i = 0; i < numDims; ++i) {
    if (nonUnitDim != i)
      unitDimIndices.push_back(i);
  }

  // Create 1D tensor type for tt.make_range
  auto tensorType1D = RankedTensorType::get({end - start}, b.getI32Type());

  // Create tt.make_range operation (1D)
  Value rangeTensor =
      triton::MakeRangeOp::create(b, loc, tensorType1D, start, end);

  // Use tt.expand_dims to restore the original shape
  // We need to insert unit dimensions at the correct positions
  Value expandedTensor = rangeTensor;
  for (int64_t unitDimIdx : unitDimIndices) {
    // Get current tensor type
    auto currentType = cast<RankedTensorType>(expandedTensor.getType());
    SmallVector<int64_t> newShape(currentType.getShape().begin(),
                                  currentType.getShape().end());
    newShape.insert(newShape.begin() + unitDimIdx, 1);
    auto expandedType = RankedTensorType::get(newShape, b.getI32Type());

    expandedTensor = triton::ExpandDimsOp::create(b, loc, expandedType,
                                                  expandedTensor, unitDimIdx);
  }
  return expandedTensor;
}

// Helper function to broadcast tensors to compatible shapes
static std::pair<Value, Value>
broadcastTensors(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  auto tensorLhsType = cast<RankedTensorType>(lhs.getType());
  auto tensorRhsType = cast<RankedTensorType>(rhs.getType());

  auto lhsShape = tensorLhsType.getShape();
  auto rhsShape = tensorRhsType.getShape();
  auto rank = lhsShape.size();

  // Check if we need broadcasting
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
      // Incompatible shapes - for now, assume they're compatible
      resultShape[i] = std::max(lhsShape[i], rhsShape[i]);
    }
  }

  if (!needsBroadcast) {
    // No broadcasting needed
    return {lhs, rhs};
  }

  // Create the broadcast result type
  auto resultType =
      RankedTensorType::get(resultShape, tensorLhsType.getElementType());

  // Broadcast lhs if needed using triton.broadcast
  Value broadcastedLhs = lhs;
  if (!llvm::equal(lhsShape, resultShape)) {
    broadcastedLhs = triton::BroadcastOp::create(builder, loc, resultType, lhs);
  }

  // Broadcast rhs if needed
  Value broadcastedRhs = rhs;
  if (!llvm::equal(rhsShape, resultShape)) {
    auto rhsResultType =
        RankedTensorType::get(resultShape, tensorRhsType.getElementType());
    broadcastedRhs =
        triton::BroadcastOp::create(builder, loc, rhsResultType, rhs);
  }

  return {broadcastedLhs, broadcastedRhs};
}

// Helper function to broadcast a scalar to match a tensor's shape
static Value broadcastScalarToTensor(OpBuilder &builder, Location loc,
                                     Value scalar, Value tensor) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto shape = tensorType.getShape();
  auto elementType = scalar.getType();

  // Use triton.splat to broadcast scalar to tensor
  auto resultType = RankedTensorType::get(shape, elementType);
  return triton::SplatOp::create(builder, loc, resultType, scalar);
}

// Helper function to ensure operands have compatible shapes
static SmallVector<Value>
ensureCompatibleShapes(OpBuilder &builder, Location loc, ValueRange values) {
  if (values.size() < 2)
    return SmallVector<Value>(values);

  SmallVector<Value> results(values);

  // we need to run two passes, to make sure we propagate all broadcasts
  for (int pass = 0; pass < 2; pass++) {
    Value lhs = results[0];
    for (size_t i = 1; i < results.size(); i++) {
      Value rhs = results[i];
      auto lhsType = lhs.getType();
      auto rhsType = rhs.getType();

      auto lhsTensorType = dyn_cast<RankedTensorType>(lhsType);
      auto rhsTensorType = dyn_cast<RankedTensorType>(rhsType);

      if (!lhsTensorType && !rhsTensorType) {
        // Both scalars, no broadcasting needed
      } else if (lhsTensorType && !rhsTensorType) {
        // LHS is tensor, RHS is scalar - broadcast RHS
        rhs = broadcastScalarToTensor(builder, loc, rhs, lhs);
      } else if (!lhsTensorType && rhsTensorType) {
        // LHS is scalar, RHS is tensor - broadcast LHS
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

// Helper to create native arith operations (works on both scalars and tensors)
static Value createNativeArithOp(OpBuilder &builder, Location loc,
                                 StringRef name, Attribute constantValue,
                                 ValueRange operands) {
  if (name == "ConstantIntOp" || name == "ConstantOp") {
    if (auto intAttr = dyn_cast<IntegerAttr>(constantValue)) {
      return arith::ConstantOp::create(builder, loc, intAttr);
    } else if (auto boolAttr = dyn_cast<BoolAttr>(constantValue)) {
      return arith::ConstantOp::create(builder, loc, boolAttr);
    }
    llvm_unreachable("Unsupported constant type");
  }

  // arith ops work natively on both scalars and tensors
  if (name == "AddIOp") {
    return arith::AddIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "SubIOp") {
    return arith::SubIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "MulIOp") {
    return arith::MulIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "DivSIOp") {
    return arith::DivSIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "DivUIOp") {
    return arith::DivUIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "RemSIOp") {
    return arith::RemSIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "RemUIOp") {
    return arith::RemUIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "AndIOp") {
    return arith::AndIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "OrIOp") {
    return arith::OrIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "XOrIOp") {
    return arith::XOrIOp::create(builder, loc, operands[0], operands[1]);
  }
  if (name == "SelectOp") {
    return arith::SelectOp::create(builder, loc, operands[0], operands[1],
                                   operands[2]);
  }
  if (name.starts_with("CmpIOp_")) {
    StringRef predStr = name.drop_front(7); // Remove "CmpIOp_"
    arith::CmpIPredicate pred;
    if (predStr == "eq")
      pred = arith::CmpIPredicate::eq;
    else if (predStr == "ne")
      pred = arith::CmpIPredicate::ne;
    else if (predStr == "slt")
      pred = arith::CmpIPredicate::slt;
    else if (predStr == "sle")
      pred = arith::CmpIPredicate::sle;
    else if (predStr == "sgt")
      pred = arith::CmpIPredicate::sgt;
    else if (predStr == "sge")
      pred = arith::CmpIPredicate::sge;
    else if (predStr == "ult")
      pred = arith::CmpIPredicate::ult;
    else if (predStr == "ule")
      pred = arith::CmpIPredicate::ule;
    else if (predStr == "ugt")
      pred = arith::CmpIPredicate::ugt;
    else if (predStr == "uge")
      pred = arith::CmpIPredicate::uge;
    else
      llvm_unreachable("Unknown comparison predicate");
    return arith::CmpIOp::create(builder, loc, pred, operands[0], operands[1]);
  }

  llvm_unreachable("Unknown arith operation");
}

// Helper to create ArithOp - generates native arith operations
static Value createArithOp(OpBuilder &builder, Location loc, Type resultType,
                           StringRef name, Attribute constantValue,
                           ValueRange operands) {

  auto newOperands = ensureCompatibleShapes(builder, loc, operands);

  if (!newOperands.empty()) {
    assert(resultType == nullptr);
    resultType = newOperands[0].getType();
    if (name.contains("SelectOp"))
      resultType = newOperands[1].getType();

    if (name.contains("Cmp")) {
      if (isa<RankedTensorType>(resultType)) {
        auto tensorType = cast<RankedTensorType>(resultType);
        resultType =
            RankedTensorType::get(tensorType.getShape(), builder.getI1Type());
      } else {
        resultType = builder.getI1Type();
      }
    }
  } else {
    assert(resultType != nullptr);
  }

  // arith ops work natively on both scalars and tensors
  return createNativeArithOp(builder, loc, name, constantValue, newOperands);
}

/// Visit affine expressions recursively and build the sequence of operations
/// that correspond to it.  Visitation functions return an Value of the
/// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  /// This internal class expects arguments to be non-null, checks must be
  /// performed at the call site.
  AffineApplyExpander(OpBuilder &builder, ValueRange dimValues,
                      ValueRange symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  Value buildBinaryExpr(AffineBinaryOpExpr expr, const std::string &opName,
                        arith::IntegerOverflowFlags overflowFlags =
                            arith::IntegerOverflowFlags::none) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;

    // Use native arith operations
    return createArithOp(builder, loc, nullptr, opName, nullptr,
                         ValueRange{lhs, rhs});
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr(expr, "AddIOp");
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr(expr, "MulIOp", arith::IntegerOverflowFlags::nsw);
  }

  /// Euclidean modulo operation: negative RHS is not allowed.
  /// Remainder of the euclidean integer division is always non-negative.
  ///
  /// Implemented as
  ///
  ///     a mod b =
  ///         let remainder = srem a, b;
  ///             negative = a < 0 in
  ///         select negative, remainder + b, remainder.
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

    Value remainder = createArithOp(builder, loc, nullptr, "RemSIOp", nullptr,
                                    ValueRange{lhs, rhs});
    Value zeroCst = createArithOp(
        builder, loc, builder.getI32Type(), "ConstantIntOp",
        builder.getIntegerAttr(builder.getI32Type(), 0), ValueRange{});
    Value isRemainderNegative =
        createArithOp(builder, loc, nullptr, "CmpIOp_slt", nullptr,
                      ValueRange{remainder, zeroCst});
    Value correctedRemainder = createArithOp(builder, loc, nullptr, "AddIOp",
                                             nullptr, ValueRange{remainder, rhs});
    Value result =
        createArithOp(builder, loc, nullptr, "SelectOp", nullptr,
                      ValueRange{isRemainderNegative, correctedRemainder, remainder});
    return result;
  }

  /// Floor division operation (rounds towards negative infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///        a floordiv b =
  ///            let negative = a < 0 in
  ///            let absolute = negative ? -a - 1 : a in
  ///            let quotient = absolute / b in
  ///                negative ? -quotient - 1 : quotient
  ///
  /// Note: this lowering does not use arith.floordivsi because the lowering of
  /// that to arith.divsi (see populateCeilFloorDivExpandOpsPatterns) generates
  /// not one but two arith.divsi. That could be changed to one divsi, but one
  /// way or another, going through arith.floordivsi will result in more complex
  /// IR because arith.floordivsi is more general than affine floordiv in that
  /// it supports negative RHS.
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

    Value zeroCst = createArithOp(
        builder, loc, builder.getI32Type(), "ConstantIntOp",
        builder.getIntegerAttr(builder.getI32Type(), 0), ValueRange{});
    Value noneCst = createArithOp(
        builder, loc, builder.getI32Type(), "ConstantIntOp",
        builder.getIntegerAttr(builder.getI32Type(), -1), ValueRange{});
    Value negative = createArithOp(builder, loc, nullptr, "CmpIOp_slt", nullptr,
                                   ValueRange{lhs, zeroCst});
    Value negatedDecremented = createArithOp(builder, loc, nullptr, "SubIOp",
                                             nullptr, ValueRange{noneCst, lhs});
    Value dividend =
        createArithOp(builder, loc, nullptr, "SelectOp", nullptr,
                      ValueRange{negative, negatedDecremented, lhs});
    Value quotient = createArithOp(builder, loc, nullptr, "DivSIOp", nullptr,
                                   ValueRange{dividend, rhs});
    Value correctedQuotient = createArithOp(builder, loc, nullptr, "SubIOp",
                                            nullptr, ValueRange{noneCst, quotient});
    Value result =
        createArithOp(builder, loc, nullptr, "SelectOp", nullptr,
                      ValueRange{negative, correctedQuotient, quotient});
    return result;
  }

  /// Ceiling division operation (rounds towards positive infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///     a ceildiv b =
  ///         let negative = a <= 0 in
  ///         let absolute = negative ? -a : a - 1 in
  ///         let quotient = absolute / b in
  ///             negative ? -quotient : quotient + 1
  ///
  /// Note: not using arith.ceildivsi for the same reason as explained in the
  /// visitFloorDivExpr comment.
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

    Value zeroCst = createArithOp(
        builder, loc, builder.getI32Type(), "ConstantIntOp",
        builder.getIntegerAttr(builder.getI32Type(), 0), ValueRange{});
    Value oneCst = createArithOp(
        builder, loc, builder.getI32Type(), "ConstantIntOp",
        builder.getIntegerAttr(builder.getI32Type(), 1), ValueRange{});
    Value nonPositive = createArithOp(builder, loc, nullptr, "CmpIOp_sle",
                                      nullptr, ValueRange{lhs, zeroCst});
    Value negated = createArithOp(builder, loc, nullptr, "SubIOp", nullptr,
                                  ValueRange{zeroCst, lhs});
    Value decremented = createArithOp(builder, loc, nullptr, "SubIOp", nullptr,
                                      ValueRange{lhs, oneCst});
    Value dividend =
        createArithOp(builder, loc, nullptr, "SelectOp", nullptr,
                      ValueRange{nonPositive, negated, decremented});
    Value quotient = createArithOp(builder, loc, nullptr, "DivSIOp", nullptr,
                                   ValueRange{dividend, rhs});
    Value negatedQuotient = createArithOp(builder, loc, nullptr, "SubIOp",
                                          nullptr, ValueRange{zeroCst, quotient});
    Value incrementedQuotient = createArithOp(builder, loc, nullptr, "AddIOp",
                                              nullptr, ValueRange{quotient, oneCst});
    Value result = createArithOp(
        builder, loc, nullptr, "SelectOp", nullptr,
        ValueRange{nonPositive, negatedQuotient, incrementedQuotient});
    return result;
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    return createArithOp(
        builder, loc, builder.getI32Type(), "ConstantIntOp",
        builder.getIntegerAttr(builder.getI32Type(), expr.getValue()),
        ValueRange{});
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
  ValueRange dimValues;
  ValueRange symbolValues;

  Location loc;
};

/// Create a sequence of operations that implement the `expr` applied to the
/// given dimension and symbol values.
static mlir::Value expandAffineExpr(OpBuilder &builder, Location loc,
                                    AffineExpr expr, ValueRange dimValues,
                                    ValueRange symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

/// Create a sequence of operations that implement the `affineMap` applied to
/// the given `operands` (as it it were an AffineApplyOp).
static FailureOr<SmallVector<Value>> expandAffineMap(OpBuilder &builder,
                                                     Location loc,
                                                     AffineMap affineMap,
                                                     ValueRange operands) {
  auto numDims = affineMap.getNumDims();
  // rocMLIR currently uses static strides/shapes, so symbols are always 0.
  assert(affineMap.getNumSymbols() == 0 &&
         "dynamic shapes (affine symbols) not yet supported");
  if (operands.size() < numDims)
    return failure();

  auto expanded = llvm::to_vector(
      llvm::map_range(affineMap.getResults(),
                      [numDims, &builder, loc, operands](AffineExpr expr) {
                        return expandAffineExpr(builder, loc, expr,
                                                operands.take_front(numDims),
                                                operands.drop_front(numDims));
                      }));
  if (llvm::all_of(expanded, [](Value v) { return v; }))
    return expanded;
  return failure();
}

static Value updateValidityAfter(OpBuilder &b, Location loc,
                                 TransformMapAttr map, ValueRange outputs) {
  Value isValid = createArithOp(b, loc, b.getI1Type(), "ConstantIntOp",
                                b.getBoolAttr(true), ValueRange{});
  ArrayRef<int64_t> lowerBounds = map.getLowerBounds();

  // unsigned < catches both negatives (as all negatives are > the bound)
  // and being too large on the right.
  auto addLowerDimUltClamp = [&](uint32_t lowerDim) {
    int64_t bound = lowerBounds[lowerDim];
    Value boundConst =
        createArithOp(b, loc, b.getI32Type(), "ConstantIntOp",
                      b.getIntegerAttr(b.getI32Type(), bound), ValueRange{});
    Value output = outputs[lowerDim];
    Value inBounds = createArithOp(b, loc, nullptr, "CmpIOp_ult", nullptr,
                                   ValueRange{output, boundConst});
    isValid = createArithOp(b, loc, nullptr, "AndIOp", nullptr,
                            ValueRange{inBounds, isValid});
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
// TransformsToPtrOp lowering.
//===----------------------------------------------------------------------===//
struct TransformsToPtrRewritePattern
    : public OpRewritePattern<TransformsToPtrOp> {
  using OpRewritePattern<TransformsToPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransformsToPtrOp op,
                                PatternRewriter &b) const override {
    using AffineResults = SmallVector<Value>;
    Location loc = op.getLoc();
    Value source = op.getSource();
    ValueRange extraIndices = op.getExtraIndices();

    // Get output shapes from result types (tensors)
    auto pointerResultType = cast<RankedTensorType>(op.getPointers().getType());
    ArrayRef<int64_t> shape = pointerResultType.getShape();

    source = isolateTransforms(b, source);

    // TODO(roctriton): buffer could be the output of input fusion instead of an
    // input tensor! Fix this when we enable fusions.
    auto [buffer, transforms, needs64BitIdx] = untransform(b, source);

    size_t bufferIdxCount = shape.size();
    if (bufferIdxCount == 0) {
      return op.emitOpError("output pointer tensor must have non-zero rank");
    }
    size_t extraIdxCount = extraIndices.size();
    if (extraIdxCount < bufferIdxCount) {
      return op.emitOpError("requires at least ")
             << bufferIdxCount << " extra indices for output tile of rank "
             << bufferIdxCount << ", but only " << extraIdxCount
             << " indices were provided";
    }
    SmallVector<Value> initValues(extraIndices);
    for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
      // Create the range values using triton.make_range
      auto rangeValue =
          makeRange(b, loc, 0, shape[dimension], shape.size(), dimension);
      initValues.push_back(rangeValue);
    }

    // TODO(roctriton): check rangeIndices match `pointers` and `mask` shapes

    // For each domain, store the sequence of composed affine maps needed to
    // compute the result coordinate, along with the transform map that
    // triggered each break in the chain. Such a break is created at any point
    // where the validity of map coordinates is impacted.
    SmallVector<std::pair<AffineMap, TransformMapAttr>> composedMaps;

    SmallVector<TransformMapAttr> toCompose;
    for (auto t : transforms.getAsRange<TransformMapAttr>()) {
      toCompose.push_back(t);
      if (mapImpactsValidity(t)) {
        AffineMap composed = composeTransforms(toCompose);
        composedMaps.emplace_back(composed, t);
        toCompose.clear();
      }
    }
    // Account for all maps after the last validity impact.
    AffineMap finalComposed = composeTransforms(toCompose);
    composedMaps.emplace_back(finalComposed, nullptr);

    // Create code to actually transform the coordinates
    AffineResults computed(initValues);
    Value isValid = createArithOp(b, loc, b.getI1Type(), "ConstantIntOp",
                                  b.getBoolAttr(true), ValueRange{});
    for (const auto &[composedMap, transform] : composedMaps) {
      if (!composedMap) // empty transformations
        continue;
      FailureOr<AffineResults> transformed =
          expandAffineMap(b, loc, composedMap, computed);
      if (failed(transformed))
        return op.emitOpError("Transforms are not well formed");
      computed.assign(*transformed);
      if (transform) { // Time for bounds checks or other validity updates
        Value validityUpdate = updateValidityAfter(b, loc, transform, computed);
        isValid = createArithOp(b, loc, nullptr, "AndIOp", nullptr,
                                ValueRange{validityUpdate, isValid});
      }
    }

    // Hoist pointer extraction to function entry to avoid redundant extractions
    // when TransformsToPtrOp is inside loops or other control flow.
    // For constant buffers (like fakeTensor used for index calculations),
    // we use a base pointer of 0 since the actual pointer value doesn't matter.
    Value baseAddr;
    {
      OpBuilder::InsertionGuard guard(b);

      bool isConstantBuffer =
          buffer.getDefiningOp<arith::ConstantOp>() != nullptr;

      if (isConstantBuffer) {
        // For constants (like fakeTensor), use base pointer of 0
        // These are only used for index calculations, not actual memory access
        baseAddr =
            arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(0));
      } else {
        // For function arguments, hoist to function entry
        auto parentFunc = op->getParentOfType<func::FuncOp>();
        b.setInsertionPointToStart(&parentFunc.front());

        // Extract the base pointer from the tensor as i32
        baseAddr =
              rock::ExtractPtrOp::create(b, loc, b.getI32Type(), buffer);
              
      }
    }
    // Use tensor.splat for broadcasting scalar to triton
    auto splatType = RankedTensorType::get(shape, b.getI32Type());
    Value baseAddrSplat = triton::SplatOp::create(b, loc, splatType, baseAddr);
    // InsertionGuard restores original insertion point here

    // add `baseAddr` using linalg.map for tensor addition
    Value pointerTensor =
        createArithOp(b, loc, nullptr, "AddIOp", nullptr,
                      {baseAddrSplat, computed[0]});

    // Create the mask tensor by broadcasting isValid to the right shape
    Value maskTensor;
    if (isa<RankedTensorType>(isValid.getType())) {
      // isValid is already a tensor, ensure it has the right shape
      auto isValidTensorType = cast<RankedTensorType>(isValid.getType());
      if (isValidTensorType.getShape() != shape) {
        // Need to broadcast using triton.broadcast
        auto maskType = RankedTensorType::get(shape, b.getI1Type());
        maskTensor = triton::BroadcastOp::create(b, loc, maskType, isValid);
      } else {
        maskTensor = isValid;
      }
    } else {
      // isValid is a scalar, splat it to tensor using tensor.splat
      auto maskType = RankedTensorType::get(shape, b.getI1Type());
      maskTensor = triton::SplatOp::create(b, loc, maskType, isValid);
    }

    // Replace the op with the tensor results
    b.replaceOp(op, {pointerTensor, maskTensor});

    return success();
  }
};

} // end anonymous namespace

void RockTransformsToPointerArithPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::TransformsToPtrOp>();
  // Note: We don't mark TransformOp as illegal. After TransformsToPtrOp
  // conversion, transform chains become dead code (each transform only used
  // by the next transform in the chain). These will be cleaned up by
  // Canonicalizer.
  target.addLegalDialect<rock::RockDialect, arith::ArithDialect,
                         tensor::TensorDialect, triton::TritonDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<TransformsToPtrRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
