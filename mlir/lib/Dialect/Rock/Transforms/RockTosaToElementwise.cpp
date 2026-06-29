//===- RockTosaToElementwise.cpp - TOSA elementwise to arith/math --------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// Convert TOSA pointwise/elementwise ops directly to arith / math ops on
// tensors.  For example:
//
//   %r = tosa.add %a, %b : tensor<64x128xf16>
//     ->
//   %r = arith.addf %a, %b : tensor<64x128xf16>
//
// This pass only runs on kernel functions (those with the rock.kernel attr).
// Non-kernel (CPU) functions are left untouched so the normal tosa-to-linalg
// pipeline can lower them through linalg.generic later.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTosaCustomOps.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTOSATOELEMENTWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {

// ===--------------------------------------------------------------------=== //
// Generic templates for 1:1 TOSA -> arith/math replacement on tensors.
// ===--------------------------------------------------------------------=== //

// Create a rock.transform that broadcasts `input` to `targetShape`.
// Dimensions where the source has size 1 and the target has size > 1 use
// Broadcast{1} (i.e. index % 1 = 0). Returns the input unchanged if shapes
// already match, or failure if the shapes are incompatible.
static FailureOr<Value>
createBroadcastTransform(PatternRewriter &rewriter, Location loc, Value input,
                         ArrayRef<int64_t> targetShape) {
  auto inputTy = cast<ShapedType>(input.getType());
  ArrayRef<int64_t> inputShape = inputTy.getShape();
  if (inputShape == targetShape)
    return input;
  if (inputShape.size() != targetShape.size())
    return failure();

  SmallVector<SmallString<8>, 8> nameStorage(targetShape.size());
  SmallVector<StringRef> dimNames;
  for (unsigned i = 0; i < targetShape.size(); ++i) {
    ("dim" + Twine(i)).toVector(nameStorage[i]);
    dimNames.push_back(nameStorage[i]);
  }

  rock::TopDownTMBuilder bcast(rewriter, dimNames, targetShape, loc);
  for (unsigned i = 0; i < targetShape.size(); ++i) {
    if (inputShape[i] == targetShape[i]) {
      bcast.passThrough({dimNames[i]});
    } else if (inputShape[i] == 1) {
      bcast.takeRemainder(dimNames[i], 1);
    } else {
      return failure();
    }
  }

  Value result = rock::TransformOp::create(rewriter, loc, input, bcast.get());
  return result;
}

// Broadcast two binary operands to the result shape of the given TOSA op.
// Works with any op that has getInput1(), getInput2(), getType(), getLoc().
template <typename OpTy>
static FailureOr<std::pair<Value, Value>>
broadcastInputs(PatternRewriter &rewriter, OpTy op) {
  auto resultShape = cast<ShapedType>(op.getType()).getShape();
  auto b1 =
      createBroadcastTransform(rewriter, op.getLoc(), op.getInput1(), resultShape);
  auto b2 =
      createBroadcastTransform(rewriter, op.getLoc(), op.getInput2(), resultShape);
  if (failed(b1) || failed(b2))
    return failure();
  return std::make_pair(*b1, *b2);
}

// Binary op that dispatches on float vs integer element type.
template <typename TosaOp, typename FloatOp, typename IntOp>
struct BinaryConverter : public OpRewritePattern<TosaOp> {
  using OpRewritePattern<TosaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = broadcastInputs(rewriter, op);
    if (failed(inputs))
      return failure();
    auto [in1, in2] = *inputs;
    auto elemTy = cast<ShapedType>(op.getType()).getElementType();
    if (isa<FloatType>(elemTy))
      rewriter.replaceOpWithNewOp<FloatOp>(op, op.getType(), in1, in2);
    else if (isa<IntegerType>(elemTy))
      rewriter.replaceOpWithNewOp<IntOp>(op, op.getType(), in1, in2);
    else
      return failure();
    return success();
  }
};

// Unary op whose ODS argument is $input1 (e.g. tosa.exp, tosa.log).
template <typename TosaOp, typename TargetOp>
struct UnaryConverter : public OpRewritePattern<TosaOp> {
  using OpRewritePattern<TosaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getType(), op.getInput1());
    return success();
  }
};

// Unary op whose ODS argument is $input (e.g. tosa.tanh, tosa.erf).
template <typename TosaOp, typename TargetOp>
struct UnaryInputConverter : public OpRewritePattern<TosaOp> {
  using OpRewritePattern<TosaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getType(), op.getInput());
    return success();
  }
};

// Integer-only binary op.
template <typename TosaOp, typename IntOp>
struct IntBinaryConverter : public OpRewritePattern<TosaOp> {
  using OpRewritePattern<TosaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = broadcastInputs(rewriter, op);
    if (failed(inputs))
      return failure();
    auto [in1, in2] = *inputs;
    rewriter.replaceOpWithNewOp<IntOp>(op, op.getType(), in1, in2);
    return success();
  }
};

// ===--------------------------------------------------------------------=== //
// Special-case patterns.
// ===--------------------------------------------------------------------=== //

// tosa.abs: math.absf (float) or max(x, -x) (int)
struct AbsConverter : public OpRewritePattern<tosa::AbsOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::AbsOp op,
                                PatternRewriter &rewriter) const override {
    auto elemTy = cast<ShapedType>(op.getType()).getElementType();
    if (isa<FloatType>(elemTy)) {
      rewriter.replaceOpWithNewOp<math::AbsFOp>(op, op.getType(),
                                                op.getInput1());
    } else if (isa<IntegerType>(elemTy)) {
      Location loc = op.getLoc();
      auto zeroAttr = DenseElementsAttr::get(
          cast<ShapedType>(op.getType()), rewriter.getIntegerAttr(elemTy, 0));
      Value zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
      Value neg = arith::SubIOp::create(rewriter, loc, op.getType(), zero,
                                        op.getInput1());
      rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, op.getInput1(), neg);
    } else {
      return failure();
    }
    return success();
  }
};

// tosa.negate: arith.negf (float) or 0 - x (int, no quantization)
struct NegateConverter : public OpRewritePattern<tosa::NegateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::NegateOp op,
                                PatternRewriter &rewriter) const override {
    auto elemTy = cast<ShapedType>(op.getType()).getElementType();
    if (isa<FloatType>(elemTy)) {
      rewriter.replaceOpWithNewOp<arith::NegFOp>(op, op.getType(),
                                                 op.getInput1());
    } else if (isa<IntegerType>(elemTy)) {
      Location loc = op.getLoc();
      auto zeroAttr = DenseElementsAttr::get(
          cast<ShapedType>(op.getType()), rewriter.getIntegerAttr(elemTy, 0));
      Value zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, op.getType(), zero,
                                                 op.getInput1());
    } else {
      return failure();
    }
    return success();
  }
};

// tosa.mul: arith.mulf (float) or arith.muli (int, shift == 0)
struct MulConverter : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = broadcastInputs(rewriter, op);
    if (failed(inputs))
      return failure();
    auto [in1, in2] = *inputs;
    auto elemTy = cast<ShapedType>(op.getType()).getElementType();
    if (isa<FloatType>(elemTy)) {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getType(), in1, in2);
      return success();
    }
    if (isa<IntegerType>(elemTy)) {
      DenseElementsAttr shiftElem;
      if (matchPattern(op.getShift(), m_Constant(&shiftElem)) &&
          shiftElem.getValues<IntegerAttr>()[0].getInt() == 0) {
        rewriter.replaceOpWithNewOp<arith::MulIOp>(op, op.getType(), in1, in2);
        return success();
      }
      return rewriter.notifyMatchFailure(op, "non-zero shift unsupported");
    }
    return failure();
  }
};

// tosa.reciprocal(tosa.rsqrt(x)) → math.sqrt(x)
// Fold the two-op sqrt decomposition (from MIGraphXToTosa) into a single op.
struct ReciprocalRsqrtToSqrtConverter
    : public OpRewritePattern<tosa::ReciprocalOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::ReciprocalOp op,
                                PatternRewriter &rewriter) const override {
    auto rsqrt = op.getInput1().getDefiningOp<tosa::RsqrtOp>();
    if (!rsqrt)
      return failure();
    rewriter.replaceOpWithNewOp<math::SqrtOp>(op, op.getType(),
                                              rsqrt.getInput1());
    if (rsqrt->use_empty())
      rewriter.eraseOp(rsqrt);
    return success();
  }
};

// tosa.reciprocal: arith.divf(splat(1.0), x)
struct ReciprocalConverter : public OpRewritePattern<tosa::ReciprocalOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::ReciprocalOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto shapedTy = cast<ShapedType>(op.getType());
    auto oneAttr = DenseElementsAttr::get(
        shapedTy, rewriter.getFloatAttr(shapedTy.getElementType(), 1.0));
    Value one = arith::ConstantOp::create(rewriter, loc, oneAttr);
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, op.getType(), one,
                                               op.getInput1());
    return success();
  }
};

// arith.negf(x) -> arith.mulf(x, -1.0)
// This supports migraphx.neg operator, which will be expanded to arith.negf.
// However, the TritonToTritonGPU conversion does not have a pattern for
// arith.negf, so we expand it here into ops that Triton supports.
//
// Note that the cleanest way would be to make Triton support arith.negf,
// however they rejected this change:
// https://github.com/triton-lang/triton/pull/9955
struct NegFTritonWorkaround : public OpRewritePattern<arith::NegFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::NegFOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto shapedTy = dyn_cast<ShapedType>(op.getType());
    if (!shapedTy)
      return failure();
    auto negOneAttr = DenseElementsAttr::get(
        shapedTy, rewriter.getFloatAttr(shapedTy.getElementType(), -1.0));
    Value negOne = arith::ConstantOp::create(rewriter, loc, negOneAttr);
    rewriter.replaceOpWithNewOp<arith::MulFOp>(op, op.getType(),
                                               op.getOperand(), negOne);
    return success();
  }
};

// tosa.sigmoid: 1 / (1 + exp(-x))
// We compute -x as (0 - x) to match both MIGraphX semantics and Triton's
// implementation, avoiding arith.negf which Triton doesn't support on tensors.
// See: triton/python/triton/language/standard.py (sigmoid)
//      triton/python/triton/language/semantic.py (minus)
// Please note that we dont want to generate arith.negf here, because
// it is not supported by Triton.
struct SigmoidConverter : public OpRewritePattern<tosa::SigmoidOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto shapedTy = cast<ShapedType>(op.getType());
    auto zeroAttr = DenseElementsAttr::get(
        shapedTy, rewriter.getFloatAttr(shapedTy.getElementType(), 0.0));
    Value zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
    auto oneAttr = DenseElementsAttr::get(
        shapedTy, rewriter.getFloatAttr(shapedTy.getElementType(), 1.0));
    Value one = arith::ConstantOp::create(rewriter, loc, oneAttr);
    Value negX =
        arith::SubFOp::create(rewriter, loc, op.getType(), zero, op.getInput());
    Value expNegX = math::ExpOp::create(rewriter, loc, op.getType(), negX);
    Value denom =
        arith::AddFOp::create(rewriter, loc, op.getType(), one, expNegX);
    rewriter.replaceOpWithNewOp<arith::DivFOp>(op, op.getType(), one, denom);
    return success();
  }
};

// tosa.select: arith.select(pred, on_true, on_false)
struct SelectConverter : public OpRewritePattern<tosa::SelectOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::SelectOp op,
                                PatternRewriter &rewriter) const override {
    auto resultShape = cast<ShapedType>(op.getType()).getShape();
    auto pred = createBroadcastTransform(rewriter, op.getLoc(), op.getInput1(),
                                         resultShape);
    auto onTrue = createBroadcastTransform(rewriter, op.getLoc(),
                                           op.getInput2(), resultShape);
    auto onFalse = createBroadcastTransform(rewriter, op.getLoc(),
                                            op.getInput3(), resultShape);
    if (failed(pred) || failed(onTrue) || failed(onFalse))
      return failure();
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, *pred, *onTrue, *onFalse);
    return success();
  }
};

// tosa.clamp (float): arith.maximumf(arith.minimumf(x, max), min)
// tosa.clamp (int):   arith.maxsi(arith.minsi(x, max), min)
struct ClampConverter : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::ClampOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto shapedTy = cast<ShapedType>(op.getType());
    auto elemTy = shapedTy.getElementType();
    Value x = op.getInput();

    if (isa<FloatType>(elemTy)) {
      auto minApf = cast<FloatAttr>(op->getAttr("min_val")).getValue();
      auto maxApf = cast<FloatAttr>(op->getAttr("max_val")).getValue();
      bool losesInfo = false;
      minApf.convert(cast<FloatType>(elemTy).getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &losesInfo);
      maxApf.convert(cast<FloatType>(elemTy).getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &losesInfo);

      Value minVal = arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(shapedTy,
                                 rewriter.getFloatAttr(elemTy, minApf)));
      Value maxVal = arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(shapedTy,
                                 rewriter.getFloatAttr(elemTy, maxApf)));
      Value clamped = arith::MinimumFOp::create(rewriter, loc, x, maxVal);
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(op, clamped, minVal);
    } else if (isa<IntegerType>(elemTy)) {
      int64_t minI =
          cast<IntegerAttr>(op->getAttr("min_val")).getValue().getSExtValue();
      int64_t maxI =
          cast<IntegerAttr>(op->getAttr("max_val")).getValue().getSExtValue();
      Value minVal = arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(shapedTy,
                                 rewriter.getIntegerAttr(elemTy, minI)));
      Value maxVal = arith::ConstantOp::create(
          rewriter, loc,
          DenseElementsAttr::get(shapedTy,
                                 rewriter.getIntegerAttr(elemTy, maxI)));
      Value clamped = arith::MinSIOp::create(rewriter, loc, x, maxVal);
      rewriter.replaceOpWithNewOp<arith::MaxSIOp>(op, clamped, minVal);
    } else {
      return failure();
    }
    return success();
  }
};

// tosa.cast: dispatch to the appropriate arith cast op
struct CastConverter : public OpRewritePattern<tosa::CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = cast<ShapedType>(op.getInput().getType()).getElementType();
    auto dstTy = cast<ShapedType>(op.getType()).getElementType();

    if (srcTy == dstTy) {
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    bool bitExtend =
        srcTy.getIntOrFloatBitWidth() < dstTy.getIntOrFloatBitWidth();

    // float -> float
    if (isa<FloatType>(srcTy) && isa<FloatType>(dstTy)) {
      if (bitExtend)
        rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, op.getType(),
                                                   op.getInput());
      else
        rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, op.getType(),
                                                     op.getInput());
      return success();
    }

    // int -> float
    if (isa<IntegerType>(srcTy) && isa<FloatType>(dstTy)) {
      if (srcTy.isInteger(1) || srcTy.isUnsignedInteger())
        rewriter.replaceOpWithNewOp<arith::UIToFPOp>(op, op.getType(),
                                                     op.getInput());
      else
        rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, op.getType(),
                                                     op.getInput());
      return success();
    }

    // float -> bool
    if (isa<FloatType>(srcTy) && dstTy.isInteger(1)) {
      auto srcShapedTy = cast<ShapedType>(op.getInput().getType());
      Value zero = arith::ConstantOp::create(
          rewriter, op.getLoc(),
          DenseElementsAttr::get(srcShapedTy,
                                 rewriter.getFloatAttr(srcTy, 0.0)));
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(
          op, arith::CmpFPredicate::UNE, op.getInput(), zero);
      return success();
    }

    // float -> int: not reachable. This pass is only invoked from the
    // MIGraphX pipeline, and the MIGraphX frontend never emits a plain
    // `tosa.cast` for fp->int
    if (isa<FloatType>(srcTy) && isa<IntegerType>(dstTy)) {
      return op.emitOpError(
          "tosa.cast from floating-point to integer is not supported by "
          "rock-tosa-to-elementwise");
    }

    // int -> bool
    if (isa<IntegerType>(srcTy) && dstTy.isInteger(1)) {
      auto srcShapedTy = cast<ShapedType>(op.getInput().getType());
      Value zero = arith::ConstantOp::create(
          rewriter, op.getLoc(),
          DenseElementsAttr::get(
              srcShapedTy,
              rewriter.getIntegerAttr(srcTy, 0)));
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(
          op, arith::CmpIPredicate::ne, op.getInput(), zero);
      return success();
    }

    // int -> int
    if (isa<IntegerType>(srcTy) && isa<IntegerType>(dstTy)) {
      if (bitExtend) {
        if (srcTy.isInteger(1) || srcTy.isUnsignedInteger())
          rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, op.getType(),
                                                      op.getInput());
        else
          rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, op.getType(),
                                                      op.getInput());
      } else if (srcTy.getIntOrFloatBitWidth() > dstTy.getIntOrFloatBitWidth())
        rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, op.getType(),
                                                     op.getInput());
      else
        rewriter.replaceOp(op, op.getInput());
      return success();
    }
  
    return failure();
  }
};

// tosa.custom with domain "rocmlir": unsigned_cast, unsigned_div,
// and fp_to_int_cast.
// These are custom TOSA ops that represent operations which standard TOSA
// doesn't support or where we need to override upstream lowering behavior.
struct CustomOpConverter : public OpRewritePattern<tosa::CustomOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::CustomOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME)
      return failure();

    auto outType = cast<RankedTensorType>(op.getResults().front().getType());
    Type outElemType = outType.getElementType();

    if (op.getOperatorName() == ROCK_CUSTOMOP_UNSIGNED_CAST) {
      Value input = op.getInputList()[0];
      Type inElemType =
          cast<RankedTensorType>(input.getType()).getElementType();
      if (isa<IntegerType>(inElemType)) {
        if (isa<FloatType>(outElemType)) {
          rewriter.replaceOpWithNewOp<arith::UIToFPOp>(op, outType, input);
        } else if (outElemType.getIntOrFloatBitWidth() >
                   inElemType.getIntOrFloatBitWidth()) {
          rewriter.replaceOpWithNewOp<arith::ExtUIOp>(op, outType, input);
        } else if (outElemType.getIntOrFloatBitWidth() <
                   inElemType.getIntOrFloatBitWidth()) {
          rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, outType, input);
        } else {
          rewriter.replaceOp(op, input);
        }
      } else {
        Value result = rock::createClampedFPToInt(
            rewriter, op.getLoc(), input, outElemType, /*isUnsigned=*/true);
        rewriter.replaceOp(op, result);
      }
      return success();
    }

    if (op.getOperatorName() == ROCK_CUSTOMOP_FP_TO_INT_CAST) {
      Value input = op.getInputList()[0];
      Value result = rock::createClampedFPToInt(
          rewriter, op.getLoc(), input, outElemType, /*isUnsigned=*/false);
      rewriter.replaceOp(op, result);
      return success();
    }

    if (op.getOperatorName() == ROCK_CUSTOMOP_UNSIGNED_DIV) {
      rewriter.replaceOpWithNewOp<arith::DivUIOp>(
          op, outType, op.getInputList()[0], op.getInputList()[1]);
      return success();
    }

    return failure();
  }
};

// Comparison ops: tosa.greater / greater_equal / equal
template <typename TosaOp, arith::CmpFPredicate FPred,
          arith::CmpIPredicate IPred>
struct CmpConverter : public OpRewritePattern<TosaOp> {
  using OpRewritePattern<TosaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = broadcastInputs(rewriter, op);
    if (failed(inputs))
      return failure();
    auto [in1, in2] = *inputs;
    auto elemTy = cast<ShapedType>(in1.getType()).getElementType();
    if (isa<FloatType>(elemTy))
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, FPred, in1, in2);
    else if (isa<IntegerType>(elemTy))
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, IPred, in1, in2);
    else
      return failure();
    return success();
  }
};

// ===--------------------------------------------------------------------=== //
// Pass definition.
// ===--------------------------------------------------------------------=== //

struct RockTosaToElementwise
    : public rock::impl::RockTosaToElementwisePassBase<RockTosaToElementwise> {
  void runOnOperation() override {
    auto func = getOperation();
    if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    // check if the target has dedicated tanh instructions, in that case the
    // Triton pipeline emits those instructions directly from math.tanh. On such targets we keep
    // `math.tanh` instead of expanding it into elementary ops below.
    StringAttr arch = rock::getArchValueOnFunc(func);
    bool hasHardwareTanh = rock::archHasHardwareTanh(arch);

    target.addLegalDialect<arith::ArithDialect, math::MathDialect,
                           tensor::TensorDialect>();
    // Mark ops that Triton's TritonToTritonGPU conversion cannot handle as
    // illegal so the workaround patterns below get applied to them.
    target.addDynamicallyLegalOp<math::TanhOp>([&](math::TanhOp op) {
      return hasHardwareTanh || !isa<ShapedType>(op.getType());
    });
    target.addDynamicallyLegalOp<math::PowFOp>(
        [](math::PowFOp op) { return !isa<ShapedType>(op.getType()); });
    target.addDynamicallyLegalOp<arith::NegFOp>(
        [](arith::NegFOp op) { return !isa<ShapedType>(op.getType()); });
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isa<tosa::TosaDialect>(op->getDialect());
    });
    target.addLegalOp<tosa::ConstOp, tosa::ConstShapeOp>();
    target.addDynamicallyLegalOp<tosa::CustomOp>([](tosa::CustomOp op) {
      if (op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME)
        return true;
      StringRef name = op.getOperatorName();
      return name != ROCK_CUSTOMOP_UNSIGNED_CAST &&
             name != ROCK_CUSTOMOP_UNSIGNED_DIV &&
             name != ROCK_CUSTOMOP_FP_TO_INT_CAST;
    });

    // --- Binary float / int ---
    patterns.add<
        BinaryConverter<tosa::AddOp, arith::AddFOp, arith::AddIOp>,
        BinaryConverter<tosa::SubOp, arith::SubFOp, arith::SubIOp>,
        BinaryConverter<tosa::MaximumOp, arith::MaximumFOp, arith::MaxSIOp>,
        BinaryConverter<tosa::MinimumOp, arith::MinimumFOp, arith::MinSIOp>>(
        ctx);

    // --- Binary float-only ---
    patterns.add<IntBinaryConverter<tosa::BitwiseAndOp, arith::AndIOp>,
                 IntBinaryConverter<tosa::BitwiseOrOp, arith::OrIOp>,
                 IntBinaryConverter<tosa::BitwiseXorOp, arith::XOrIOp>,
                 IntBinaryConverter<tosa::LogicalLeftShiftOp, arith::ShLIOp>,
                 IntBinaryConverter<tosa::LogicalRightShiftOp, arith::ShRUIOp>,
                 IntBinaryConverter<tosa::IntDivOp, arith::DivSIOp>>(ctx);

    // --- Binary float-only (ODS: $input1, $input2) ---
    patterns.add<IntBinaryConverter<tosa::PowOp, math::PowFOp>>(ctx);

    // --- Unary float (ODS: $input1) ---
    patterns.add<UnaryConverter<tosa::ExpOp, math::ExpOp>,
                 UnaryConverter<tosa::LogOp, math::LogOp>,
                 UnaryConverter<tosa::RsqrtOp, math::RsqrtOp>,
                 UnaryConverter<tosa::SinOp, math::SinOp>,
                 UnaryConverter<tosa::CosOp, math::CosOp>,
                 UnaryConverter<tosa::CeilOp, math::CeilOp>,
                 UnaryConverter<tosa::FloorOp, math::FloorOp>>(ctx);

    // --- Unary float (ODS: $input) ---
    patterns.add<UnaryInputConverter<tosa::TanhOp, math::TanhOp>,
                 UnaryInputConverter<tosa::ErfOp, math::ErfOp>>(ctx);

    // --- Comparisons ---
    patterns.add<CmpConverter<tosa::GreaterOp, arith::CmpFPredicate::OGT,
                              arith::CmpIPredicate::sgt>,
                 CmpConverter<tosa::GreaterEqualOp, arith::CmpFPredicate::OGE,
                              arith::CmpIPredicate::sge>,
                 CmpConverter<tosa::EqualOp, arith::CmpFPredicate::OEQ,
                              arith::CmpIPredicate::eq>>(ctx);

    // --- Special cases ---
    patterns.add<ReciprocalRsqrtToSqrtConverter>(ctx, /*benefit=*/2);
    patterns.add<AbsConverter, NegateConverter, MulConverter,
                 ReciprocalConverter, SigmoidConverter, SelectConverter,
                 ClampConverter, CastConverter, CustomOpConverter>(ctx);

    // --- Triton workarounds ---
    // The Triton TritonToTritonGPU conversion is missing patterns for
    // math.tanh and math.powf so we use upstream
    // math::populateExpansionPatterns to expand them into ops Triton supports.
    //
    // if the target has dedicated tanh instructions then we keep `math.tanh`
    // and let the Triton pipeline lower it to v_tanh_* (via llvm.amdgcn.tanh)
    // instead of expanding it here.
    SmallVector<StringRef> opsToExpand = {"powf"};
    if (!hasHardwareTanh)
      opsToExpand.push_back("tanh");
    math::populateExpansionPatterns(patterns, opsToExpand);

    // This is to support migraphx.neg operator, which will be expanded to
    // arith.negf.
    patterns.add<NegFTritonWorkaround>(ctx);

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

