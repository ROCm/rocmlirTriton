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
//     →
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
#include "mlir/Dialect/Rock/IR/Rock.h"
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
// Generic templates for 1:1 TOSA → arith/math replacement on tensors.
// ===--------------------------------------------------------------------=== //

// Binary op that dispatches on float vs integer element type.
template <typename TosaOp, typename FloatOp, typename IntOp>
struct BinaryConverter : public OpRewritePattern<TosaOp> {
  using OpRewritePattern<TosaOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const override {
    auto elemTy = cast<ShapedType>(op.getType()).getElementType();
    if (isa<FloatType>(elemTy))
      rewriter.replaceOpWithNewOp<FloatOp>(op, op.getType(), op.getInput1(),
                                           op.getInput2());
    else if (isa<IntegerType>(elemTy))
      rewriter.replaceOpWithNewOp<IntOp>(op, op.getType(), op.getInput1(),
                                         op.getInput2());
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
    rewriter.replaceOpWithNewOp<IntOp>(op, op.getType(), op.getInput1(),
                                       op.getInput2());
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
    auto elemTy = cast<ShapedType>(op.getType()).getElementType();
    if (isa<FloatType>(elemTy)) {
      rewriter.replaceOpWithNewOp<arith::MulFOp>(
          op, op.getType(), op.getInput1(), op.getInput2());
      return success();
    }
    if (isa<IntegerType>(elemTy)) {
      DenseElementsAttr shiftElem;
      if (matchPattern(op.getShift(), m_Constant(&shiftElem)) &&
          shiftElem.getValues<IntegerAttr>()[0].getInt() == 0) {
        rewriter.replaceOpWithNewOp<arith::MulIOp>(
            op, op.getType(), op.getInput1(), op.getInput2());
        return success();
      }
      return rewriter.notifyMatchFailure(op, "non-zero shift unsupported");
    }
    return failure();
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

// tosa.sigmoid: 1 / (1 + exp(-x))
struct SigmoidConverter : public OpRewritePattern<tosa::SigmoidOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto shapedTy = cast<ShapedType>(op.getType());
    auto oneAttr = DenseElementsAttr::get(
        shapedTy, rewriter.getFloatAttr(shapedTy.getElementType(), 1.0));
    Value one = arith::ConstantOp::create(rewriter, loc, oneAttr);
    Value negX =
        arith::NegFOp::create(rewriter, loc, op.getType(), op.getInput());
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
    rewriter.replaceOpWithNewOp<arith::SelectOp>(
        op, op.getInput1(), op.getInput2(), op.getInput3());
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

    // float → float
    if (isa<FloatType>(srcTy) && isa<FloatType>(dstTy)) {
      if (bitExtend)
        rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, op.getType(),
                                                   op.getInput());
      else
        rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, op.getType(),
                                                     op.getInput());
      return success();
    }
    // int → float
    if (isa<IntegerType>(srcTy) && isa<FloatType>(dstTy)) {
      if (srcTy.isInteger(1) || srcTy.isUnsignedInteger())
        rewriter.replaceOpWithNewOp<arith::UIToFPOp>(op, op.getType(),
                                                     op.getInput());
      else
        rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, op.getType(),
                                                     op.getInput());
      return success();
    }
    // float → int
    if (isa<FloatType>(srcTy) && isa<IntegerType>(dstTy)) {
      rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, op.getType(),
                                                   op.getInput());
      return success();
    }
    // int → int
    if (isa<IntegerType>(srcTy) && isa<IntegerType>(dstTy)) {
      if (bitExtend)
        rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, op.getType(),
                                                    op.getInput());
      else
        rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, op.getType(),
                                                     op.getInput());
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
    auto elemTy = cast<ShapedType>(op.getInput1().getType()).getElementType();
    if (isa<FloatType>(elemTy))
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, FPred, op.getInput1(),
                                                 op.getInput2());
    else if (isa<IntegerType>(elemTy))
      rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, IPred, op.getInput1(),
                                                 op.getInput2());
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

    target.addLegalDialect<arith::ArithDialect, math::MathDialect,
                           tensor::TensorDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return !isa<tosa::TosaDialect>(op->getDialect());
    });
    target.addLegalOp<tosa::ConstOp, tosa::ConstShapeOp>();

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
    patterns
        .add<AbsConverter, NegateConverter, MulConverter, ReciprocalConverter,
             SigmoidConverter, SelectConverter, ClampConverter, CastConverter>(
            ctx);

    if (failed(applyPartialConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace