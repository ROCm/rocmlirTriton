//===- RocmlirPromoteSoftmaxPrecision.cpp - Promote softmax to f32 ------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// This pass matches the TOSA softmax normalization pattern
// (exp -> reduce_sum -> reciprocal -> mul) and promotes the reduce_sum,
// reciprocal, and normalize-multiply from f16/bf16 to f32.
//
// This improves numerical accuracy of the CPU reference path, eliminating the
// precision gap caused by f16 normalization in offline softmax (CPU) vs f32
// deferred normalization in online softmax (GPU).
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/RocmlirPromoteSoftmaxPrecision/RocmlirPromoteSoftmaxPrecision.h"
#include "mlir/Dialect/Rock/utility/tosaUtils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_ROCMLIRPROMOTESOFTMAXPRECISIONPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

/// Match the TOSA softmax normalization pattern:
///   exp -> reduce_sum -> reciprocal -> mul(exp, reciprocal)
/// When operating on f16/bf16, promote reduce_sum, reciprocal, and the
/// normalize-multiply to f32.
struct SoftmaxNormPromotionPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<RankedTensorType>(op.getOutput().getType());
    Type elemType = resultType.getElementType();

    if (!isa<Float16Type, BFloat16Type>(elemType))
      return failure();

    Value input1 = op.getInput1();
    Value input2 = op.getInput2();

    auto *defOp1 = input1.getDefiningOp();
    auto *defOp2 = input2.getDefiningOp();

    Value expVal;
    ReciprocalOp reciprocalOp;

    if (isa_and_nonnull<ExpOp>(defOp1) &&
        isa_and_nonnull<ReciprocalOp>(defOp2)) {
      expVal = input1;
      reciprocalOp = cast<ReciprocalOp>(defOp2);
    } else if (isa_and_nonnull<ReciprocalOp>(defOp1) &&
               isa_and_nonnull<ExpOp>(defOp2)) {
      expVal = input2;
      reciprocalOp = cast<ReciprocalOp>(defOp1);
    } else {
      return failure();
    }

    auto *recipInput = reciprocalOp.getInput1().getDefiningOp();
    auto reduceSumOp = dyn_cast_or_null<ReduceSumOp>(recipInput);
    if (!reduceSumOp)
      return failure();

    // reduce_sum's input must be the same exp that feeds into the mul
    if (reduceSumOp.getInput() != expVal)
      return failure();

    Location loc = op.getLoc();
    Type f32Type = rewriter.getF32Type();

    auto castExpToF32 = rock::tosa::createOpAndInfer<CastOp>(
        rewriter, loc, f32Type, expVal);

    auto newReduceSum = rock::tosa::createOpAndInfer<ReduceSumOp>(
        rewriter, loc, f32Type, castExpToF32,
        rewriter.getI32IntegerAttr(reduceSumOp.getAxis()));

    auto newReciprocal = rock::tosa::createOpAndInfer<ReciprocalOp>(
        rewriter, loc, f32Type, newReduceSum);

    auto newMul = rock::tosa::getMulOp(rewriter, loc, castExpToF32,
                                       newReciprocal, f32Type);

    auto castBack = rock::tosa::createOpAndInfer<CastOp>(
        rewriter, loc, elemType, newMul);

    rewriter.replaceOp(op, castBack);
    return success();
  }
};

struct RocmlirPromoteSoftmaxPrecisionPass
    : public impl::RocmlirPromoteSoftmaxPrecisionPassBase<
          RocmlirPromoteSoftmaxPrecisionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SoftmaxNormPromotionPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
