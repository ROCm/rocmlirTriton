//===- RocmlirPromoteSoftmaxPrecision.cpp - Promote softmax to f32 ------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// This pass matches the TOSA softmax normalization pattern
// (exp -> reduce_sum -> reciprocal -> mul) and promotes reduce_sum,
// reciprocal, and the normalize-multiply from f16/bf16 to f32.
//
// On the GPU path, reduce_sum is computed blockwise in f16
// over small blocks (e.g. 32 elements), accumulating relatively little
// rounding error per block. The final normalization is
// deferred to after GEMM1 and performed in f32.
//
// On the CPU path without this promotion, reduce_sum is a
// single-pass f16 reduction over the entire row (e.g. 384 elements),
// accumulating much more rounding error than the GPU's blockwise approach.
// The reciprocal and normalize-multiply are also performed in f16. Promoting
// all three operations to f32 compensates for the structural difference
// (single-pass vs blockwise) and aligns the normalization precision with the
// GPU's f32 final division.
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
/// normalize-multiply to f32. The f32 single-pass reduce_sum compensates for
/// the structural difference between the CPU's single-pass reduction and the
/// GPU's blockwise f16 reduction, while f32 reciprocal and mul match the
/// GPU's f32 final division.
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

    if (reduceSumOp.getInput() != expVal)
      return failure();

    Location loc = op.getLoc();
    Type f32Type = rewriter.getF32Type();

    // Cast exp output from f16/bf16 to f32.
    auto castExpToF32 =
        rock::tosa::createOpAndInfer<CastOp>(rewriter, loc, f32Type, expVal);

    // ReduceSum in f32 (compensates for CPU's single-pass vs GPU's blockwise).
    auto newReduceSum = rock::tosa::createOpAndInfer<ReduceSumOp>(
        rewriter, loc, f32Type, castExpToF32,
        rewriter.getI32IntegerAttr(reduceSumOp.getAxis()));

    // Reciprocal in f32 (matches GPU's f32 final division).
    auto newReciprocal = rock::tosa::createOpAndInfer<ReciprocalOp>(
        rewriter, loc, f32Type, newReduceSum);

    // Normalize in f32: exp_f32 * reciprocal_f32.
    auto newMul = rock::tosa::getMulOp(rewriter, loc, castExpToF32,
                                       newReciprocal, f32Type);

    // Cast back to original element type for downstream matmul.
    auto castBack =
        rock::tosa::createOpAndInfer<CastOp>(rewriter, loc, elemType, newMul);

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
