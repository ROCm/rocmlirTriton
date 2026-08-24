//===- MIGraphXTosaSimplify.cpp - add simplification to tosa ops  ------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// ============================================================

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Passes.h"
#include "mlir/Dialect/Rock/IR/RockTosaCustomOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace migraphx {
#define GEN_PASS_DEF_MIGRAPHXTOSASIMPLIFYPASS
#include "mlir/Dialect/MIGraphX/Passes.h.inc"
} // namespace migraphx
} // namespace mlir

using namespace mlir;
using namespace mlir::migraphx;

namespace {

// Check if the cast is redundant (i.e., the input and output types are the
// same). If so, we can eliminate the cast operation.
class EliminateCastOp final : public OpRewritePattern<tosa::CastOp> {
public:
  using OpRewritePattern<tosa::CastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::CastOp op,
                                PatternRewriter &b) const override {

    Value input = op.getInput();
    Type inputType = input.getType();
    Type outputType = op.getOutput().getType();
    // avoid eliminating cast between float8e8m0fnu types
    // this is required to preserve getting realistic quantized results when
    // running on host and comparing them with GPU results
    if (isa<Float8E8M0FNUType>(cast<ShapedType>(inputType).getElementType()) ||
        isa<Float8E8M0FNUType>(cast<ShapedType>(outputType).getElementType())) {
      return failure();
    }
    if (inputType == outputType) {
      // If we find a cast that leads to the same type, we can eliminate it.
      b.replaceOp(op, input);
      return success();
    }
    while (input.getDefiningOp<tosa::CastOp>()) {
      input = input.getDefiningOp<tosa::CastOp>().getInput();
      inputType = input.getType();
      if (inputType == outputType) {
        // If we find a cast that leads to the same type, we can eliminate it.
        b.replaceOp(op, input);
        return success();
      }
    }
    return failure();
  }
};

// Given this pattern:
//
// %1 = tosa.cast %0 : (tensor<100xi1>) -> tensor<100xf16>
// %2 = tosa.custom %1 {
//   domain_name = "rocmlir",
//   implementation_attrs = "",
//   operator_name = "fp_to_int_cast"
// } : (tensor<100xf16>) -> tensor<100xi8>
//
// we can rewrite it as:
//
// %2 = tosa.cast %0 : (tensor<100xi1>) -> tensor<100xi8>
//
// Note that when widening an i1 it yields exactly 0.0 or 1.0,
// so the NaN check and the range clamp is useless.
// This pattern is frequently used by LeakyReLU ops in MIGraphX.
class SimplifyBoolFpToIntCast final : public OpRewritePattern<tosa::CustomOp> {
public:
  using OpRewritePattern<tosa::CustomOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::CustomOp op,
                                PatternRewriter &b) const override {
    if (op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME ||
        op.getOperatorName() != ROCK_CUSTOMOP_FP_TO_INT_CAST)
      return failure();
    if (op.getInputList().size() != 1 || op.getResults().size() != 1)
      return failure();

    auto widening = op.getInputList().front().getDefiningOp<tosa::CastOp>();
    if (!widening)
      return failure();

    Value boolInput = widening.getInput();
    if (!cast<ShapedType>(boolInput.getType()).getElementType().isInteger(1))
      return failure();

    auto outputType = cast<ShapedType>(op.getResults().front().getType());
    Type outputElemType = outputType.getElementType();
    // A signed 1-bit destination can hold 0 but not 1, so clamping to its range
    // and extending the boolean disagree there.
    if (!isa<IntegerType>(outputElemType) || outputElemType.isInteger(1))
      return failure();

    b.replaceOpWithNewOp<tosa::CastOp>(op, outputType, boolInput);
    return success();
  }
};

struct MIGraphXTosaSimplify
    : public migraphx::impl::MIGraphXTosaSimplifyPassBase<
          MIGraphXTosaSimplify> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<EliminateCastOp, SimplifyBoolFpToIntCast>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
