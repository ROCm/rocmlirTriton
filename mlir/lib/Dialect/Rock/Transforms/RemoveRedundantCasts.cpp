//===- RemoveRedundantCasts.cpp - Remove redundant float casts -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates pure-SSA float-cast round-trips in kernel functions.
// Two patterns are handled, both as independent local rewrites on the outer
// cast op (the inner cast is left for DCE so its other consumers, if any,
// are untouched):
//
//   1. `arith.extf(arith.truncf %wide) -> %wide`
//   2. `arith.truncf(arith.extf %narrow) -> %narrow`
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREMOVEREDUNDANTCASTSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-remove-redundant-casts"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Fold `arith.extf(arith.truncf %wide)` -> `%wide` when the round-trip
/// starts and ends at the same wide precision. The rewrite is applied to a
/// single `arith.extf` consumer at a time; the underlying `arith.truncf` is
//  left intact and may keep producing the narrow value for any other consumer
//  (e.g. a narrow `ds_store_b16` or a `tt.dot` operand). If the trunc loses its
/// last consumer it is removed.
///
/// This rewrite is precision-recovering rather than bit-equivalent: a wide
/// value outside the narrow representable range would have been saturated to
/// inf by the trunc; bypassing the trunc preserves the original wide value.
struct ExtfOfTruncfPattern : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern<arith::ExtFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp extOp,
                                PatternRewriter &rewriter) const override {
    auto truncOp = extOp.getIn().getDefiningOp<arith::TruncFOp>();
    if (!truncOp)
      return failure();

    if (extOp.getOut().getType() != truncOp.getIn().getType())
      return rewriter.notifyMatchFailure(
          extOp, "extf result type does not match truncf input type");

    LLVM_DEBUG(llvm::dbgs()
               << "folding extf(truncf %wide) -> %wide at " << extOp << "\n");
    rewriter.replaceOp(extOp, truncOp.getIn());
    return success();
  }
};

/// Fold `arith.truncf(arith.extf %narrow)` -> `%narrow` when the round-trip
/// starts and ends at the same narrow precision.
///
/// Unconditionally safe: `arith.extf` is lossless and rounding the result
/// back to the original narrow type is the identity for every bit pattern.
struct TruncfOfExtfPattern : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp truncOp,
                                PatternRewriter &rewriter) const override {
    auto extOp = truncOp.getIn().getDefiningOp<arith::ExtFOp>();
    if (!extOp)
      return failure();

    if (truncOp.getOut().getType() != extOp.getIn().getType())
      return rewriter.notifyMatchFailure(
          truncOp, "truncf result type does not match extf input type");

    LLVM_DEBUG(llvm::dbgs() << "folding truncf(extf %narrow) -> %narrow at "
                            << truncOp << "\n");
    rewriter.replaceOp(truncOp, extOp.getIn());
    return success();
  }
};

struct RockRemoveRedundantCastsPass
    : public rock::impl::RockRemoveRedundantCastsPassBase<
          RockRemoveRedundantCastsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockRemoveRedundantCastsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.add<ExtfOfTruncfPattern, TruncfOfExtfPattern>(&getContext());

  if (failed(
          applyPatternsGreedily(funcOp.getOperation(), std::move(patterns))))
    return signalPassFailure();
}
