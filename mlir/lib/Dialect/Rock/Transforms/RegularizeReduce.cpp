// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

namespace mlir::rock {
#define GEN_PASS_DEF_ROCKREGULARIZEREDUCEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace mlir::rock

using namespace mlir;
using namespace mlir::rock;

namespace {

class RegularizeReducePass
    : public mlir::rock::impl::RockRegularizeReducePassBase<
          RegularizeReducePass> {
public:
  void runOnOperation() override;
};

class MoveTransformsToInput : public OpRewritePattern<TransformOp> {
public:
  using OpRewritePattern<TransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransformOp tOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace

LogicalResult
MoveTransformsToInput::matchAndRewrite(TransformOp tOp,
                                       PatternRewriter &rewriter) const {
  Operation *fusionOp = tOp.getInput().getDefiningOp();
  if (!fusionOp || !isFusionOp(fusionOp))
    return rewriter.notifyMatchFailure(tOp,
                                       "require defining op to be fusion op");

  auto outType = tOp.getType();
  auto transform = tOp.getTransformAttr();
  rewriter.setInsertionPoint(fusionOp);

  IRMapping mapping;
  for (Value operand : fusionOp->getOperands()) {
    auto transformOutType = RankedTensorType::get(
        outType.getShape(), getElementTypeOrSelf(operand));
    auto newOperand = TransformOp::create(rewriter, tOp.getLoc(),
                                          transformOutType, operand, transform);
    mapping.map(operand, newOperand);
  }

  Operation *newFusionOp = rewriter.clone(*fusionOp, mapping);
  rewriter.modifyOpInPlace(newFusionOp,
                           [&] { newFusionOp->getResult(0).setType(outType); });
  rewriter.replaceOp(tOp, newFusionOp);

  return success();
}

void RegularizeReducePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!funcOp->hasAttr("rock.kernel") || !funcOp->hasAttr("rock.reduce"))
    return;

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MoveTransformsToInput>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    return signalPassFailure();
}
