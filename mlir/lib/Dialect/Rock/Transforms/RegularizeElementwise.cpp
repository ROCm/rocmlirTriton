//===- RegularizeElementwise.cpp - Push inter-fusion transforms to inputs -===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "ElementwiseUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEELEMENTWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockRegularizeElementwisePass
    : public rock::impl::RockRegularizeElementwisePassBase<
          RockRegularizeElementwisePass> {
  void runOnOperation() override;
};

} // namespace

class ReorderTransforms : public OpRewritePattern<TransformOp> {
public:
  using OpRewritePattern<TransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransformOp tOp,
                                PatternRewriter &rewriter) const override {
    Operation *fusionOp = tOp.getInput().getDefiningOp();
    if (!fusionOp || !isFusionOp(fusionOp))
      return rewriter.notifyMatchFailure(tOp, "not defined by fusion op");

    TransformMapAttr transformAttr = tOp.getTransform();
    auto transformedType = cast<RankedTensorType>(tOp.getResult().getType());

    SmallVector<Value> newOperands;
    for (Value operand : fusionOp->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType) {
        newOperands.push_back(operand);
        continue;
      }
      auto newType = RankedTensorType::get(transformedType.getShape(),
                                           operandType.getElementType());
      Value transformed = TransformOp::create(rewriter, tOp.getLoc(), newType,
                                              operand, transformAttr);
      newOperands.push_back(transformed);
    }

    Operation *cloned = rewriter.clone(*fusionOp);
    for (auto [idx, operand] : llvm::enumerate(newOperands))
      cloned->setOperand(idx, operand);
    cloned->getResult(0).setType(transformedType);
    rewriter.replaceOp(tOp, cloned);
    return success();
  }
};

void RockRegularizeElementwisePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!isStandaloneElementwiseKernel(funcOp))
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ReorderTransforms>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}
