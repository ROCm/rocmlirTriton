//===- RegularizeNonGemm.cpp - Push transforms to nongemm inputs --------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// A nongemm kernel is a graph of views and elementwise ops, so a view can sit
// anywhere in it: between two elementwise ops, between an elementwise op and
// the reduction, or on an input. The passes that follow want one shape per
// kernel, with every view already applied to the leaves.
//
// This pass gets there by repeatedly swapping a `rock.transform` with the
// fusion op feeding it, which moves views towards the inputs and leaves the
// elementwise ops stacked on the viewed shape.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "NonGemmUtils.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZENONGEMMPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockRegularizeNonGemmPass
    : public rock::impl::RockRegularizeNonGemmPassBase<
          RockRegularizeNonGemmPass> {
  void runOnOperation() override;
};

/// Rewrite `transform(fusion(a, b, ...))` into
/// `fusion(transform(a), transform(b), ...)`.
///
/// A view and an elementwise op commute: the view only renumbers coordinates
/// and the op only maps the value at each one. Applying the view to every
/// operand instead therefore computes the same tensor, and repeating it walks
/// the view down to the leaves.
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

    // The view is built in front of the op it is being pushed through, so
    // that the operands it reads still dominate it.
    rewriter.setInsertionPoint(fusionOp);

    SmallVector<Value> newOperands;
    for (Value operand : fusionOp->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      // A scalar operand, such as the bound of a clamp, is not indexed and so
      // has no view to carry.
      if (!operandType) {
        newOperands.push_back(operand);
        continue;
      }
      // Only the shape travels: an op like `arith.extf` has operands and a
      // result of different element types.
      auto newType = RankedTensorType::get(transformedType.getShape(),
                                           operandType.getElementType());
      newOperands.push_back(TransformOp::create(rewriter, tOp.getLoc(), newType,
                                                operand, transformAttr));
    }

    Operation *cloned = rewriter.clone(*fusionOp);
    for (auto [idx, operand] : llvm::enumerate(newOperands))
      cloned->setOperand(idx, operand);
    cloned->getResult(0).setType(transformedType);
    rewriter.replaceOp(tOp, cloned);
    return success();
  }
};

} // namespace

void RockRegularizeNonGemmPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!isStandaloneNonGemmKernel(funcOp))
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<ReorderTransforms>(ctx);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    signalPassFailure();
}
