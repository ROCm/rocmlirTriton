//===- RegularizeElementwise.cpp - Push inter-fusion transforms to inputs -===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEELEMENTWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

static bool isStandaloneElementwiseKernel(func::FuncOp funcOp) {
  return funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
         funcOp->hasAttr(rock::ElementwiseAttr::getMnemonic());
}

struct RockRegularizeElementwisePass
    : public rock::impl::RockRegularizeElementwisePassBase<
          RockRegularizeElementwisePass> {
  void runOnOperation() override;
};

} // namespace

void RockRegularizeElementwisePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!isStandaloneElementwiseKernel(funcOp))
    return;

  bool changed = true;
  while (changed) {
    changed = false;

    SmallVector<TransformOp> toProcess;
    funcOp.walk([&](TransformOp tOp) {
      if (Operation *def = tOp.getInput().getDefiningOp();
          def && isFusionOp(def))
        toProcess.push_back(tOp);
    });

    for (TransformOp tOp : toProcess) {
      Operation *fusionOp = tOp.getInput().getDefiningOp();
      if (!fusionOp || !isFusionOp(fusionOp))
        continue;

      TransformMapAttr transformAttr = tOp.getTransform();
      auto transformedType = cast<RankedTensorType>(tOp.getResult().getType());
      OpBuilder builder(fusionOp);
      Location loc = fusionOp->getLoc();

      SmallVector<Value> newOperands;
      for (Value operand : fusionOp->getOperands()) {
        auto operandType = dyn_cast<RankedTensorType>(operand.getType());
        if (!operandType) {
          newOperands.push_back(operand);
          continue;
        }
        auto newType = RankedTensorType::get(transformedType.getShape(),
                                             operandType.getElementType());
        Value transformed =
            TransformOp::create(builder, loc, newType, operand, transformAttr);
        newOperands.push_back(transformed);
      }

      Operation *cloned = builder.clone(*fusionOp);
      for (auto [idx, operand] : llvm::enumerate(newOperands))
        cloned->setOperand(idx, operand);
      cloned->getResult(0).setType(transformedType);

      tOp.getResult().replaceAllUsesWith(cloned->getResult(0));
      tOp->erase();
      if (fusionOp->use_empty())
        fusionOp->erase();
      changed = true;
    }
  }
}
