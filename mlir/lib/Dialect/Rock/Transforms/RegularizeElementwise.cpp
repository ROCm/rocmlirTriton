//===- RegularizeElementwise.cpp - Push inter-fusion transforms to inputs -===//
//
// Copyright 2026 Advanced Micro Devices.
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
//
// For standalone elementwise kernels, rock.transform ops can appear between
// fusion ops (arith/math elementwise ops) when the original MIGraphX graph
// had transforms between fusions (e.g. reshape, transpose).
//
// This pass pushes those inter-fusion transforms upward through fusion ops
// so that all transforms end up at the input level (before any fusion op).
// This is valid because for any elementwise op f:
//
//   transform(f(x, y)) == f(transform(x), transform(y))
//
// The pass iterates until no more inter-fusion transforms remain.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEELEMENTWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-regularize-elementwise"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockRegularizeElementwisePass
    : public rock::impl::RockRegularizeElementwisePassBase<
          RockRegularizeElementwisePass> {
  void runOnOperation() override;
};

} // namespace

void RockRegularizeElementwisePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!isElementwiseKernel(funcOp))
    return;

  bool changed = true;
  while (changed) {
    changed = false;
    // Collect transform ops whose input is produced by a fusion op.
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
      for (auto [idx, newOp] : llvm::enumerate(newOperands))
        cloned->setOperand(idx, newOp);
      cloned->getResult(0).setType(transformedType);

      tOp.getResult().replaceAllUsesWith(cloned->getResult(0));
      tOp->erase();
      if (fusionOp->use_empty())
        fusionOp->erase();

      changed = true;
    }
  }
}
