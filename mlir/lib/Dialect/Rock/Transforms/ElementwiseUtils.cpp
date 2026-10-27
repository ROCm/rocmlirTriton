//===- ElementwiseUtils.cpp - Common helpers for elementwise passes ------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "ElementwiseUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

using namespace mlir;
using namespace mlir::rock;

bool mlir::rock::isStandaloneElementwiseKernel(func::FuncOp funcOp) {
  return funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
         funcOp->hasAttr(rock::ElementwiseAttr::getMnemonic());
}

int64_t mlir::rock::getElementwiseTileSize(func::FuncOp funcOp) {
  if (auto blockSize =
          funcOp->getAttrOfType<IntegerAttr>(rock::BlockSizeAttr::getMnemonic()))
    return blockSize.getInt();
  return 256;
}

llvm::SetVector<Value> mlir::rock::getElementwiseKernelInputs(
    func::FuncOp funcOp) {
  llvm::SetVector<Value> leafInputs;
  funcOp.walk([&](Operation *op) {
    if (!isFusionOp(op))
      return;
    for (Value operand : op->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      SmallVector<TransformMapAttr> transforms;
      auto [underlying, _] = rock::untransform(operand, transforms);
      if (!isa<BlockArgument>(underlying) &&
          !underlying.getDefiningOp<arith::ConstantOp>())
        continue;
      leafInputs.insert(operand);
    }
  });
  return leafInputs;
}
