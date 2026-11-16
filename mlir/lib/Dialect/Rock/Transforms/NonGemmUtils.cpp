//===- NonGemmUtils.cpp - Common helpers for the nongemm passes ---------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "NonGemmUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::rock;

bool mlir::rock::isStandaloneNonGemmKernel(func::FuncOp funcOp) {
  return funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
         funcOp->hasAttr(rock::NonGemmAttr::getMnemonic());
}

bool mlir::rock::hasReduce(func::FuncOp funcOp) {
  return funcOp.walk([](ReduceOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

NonGemmParamsAttr mlir::rock::chooseNonGemmParams(MLIRContext *ctx,
                                                  Type elemType,
                                                  int64_t numRows,
                                                  int64_t numCols,
                                                  int64_t waveSize,
                                                  int64_t maxTileCols) {
  // The widest load a lane can issue is 128 bits. Expressed in elements, this
  // is how many of them a thread wants to hold for the load to be a single
  // `dwordx4` rather than a string of narrower ones.
  int64_t vecWidth = std::max<int64_t>(
      1, 128 / std::max<unsigned>(1, elemType.getIntOrFloatBitWidth()));
  int64_t threads = kNonGemmNumWaves * waveSize;

  int64_t colsPerBlock =
      std::min<int64_t>(llvm::bit_ceil(static_cast<uint64_t>(numCols)),
                        llvm::bit_ceil(static_cast<uint64_t>(maxTileCols)));

  // Spend the rest of the tile on rows, so a thread ends up with about
  // `vecWidth` elements whatever the reduction extent is. A short reduction
  // therefore packs many rows into a workgroup and keeps the combining inside
  // a wave; a long one drops to a single row and combines through LDS once.
  // A kernel with no reduce has `numRows == 1` and so lands at one row here.
  int64_t rowsPerBlock = llvm::bit_ceil(static_cast<uint64_t>(
      std::max<int64_t>(1, threads * vecWidth / colsPerBlock)));
  rowsPerBlock = std::min<int64_t>(
      rowsPerBlock, llvm::bit_ceil(static_cast<uint64_t>(numRows)));

  return NonGemmParamsAttr::get(ctx, rowsPerBlock, colsPerBlock,
                                kNonGemmNumWaves, kNonGemmNumStages);
}

int64_t mlir::rock::getElementwiseTileSize(func::FuncOp funcOp) {
  if (auto blockSize = funcOp->getAttrOfType<IntegerAttr>(
          rock::BlockSizeAttr::getMnemonic()))
    return blockSize.getInt();
  return 256;
}

llvm::SetVector<Value>
mlir::rock::getElementwiseKernelInputs(func::FuncOp funcOp) {
  llvm::SetVector<Value> leafInputs;
  funcOp.walk([&](Operation *op) {
    if (!isFusionOp(op))
      return;
    for (Value operand : op->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      SmallVector<TransformMapAttr> transforms;
      auto [underlying, _] = rock::untransform(operand, transforms);
      if (isa<BlockArgument>(underlying) ||
          underlying.getDefiningOp<arith::ConstantOp>())
        leafInputs.insert(operand);
    }
  });
  return leafInputs;
}
