//===- NonGemmUtils.cpp - Common helpers for the nongemm passes ---------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "NonGemmUtils.h"

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

NonGemmParamsAttr
mlir::rock::chooseNonGemmParams(MLIRContext *ctx, Type elemType, bool hasReduce,
                                int64_t numRows, int64_t numCols,
                                int64_t waveSize, int64_t maxTileCols) {
  // The widest load a lane can issue is 128 bits. Expressed in elements, this
  // is how many of them a thread wants to hold for the load to be a single
  // `dwordx4` rather than a string of narrower ones.
  int64_t vecWidth = std::max<int64_t>(
      1, 128 / std::max<unsigned>(1, elemType.getIntOrFloatBitWidth()));
  int64_t threads = kNonGemmNumWaves * waveSize;
  // What one workgroup holds when every lane issues one full-width load.
  int64_t budget = threads * vecWidth;

  // A reduction is allowed to overspend that budget, up to the register limit
  // `maxTileCols`: its row has to be summed by one workgroup to keep a single
  // writer per output, so a long row buys fewer, fatter blocks. A map is
  // under no such obligation, and a tile past the budget would only mean
  // fewer blocks each doing more, so it stops there.
  int64_t cap = hasReduce ? maxTileCols : budget;
  int64_t colsPerBlock =
      std::min<int64_t>(llvm::bit_ceil(static_cast<uint64_t>(numCols)),
                        llvm::bit_ceil(static_cast<uint64_t>(cap)));

  // Spend the rest of the tile on rows, so a thread ends up with about
  // `vecWidth` elements whatever the reduction extent is. A short reduction
  // therefore packs many rows into a workgroup and keeps the combining inside
  // a wave; a long one drops to a single row and combines through LDS once.
  // A map has `numRows == 1` and so lands at one row here.
  int64_t rowsPerBlock = llvm::bit_ceil(
      static_cast<uint64_t>(std::max<int64_t>(1, budget / colsPerBlock)));
  rowsPerBlock = std::min<int64_t>(
      rowsPerBlock, llvm::bit_ceil(static_cast<uint64_t>(numRows)));

  return NonGemmParamsAttr::get(ctx, rowsPerBlock, colsPerBlock,
                                kNonGemmNumWaves, kNonGemmNumStages);
}
