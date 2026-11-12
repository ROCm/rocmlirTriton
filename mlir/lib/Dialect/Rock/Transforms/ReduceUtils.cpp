//===- ReduceUtils.cpp - Common helpers for the reduce passes ------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "ReduceUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::rock;

bool mlir::rock::isStandaloneReduceKernel(func::FuncOp funcOp) {
  return funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
         funcOp->hasAttr(rock::ReduceAttr::getMnemonic());
}

ReduceParamsAttr mlir::rock::chooseReduceParams(ReduceOp reduceOp,
                                                int64_t numRows,
                                                int64_t numCols,
                                                int64_t waveSize,
                                                int64_t maxTileCols) {
  Type elemType = getElementTypeOrSelf(reduceOp.getIn().getType());
  // The widest load a lane can issue is 128 bits. Expressed in elements, this
  // is how many of them a thread wants to hold for the load to be a single
  // `dwordx4` rather than a string of narrower ones.
  int64_t vecWidth = std::max<int64_t>(
      1, 128 / std::max<unsigned>(1, elemType.getIntOrFloatBitWidth()));
  int64_t threads = kReduceNumWaves * waveSize;

  int64_t colsPerBlock =
      std::min<int64_t>(llvm::bit_ceil(static_cast<uint64_t>(numCols)),
                        llvm::bit_ceil(static_cast<uint64_t>(maxTileCols)));

  // Spend the rest of the tile on rows, so a thread ends up with about
  // `vecWidth` elements whatever the reduction extent is. A short reduction
  // therefore packs many rows into a workgroup and keeps the combining inside
  // a wave; a long one drops to a single row and combines through LDS once.
  int64_t rowsPerBlock = llvm::bit_ceil(static_cast<uint64_t>(
      std::max<int64_t>(1, threads * vecWidth / colsPerBlock)));
  rowsPerBlock = std::min<int64_t>(
      rowsPerBlock, llvm::bit_ceil(static_cast<uint64_t>(numRows)));

  return ReduceParamsAttr::get(reduceOp.getContext(), rowsPerBlock,
                               colsPerBlock, kReduceNumWaves);
}
