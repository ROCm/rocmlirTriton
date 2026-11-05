//===- ReduceUtils.cpp - Common helpers for the reduce passes ------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "ReduceUtils.h"

using namespace mlir;
using namespace mlir::rock;

bool mlir::rock::isStandaloneReduceKernel(func::FuncOp funcOp) {
  return funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
         funcOp->hasAttr(rock::ReduceAttr::getMnemonic());
}
