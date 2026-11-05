//===- ReduceUtils.h - Common helpers for the reduce passes --------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TRANSFORMS_REDUCEUTILS_H
#define MLIR_DIALECT_ROCK_TRANSFORMS_REDUCEUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace rock {

/// Whether `funcOp` is a standalone reduce kernel, the kind the reduce
/// pipeline owns.
bool isStandaloneReduceKernel(func::FuncOp funcOp);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TRANSFORMS_REDUCEUTILS_H
