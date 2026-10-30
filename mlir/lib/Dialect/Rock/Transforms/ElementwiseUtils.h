//===- ElementwiseUtils.h - Common helpers for elementwise passes --------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TRANSFORMS_ELEMENTWISEUTILS_H
#define MLIR_DIALECT_ROCK_TRANSFORMS_ELEMENTWISEUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace rock {

/// True when this is a standalone elementwise GPU kernel function.
bool isStandaloneElementwiseKernel(func::FuncOp funcOp);

/// Tile size used by standalone elementwise passes.
/// Falls back to 256 when rock.block_size is absent.
int64_t getElementwiseTileSize(func::FuncOp funcOp);

/// Returns leaf tensor inputs of the elementwise fusion DAG.
/// Leaves are block arguments or arith constants reached via transform chains.
llvm::SetVector<Value> getElementwiseKernelInputs(func::FuncOp funcOp);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TRANSFORMS_ELEMENTWISEUTILS_H
