//===- GetRockInfo.h - functions used to calculate information about Rock ops
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_ROCK_IR_GETROCKINFO_H
#define MLIR_DIALECT_ROCK_IR_GETROCKINFO_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
class Operation;
class Type;

namespace rock {

// Return the enclosing function-like op (`func.func`, `gpu.func`, or
// `llvm.func`) of `op`.  If `op` is itself a function-like op, returns it.
// Returns a null `FunctionOpInterface` if no enclosing function exists.
FunctionOpInterface getParentFuncOp(Operation *op);

// Get the arch attribute from the function, falling back to its enclosing
// symbol-table parent (e.g. ModuleOp or gpu::GPUModuleOp).
FailureOr<StringAttr> getArchOnFunc(FunctionOpInterface func);
// Get the arch attribute, asserting if it cannot be found.
StringAttr getArchValueOnFunc(FunctionOpInterface func);

// Get the num_cu attribute, looking on the function and then its enclosing
// symbol-table parent.
FailureOr<int64_t> getNumCUOnFunc(FunctionOpInterface func);
// Get the num_cu attribute, falling back to the per-arch minimum if missing.
int64_t getNumCUValueOnFunc(FunctionOpInterface func);

// Get the num_chiplets attribute, looking on the function and then its
// enclosing symbol-table parent.
FailureOr<int64_t> getNumChipletsOnFunc(FunctionOpInterface func);
// Get the num_chiplets attribute, falling back to the per-arch maximum if
// missing.
int64_t getNumChipletsValueOnFunc(FunctionOpInterface func);

// Convenience overloads that look up the attribute on `op`'s parent function.
FailureOr<StringAttr> getArch(Operation *op);
StringAttr getArchValue(Operation *op);
FailureOr<int64_t> getNumCU(Operation *op);
int64_t getNumCUValue(Operation *op);
FailureOr<int64_t> getNumChiplets(Operation *op);
int64_t getNumChipletsValue(Operation *op);

} // End namespace rock
} // End namespace mlir
#endif // MLIR_DIALECT_ROCK_IR_GETROCKINFO_H
