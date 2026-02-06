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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
class Operation;
class Type;

namespace rock {

// This function returns the func or gpu.func of a given op
Operation *getParentFuncOp(Operation *op);

// Get the arch from the op
FailureOr<StringAttr> getArch(Operation *op);

// Get the arch from the op and error out if it cannot be found
StringAttr getArchValue(Operation *op);

// Get the num_cu from the op
FailureOr<int64_t> getNumCU(Operation *op);

// Get the num_cu from the op, and error out if it cannot be found
int64_t getNumCUValue(Operation *op);

// Get the num_chiplets from the op
FailureOr<int64_t> getNumChiplets(Operation *op);

// Get the num_chiplets from the op, and error out if it cannot be found
int64_t getNumChipletsValue(Operation *op);

} // End namespace rock
} // End namespace mlir
#endif // MLIR_DIALECT_ROCK_IR_GETROCKINFO_H
