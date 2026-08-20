//===------- GetRockInfo.cpp - Utility functions to get Rock Op info ------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/GetRockInfo.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "rock-info-utils"

using namespace mlir;
using namespace mlir::rock;

FunctionOpInterface mlir::rock::getParentFuncOp(Operation *op) {
  if (auto func = dyn_cast<FunctionOpInterface>(op))
    return func;
  return op->getParentOfType<FunctionOpInterface>();
}

namespace {
// Look up an attribute on a function-like op, falling back to the nearest
// enclosing symbol-table parent (typically `ModuleOp` or `gpu::GPUModuleOp`).
// Rock kernel parameters such as `rock.arch`, `rock.num_cu`, etc. now live on
// the function or the module; they are no longer attached to individual rock
// ops.
template <typename RetAttrType>
FailureOr<RetAttrType> getAttrFromFuncOrParent(FunctionOpInterface func,
                                               StringRef attrName) {
  if (!func)
    return failure();
  if (auto attr = func->getAttrOfType<RetAttrType>(attrName))
    return attr;
  if (auto symbolTableOp = func->getParentWithTrait<OpTrait::SymbolTable>()) {
    if (auto attr = symbolTableOp->getAttrOfType<RetAttrType>(attrName))
      return attr;
  }
  return failure();
}
} // namespace

FailureOr<StringAttr> mlir::rock::getArchOnFunc(FunctionOpInterface func) {
  return getAttrFromFuncOrParent<StringAttr>(func,
                                             rock::ArchAttr::getMnemonic());
}

FailureOr<StringAttr> mlir::rock::getArch(Operation *op) {
  return rock::getArchOnFunc(rock::getParentFuncOp(op));
}

StringAttr mlir::rock::getArchValueOnFunc(FunctionOpInterface func) {
  auto maybeArch = rock::getArchOnFunc(func);
  if (failed(maybeArch))
    llvm_unreachable("No 'arch' attribute on kernel");

  return maybeArch.value();
}

StringAttr mlir::rock::getArchValue(Operation *op) {
  return rock::getArchValueOnFunc(rock::getParentFuncOp(op));
}

FailureOr<int64_t> mlir::rock::getNumCUOnFunc(FunctionOpInterface func) {
  FailureOr<StringAttr> maybeArch = getArchOnFunc(func);
  if (failed(maybeArch)) {
    LLVM_DEBUG(llvm::dbgs() << "arch not found\n");
    return failure();
  }
  StringAttr arch = maybeArch.value();
  FailureOr<IntegerAttr> maybeNumCU = getAttrFromFuncOrParent<IntegerAttr>(
      func, rock::NumCUAttr::getMnemonic());
  if (failed(maybeNumCU)) {
    return failure();
  }
  IntegerAttr numCU = maybeNumCU.value();
  int64_t minNumCU = rock::getMinNumCU(arch);
  if (numCU.getValue().getSExtValue() < minNumCU) {
    return func->emitError()
           << "num_cu=" << numCU
           << " cannot be lower than arch minNumCU=" << minNumCU;
  }
  return numCU.getValue().getSExtValue();
}

FailureOr<int64_t> mlir::rock::getNumCU(Operation *op) {
  return rock::getNumCUOnFunc(rock::getParentFuncOp(op));
}

int64_t mlir::rock::getNumCUValueOnFunc(FunctionOpInterface func) {
  auto maybeCU = rock::getNumCUOnFunc(func);
  if (succeeded(maybeCU)) {
    return maybeCU.value();
  }

  // Otherwise, we will need to get the minimum CU value from the architecture
  auto archStr = rock::getArchValueOnFunc(func);
  int64_t minCU = rock::getMinNumCU(archStr);
  LLVM_DEBUG(llvm::dbgs() << "Could not find num_cu, defaulting to minimum "
                          << "CU value for " << archStr << ": " << minCU
                          << "\n");
  return minCU;
}

int64_t mlir::rock::getNumCUValue(Operation *op) {
  return rock::getNumCUValueOnFunc(rock::getParentFuncOp(op));
}

FailureOr<int64_t> mlir::rock::getNumChipletsOnFunc(FunctionOpInterface func) {
  FailureOr<StringAttr> maybeArch = getArchOnFunc(func);
  if (failed(maybeArch)) {
    LLVM_DEBUG(llvm::dbgs() << "arch not found\n");
    return failure();
  }
  StringAttr arch = maybeArch.value();
  FailureOr<IntegerAttr> maybeNumChiplets =
      getAttrFromFuncOrParent<IntegerAttr>(
          func, rock::NumChipletsAttr::getMnemonic());
  if (failed(maybeNumChiplets)) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find num_chiplets\n");
    return failure();
  }
  IntegerAttr numChiplets = maybeNumChiplets.value();
  if (numChiplets.getValue().getSExtValue() <= 0) {
    return func->emitError() << "num_chiplets must be greater than zero";
  }
  int64_t maxNumChiplets = rock::getMaxNumChiplets(arch);
  if (numChiplets.getValue().getSExtValue() > maxNumChiplets) {
    return func->emitError()
           << "num_chiplets=" << numChiplets
           << " cannot be greater than arch maxNumChiplets=" << maxNumChiplets;
  }
  return numChiplets.getValue().getSExtValue();
}

FailureOr<int64_t> mlir::rock::getNumChiplets(Operation *op) {
  return rock::getNumChipletsOnFunc(rock::getParentFuncOp(op));
}

int64_t mlir::rock::getNumChipletsValueOnFunc(FunctionOpInterface func) {
  auto maybeChiplets = rock::getNumChipletsOnFunc(func);
  if (succeeded(maybeChiplets)) {
    return maybeChiplets.value();
  }

  // Otherwise, infer chiplets from the effective CU count. This keeps omitted
  // chiplets consistent with an explicitly supplied or architecture-default CU
  // count.
  auto archStr = rock::getArchValueOnFunc(func);
  int64_t numCU = rock::getNumCUValueOnFunc(func);
  int64_t chiplets = rock::inferNumChiplets(archStr, numCU);
  LLVM_DEBUG(llvm::dbgs() << "Could not find num_chiplets, inferring "
                          << chiplets << " chiplets for " << archStr << " with "
                          << numCU << " CUs\n");
  return chiplets;
}

int64_t mlir::rock::getNumChipletsValue(Operation *op) {
  return rock::getNumChipletsValueOnFunc(rock::getParentFuncOp(op));
}
