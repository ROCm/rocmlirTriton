//===- DetachReattach.h - Module function detach/reattach utilities -------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header provides utilities for temporarily detaching functions from a
// module and reattaching them later. This is useful for running passes on a
// subset of functions in isolation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UTILS_DETACHREATTACH_H
#define MLIR_UTILS_DETACHREATTACH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

/// Holds functions that have been detached from a module, along with their
/// stub declarations that were left behind.
struct DetachedFuncs {
  struct Entry {
    Operation *realFunc;
    func::FuncOp stub;
  };
  llvm::SmallVector<Entry> entries;
};

/// Physically remove functions matching `shouldDetach` from the module,
/// leaving private declaration stubs behind so that symbol references
/// (e.g. func.call) remain valid during pass execution.
///
/// Other module-level operations (globals, etc.) are left untouched.
DetachedFuncs detachFuncs(ModuleOp module,
                          function_ref<bool(func::FuncOp)> shouldDetach);

/// Re-insert previously detached functions into the module, replacing
/// their stubs. If a stub's signature diverged from the real function
/// during pipeline execution (e.g. type conversion passes), asserts
/// that no callers of that stub remain.
void reattachFuncs(ModuleOp module, DetachedFuncs &detached);

} // namespace mlir

#endif // MLIR_UTILS_DETACHREATTACH_H
