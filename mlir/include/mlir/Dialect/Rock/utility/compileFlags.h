// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#ifndef MLIR_DIALECT_ROCK_UTILITY_COMPILEFLAGS_H
#define MLIR_DIALECT_ROCK_UTILITY_COMPILEFLAGS_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace rock {

/// `rock.disable-fast-math` on `func`, or the pass/pipeline option that
/// carries the same request.
inline bool disablesFastMath(Operation *func, bool passOption) {
  return passOption ||
         (func && func->hasAttr(DisableFastMathAttr::getMnemonic()));
}

/// `rock.disable-nnan`, or a full fast-math disable, which skips the flag
/// pass and therefore never attaches `nnan` either.
inline bool disablesNnan(Operation *func) {
  return func && (func->hasAttr(DisableNnanAttr::getMnemonic()) ||
                  func->hasAttr(DisableFastMathAttr::getMnemonic()));
}

/// True when a top-level kernel function (`rock.kernel`) carries `mnemonic`.
/// Nested funcs are ignored: the driver and the MIGraphX C API decide the
/// pipeline from the kernel they were asked to compile, which is top-level.
/// A pass that runs on one func still reads that func's own attribute.
inline bool anyKernelHasAttr(ModuleOp module, StringRef mnemonic) {
  if (!module)
    return false;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func->hasAttr(KernelAttr::getMnemonic()) && func->hasAttr(mnemonic))
      return true;
  }
  return false;
}

inline bool kernelDisablesFastMath(ModuleOp module) {
  return anyKernelHasAttr(module, DisableFastMathAttr::getMnemonic());
}

/// `rock.debug` on the module or on any op still in it. The attribute starts
/// on the kernel `func.func` and is copied onto `tt.func`, so either spelling
/// counts.
inline bool moduleRequestsDebug(ModuleOp module) {
  if (!module)
    return false;
  if (module->hasAttr(DebugAttr::getMnemonic()))
    return true;
  bool found = false;
  module.walk([&](Operation *op) {
    if (op->hasAttr(DebugAttr::getMnemonic())) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_COMPILEFLAGS_H
