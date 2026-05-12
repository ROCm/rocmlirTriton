//===- DetachReattach.cpp - Module function detach/reattach utilities -----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/utils/DetachReattach.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

DetachedFuncs mlir::detachFuncs(ModuleOp module,
                                function_ref<bool(func::FuncOp)> shouldDetach) {
  DetachedFuncs detached;

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
    if (!shouldDetach(funcOp))
      continue;

    OpBuilder stubBuilder(funcOp);
    auto stub =
        func::FuncOp::create(stubBuilder, funcOp.getLoc(), funcOp.getName(),
                             funcOp.getFunctionType());
    stub.setVisibility(SymbolTable::Visibility::Private);

    funcOp->remove();
    detached.entries.push_back(
        {funcOp, stub.getSymNameAttr(), funcOp.getFunctionType()});
  }

  return detached;
}

void mlir::reattachFuncs(ModuleOp module, DetachedFuncs &detached) {
  for (auto &entry : detached.entries) {
    auto *realFunc = entry.realFunc;
    // Re-resolve the stub by symbol name: the host pipeline may have erased
    // and replaced the original stub op (e.g. ConvertFuncToLLVMPass rewrites
    // every `func.func` into an `llvm.func`).
    Operation *stub = SymbolTable::lookupSymbolIn(module, entry.stubName);
    assert(stub && "detached function stub disappeared from module");

    auto funcStub = dyn_cast<func::FuncOp>(stub);
    if (funcStub && funcStub.getFunctionType() != entry.originalType)
      assert(SymbolTable::symbolKnownUseEmpty(funcStub, module) &&
             "detached function signature changed but stub still has callers");

    stub->getBlock()->getOperations().insert(stub->getIterator(), realFunc);
    stub->erase();
  }
  detached.entries.clear();
}
