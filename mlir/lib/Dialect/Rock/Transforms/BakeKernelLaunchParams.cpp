//===- BakeKernelLaunchParams.cpp - Static LDS + remove workspace args ----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Triton's lowering pipeline creates two runtime-configurable artifacts that
// can instead be resolved statically at compile time on AMDGPU:
//
// 1. **Dynamic shared memory (LDS)**: Triton creates an external global
//    `@global_smem = external addrspace(3) global [0 x i8]` and expects the
//    host to pass the shared-memory size via hipExtModuleLaunchKernel's
//    `sharedMem` parameter.  This pass replaces the global with a statically
//    sized one (`internal addrspace(3) global [N x i8] undef`) so the AMDGPU
//    backend bakes the size into `.amdhsa_group_segment_fixed_size`.
//
// 2. **Unused workspace arguments**: Triton appends two extra pointer args
//    (global-scratch and profile-scratch) to every kernel. Our pipeline never
//    uses them (they are always null at launch).  If they have no uses in the
//    IR, this pass strips them from the kernel signature.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_BAKEKERNELLAUNCHPARAMSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "bake-kernel-launch-params"

using namespace mlir;

namespace {

struct BakeKernelLaunchParamsPass
    : public rock::impl::BakeKernelLaunchParamsPassBase<
          BakeKernelLaunchParamsPass> {
  using BakeKernelLaunchParamsPassBase::BakeKernelLaunchParamsPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // ---------------------------------------------------------------
    // Step 1: Convert @global_smem from dynamic (external, size 0)
    //         to static (internal, size = ttg.shared).
    // ---------------------------------------------------------------
    int64_t sharedMemSize = 0;
    if (auto attr = moduleOp->getAttrOfType<IntegerAttr>("ttg.shared"))
      sharedMemSize = attr.getInt();

    if (sharedMemSize < 0) {
      moduleOp.emitError("ttg.shared is negative (") << sharedMemSize << ")";
      return signalPassFailure();
    }

    bool foundGlobalSmem = false;
    for (auto globalOp : moduleOp.getOps<LLVM::GlobalOp>()) {
      if (globalOp.getSymName() != "global_smem")
        continue;
      foundGlobalSmem = true;

      if (sharedMemSize > 0) {
        auto elemTy = IntegerType::get(ctx, 8);
        auto newArrayTy = LLVM::LLVMArrayType::get(elemTy, sharedMemSize);

        globalOp.setGlobalType(newArrayTy);
        globalOp.setLinkage(LLVM::Linkage::Internal);
        globalOp.setValueAttr(LLVM::UndefAttr::get(ctx));

        LLVM_DEBUG(llvm::dbgs() << "Converted @global_smem to static LDS of "
                                << sharedMemSize << " bytes\n");
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "ttg.shared is 0; leaving @global_smem as-is\n");
      }
      break; // There is exactly one @global_smem per module.
    }

    if (!foundGlobalSmem) {
      moduleOp.emitError("@global_smem not found in module");
      return signalPassFailure();
    }

    // Always remove ttg.shared — after this pass, LDS is either statically
    // baked into the binary or zero. Downstream code should not rely on it.
    moduleOp->removeAttr("ttg.shared");

    // ---------------------------------------------------------------
    // Step 2: Remove unused trailing workspace arguments from the
    //         kernel LLVM function.  Triton's amendFuncOp appends
    //         exactly two ptr<1> args (global-scratch and
    //         profile-scratch).  We cap removal at that count so
    //         real kernel data args are never touched.
    // ---------------------------------------------------------------
    constexpr unsigned kMaxWorkspaceArgs = 2;

    for (auto funcOp :
         llvm::make_early_inc_range(moduleOp.getOps<LLVM::LLVMFuncOp>())) {
      if (funcOp.isExternal())
        continue;
      // Only process kernel functions (external linkage, non-declaration).
      if (funcOp.getLinkage() != LLVM::Linkage::External)
        continue;

      unsigned numArgs = funcOp.getNumArguments();
      if (numArgs < 2)
        continue;

      unsigned numToRemove = 0;
      for (unsigned i = numArgs; i > 0 && numToRemove < kMaxWorkspaceArgs;
           --i) {
        BlockArgument arg = funcOp.getArgument(i - 1);
        if (!arg.use_empty())
          break;
        // Triton workspace args are ptr<1> (global address space).
        auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(arg.getType());
        if (!ptrTy || ptrTy.getAddressSpace() != 1)
          break;
        ++numToRemove;
      }

      if (numToRemove == 0)
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "Removing " << numToRemove << " unused trailing args from "
                 << funcOp.getName() << "\n");

      // Build the new function type without the trailing args.
      auto oldFnTy = funcOp.getFunctionType();
      unsigned newNumArgs = numArgs - numToRemove;
      SmallVector<Type> newParamTypes;
      newParamTypes.reserve(newNumArgs);
      for (unsigned i = 0; i < newNumArgs; ++i)
        newParamTypes.push_back(oldFnTy.getParamType(i));
      auto newFnTy = LLVM::LLVMFunctionType::get(
          oldFnTy.getReturnType(), newParamTypes, oldFnTy.isVarArg());

      // Erase block arguments from the entry block (reverse order).
      Block &entryBlock = funcOp.getBody().front();
      for (unsigned i = numArgs; i > newNumArgs; --i)
        entryBlock.eraseArgument(i - 1);

      funcOp.setFunctionType(newFnTy);

      // Also trim arg attrs if present.
      if (auto argAttrs = funcOp.getArgAttrsAttr()) {
        SmallVector<Attribute> trimmed(argAttrs.begin(),
                                       argAttrs.begin() + newNumArgs);
        funcOp.setArgAttrsAttr(ArrayAttr::get(ctx, trimmed));
      }
    }
  }
};

} // namespace
