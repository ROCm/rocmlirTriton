//===- ResolveKernelLaunchParams.cpp - Static LDS + remove workspace args -===//
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
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"

#include <cassert>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_RESOLVEKERNELLAUNCHPARAMSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "resolve-kernel-launch-params"

using namespace mlir;

namespace {

/// Reject launches too large for an HSA dispatch packet, whose `grid_size_x`
/// is a uint32_t counted in work-items
static LogicalResult validateKernelLaunchDimensions(ModuleOp moduleOp) {
  bool hasGridMetadata = llvm::any_of(
      moduleOp.getOps<LLVM::LLVMFuncOp>(), [&](LLVM::LLVMFuncOp funcOp) {
        return funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
               moduleOp->hasAttr(
                   rock::GridSizeAttr::getModuleAttrName(funcOp.getName()));
      });
  if (!hasGridMetadata)
    return success();

  SmallVector<rock::KernelInfo> kernels;
  if (failed(rock::collectKernelInfo(moduleOp, kernels)))
    return moduleOp.emitError(
        "could not validate kernel launch dimensions because kernel metadata "
        "collection failed");

  // The dispatch packet counts work-items, not workgroups, so it is the whole
  // grid * block * cluster product that has to fit in a uint32.
  for (rock::KernelInfo &kernel : kernels) {
    assert(kernel.gridSize > 0 && "expected a positive kernel grid size");
    assert(kernel.blockSize > 0 && "expected a positive kernel block size");
    assert(kernel.clusterSize > 0 && "expected a positive kernel cluster size");
    if (kernel.gridSize > rock::maxHardwareGridSize) {
      rock::markAsNotApplicable(moduleOp);
      return kernel.llvmFunc.emitOpError()
             << "grid size " << kernel.gridSize
             << " exceeds the runtime grid size limit of "
             << rock::maxHardwareGridSize;
    }

    if (kernel.blockSize > rock::maxHardwareWorkgroupSize) {
      rock::markAsNotApplicable(moduleOp);
      return kernel.llvmFunc.emitOpError()
             << "block size " << kernel.blockSize
             << " exceeds the AMDGPU workgroup size limit of "
             << rock::maxHardwareWorkgroupSize;
    }

    std::optional<int64_t> workItems =
        llvm::checkedMul(kernel.gridSize, kernel.blockSize);
    if (workItems)
      workItems = llvm::checkedMul(*workItems, kernel.clusterSize);
    if (workItems && *workItems <= rock::maxHardwareGridSize)
      continue;

    rock::markAsNotApplicable(moduleOp);
    return kernel.llvmFunc.emitOpError()
           << "launch dimensions (grid size " << kernel.gridSize
           << ", block size " << kernel.blockSize << ", cluster size "
           << kernel.clusterSize << ") exceed the runtime limit of "
           << rock::maxHardwareGridSize << " work-items";
  }

  return success();
}

struct ResolveKernelLaunchParamsPass
    : public rock::impl::ResolveKernelLaunchParamsPassBase<
          ResolveKernelLaunchParamsPass> {
  using ResolveKernelLaunchParamsPassBase::ResolveKernelLaunchParamsPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    // ---------------------------------------------------------------
    // Step 1: Convert @global_smem from dynamic (external, size 0)
    //         to static (internal, size = ttg.shared).
    // ---------------------------------------------------------------
    auto sharedAttr = moduleOp->getAttrOfType<IntegerAttr>("ttg.shared");
    if (!sharedAttr) {
      moduleOp.emitError("ttg.shared attribute not found on module");
      return signalPassFailure();
    }
    int64_t sharedMemSize = sharedAttr.getInt();

    if (sharedMemSize < 0) {
      moduleOp.emitError("ttg.shared is negative (") << sharedMemSize << ")";
      return signalPassFailure();
    }

    // Find the target architecture from rock.arch on any kernel function
    // (`rock::getArchOnFunc` walks up to the module attribute if needed).
    StringRef archStr;
    for (auto funcOp : moduleOp.getOps<LLVM::LLVMFuncOp>()) {
      if (auto arch = rock::getArchOnFunc(funcOp); succeeded(arch)) {
        archStr = *arch;
        break;
      }
    }
    if (archStr.empty()) {
      moduleOp.emitError("rock.arch not found on kernel function or module");
      return signalPassFailure();
    }

    int64_t maxLDS = rock::getLDSSize(archStr);
    if (sharedMemSize > maxLDS) {
      rock::markAsNotApplicable(moduleOp);
      mlir::emitError(moduleOp.getLoc(), "ttg.shared (")
          << sharedMemSize << ") exceeds LDS limit (" << maxLDS << ") for "
          << archStr;
      return signalPassFailure();
    }

    if (failed(validateKernelLaunchDimensions(moduleOp)))
      return signalPassFailure();

    auto globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>("global_smem");
    if (!globalOp) {
      moduleOp.emitError("@global_smem not found in module");
      return signalPassFailure();
    }

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

    // Always remove ttg.shared — after this pass, LDS is either statically
    // baked into the binary or zero. Downstream code should not rely on it.
    moduleOp->removeAttr("ttg.shared");

    // ---------------------------------------------------------------
    // Step 2: Remove the two trailing workspace arguments that
    //         Triton's amendFuncOp appends to every kernel
    //         (global-scratch ptr<1> and profile-scratch ptr<1>).
    //         These must be unused; if they have uses the pass fails
    //         because downstream code (tuning driver, RockEmitGpuBinaryPass)
    //         assumes a 1:1 mapping between host args and kernel args.
    // ---------------------------------------------------------------
    constexpr unsigned kWorkspaceArgs = 2;

    for (auto funcOp :
         llvm::make_early_inc_range(moduleOp.getOps<LLVM::LLVMFuncOp>())) {
      if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
        continue;

      unsigned numArgs = funcOp.getNumArguments();
      if (numArgs < kWorkspaceArgs) {
        funcOp.emitError("kernel '")
            << funcOp.getName() << "' has fewer than " << kWorkspaceArgs
            << " arguments — expected trailing workspace args from "
               "amendFuncOp";
        return signalPassFailure();
      }

      // The last kWorkspaceArgs args must be ptr<1> (from amendFuncOp).
      for (unsigned i = numArgs - kWorkspaceArgs; i < numArgs; ++i) {
        auto ptrTy =
            dyn_cast<LLVM::LLVMPointerType>(funcOp.getArgument(i).getType());
        if (!ptrTy || ptrTy.getAddressSpace() != 1) {
          funcOp.emitError("expected trailing workspace arg %arg")
              << i << " to be ptr<1>, got " << funcOp.getArgument(i).getType();
          return signalPassFailure();
        }
      }

      // All workspace args must be unused.
      for (unsigned i = numArgs - kWorkspaceArgs; i < numArgs; ++i) {
        BlockArgument arg = funcOp.getArgument(i);
        if (!arg.use_empty()) {
          funcOp.emitError("workspace argument %arg")
              << i
              << " has unexpected uses — cannot strip trailing "
                 "workspace args from kernel '"
              << funcOp.getName() << "'";
          return signalPassFailure();
        }
      }

      // Guard against callers inside the module — if someone calls this
      // kernel, changing its signature would silently break the call site.
      auto uses = SymbolTable::getSymbolUses(funcOp, moduleOp);
      if (uses && !uses->empty()) {
        funcOp.emitError("kernel '")
            << funcOp.getName()
            << "' still has callers in the module — cannot strip workspace "
               "args (pipeline ordering issue?)";
        return signalPassFailure();
      }

      LLVM_DEBUG(llvm::dbgs() << "Removing " << kWorkspaceArgs
                              << " unused trailing workspace args from "
                              << funcOp.getName() << "\n");

      unsigned newNumArgs = numArgs - kWorkspaceArgs;

      // Build the new function type without the trailing args.
      auto oldFnTy = funcOp.getFunctionType();
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
