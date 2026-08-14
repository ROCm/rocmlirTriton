//===- RockPrepareLLVM.cpp - prepares the generated code for LLVM       ---===//
//
// Copyright 2026 AMD
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Annotates LLVM dialect IR for more efficient AMDGPU codegen:
//   1. GEP inbounds flags
//   2. Invariant loads + alias scope metadata (AMDGPU backend workaround)
//   3. Atomic RMW metadata for native hardware atomics
//   4. amdgpu-no-heap-ptr (drops the unused device-heap implicit arg, which
//      otherwise makes the runtime launch a one-time __amd_rocclr_initHeap).
//      Always added: Rock kernels never call a device-side allocator
//      (malloc/free/__ockl_dm_*), so they can never reach the rocclr heap.
//
// IMPORTANT: the set of attributes/metadata produced here is part of the
// rocmlirTriton kernel ABI documented in `docs/kernel_memory_assumptions.md`.
// If you add, remove, or change the meaning of any annotation below, please
// update that document (and the matching tablegen description in
// `mlir/include/mlir/Dialect/Rock/Passes.td`) so external callers stay in
// sync with what the compiler actually promises/requires.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPREPARELLVMPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-prepare-llvm"

using namespace mlir;

namespace {

struct RockPrepareLLVMPass
    : public rock::impl::RockPrepareLLVMPassBase<RockPrepareLLVMPass> {
  using RockPrepareLLVMPassBase::RockPrepareLLVMPassBase;
  void runOnOperation() override;
};
} // end namespace

// Trace a pointer value back to its function argument through
// addrspacecast / make_buffer_rsrc / gep chains.
static BlockArgument traceToArg(Value pointer, LLVM::LLVMFuncOp func,
                                DenseMap<Value, BlockArgument> &cache) {
  auto cached = cache.find(pointer);
  if (cached != cache.end())
    return cached->second;
  BlockArgument res = nullptr;
  if (auto cast = pointer.getDefiningOp<LLVM::AddrSpaceCastOp>())
    res = traceToArg(cast.getArg(), func, cache);
  else if (auto asBuffer = pointer.getDefiningOp<ROCDL::MakeBufferRsrcOp>())
    res = traceToArg(asBuffer.getBase(), func, cache);
  else if (auto gep = pointer.getDefiningOp<LLVM::GEPOp>())
    res = traceToArg(gep.getBase(), func, cache);
  else if (auto arg = dyn_cast<BlockArgument>(pointer)) {
    auto ptrType = dyn_cast<LLVM::LLVMPointerType>(arg.getType());
    unsigned addrSpace = ~0;
    if (ptrType)
      addrSpace = ptrType.getAddressSpace();
    if (arg.getOwner() == &func.front() && (addrSpace == 0 || addrSpace == 1))
      res = arg;
  }

  cache.insert({pointer, res});
  return res;
}

// 1. Mark GEPs as inbounds (except buffer fat pointers in addrspace 7).
static void markGEPsInbounds(LLVM::LLVMFuncOp func) {
  func.walk([](LLVM::GEPOp gepOp) {
    if (cast<LLVM::LLVMPointerType>(gepOp.getType()).getAddressSpace() != 7)
      gepOp.setNoWrapFlags(gepOp.getNoWrapFlags() |
                           LLVM::GEPNoWrapFlags::inbounds);
  });
}

// 2. Annotate memory accesses with invariant-load and alias-scope metadata.
//
// We'd like to reinforce that the loads we're doing from readonly
// arguments are invariant - concurrent modification of any input we read is
// undefined behavior.
//
// The second set of annotations has to do with a deficiency in the AMDGPU
// backend. Specifically, the `noalias` attributes on kernel arguments
// get discarded in the backend as the function is rewritten to include the
// actual kernel argument loads being performed.
static void annotateMemoryAccesses(LLVM::LLVMFuncOp func, OpBuilder &b) {
  size_t n = func.getNumArguments();
  llvm::SmallBitVector isReadonly(n);
  auto domain =
      b.getAttr<LLVM::AliasScopeDomainAttr>(b.getStringAttr(func.getSymName()));
  llvm::SmallVector<ArrayAttr> aliasScopes;
  aliasScopes.reserve(n);
  llvm::SmallVector<ArrayAttr> noaliasScopes;
  noaliasScopes.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (func.getArgAttr(i, LLVM::LLVMDialect::getReadonlyAttrName()))
      isReadonly[i] = true;
    if (!isa<LLVM::LLVMPointerType>(func.getArgument(i).getType())) {
      aliasScopes.push_back(nullptr);
      continue;
    }
    // RockAnalyzeMemoryUse sets `llvm.noalias` on every pointer-typed kernel
    // argument before this pass runs, so generating per-argument alias scopes
    // is always sound. Assert this invariant to catch any future pipeline
    // change that would invalidate it.
    assert(func.getArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName()) &&
           "RockPrepareLLVM expects every pointer kernel argument to be "
           "marked llvm.noalias (set by RockAnalyzeMemoryUse)");
    auto aliasScope =
        LLVM::AliasScopeAttr::get(domain, b.getStringAttr("arg" + Twine(i)));
    aliasScopes.push_back(b.getArrayAttr(aliasScope));
  }
  {
    SmallVector<Attribute> allButOneScope;
    allButOneScope.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      for (auto [j, val] : llvm::enumerate(aliasScopes)) {
        if (j != i && val)
          allButOneScope.push_back(val[0]);
      }
      noaliasScopes.push_back(b.getArrayAttr(allButOneScope));
      allButOneScope.clear();
    }
  }
  llvm::DenseMap<Value, BlockArgument> toArgCache;

  auto mergeScopes = [&](ArrayAttr existing,
                         ArrayAttr additional) -> ArrayAttr {
    SmallVector<Attribute> mergedScopes;
    if (existing)
      mergedScopes.append(existing.begin(), existing.end());
    if (additional)
      mergedScopes.append(additional.begin(), additional.end());
    if (mergedScopes.empty())
      return nullptr;
    return b.getArrayAttr(mergedScopes);
  };

  // TODO(AIROCMLIR-802): Consider whether "nontemporal" is useful here (see
  // rocMLIR's AnalyzeMemoryUse.cpp comments).
  func.walk([&](LLVM::AliasAnalysisOpInterface aliasIface) {
    Operation *aliasOp = aliasIface.getOperation();
    Value ptrArg;
    for (Value arg : aliasOp->getOperands()) {
      if (isa<LLVM::LLVMPointerType>(arg.getType())) {
        ptrArg = arg;
        break;
      }
    }
    if (!ptrArg)
      return;
    BlockArgument funcArg = traceToArg(ptrArg, func, toArgCache);
    if (!funcArg)
      return;
    unsigned argNo = funcArg.getArgNumber();
    if (auto load = dyn_cast<LLVM::LoadOp>(aliasOp))
      load.setInvariant(isReadonly[argNo]);

    if (ArrayAttr mergedAliasScopes =
            mergeScopes(aliasIface.getAliasScopesOrNull(), aliasScopes[argNo]))
      aliasIface.setAliasScopes(mergedAliasScopes);

    if (ArrayAttr mergedNoAliasScopes = mergeScopes(
            aliasIface.getNoAliasScopesOrNull(), noaliasScopes[argNo]))
      aliasIface.setNoAliasScopes(mergedNoAliasScopes);
  });
}

// 3. Relax atomics. We set the atomic order on read-modify-write
// operations to `monotonic`, which is the extent of the guarantees
// we need about them, and we set the syncscope to "agent-one-as":
// per the memory model, this sync scope means we get our atomic guarantees
// (the monotonicity / lack of data races) above with other atomics executing
// on the GPU, but not with those executing on, say, the host (which is
// a situation we won't be in). We also guarantee that a pointer won't be
// accessed through multiple address spaces.
//
// We also set the metadata for indicating that the arguments to atomics won't
// be host memory or fine-grained memory, and that we don't care about
// the denormal mode. These are needed to ensure that efficient instructions
// (which are unsafe in the absence of this metadata) are selected,
// especially once the old function-level attributes for controlling this go
// away.
static void relaxAtomics(LLVM::LLVMFuncOp func, OpBuilder &b,
                         bool allowFlushDenorm) {
  auto *dialect = func.getContext()->getLoadedDialect<ROCDL::ROCDLDialect>();
  auto noRemoteMemHelper = dialect->getNoRemoteMemoryAttrHelper();
  auto noFineMemHelper = dialect->getNoFineGrainedMemoryAttrHelper();
  auto ignoreDenormalModeHelper = dialect->getIgnoreDenormalModeAttrHelper();
  auto unitAttr = b.getUnitAttr();
  func.walk([&](LLVM::AtomicRMWOp op) {
    op.setSyncscope("agent-one-as");
    op.setOrdering(LLVM::AtomicOrdering::monotonic);
    noRemoteMemHelper.setAttr(op, unitAttr);
    noFineMemHelper.setAttr(op, unitAttr);
    if (allowFlushDenorm)
      ignoreDenormalModeHelper.setAttr(op, unitAttr);
  });
  func.walk([&](LLVM::AtomicCmpXchgOp op) {
    op.setSyncscope("agent-one-as");
    op.setSuccessOrdering(LLVM::AtomicOrdering::monotonic);
    op.setFailureOrdering(LLVM::AtomicOrdering::monotonic);
    noRemoteMemHelper.setAttr(op, unitAttr);
    noFineMemHelper.setAttr(op, unitAttr);
  });
}

// Mark the kernel `amdgpu-no-heap-ptr` via the LLVM-dialect `passthrough`
// attribute. Dropping the hidden_heap_v1 implicit argument from the kernel
// ABI stops the HIP runtime from launching the  __amd_rocclr_initHeap setup
// kernel at module load. This is always sound for Rock kernels: they never
// call a device-side allocator (malloc/free/__ockl_dm_*), so they can never
// reach the rocclr device heap.
static void dropDeviceHeapArg(LLVM::LLVMFuncOp func) {
  StringRef attrName = "amdgpu-no-heap-ptr";
  SmallVector<Attribute> passthrough;
  if (ArrayAttr existing = func.getPassthroughAttr()) {
    for (Attribute attr : existing) {
      if (auto str = dyn_cast<StringAttr>(attr);
          str && str.getValue() == attrName)
        return;
      passthrough.push_back(attr);
    }
  }
  passthrough.push_back(StringAttr::get(func.getContext(), attrName));
  func.setPassthroughAttr(ArrayAttr::get(func.getContext(), passthrough));
}

void RockPrepareLLVMPass::runOnOperation() {
  LLVM::LLVMFuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic())) {
    LLVM_DEBUG(llvm::dbgs() << "RockPrepareLLVM: skipping non-kernel function "
                            << func.getSymName() << "\n");
    return;
  }

  OpBuilder b(&getContext());
  markGEPsInbounds(func);
  annotateMemoryAccesses(func, b);
  relaxAtomics(func, b, allowFlushDenorm);
  dropDeviceHeapArg(func);
}
