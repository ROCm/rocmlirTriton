//===- AnalyzeMemoryUse.cpp - Analyze kernel arg memory usage -------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// Walks kernel function arguments to determine their memory access patterns
// (readonly, writeonly, readwrite) and sets LLVM attributes + Triton metadata
// accordingly.
//
// IMPORTANT: the set of attributes/metadata produced here is part of the
// rocmlirTriton kernel ABI documented in `docs/kernel_assumptions.md`.
// If you add, remove, or change the meaning of any attribute below, please
// update that document (and the matching tablegen description in
// `mlir/include/mlir/Dialect/Rock/Passes.td`) so external callers stay in
// sync with what the compiler actually promises/requires.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKANALYZEMEMORYUSEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-analyze-memory-use"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockAnalyzeMemoryUsePass
    : public rock::impl::RockAnalyzeMemoryUsePassBase<
          RockAnalyzeMemoryUsePass> {
  void runOnOperation() override;
};

} // end namespace

void RockAnalyzeMemoryUsePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  size_t n = func.getNumArguments();
  // Track which arguments are read from and written to.
  llvm::SmallBitVector isWritten(n);
  llvm::SmallBitVector isRead(n);
  llvm::SmallBitVector isAtomicWrite(n);

  // Walk all blockwise_store ops to find which args are written.
  auto storeResult = func.walk([&](rock::BlockwiseStoreOp storeOp) {
    FailureOr<BlockArgument> destArg =
        rock::findBlockArgument(storeOp.getDest());
    if (failed(destArg) || destArg->getOwner() != &func.front()) {
      storeOp.emitError("cannot trace store destination to kernel argument");
      return WalkResult::interrupt();
    }
    unsigned argNo = destArg->getArgNumber();
    isWritten[argNo] = true;
    if (storeOp.getStoreMethod() != StoreMethod::Set) {
      isAtomicWrite[argNo] = true;
      isRead[argNo] = true;
    }
    return WalkResult::advance();
  });
  if (storeResult.wasInterrupted())
    return signalPassFailure();

  // Walk all blockwise_load ops to find which args are read.
  auto loadResult = func.walk([&](rock::BlockwiseLoadOp loadOp) {
    FailureOr<BlockArgument> srcArg =
        rock::findBlockArgument(loadOp.getSource());
    if (succeeded(srcArg) && srcArg->getOwner() == &func.front()) {
      isRead[srcArg->getArgNumber()] = true;
      return WalkResult::advance();
    }

    // Dense compiler constants use the same blockwise-load machinery as
    // kernel inputs, but are backed by internal GPU globals and therefore do
    // not participate in kernel-argument memory attributes.
    SmallVector<TransformMapAttr> transforms;
    Value root;
    std::tie(root, std::ignore) =
        rock::untransform(loadOp.getSource(), transforms);
    if (!rock::isDenseNonSplatConstant(root)) {
      loadOp.emitError("cannot trace load source to kernel argument");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (loadResult.wasInterrupted())
    return signalPassFailure();

  // A non-atomic arg must not be both read and written.
  for (size_t i = 0; i < n; ++i) {
    if (isRead[i] && isWritten[i] && !isAtomicWrite[i]) {
      func.emitError("kernel argument ")
          << i << " is both read and written without atomic semantics";
      return signalPassFailure();
    }
  }

  OpBuilder b(&getContext());
  Attribute unit = b.getUnitAttr();
  // Walk through kernel arguments, attaching the relevant attributes
  for (size_t i = 0; i < n; ++i) {
    auto argType = func.getArgument(i).getType();
    if (!isa<RankedTensorType>(argType))
      continue;

    bool argIsRead = isRead[i];
    bool argIsWritten = isWritten[i];
    bool argIsUnused = !argIsRead && !argIsWritten;

    LLVM_DEBUG(llvm::dbgs()
               << "  arg" << i << ": "
               << (argIsUnused                 ? "unused"
                   : argIsRead && argIsWritten ? "readwrite (atomic)"
                   : argIsRead                 ? "readonly"
                                               : "writeonly")
               << "\n");

    // Note: we'll need to go back in and add alias scopes after LLVM
    // translation because the AMDGPU backend currently chucks out `noalias`
    // on kernel arguments because it's hard to lower.
    if (argIsUnused)
      // Unused pointer argument is readnone.
      func.setArgAttr(i, LLVM::LLVMDialect::getReadnoneAttrName(), unit);
    else if (!argIsWritten)
      func.setArgAttr(i, LLVM::LLVMDialect::getReadonlyAttrName(), unit);
    else if (!argIsRead)
      func.setArgAttr(i, LLVM::LLVMDialect::getWriteOnlyAttrName(), unit);

    func.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(), unit);
    func.setArgAttr(i, LLVM::LLVMDialect::getNoCaptureAttrName(), unit);
    func.setArgAttr(i, LLVM::LLVMDialect::getNoFreeAttrName(), unit);
    func.setArgAttr(i, LLVM::LLVMDialect::getNonNullAttrName(), unit);
    func.setArgAttr(i, LLVM::LLVMDialect::getNoUndefAttrName(), unit);

    // Note: `inreg` for SGPR preloading is set by TritonToHsaco.cpp on all
    // arguments (tensor + scalar), so we don't set it here.

    // As near as we can tell, there's no universe in which global pointers
    // aren't aligned to 16 bytes.
    // TODO: upstream to Triton: AMD's FuncOpToLLVM could derive llvm.align
    // from tt.divisibility so we don't have to set it here.
    func.setArgAttr(i, LLVM::LLVMDialect::getAlignAttrName(),
                    b.getI64IntegerAttr(16));

    // Triton metadata: divisibility enables better vectorization.
    // We use 16 because 128-bit is the maximum vector load/store width on
    // modern GPUs, and we use gpu malloc so this is always the case.
    func.setArgAttr(i, "tt.divisibility", b.getI32IntegerAttr(16));

    // Anyone lying about the size of their input deserves exactly what they
    // get.
    auto tensorType = cast<RankedTensorType>(argType);
    if (tensorType.hasStaticShape()) {
      int64_t sizeInBits =
          tensorType.getNumElements() * tensorType.getElementTypeBitWidth();
      int64_t sizeInBytes = llvm::divideCeil(sizeInBits, 8);
      func.setArgAttr(i, LLVM::LLVMDialect::getDereferenceableAttrName(),
                      b.getI64IntegerAttr(sizeInBytes));

      // Tensor fits in 32-bit address range, enable buffer ops optimization.
      constexpr int64_t k2GBLimit = (1LL << 31);
      if (sizeInBytes < k2GBLimit)
        func.setArgAttr(i, "tt.pointer_range", b.getI32IntegerAttr(32));
    }
  }
}
