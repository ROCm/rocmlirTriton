//===- KernelMetadataTranslation.h - Kernel metadata attrs ------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides the LLVM translation interface that accepts the kernel
// metadata Triton leaves on function parameters. It is shared by every entry
// point that translates a kernel to LLVM IR: the dialect registration used by
// rocmlir-opt and rocmlir-driver, and the standalone triton-to-hsaco
// translation used by rocmlir-translate.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSLATION_KERNELMETADATATRANSLATION_H
#define MLIR_TRANSLATION_KERNELMETADATATRANSLATION_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace rock {

/// Accepts the kernel metadata Triton leaves on function parameters, which has
/// no LLVM IR counterpart and is dropped during translation. Without an
/// interface claiming the namespace, that goes through
/// `LLVMTranslationInterface::convertParameterAttr`, which warns once per
/// attribute via `Operation::emitWarning` - attaching the whole kernel to every
/// diagnostic and dominating the translation. Function and module attributes
/// need no interface; `amendOperation` already ignores them silently.
///
/// Rock is deliberately not registered. `rock.prefill` is the only `rock`
/// parameter attribute, and RockTensorToTritonPtrPass leaves it behind once it
/// has been recorded as a module attribute, so no `rock` parameter attribute
/// should reach LLVM translation. If one ever does, the warning is the signal
/// we want rather than something to suppress.
class KernelMetadataLLVMTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertParameterAttr(LLVM::LLVMFuncOp function, int argIdx,
                       NamedAttribute attr,
                       LLVM::ModuleTranslation &moduleTranslation) const final {
    return success();
  }
};

/// Register the interface for `tt`, the one dialect whose parameter attributes
/// (`tt.divisibility` and friends) are still on the kernel when we translate to
/// LLVM IR.
inline void
registerKernelMetadataDialectTranslation(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, triton::TritonDialect *dialect) {
    dialect->addInterfaces<KernelMetadataLLVMTranslationInterface>();
  });
}

} // namespace rock
} // namespace mlir

#endif // MLIR_TRANSLATION_KERNELMETADATATRANSLATION_H
