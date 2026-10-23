//===- TritonToHsaco.h - Triton LLVM IR to HSACO translation ----*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides registration calls for Triton LLVM dialect IR to HSACO binary
// translation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSLATION_TRITONTOHSACO_H
#define MLIR_TRANSLATION_TRITONTOHSACO_H

#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/OptimizationLevel.h"
#include <string>
#include <vector>

namespace mlir {
namespace rock {

/// Options for Triton to HSACO translation
struct TritonToHsacoOptions {
  std::string triple;
  std::string arch;
  std::string features;
  int optLevel;
  int numWarps;
  int numCTAs;
  int wavesPerEU;
  bool enableFpFusion = true;
  bool allowFlushDenorm = true;
  bool scalarizePackedFops = false; // Scalarize packed float ops
  std::string llvmFnAttrs;
  int useExpertScheduling = kKnobDefault;
  std::vector<std::string> externLibPaths; // Paths to external device libraries
                                           // (ocml.bc, ockl.bc, etc.)
};

/// Translate a Triton LLVM dialect module to HSACO binary.
/// This implements the functionality from Triton's compiler.py:
/// - make_llir() (part 2): LLVM-IR (MLIR) -> LLVM-IR (LLVM)
/// - make_amdgcn(): LLVM -> AMDGCN assembly
/// - make_hsaco(): AMDGCN assembly -> HSACO binary
///
/// Returns the HSACO binary on success, or failure on error.
FailureOr<std::string>
translateTritonToHsaco(ModuleOp module, const TritonToHsacoOptions &options);

/// Register the translation with mlir-translate.
void registerTritonToHsacoTranslation();

} // namespace rock
} // namespace mlir

#endif // MLIR_TRANSLATION_TRITONTOHSACO_H
