//===- Schedules.h - CPU lowering transform schedules -----------*- C++ -*-===//
//
// Copyright 2026 Advanced Micro Devices.
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
// =============================================================================
//
// This header declares the transform schedules used for CPU lowering.
// The schedules are MLIR transform dialect modules that define the
// optimization and lowering pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_H
#define MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;

namespace cpu {

/// Container holding all parsed transform schedule modules.
/// These modules are used by the CpuLowerVerifierPass to apply
/// transform sequences to CPU verifier functions.
struct TransformSchedules {
  OwningOpRef<ModuleOp> preModule;               // canonicalize + cse
  OwningOpRef<ModuleOp> postModule;              // LICM + hoisting
  OwningOpRef<ModuleOp> fusedConvToMatmulModule; // fused conv -> matmul shape
  OwningOpRef<ModuleOp> tilingModule;            // tiling
  OwningOpRef<ModuleOp> vectorizationModule;     // vectorization
  OwningOpRef<ModuleOp> unrollModule;            // loop unrolling
  OwningOpRef<ModuleOp> lowerToLLVMModule; // bufferization + LLVM lowering
};

/// Parse and create all transform schedules.
/// Returns failure if any schedule fails to parse.
/// The context must have the transform dialect and extensions registered.
FailureOr<TransformSchedules> createTransformSchedules(MLIRContext *ctx);

/// Apply a transform sequence module to a target module.
/// After applying the transform, any nested module structure is automatically
/// unwrapped (some transforms wrap their result in an additional module).
/// Returns failure if the transform interpreter fails.
LogicalResult applyTransformSequence(OwningOpRef<ModuleOp> &targetModule,
                                     ModuleOp transformModule,
                                     StringRef sequenceName,
                                     StringRef funcName);

/// Register all transform dialect extensions needed for the schedules.
/// This should be called from getDependentDialects().
void registerScheduleDialectExtensions(DialectRegistry &registry);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_H
