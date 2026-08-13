//===- FusedConvToMatmulSchedule.h - Fused conv -> matmul shape -*- C++ -*-===//
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

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_FUSEDCONVTOMATMULSCHEDULE_H
#define MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_FUSEDCONVTOMATMULSCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;

namespace cpu {

/// This schedule consumes the 8-D fused conv `linalg.generic` emitted by the
/// `--cpu-conv-to-gemm` pass (matched via the `rock.cpu_fused_conv` attr) and
/// drives it toward a matmul-shaped inner kernel that the existing tiling
/// and vectorization schedules can recognise.
OwningOpRef<ModuleOp> buildFusedConvToMatmulSchedule(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_FUSEDCONVTOMATMULSCHEDULE_H
