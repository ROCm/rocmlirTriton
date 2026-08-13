//===- UnrollSchedule.h - Unroll transform schedule -*- C++ -*-===//
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
// =============================================================================
//
// This header declares the function that builds the unroll transform
// schedule using the MLIR C++ API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_UNROLLSCHEDULE_H
#define MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_UNROLLSCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;

namespace cpu {

/// Build the unroll transform module.
/// This schedule unrolls the innermost loop containing vector.contract ops.
///
/// Equivalent to:
///   %0 = transform.structured.match ops{["vector.contract"]} in %arg0
///   %1 = transform.get_parent_op %0 {op_name = "scf.for"}
///   transform.loop.unroll %1 {factor = 4}
OwningOpRef<ModuleOp> buildUnrollSchedule(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_UNROLLSCHEDULE_H
