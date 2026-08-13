//===- VectorizationSchedule.h - Vectorization transform schedule -*- C++ -*-=//
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
// This header declares the function that builds the vectorization transform
// schedule using the MLIR C++ API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_VECTORIZATIONSCHEDULE_H
#define MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_VECTORIZATIONSCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;

namespace cpu {

/// Build the vectorization transform module.
/// This schedule vectorizes linalg ops marked with rock.matmul attribute.
///
/// Equivalent to:
///   %0 = transform.structured.match attributes {rock.matmul} in %arg0
///   %1 = transform.get_parent_op %0 {isolated_from_above}
///   %2 = transform.structured.vectorize_children_and_apply_patterns %1
///        {vectorize_nd_extract, vectorize_padding}
OwningOpRef<ModuleOp> buildVectorizationSchedule(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_VECTORIZATIONSCHEDULE_H
