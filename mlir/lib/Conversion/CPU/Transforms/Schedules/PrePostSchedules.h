//===- PrePostSchedules.h - Pre/Post transform schedules -------- C++ -*-===//
//
// Copyright 2026 Advanced Micro Devices.
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
// This header declares functions that build the pre and post transform
// schedules using the MLIR C++ API. These schedules are applied before and
// after each main transform phase (tiling, vectorization, lowering).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_PREPOSTSCHEDULES_H
#define MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_PREPOSTSCHEDULES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;

namespace cpu {

/// Build the pre-sequence transform module.
/// This schedule applies canonicalization and CSE before each main transform.
///
/// Equivalent to:
///   %func = transform.structured.match ops{["func.func"]} in %arg0
///   %1 = transform.apply_registered_pass "canonicalize" to %func
///   transform.apply_cse to %1
OwningOpRef<ModuleOp> buildPreSchedule(MLIRContext *ctx);

/// Build the post-sequence transform module.
/// This schedule applies LICM, vector hoisting, and canonicalization after
/// each main transform.
///
/// Equivalent to:
///   %func = transform.structured.match ops{["func.func"]} in %arg0
///   %loops = transform.structured.match interface{LoopLikeInterface} in %func
///   transform.apply_licm to %loops
///   %1 = transform.structured.hoist_redundant_vector_transfers %func
///   %2 = transform.structured.hoist_redundant_vector_broadcasts %1
///   %3 = transform.apply_registered_pass "canonicalize" to %2
OwningOpRef<ModuleOp> buildPostSchedule(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_PREPOSTSCHEDULES_H
