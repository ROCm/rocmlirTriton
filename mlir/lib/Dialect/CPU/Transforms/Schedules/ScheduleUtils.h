//===- ScheduleUtils.h - Common utilities for transform schedules ---------===//
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
// This header declares common utility functions used by transform schedule
// builders.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_SCHEDULEUTILS_H
#define MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_SCHEDULEUTILS_H

#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;

namespace cpu {

/// Create a module with the transform.with_named_sequence attribute.
/// This is the standard setup for transform schedule modules.
OwningOpRef<ModuleOp> createTransformModule(MLIRContext *ctx);

/// Get the !transform.any_op type, commonly used for transform handles.
transform::AnyOpType getAnyOpType(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_SCHEDULEUTILS_H
