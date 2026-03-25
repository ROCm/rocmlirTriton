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

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"

#include <functional>

namespace mlir {
class MLIRContext;

namespace cpu {

/// Callback type for building the body of a transform sequence.
/// The callback receives the ImplicitLocOpBuilder and the block argument
/// (the root handle).
using TransformBodyBuilder =
    std::function<void(ImplicitLocOpBuilder &, BlockArgument)>;

/// Create a module with the transform.with_named_sequence attribute.
/// This is the standard setup for transform schedule modules.
OwningOpRef<ModuleOp> createTransformModule(MLIRContext *ctx);

/// Get the !transform.any_op type, commonly used for transform handles.
transform::AnyOpType getAnyOpType(MLIRContext *ctx);

/// Create a complete transform module with a named sequence called
/// "__transform_main". The bodyBuilder callback is invoked to populate
/// the sequence body. YieldOp is automatically added at the end.
OwningOpRef<ModuleOp> buildTransformModule(MLIRContext *ctx,
                                           TransformBodyBuilder bodyBuilder);

/// Create a DictionaryAttr containing the iterator_types attribute for
/// matching matmul-like operations (pattern: [parallel, parallel, parallel,
/// reduction]).
DictionaryAttr getMatmulIteratorTypesAttr(MLIRContext *ctx);

/// Create a transform::MatchOp that matches linalg.generic ops with matmul
/// iterator types pattern [parallel, parallel, parallel, reduction].
transform::MatchOp createMatchMatmulOp(ImplicitLocOpBuilder &ib,
                                       MLIRContext *ctx, Value target);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_SCHEDULEUTILS_H
