//===- TilingSchedule.h - Tiling transform schedule ------------- C++ -*-===//
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
// This header declares the function that builds the tiling transform
// schedule using the MLIR C++ API.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_TILINGSCHEDULE_H
#define MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_TILINGSCHEDULE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;

namespace cpu {

/// Build the tiling (optimization) transform module.
/// This schedule tiles matmul operations for CPU cache efficiency.
///
/// The tiling strategy is:
/// 1. Fuse + tile outer loops: [1, 256, 64] (batch, M, N)
/// 2. Tile reduction dimension: [0, 0, 0, 64] (K)
/// 3. Tile microkernel: [0, 8, 8, 1] (register blocking)
/// 4. Apply canonicalization and lower-affine
OwningOpRef<ModuleOp> buildTilingSchedule(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_TILINGSCHEDULE_H
