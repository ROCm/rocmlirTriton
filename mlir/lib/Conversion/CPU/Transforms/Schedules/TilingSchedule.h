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

#include <cstdint>

namespace mlir {
class MLIRContext;

namespace cpu {

/// Tile sizes used by the matmul portion of the tiling schedule.
///
/// The defaults match the historical hard-coded values. Dimension order
/// matches the matmul `linalg.generic` iteration space (G, M, N, K), where
/// G/M/N are parallel and K is the reduction.
///
/// All tile sizes must divide their corresponding problem dim cleanly, so
/// the generated `scf.for` loops have static post-tile shapes (otherwise
/// `transform.structured.vectorize` would fail). The caller is responsible
/// for choosing divisors -- see `chooseMatmulTileSizes` in LowerCpuVerifier.cpp.
struct MatmulTileSizes {
  /// Outer fuse tile (G, M, N); applied with interchange [0, 2, 1] so the
  /// loop nest order becomes G -> N -> M.
  int64_t gFuse = 1;
  int64_t mFuse = 256;
  int64_t nFuse = 64;
  /// K reduction tile applied by the second `tile_using_for`.
  int64_t kTile = 64;
  /// Innermost register-blocking micro-tile (M, N, K).
  int64_t microTileM = 8;
  int64_t microTileN = 8;
  int64_t microTileK = 8;
};

/// Build the tiling (optimization) transform module.
/// This schedule tiles matmul operations for CPU cache efficiency.
///
/// The tiling strategy is:
/// 1. Fuse + tile outer loops: [gFuse, mFuse, nFuse] (G, M, N)
/// 2. Tile reduction dimension: [0, 0, 0, kTile] (K)
/// 3. Tile microkernel: [0, microTileM, microTileN, microTileK]
/// 4. Apply canonicalization and lower-affine
OwningOpRef<ModuleOp> buildTilingSchedule(MLIRContext *ctx,
                                          const MatmulTileSizes &tileSizes);

/// Convenience overload: build the tiling schedule with the default tile
/// sizes from `MatmulTileSizes{}`. Kept for callers that don't have a
/// `func::FuncOp` available to drive shape-aware tile selection.
OwningOpRef<ModuleOp> buildTilingSchedule(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_TILINGSCHEDULE_H
