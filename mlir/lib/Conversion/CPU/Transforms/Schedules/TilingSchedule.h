//===- TilingSchedule.h - Tiling transform schedule ------------- C++ -*-===//
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

/// Hold data used during tiling schedule.
struct MatmulTileSizes {
  /// AVX is 256 bits = 8 fp32 lanes.
  static constexpr int64_t kVectorSize = 8;

  /// Tile sizes for M,N,K.
  int64_t mFuse = 256;
  int64_t nFuse = 64;
  int64_t kTile = 64;
  /// Innermost register-blocking micro-tile (M, N, K). Defaults to the
  /// SIMD vector width so the inner kernel maps to a single AVX register.
  int64_t microTileM = kVectorSize;
  int64_t microTileN = kVectorSize;
  int64_t microTileK = kVectorSize;

  /// Per-dim divisibility: true when the corresponding problem dim is a
  /// clean multiple of the chosen tile, false otherwise.
  bool mDivisible = true;
  bool nDivisible = true;
  bool kDivisible = true;

  /// Position of the (M, N, K) iter dims within the matmul's iter-space.
  unsigned mDim = 0;
  unsigned nDim = 1;
  unsigned kDim = 2;
};

/// Build the tiling transform module.
/// This schedule tiles and peels matmul ops.
///
/// The tiling strategy is:
/// 1. Fuse + tile outer loops: [mFuse, nFuse] (M, N)
/// 2. Peel the M loop if it's not a multiple of mFuse.
/// 3. Tile reduction dimension: [0, 0, kTile] (K)
/// 4. Peel the K loop if it's not a multiple of kTile.
/// 5. Tile microkernel: [microTileM, microTileN, microTileK]
/// 6. Apply canonicalization and lower-affine
OwningOpRef<ModuleOp> buildTilingSchedule(MLIRContext *ctx,
                                          const MatmulTileSizes &tileSizes);

/// Convenience overload: build the tiling schedule with the default tile
/// sizes from `MatmulTileSizes{}`. Kept for callers that don't have a
/// `func::FuncOp` available to drive shape-aware tile selection.
OwningOpRef<ModuleOp> buildTilingSchedule(MLIRContext *ctx);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_SCHEDULES_TILINGSCHEDULE_H
