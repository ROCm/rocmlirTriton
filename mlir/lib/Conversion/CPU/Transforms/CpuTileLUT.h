//===- CpuTileLUT.h - Per-CPU matmul tile-size lookup -----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A lookup table that maps the host CPU identity (as reported by LLVM's
// `sys::getHostCPUName()`, e.g. `sapphirerapids`, `znver4`, `icelake-server`)
// and the matmul problem shape `(M, N, K)` to a hand-tuned set of outer-loop
// tile sizes `(mFuse, nFuse, kTile)` for the CPU verifier matmul.
//
// The LUT is consulted before the generic divisor-ladder heuristic in
// `LowerCpuVerifier.cpp`. When no entry exists for the current host CPU the
// caller falls back to the ladder, so adding hardware is purely additive --
// no shape is regressed by introducing a new entry.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CPU_TRANSFORMS_CPU_TILE_LUT_H
#define MLIR_DIALECT_CPU_TRANSFORMS_CPU_TILE_LUT_H

#include <cstdint>
#include <optional>

namespace mlir {
namespace cpu {

/// A triple of outer-loop tile sizes for the CPU verifier matmul. These
/// correspond to `MatmulTileSizes::{mFuse,nFuse,kTile}` and are *not* the
/// innermost register-blocking micro-tiles (those stay at SIMD width).
struct CpuTileTriple {
  int64_t mFuse;
  int64_t nFuse;
  int64_t kTile;
};

/// Look up tile sizes for the *current* host CPU and the given matmul shape.
/// Returns `std::nullopt` when no LUT entry covers the host -- in which case
/// the caller should fall back to the divisor-ladder heuristic.
///
/// Dynamic dims are passed through as `ShapedType::kDynamic`; LUT entries
/// are free to ignore those dims and return whatever default the CPU prefers.
std::optional<CpuTileTriple> lookupHostCpuTileSizes(int64_t M, int64_t N,
                                                    int64_t K);

} // namespace cpu
} // namespace mlir

#endif // MLIR_DIALECT_CPU_TRANSFORMS_CPU_TILE_LUT_H
