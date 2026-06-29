//===- KnobUtils.h - Triton knob constants and validation -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The knobs are serialized into the `gemm:v3:` / `attn:v3:` perfConfig
// schema and threaded through `TritonOptions` / `BackendOptions` in
// `Pipelines.h`. They split into two shapes:
//
//   - Five knobs, which legal values are:
//     `kKnobDefault` (-1, "use the arch default"), `0` (force off), or
//     `1` (force on). Validate with `isValidKnobBoolean`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_KNOBUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_KNOBUTILS_H

#include <cstdint>

namespace mlir {
namespace rock {

/// Tri-state sentinel for the five Triton knob fields in the perfConfig
/// (`useAsyncCopy`, `useBlockPingpong`, `useInThreadTranspose`,
/// `useBufferOps`, `useBufferAtomics`) plus the debug-only
/// `TritonOptions::bufferOpsAnalyzeSmallTensorRange` override (which lives
/// outside the perfConfig). A field set to
/// `kKnobDefault` means "use the per-arch default"; `0` and `1` mean
/// explicit off/on. Lives here (rather than in `Pipelines.h`) so
/// the IR / Tuning libraries can consume it without pulling in
/// `mlir/Pass/PassOptions.h`.
inline constexpr int kKnobDefault = -1;

/// Returns true iff `value` is a legal value for any of the
/// boolean Triton knobs.
bool isValidKnobBoolean(int64_t value);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_KNOBUTILS_H
