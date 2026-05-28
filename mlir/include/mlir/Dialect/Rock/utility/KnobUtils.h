//===- KnobUtils.h - Triton knob constants and validation -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The knobs are serialized into the `gemm:v2:` / `attn:v2:` perfConfig
// schema and threaded through `TritonOptions` / `BackendOptions` in
// `Pipelines.h`. They split into two shapes:
//
//   - Six knobs, which legal values are:
//     `kKnobDefault` (-1, "use the arch default"), `0` (force off), or
//     `1` (force on). Validate with `isValidKnobBoolean`.
//   - One bitfield (`scheduleHint`) that mirrors upstream's comma-
//     separated `HIPOptions.schedule_hint` string. Each variant gets
//     a stable bit; combinations are expressed by OR-ing the bits.
//     Validate with `isValidScheduleHintBitfield` and expand to the
//     upstream-style variant-string list with
//     `expandScheduleHintBitfield`.
//
// The bit positions in the `scheduleHint` encoding are append-only;
// new variants must claim a new bit and extend `kScheduleHintBitTable`
// in `KnobUtils.cpp` in lockstep.
//
// Recognised `scheduleHint` variants:
//   - TTGIR variants from Triton's own `SchedHint` TableGen enum
//     (currently just `attention`). On a future Triton bump, new enum
//     entries must be wired in by claiming a new bit.
//   - `memory-bound-attention` is an LLIR-only variant that upstream
//     Triton recognises via a literal string match in compiler.py and
//     translates into the `amdgpu-sched-strategy=iterative-ilp` kernel
//     attribute. It has no TableGen enum entry, so we keep its bit
//     assignment hard-coded here.
//
// Bit encoding (perfConfig serialization, append-only):
//   kKnobDefault (-1)                       = use the arch default
//                                             (today equivalent to "none")
//   kScheduleHintNone (0)                   = explicit "no hints"
//   kScheduleHintAttention (0x1)            = bit for "attention"
//   kScheduleHintMemoryBoundAttention (0x2) = bit for "memory-bound-attention"
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_KNOBUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_KNOBUTILS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace rock {

/// Tri-state sentinel for the six Triton knob fields in the perfConfig
/// (`useAsyncCopy`, `useBlockPingpong`, `useInThreadTranspose`,
/// `useBufferOps`, `useBufferAtomics`, `scheduleHint`) plus the debug-only
/// `TritonOptions::bufferOpsAnalyzeSmallTensorRange` override (which lives
/// outside the perfConfig). A field set to
/// `kKnobDefault` means "use the per-arch default"; `0` and `1` mean
/// explicit off/on for the boolean knobs, and `scheduleHint` uses the bit
/// encoding declared below. Lives here (rather than in `Pipelines.h`) so
/// the IR / Tuning libraries can consume it without pulling in
/// `mlir/Pass/PassOptions.h`.
inline constexpr int kKnobDefault = -1;

/// The literal LLIR-only schedule-hint variant. Upstream Triton recognises
/// it via a string match in compiler.py and emits the kernel attribute
/// `amdgpu-sched-strategy=iterative-ilp`; it intentionally does not have
/// a TableGen enum entry.
inline constexpr llvm::StringRef kMemoryBoundAttentionHint =
    "memory-bound-attention";

/// Stable bit encoding of `scheduleHint` for the perfConfig.
/// Append-only: never renumber existing bits.
inline constexpr int64_t kScheduleHintNone = 0;
inline constexpr int64_t kScheduleHintAttention = 1 << 0;
inline constexpr int64_t kScheduleHintMemoryBoundAttention = 1 << 1;

/// Union of all bits claimed by `kScheduleHint*` above. Useful when
/// validating perfConfig strings.
inline constexpr int64_t kAllScheduleHintBits =
    kScheduleHintAttention | kScheduleHintMemoryBoundAttention;

/// Returns true iff `value` is a legal value for any of the
/// boolean Triton knobs.
bool isValidKnobBoolean(int64_t value);

/// Returns true iff `bitfield` is a legal `scheduleHint` value:
///   - `kKnobDefault` (`-1`, "use the arch default"), or
///   - `kScheduleHintNone` (`0`, explicit "no hints"), or
///   - a non-negative value whose set bits are a subset of
///     `kAllScheduleHintBits`.
bool isValidScheduleHintBitfield(int64_t bitfield);

/// Iterate the set bits of `bitfield` in stable order, appending each
/// variant's string name to `hints`. `kKnobDefault` (`-1`) and
/// `kScheduleHintNone` produce an empty `hints` vector. Values that
/// fail `isValidScheduleHintBitfield` (negative non-sentinels or
/// unknown high bits) return `failure()`.
LogicalResult
expandScheduleHintBitfield(int64_t bitfield,
                           llvm::SmallVectorImpl<std::string> &hints);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_KNOBUTILS_H
