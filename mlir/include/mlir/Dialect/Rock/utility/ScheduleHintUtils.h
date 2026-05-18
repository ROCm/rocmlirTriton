//===- ScheduleHintUtils.h - Parse Triton scheduleHint strings ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Centralised parser for the `scheduleHint` knob mirrored from upstream
// Triton's `HIPOptions.schedule_hint`
// (external/triton/third_party/amd/backend/compiler.py).
//
// rocmlirTriton's `scheduleHint` is serialized into the perfConfig as an
// `int64_t` **bitfield**. Each TTGIR or LLIR variant gets a stable bit;
// the bitfield can express combinations of variants (e.g.
// `attention | memory-bound-attention`), matching upstream's
// comma-separated `HIPOptions.schedule_hint` string. The bit positions
// are append-only -- new variants must be assigned a new bit, never
// reuse an existing one.
//
// The set of accepted tokens has two sources:
//   - TTGIR variants come from Triton's own `SchedHint` TableGen enum
//     (currently `attention`); we look them up with the auto-generated
//     `triton::amdgpu::symbolizeSchedHint`. New enum entries Triton adds
//     on a future bump are picked up automatically by the string parser,
//     but a new bit must also be added to the table below for them to
//     round-trip through the perfConfig.
//   - `memory-bound-attention` is an LLIR-only variant that upstream
//     Triton recognises via a literal string match in compiler.py and
//     translates into the `amdgpu-sched-strategy=iterative-ilp` kernel
//     attribute. It has no TableGen enum entry, so we hard-code it here.
//
// Bit encoding (perfConfig serialization, append-only):
//   kKnobDefault (-1)                  = use the arch default (today
//                                         equivalent to "none")
//   kScheduleHintNone (0)              = explicit "no hints"
//   kScheduleHintAttention (0x1)       = bit for "attention"
//   kScheduleHintMemoryBoundAttention  = bit for "memory-bound-attention"
//   (0x2)
//
// Note: upstream Triton's `SchedHint` TableGen enum has its own integer
// values (none=0, attention=2). Those are *not* used as bit positions
// here because they're not powers of two and would alias. The mapping
// from variant string to bit is local to this header.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_SCHEDULEHINTUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_SCHEDULEHINTUTILS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace rock {

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

/// Parse a `scheduleHint` string into the set of variants it requests.
///
/// `raw` mirrors upstream Triton's `HIPOptions.schedule_hint`: either
/// the sentinel `"none"` (case-insensitive) or a comma-separated list of
/// variants. Each token is validated against:
///   - Triton's `SchedHint` TableGen enum (today: `attention`), via the
///     auto-generated `triton::amdgpu::symbolizeSchedHint`; or
///   - the LLIR-only literal `kMemoryBoundAttentionHint`.
///
/// On success, `hints` is filled with the lowercased, trimmed variant
/// names in the order they appeared. The `"none"` sentinel yields an
/// empty vector. On failure (unknown variant or empty token), an error
/// is emitted to `llvm::errs()` and `hints` is left in an unspecified
/// state.
LogicalResult parseScheduleHint(llvm::StringRef raw,
                                llvm::SmallVectorImpl<std::string> &hints);

/// Convert a `scheduleHint` string into its stable bitfield encoding.
/// The `"none"` sentinel maps to `kScheduleHintNone` (0). Each
/// recognised variant contributes its bit. Unknown variants emit an
/// error and return `failure()`.
FailureOr<int64_t> scheduleHintToBitfield(llvm::StringRef raw);

/// Inverse of `scheduleHintToBitfield`. Iterates set bits in stable
/// order (least significant first) and joins their variant names with
/// commas. `kKnobDefault` (`-1`) and `kScheduleHintNone` both stringify
/// to `"none"`. Unknown bits in the input return `failure()`.
FailureOr<std::string> scheduleHintBitfieldToString(int64_t bitfield);

/// Iterate the set bits of `bitfield` in stable order, appending each
/// variant's string name to `hints`. `kKnobDefault` (`-1`) and
/// `kScheduleHintNone` produce an empty `hints` vector. Unknown bits
/// return `failure()`. This is the canonical helper for replicating
/// upstream's `for hint in options.schedule_hint.split(","):` loop on
/// our bitfield representation.
LogicalResult
expandScheduleHintBitfield(int64_t bitfield,
                           llvm::SmallVectorImpl<std::string> &hints);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_SCHEDULEHINTUTILS_H
