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
// The set of accepted tokens has two sources:
//   - TTGIR variants come from Triton's own `SchedHint` TableGen enum
//     (currently `attention`); we look them up with the auto-generated
//     `triton::amdgpu::symbolizeSchedHint`. New enum entries Triton adds
//     on a future bump are picked up automatically.
//   - `memory-bound-attention` is an LLIR-only variant that upstream
//     Triton recognises via a literal string match in compiler.py and
//     translates into the `amdgpu-sched-strategy=iterative-ilp` kernel
//     attribute. It has no TableGen enum entry, so we hard-code it here.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_SCHEDULEHINTUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_SCHEDULEHINTUTILS_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {
namespace rock {

/// The literal LLIR-only schedule-hint variant. Upstream Triton recognises
/// it via a string match in compiler.py and emits the kernel attribute
/// `amdgpu-sched-strategy=iterative-ilp`; it intentionally does not have
/// a TableGen enum entry.
inline constexpr llvm::StringRef kMemoryBoundAttentionHint =
    "memory-bound-attention";

/// Parse a `scheduleHint` string into the set of variants it requests.
///
/// `raw` mirrors upstream Triton's `HIPOptions.schedule_hint`: either the
/// sentinel `"none"` (case-insensitive) or a comma-separated list of
/// variants. Each token is validated against:
///   - Triton's `SchedHint` TableGen enum (today: `attention`), via the
///     auto-generated `triton::amdgpu::symbolizeSchedHint`; or
///   - the LLIR-only literal `kMemoryBoundAttentionHint`.
///
/// On success, `hints` is filled with the lowercased, trimmed variant
/// names in the order they appeared (duplicates preserved). The `"none"`
/// sentinel yields an empty vector. On failure (unknown variant or empty
/// token), an error is emitted to `llvm::errs()` and `hints` is left in
/// an unspecified state.
LogicalResult parseScheduleHint(llvm::StringRef raw,
                                llvm::SmallVectorImpl<std::string> &hints);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_SCHEDULEHINTUTILS_H
