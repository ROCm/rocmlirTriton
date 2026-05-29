//===- KnobUtils.cpp - Triton knob constants and validation ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/KnobUtils.h"

#include "llvm/Support/raw_ostream.h"

#include <array>
#include <utility>

using namespace mlir;

namespace mlir {
namespace rock {

namespace {

// Stable bit -> variant name table. Append-only; never reorder.
constexpr std::array<std::pair<int64_t, llvm::StringRef>, 2>
    kScheduleHintBitTable = {{
        {kScheduleHintAttention, "attention"},
        {kScheduleHintMemoryBoundAttention, kMemoryBoundAttentionHint},
    }};

// Returns the union of all bits known to the table. Bits set in a
// bitfield outside this mask are unrecognised and should be rejected by
// `isValidScheduleHintBitfield`.
constexpr int64_t allKnownBits() {
  int64_t mask = 0;
  for (auto [bit, name] : kScheduleHintBitTable)
    mask |= bit;
  return mask;
}

static_assert(allKnownBits() == kAllScheduleHintBits,
              "kScheduleHintBitTable and kAllScheduleHintBits differ; "
              "update both when adding a schedule-hint variant.");

} // namespace

bool isValidKnobBoolean(int64_t value) {
  return value == kKnobDefault || value == 0 || value == 1;
}

bool isValidScheduleHintBitfield(int64_t bitfield) {
  if (bitfield == kKnobDefault)
    return true;
  if (bitfield < 0)
    return false;
  return (bitfield & ~kAllScheduleHintBits) == 0;
}

LogicalResult
expandScheduleHintBitfield(int64_t bitfield,
                           llvm::SmallVectorImpl<std::string> &hints) {
  hints.clear();
  if (!isValidScheduleHintBitfield(bitfield)) {
    llvm::errs() << "scheduleHint bitfield " << bitfield
                 << " is invalid; expected kKnobDefault (-1), "
                    "kScheduleHintNone (0), or a subset of bitmask "
                 << kAllScheduleHintBits << "\n";
    return failure();
  }
  if (bitfield == kKnobDefault || bitfield == kScheduleHintNone)
    return success();
  for (auto [bit, name] : kScheduleHintBitTable) {
    if (bitfield & bit)
      hints.push_back(name.str());
  }
  return success();
}

} // namespace rock
} // namespace mlir
