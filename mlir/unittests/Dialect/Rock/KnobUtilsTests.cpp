//===- KnobUtilsTests.cpp - Tests for Triton knob utilities ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coverage for the seven Triton knob fields: `isValidKnobBoolean` for
// the six tri-state booleans and the `scheduleHint` bitfield encoding
// (validity rule + stable-order expansion). The encoding is part of
// the perfConfig wire format, so each round-trip is effectively a
// contract test; regressions here will silently change tuning DB
// semantics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/KnobUtils.h"

#include "gtest/gtest.h"

#include <limits>

using namespace mlir;
using namespace mlir::rock;

TEST(KnobUtilsTest, IsValidKnobBoolean) {
  EXPECT_TRUE(isValidKnobBoolean(kKnobDefault));
  EXPECT_TRUE(isValidKnobBoolean(0));
  EXPECT_TRUE(isValidKnobBoolean(1));

  // Anything outside {-1, 0, 1} is rejected. The implicit `int64_t ->
  // bool` conversion in the Pipelines.cpp helpers would otherwise
  // silently coerce e.g. `2` to "force on", which is the bug this
  // helper exists to prevent.
  EXPECT_FALSE(isValidKnobBoolean(2));
  EXPECT_FALSE(isValidKnobBoolean(-2));
  EXPECT_FALSE(isValidKnobBoolean(std::numeric_limits<int64_t>::max()));
  EXPECT_FALSE(isValidKnobBoolean(std::numeric_limits<int64_t>::min()));
}

TEST(KnobUtilsTest, BitfieldRejectsUnknownBits) {
  // Bit 31 has no variant assignment today. Surfacing it as an error
  // forces the caller to update kScheduleHintBitTable rather than
  // silently dropping the hint.
  int64_t unknown = static_cast<int64_t>(1) << 31;
  llvm::SmallVector<std::string, 2> hints;
  EXPECT_TRUE(failed(expandScheduleHintBitfield(unknown, hints)));
}

TEST(KnobUtilsTest, ExpandBitfieldStableOrder) {
  // Stable order matters: makeTtgir() iterates the expansion to add
  // passes, and the resulting pipeline must not depend on the order
  // hints appeared in the user-facing string.
  llvm::SmallVector<std::string, 2> hints;
  ASSERT_TRUE(succeeded(expandScheduleHintBitfield(
      kScheduleHintAttention | kScheduleHintMemoryBoundAttention, hints)));
  ASSERT_EQ(hints.size(), 2u);
  EXPECT_EQ(hints[0], "attention");
  EXPECT_EQ(hints[1], "memory-bound-attention");

  hints.clear();
  ASSERT_TRUE(
      succeeded(expandScheduleHintBitfield(kScheduleHintAttention, hints)));
  ASSERT_EQ(hints.size(), 1u);
  EXPECT_EQ(hints[0], "attention");

  hints.clear();
  ASSERT_TRUE(succeeded(
      expandScheduleHintBitfield(kScheduleHintMemoryBoundAttention, hints)));
  ASSERT_EQ(hints.size(), 1u);
  EXPECT_EQ(hints[0], "memory-bound-attention");

  hints.clear();
  ASSERT_TRUE(succeeded(expandScheduleHintBitfield(kKnobDefault, hints)));
  EXPECT_TRUE(hints.empty());

  hints.clear();
  ASSERT_TRUE(succeeded(expandScheduleHintBitfield(kScheduleHintNone, hints)));
  EXPECT_TRUE(hints.empty());
}

TEST(KnobUtilsTest, IsValidScheduleHintBitfield) {
  // The two sentinels and every individual variant bit are valid, as
  // is any union of variant bits.
  EXPECT_TRUE(isValidScheduleHintBitfield(kKnobDefault));
  EXPECT_TRUE(isValidScheduleHintBitfield(kScheduleHintNone));
  EXPECT_TRUE(isValidScheduleHintBitfield(kScheduleHintAttention));
  EXPECT_TRUE(isValidScheduleHintBitfield(kScheduleHintMemoryBoundAttention));
  EXPECT_TRUE(isValidScheduleHintBitfield(kAllScheduleHintBits));

  // Negative values other than `kKnobDefault` are invalid; -1 is the
  // only sentinel. Catching this matters because an int64_t -1 is
  // two's-complement-all-ones, so a naive `value & someBit` test would
  // spuriously fire on the sentinel.
  EXPECT_FALSE(isValidScheduleHintBitfield(-2));
  EXPECT_FALSE(
      isValidScheduleHintBitfield(std::numeric_limits<int64_t>::min()));

  // Any bit outside `kAllScheduleHintBits` is unrecognised. Pick a
  // high bit so this stays robust as new variants are appended.
  EXPECT_FALSE(isValidScheduleHintBitfield(static_cast<int64_t>(1) << 31));
  EXPECT_FALSE(isValidScheduleHintBitfield(kAllScheduleHintBits + 1));
}
