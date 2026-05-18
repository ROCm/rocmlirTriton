//===- ScheduleHintUtilsTests.cpp - Tests for scheduleHint bitfield -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coverage for the multi-select bitfield encoding mirrored from upstream
// Triton's `HIPOptions.schedule_hint`. The encoding is part of the
// perfConfig wire format, so each round-trip is effectively a contract
// test; regressions here will silently change tuning DB semantics.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/ScheduleHintUtils.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
constexpr int64_t kKnobDefault = -1;
}

TEST(ScheduleHintUtilsTest, ParseNoneSentinel) {
  llvm::SmallVector<std::string, 2> hints;
  ASSERT_TRUE(succeeded(parseScheduleHint("none", hints)));
  EXPECT_TRUE(hints.empty());
  ASSERT_TRUE(succeeded(parseScheduleHint("NONE", hints)));
  EXPECT_TRUE(hints.empty());
}

TEST(ScheduleHintUtilsTest, ParseSingleVariants) {
  llvm::SmallVector<std::string, 2> hints;
  ASSERT_TRUE(succeeded(parseScheduleHint("attention", hints)));
  ASSERT_EQ(hints.size(), 1u);
  EXPECT_EQ(hints[0], "attention");

  ASSERT_TRUE(succeeded(parseScheduleHint("memory-bound-attention", hints)));
  ASSERT_EQ(hints.size(), 1u);
  EXPECT_EQ(hints[0], "memory-bound-attention");
}

TEST(ScheduleHintUtilsTest, ParseCommaSeparatedCombination) {
  llvm::SmallVector<std::string, 2> hints;
  ASSERT_TRUE(succeeded(
      parseScheduleHint("attention,memory-bound-attention", hints)));
  ASSERT_EQ(hints.size(), 2u);
  EXPECT_EQ(hints[0], "attention");
  EXPECT_EQ(hints[1], "memory-bound-attention");

  // Whitespace and casing tolerance, matching upstream's `.lower()` + split.
  ASSERT_TRUE(succeeded(
      parseScheduleHint("  Attention , MEMORY-BOUND-ATTENTION  ", hints)));
  ASSERT_EQ(hints.size(), 2u);
  EXPECT_EQ(hints[0], "attention");
  EXPECT_EQ(hints[1], "memory-bound-attention");
}

TEST(ScheduleHintUtilsTest, ParseRejectsUnknownAndEmptyTokens) {
  llvm::SmallVector<std::string, 2> hints;
  EXPECT_TRUE(failed(parseScheduleHint("bogus", hints)));
  EXPECT_TRUE(failed(parseScheduleHint("attention,", hints)));
  EXPECT_TRUE(failed(parseScheduleHint(",attention", hints)));
}

TEST(ScheduleHintUtilsTest, StringToBitfield) {
  auto none = scheduleHintToBitfield("none");
  ASSERT_TRUE(succeeded(none));
  EXPECT_EQ(*none, kScheduleHintNone);

  auto attn = scheduleHintToBitfield("attention");
  ASSERT_TRUE(succeeded(attn));
  EXPECT_EQ(*attn, kScheduleHintAttention);

  auto mba = scheduleHintToBitfield("memory-bound-attention");
  ASSERT_TRUE(succeeded(mba));
  EXPECT_EQ(*mba, kScheduleHintMemoryBoundAttention);

  auto combo = scheduleHintToBitfield("attention,memory-bound-attention");
  ASSERT_TRUE(succeeded(combo));
  EXPECT_EQ(*combo,
            kScheduleHintAttention | kScheduleHintMemoryBoundAttention);

  // Token order in the input must not affect the bitfield (set semantics).
  auto reverseCombo =
      scheduleHintToBitfield("memory-bound-attention,attention");
  ASSERT_TRUE(succeeded(reverseCombo));
  EXPECT_EQ(*reverseCombo, *combo);
}

TEST(ScheduleHintUtilsTest, BitfieldToStringRoundTrip) {
  auto none = scheduleHintBitfieldToString(kScheduleHintNone);
  ASSERT_TRUE(succeeded(none));
  EXPECT_EQ(*none, "none");

  // `kKnobDefault` is *not* a hint state; the canonical string for it
  // is also "none", matching upstream's default `HIPOptions.schedule_hint`.
  auto def = scheduleHintBitfieldToString(kKnobDefault);
  ASSERT_TRUE(succeeded(def));
  EXPECT_EQ(*def, "none");

  auto attn = scheduleHintBitfieldToString(kScheduleHintAttention);
  ASSERT_TRUE(succeeded(attn));
  EXPECT_EQ(*attn, "attention");

  auto combo = scheduleHintBitfieldToString(kScheduleHintAttention |
                                            kScheduleHintMemoryBoundAttention);
  ASSERT_TRUE(succeeded(combo));
  EXPECT_EQ(*combo, "attention,memory-bound-attention");
}

TEST(ScheduleHintUtilsTest, BitfieldRejectsUnknownBits) {
  // Bit 31 has no variant assignment today. Surfacing it as an error
  // forces the caller to update kScheduleHintBitTable rather than
  // silently dropping the hint.
  int64_t unknown = static_cast<int64_t>(1) << 31;
  llvm::SmallVector<std::string, 2> hints;
  EXPECT_TRUE(failed(expandScheduleHintBitfield(unknown, hints)));
  EXPECT_TRUE(failed(scheduleHintBitfieldToString(unknown)));
}

TEST(ScheduleHintUtilsTest, ExpandBitfieldStableOrder) {
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
  ASSERT_TRUE(succeeded(expandScheduleHintBitfield(kKnobDefault, hints)));
  EXPECT_TRUE(hints.empty());

  hints.clear();
  ASSERT_TRUE(succeeded(expandScheduleHintBitfield(kScheduleHintNone, hints)));
  EXPECT_TRUE(hints.empty());
}
