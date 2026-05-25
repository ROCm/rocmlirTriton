//===- PerfConfigParsingTests.cpp - Tests for GemmParams/GemmGemmParams ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct PerfConfigTestEnv {
  MLIRContext ctx;

  PerfConfigTestEnv() { ctx.getOrLoadDialect<RockDialect>(); }

  StringAttr str(StringRef s) { return StringAttr::get(&ctx, s); }
};
} // namespace

// --- GemmParamsAttr: v1 back-compat ---

TEST(PerfConfigParsingTest, GemmParamsValidV1BackCompat) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:v1:128,128,16,1,1,4,32,1,2,0,1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlock(), 128);
  EXPECT_EQ(attr.getNPerBlock(), 128);
  EXPECT_EQ(attr.getKPerBlock(), 16);
  EXPECT_EQ(attr.getKpack(), 1);
  EXPECT_EQ(attr.getNumCTAs(), 1);
  EXPECT_EQ(attr.getNumWaves(), 4);
  EXPECT_EQ(attr.getMatrixInstrNonkdim(), 32);
  EXPECT_EQ(attr.getSplitKFactor(), 1);
  EXPECT_EQ(attr.getNumStages(), 2);
  EXPECT_EQ(attr.getWavesPerEU(), 0);
  EXPECT_EQ(attr.getGridGroupSize(), 1);
  // v1 strings predate the knob fields; the parser must default all 7
  // knobs to `kKnobDefault` so older tuning DBs continue to work.
  EXPECT_EQ(attr.getUseAsyncCopy(), kKnobDefault);
  EXPECT_EQ(attr.getUseBlockPingpong(), kKnobDefault);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferAtomics(), kKnobDefault);
  EXPECT_EQ(attr.getBufferOpsAnalyzeSmallTensorRange(), kKnobDefault);
  EXPECT_EQ(attr.getScheduleHint(), kKnobDefault);
}

// --- GemmParamsAttr: v2 ---

TEST(PerfConfigParsingTest, GemmParamsValidV2AllDefaults) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,-1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlock(), 128);
  EXPECT_EQ(attr.getGridGroupSize(), 1);
  EXPECT_EQ(attr.getUseAsyncCopy(), kKnobDefault);
  EXPECT_EQ(attr.getUseBlockPingpong(), kKnobDefault);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferAtomics(), kKnobDefault);
  EXPECT_EQ(attr.getBufferOpsAnalyzeSmallTensorRange(), kKnobDefault);
  EXPECT_EQ(attr.getScheduleHint(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmParamsValidV2MixedKnobs) {
  PerfConfigTestEnv e;
  // Knob block: useAsyncCopy=1, useBlockPingpong=0, useInThreadTranspose=-1,
  // useBufferOps=1, useBufferAtomics=0, bufferOpsAnalyzeSmallTensorRange=1,
  // scheduleHint=2 (memory-bound-attention).
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,1,0,-1,1,0,1,2"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseAsyncCopy(), 1);
  EXPECT_EQ(attr.getUseBlockPingpong(), 0);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), 1);
  EXPECT_EQ(attr.getUseBufferAtomics(), 0);
  EXPECT_EQ(attr.getBufferOpsAnalyzeSmallTensorRange(), 1);
  EXPECT_EQ(attr.getScheduleHint(), 2);
}

TEST(PerfConfigParsingTest, GemmParamsV2RoundTrip) {
  PerfConfigTestEnv e;
  StringRef original = "gemm:v2:128,128,16,1,1,4,32,1,2,0,1,1,0,-1,1,0,1,2";
  auto attr = GemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  StringAttr serialized = attr.getPerfConfigAttr();
  EXPECT_EQ(serialized.strref(), original);
}

TEST(PerfConfigParsingTest, GemmParamsV2ScheduleHintCombination) {
  PerfConfigTestEnv e;
  // scheduleHint is a bitfield (see KnobUtils.h). 3 = attention |
  // memory-bound-attention, mirroring upstream's
  // `schedule_hint="attention,memory-bound-attention"`. The round-trip
  // must preserve the combined value as-is.
  StringRef original =
      "gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,3";
  auto attr = GemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getScheduleHint(), 3);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(), original);
}

// --- GemmParamsAttr: v2 knob-range validation ---

TEST(PerfConfigParsingTest, GemmParamsV2RejectsBoolKnobAboveOne) {
  PerfConfigTestEnv e;
  // useBlockPingpong = 2 -- not in {-1, 0, 1}. Without validation this
  // round-trips through and is truthy-coerced to "force on" in
  // Pipelines.cpp; we want it rejected at parse time instead.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,2,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV2RejectsBoolKnobBelowMinusOne) {
  PerfConfigTestEnv e;
  // useAsyncCopy = -2 -- not in {-1, 0, 1}.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-2,-1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV2RejectsScheduleHintUnknownBit) {
  PerfConfigTestEnv e;
  // scheduleHint = 4 -- bit 2 is not in `kAllScheduleHintBits` today.
  // Accepting it would silently lose the hint downstream.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,4"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest,
     GemmParamsV2RejectsScheduleHintNegativeNonSentinel) {
  PerfConfigTestEnv e;
  // scheduleHint = -2 -- only `-1` (kKnobDefault) is a legal negative
  // value.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,-2"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV2TooFewParams) {
  PerfConfigTestEnv e;
  // v2 expects 18 fields (11 tunables + 7 knobs). Missing the last knob.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV2TooManyParams) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,-1,99"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsUnknownVersionV3) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:v3:128,128,16,1,1,4,32,1,2,0,1"));
  EXPECT_FALSE(attr);
}

// --- GemmParamsAttr: shared error paths (unchanged from v1 era) ---

TEST(PerfConfigParsingTest, GemmParamsWrongPrefix) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("attn:v1:128,128,16,1,1,4,32,1,2,0,1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV1TooFewParams) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:v1:128,128,16"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV1TooManyParams) {
  PerfConfigTestEnv e;
  auto attr =
      GemmParamsAttr::get(e.str("gemm:v1:128,128,16,1,1,4,32,1,2,0,1,99"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsEmptyString) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str(""));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsGarbage) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("not_a_perf_config"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsNonIntegerParam) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:v1:128,abc,16,1,1,4,32,1,2,0,1"));
  EXPECT_FALSE(attr);
}

// --- GemmGemmParamsAttr: v1 back-compat ---

TEST(PerfConfigParsingTest, GemmGemmParamsValidV1BackCompat) {
  PerfConfigTestEnv e;
  auto attr =
      GemmGemmParamsAttr::get(e.str("attn:v1:64,64,32,2,1,2,16,1,1,0,1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlockG0(), 64);
  EXPECT_EQ(attr.getNPerBlockG0(), 64);
  EXPECT_EQ(attr.getKPerBlock(), 32);
  EXPECT_EQ(attr.getKpack(), 2);
  EXPECT_EQ(attr.getNumCTAs(), 1);
  EXPECT_EQ(attr.getNumWaves(), 2);
  EXPECT_EQ(attr.getMatrixInstrNonkdim(), 16);
  EXPECT_EQ(attr.getSplitKFactor(), 1);
  EXPECT_EQ(attr.getNumStages(), 1);
  EXPECT_EQ(attr.getWavesPerEU(), 0);
  EXPECT_EQ(attr.getGridGroupSize(), 1);
  EXPECT_EQ(attr.getUseAsyncCopy(), kKnobDefault);
  EXPECT_EQ(attr.getUseBlockPingpong(), kKnobDefault);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferAtomics(), kKnobDefault);
  EXPECT_EQ(attr.getBufferOpsAnalyzeSmallTensorRange(), kKnobDefault);
  EXPECT_EQ(attr.getScheduleHint(), kKnobDefault);
}

// --- GemmGemmParamsAttr: v2 ---

TEST(PerfConfigParsingTest, GemmGemmParamsValidV2AllDefaults) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1,-1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlockG0(), 64);
  EXPECT_EQ(attr.getScheduleHint(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmGemmParamsValidV2MixedKnobs) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,1,0,-1,1,0,1,1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseAsyncCopy(), 1);
  EXPECT_EQ(attr.getUseBlockPingpong(), 0);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), 1);
  EXPECT_EQ(attr.getUseBufferAtomics(), 0);
  EXPECT_EQ(attr.getBufferOpsAnalyzeSmallTensorRange(), 1);
  EXPECT_EQ(attr.getScheduleHint(), 1);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV2RoundTrip) {
  PerfConfigTestEnv e;
  StringRef original = "attn:v2:64,64,32,2,1,2,16,1,1,0,1,1,0,-1,1,0,1,1";
  auto attr = GemmGemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  StringAttr serialized = attr.getPerfConfigAttr();
  EXPECT_EQ(serialized.strref(), original);
}

// --- GemmGemmParamsAttr: v2 knob-range validation ---

TEST(PerfConfigParsingTest, GemmGemmParamsV2RejectsBoolKnobAboveOne) {
  PerfConfigTestEnv e;
  // useBlockPingpong = 2 -- not in {-1, 0, 1}.
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,-1,2,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV2RejectsScheduleHintUnknownBit) {
  PerfConfigTestEnv e;
  // scheduleHint = 4 -- unknown bit.
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1,4"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV2TooFewParams) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsUnknownVersionV3) {
  PerfConfigTestEnv e;
  auto attr =
      GemmGemmParamsAttr::get(e.str("attn:v3:64,64,32,2,1,2,16,1,1,0,1"));
  EXPECT_FALSE(attr);
}

// --- GemmGemmParamsAttr: shared error paths ---

TEST(PerfConfigParsingTest, GemmGemmParamsWrongPrefix) {
  PerfConfigTestEnv e;
  auto attr =
      GemmGemmParamsAttr::get(e.str("gemm:v1:64,64,32,2,1,2,16,1,1,0,1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV1TooFewParams) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(e.str("attn:v1:64,64"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsEmptyString) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(e.str(""));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsNonIntegerParam) {
  PerfConfigTestEnv e;
  auto attr =
      GemmGemmParamsAttr::get(e.str("attn:v1:64,64,32,2,1,2,16,x,1,0,1"));
  EXPECT_FALSE(attr);
}
