//===- PerfConfigParsingTests.cpp - Tests for GemmParams/GemmGemmParams ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include "gtest/gtest.h"

#include <array>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::rock;

namespace {
struct PerfConfigTestEnv {
  MLIRContext ctx;

  PerfConfigTestEnv() { ctx.getOrLoadDialect<RockDialect>(); }

  StringAttr str(StringRef s) { return StringAttr::get(&ctx, s); }
};

struct WarningCapture {
  std::vector<std::string> warnings;
  ScopedDiagnosticHandler handler;

  WarningCapture(MLIRContext &ctx)
      : handler(&ctx, [this](Diagnostic &diag) {
          if (diag.getSeverity() == DiagnosticSeverity::Warning) {
            warnings.push_back(diag.str());
            return success();
          }
          return failure();
        }) {}
};

void expectScheduleHintWarning(const WarningCapture &capture,
                               int64_t scheduleHint) {
  ASSERT_EQ(capture.warnings.size(), 1u);
  const std::string &warning = capture.warnings.front();
  EXPECT_NE(warning.find("scheduleHint=" + std::to_string(scheduleHint)),
            std::string::npos)
      << warning;
  EXPECT_NE(
      warning.find("scheduleHint is no longer supported and will be ignored"),
      std::string::npos)
      << warning;
}

// Builds the canonical named form from `values`, which are in schema order.
// Trailing fields a caller leaves out take their schema default, so appending a
// knob does not have to touch every test. The keys come from the attribute's
// own schema rather than a copy of it, so these helpers stay usable when a
// field is added; what the keys and their order actually are is pinned by the
// `...NamedSchemaIsPinned` tests below, which is where a schema change is meant
// to show up.
std::string perfConfigNamed(StringRef prefix, ArrayRef<StringRef> keys,
                            ArrayRef<int64_t> defaults,
                            ArrayRef<int64_t> values) {
  assert(values.size() <= keys.size() && "too many perfConfig fields");
  std::string out = (prefix + ":").str();
  for (size_t i = 0, e = keys.size(); i < e; ++i) {
    int64_t value = i < values.size() ? values[i] : defaults[i];
    out += (Twine(i ? "," : "") + keys[i] + "=" + Twine(value)).str();
  }
  return out;
}

std::string gemmNamed(ArrayRef<int64_t> f) {
  return perfConfigNamed(GemmParamsAttr::getPerfConfigPrefix(),
                         GemmParamsAttr::getPerfConfigKeys(),
                         GemmParamsAttr::getPerfConfigDefaults(), f);
}

std::string attnNamed(ArrayRef<int64_t> f) {
  return perfConfigNamed(GemmGemmParamsAttr::getPerfConfigPrefix(),
                         GemmGemmParamsAttr::getPerfConfigKeys(),
                         GemmGemmParamsAttr::getPerfConfigDefaults(), f);
}

// `llvm::join` only handles strings, and gmock's container matchers aren't
// available here.
std::string joinInts(ArrayRef<int64_t> values) {
  std::string out;
  for (int64_t value : values)
    out += (out.empty() ? "" : ",") + std::to_string(value);
  return out;
}
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
  // v1 strings predate the knob fields; the parser must default all
  // knobs to `kKnobDefault` so older tuning DBs continue to work.
  EXPECT_EQ(attr.getUseAsyncCopy(), kKnobDefault);
  EXPECT_EQ(attr.getUseBlockPingpong(), kKnobDefault);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferAtomics(), kKnobDefault);
  // v1 predates the v4 `useReductionLayout` knob; it defaults to the knob
  // default (-1 = heuristic / currently off).
  EXPECT_EQ(attr.getUseReductionLayout(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
}

// --- GemmParamsAttr: v2 read-only back-compat ---
//
// v2 carried a trailing `scheduleHint` bitfield (the 6th knob field). That
// knob was removed in v3; v2 strings are still accepted read-only, with the
// trailing token parsed and discarded. Re-serialization always emits the
// canonical v5 form.

TEST(PerfConfigParsingTest, GemmParamsV2BackCompatDiscardsScheduleHint) {
  PerfConfigTestEnv e;
  WarningCapture capture(e.ctx);
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlock(), 128);
  EXPECT_EQ(attr.getGridGroupSize(), 1);
  EXPECT_EQ(attr.getUseAsyncCopy(), kKnobDefault);
  EXPECT_EQ(attr.getUseBlockPingpong(), kKnobDefault);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferAtomics(), kKnobDefault);
  // v2 predates the v4 `useReductionLayout` knob; it defaults to the knob
  // default (-1 = heuristic / currently off).
  EXPECT_EQ(attr.getUseReductionLayout(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
  EXPECT_TRUE(capture.warnings.empty());
}

TEST(PerfConfigParsingTest, GemmParamsV2MixedKnobsReserializeAsV5) {
  PerfConfigTestEnv e;
  WarningCapture capture(e.ctx);
  // Knob block: useAsyncCopy=1, useBlockPingpong=0, useInThreadTranspose=-1,
  // useBufferOps=1, useBufferAtomics=0, plus a trailing scheduleHint=2 that
  // must be parsed and discarded.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,1,0,-1,1,0,2"));
  ASSERT_TRUE(attr);
  expectScheduleHintWarning(capture, 2);
  EXPECT_EQ(attr.getUseAsyncCopy(), 1);
  EXPECT_EQ(attr.getUseBlockPingpong(), 0);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), 1);
  EXPECT_EQ(attr.getUseBufferAtomics(), 0);
  // Re-serialization drops the trailing scheduleHint and emits the canonical
  // named form with the newer knobs defaulted to -1.
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            gemmNamed({128, 128, 16, 1, 1, 4, 32, 1, 2, 0, 1, 1, 0, -1, 1, 0,
                       -1, -1}));
}

TEST(PerfConfigParsingTest, GemmParamsV2IgnoresArbitraryScheduleHintValue) {
  PerfConfigTestEnv e;
  WarningCapture capture(e.ctx);
  // The trailing scheduleHint token is no longer validated; any integer is
  // accepted and discarded (here 4, which was an "unknown bit" pre-v3).
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,4"));
  ASSERT_TRUE(attr);
  expectScheduleHintWarning(capture, 4);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            gemmNamed({128, 128, 16, 1, 1, 4, 32, 1, 2, 0, 1, -1, -1, -1, -1,
                       -1, -1, -1}));
}

TEST(PerfConfigParsingTest, GemmParamsV2RejectsBoolKnobAboveOne) {
  PerfConfigTestEnv e;
  // useBlockPingpong = 2 -- not in {-1, 0, 1}. The bool knobs are still
  // validated on the v2 back-compat path.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,2,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV2TooFewParams) {
  PerfConfigTestEnv e;
  // v2 expects 17 fields (11 tunables + 6 knobs). Missing the last knob.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV2TooManyParams) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,99"));
  EXPECT_FALSE(attr);
}

// --- GemmParamsAttr: v3 (back-compat) ---

TEST(PerfConfigParsingTest, GemmParamsValidV3AllDefaults) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v3:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlock(), 128);
  EXPECT_EQ(attr.getGridGroupSize(), 1);
  EXPECT_EQ(attr.getUseAsyncCopy(), kKnobDefault);
  EXPECT_EQ(attr.getUseBlockPingpong(), kKnobDefault);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferAtomics(), kKnobDefault);
  // v3 predates the v4 `useReductionLayout` knob; it defaults to the knob
  // default (-1 = heuristic / currently off).
  EXPECT_EQ(attr.getUseReductionLayout(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmParamsV3ReserializesAsV5) {
  PerfConfigTestEnv e;
  // A v3 string is accepted on input; serialization always emits the canonical
  // v5 form with the newer knobs defaulted to -1.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v3:128,128,16,1,1,4,32,1,2,0,1,1,0,-1,1,0"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            gemmNamed({128, 128, 16, 1, 1, 4, 32, 1, 2, 0, 1, 1, 0, -1, 1, 0,
                       -1, -1}));
}

TEST(PerfConfigParsingTest, GemmParamsV3RejectsBoolKnobAboveOne) {
  PerfConfigTestEnv e;
  // useBlockPingpong = 2 -- not in {-1, 0, 1}.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v3:128,128,16,1,1,4,32,1,2,0,1,-1,2,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV3TooFewParams) {
  PerfConfigTestEnv e;
  // v3 expects 16 fields (11 tunables + 5 knobs).
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v3:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV3TooManyParams) {
  PerfConfigTestEnv e;
  // A stray trailing field (e.g. an old v2 scheduleHint) is rejected for v3.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v3:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

// --- GemmParamsAttr: v4 (adds the trailing `useReductionLayout` knob) ---

TEST(PerfConfigParsingTest, GemmParamsV4ReadsReductionLayout) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v4:128,128,16,1,1,4,32,1,2,0,1,1,0,-1,1,0,1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseAsyncCopy(), 1);
  EXPECT_EQ(attr.getUseBufferAtomics(), 0);
  EXPECT_EQ(attr.getUseReductionLayout(), 1);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmParamsV4ReserializesAsV5) {
  PerfConfigTestEnv e;
  StringRef original = "gemm:v4:128,128,16,1,1,4,32,1,2,0,1,1,0,-1,1,0,1";
  auto attr = GemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            gemmNamed({128, 128, 16, 1, 1, 4, 32, 1, 2, 0, 1, 1, 0, -1, 1, 0, 1,
                       -1}));
}

TEST(PerfConfigParsingTest,
     GemmParamsV4ExplicitOffReductionLayoutReserializesAsV5) {
  PerfConfigTestEnv e;
  StringRef original = "gemm:v4:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,0";
  auto attr = GemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseReductionLayout(), 0);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            gemmNamed({128, 128, 16, 1, 1, 4, 32, 1, 2, 0, 1, -1, -1, -1, -1,
                       -1, 0, -1}));
}

TEST(PerfConfigParsingTest, GemmParamsV4RejectsBadReductionLayout) {
  PerfConfigTestEnv e;
  // useReductionLayout = 2 -- not in {-1, 0, 1}.
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v4:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,2"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV4ReadsReductionLayoutSentinel) {
  PerfConfigTestEnv e;
  // useReductionLayout is a tri-state gate like the other knobs: the
  // `kKnobDefault` (-1 = heuristic / currently off) sentinel is accepted and
  // defaults the v5 OptimizeEpilogue knob.
  StringRef original = "gemm:v4:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1";
  auto attr = GemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseReductionLayout(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            gemmNamed({128, 128, 16, 1, 1, 4, 32, 1, 2, 0, 1, -1, -1, -1, -1,
                       -1, -1, -1}));
}

TEST(PerfConfigParsingTest, GemmParamsV4TooFewParams) {
  PerfConfigTestEnv e;
  // v4 expects 17 fields (11 tunables + 5 knobs + useReductionLayout).
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v4:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV4TooManyParams) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v4:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,0,99"));
  EXPECT_FALSE(attr);
}

// --- GemmParamsAttr: v5 (adds the trailing `useOptimizeEpilogue` knob) ---

TEST(PerfConfigParsingTest, GemmParamsV5ReadsOptimizeEpilogue) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v5:128,128,16,1,1,4,32,1,2,0,1,1,0,-1,1,0,1,0"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseReductionLayout(), 1);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), 0);
}

TEST(PerfConfigParsingTest, GemmParamsV5ReserializesToNamedForm) {
  PerfConfigTestEnv e;
  StringRef original =
      "gemm:v5:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,1";
  auto attr = GemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), 1);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            gemmNamed({128, 128, 16, 1, 1, 4, 32, 1, 2, 0, 1, -1, -1, -1, -1,
                       -1, -1, 1}));
}

TEST(PerfConfigParsingTest, GemmParamsV5RejectsBadOptimizeEpilogue) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v5:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,2"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV5TooFewParams) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v5:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsV5TooManyParams) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(
      e.str("gemm:v5:128,128,16,1,1,4,32,1,2,0,1,-1,-1,-1,-1,-1,-1,1,99"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsUnknownVersionV6) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:v6:128,128,16,1,1,4,32,1,2,0,1"));
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
  // Pre-v6 configs have no nPerBlockG1 field; it decodes to 0 ("untiled").
  EXPECT_EQ(attr.getNPerBlockG1(), 0);
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
  // v1 predates the v4 `useReductionLayout` knob; it defaults to the knob
  // default (-1 = heuristic / currently off).
  EXPECT_EQ(attr.getUseReductionLayout(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
}

// --- GemmGemmParamsAttr: v2 read-only back-compat ---

TEST(PerfConfigParsingTest, GemmGemmParamsV2BackCompatDiscardsScheduleHint) {
  PerfConfigTestEnv e;
  WarningCapture capture(e.ctx);
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlockG0(), 64);
  EXPECT_EQ(attr.getUseBufferAtomics(), kKnobDefault);
  // v2 predates the v4 `useReductionLayout` knob; it defaults to the knob
  // default (-1 = heuristic / currently off).
  EXPECT_EQ(attr.getUseReductionLayout(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
  EXPECT_TRUE(capture.warnings.empty());
}

TEST(PerfConfigParsingTest, GemmGemmParamsV2MixedKnobsReserializeAsV6) {
  PerfConfigTestEnv e;
  WarningCapture capture(e.ctx);
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,1,0,-1,1,0,1"));
  ASSERT_TRUE(attr);
  expectScheduleHintWarning(capture, 1);
  EXPECT_EQ(attr.getUseAsyncCopy(), 1);
  EXPECT_EQ(attr.getUseBlockPingpong(), 0);
  EXPECT_EQ(attr.getUseInThreadTranspose(), kKnobDefault);
  EXPECT_EQ(attr.getUseBufferOps(), 1);
  EXPECT_EQ(attr.getUseBufferAtomics(), 0);
  // Reserializes in the canonical named form with nPerBlockG1=0 inserted (v2
  // has no such field) and the trailing scheduleHint dropped.
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 0, 32, 2, 1, 2, 16, 1, 1, 0, 1, 1, 0, -1, 1, 0,
                       -1, -1}));
}

TEST(PerfConfigParsingTest, GemmGemmParamsV2IgnoresArbitraryScheduleHintValue) {
  PerfConfigTestEnv e;
  WarningCapture capture(e.ctx);
  // The trailing scheduleHint token is accepted and discarded (here 4).
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,4"));
  ASSERT_TRUE(attr);
  expectScheduleHintWarning(capture, 4);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 0, 32, 2, 1, 2, 16, 1, 1, 0, 1, -1, -1, -1, -1,
                       -1, -1, -1}));
}

TEST(PerfConfigParsingTest, GemmGemmParamsV2RejectsBoolKnobAboveOne) {
  PerfConfigTestEnv e;
  // useBlockPingpong = 2 -- not in {-1, 0, 1}.
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,-1,2,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV2TooFewParams) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

// --- GemmGemmParamsAttr: v3 (back-compat) ---

TEST(PerfConfigParsingTest, GemmGemmParamsValidV3AllDefaults) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v3:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlockG0(), 64);
  EXPECT_EQ(attr.getUseBufferAtomics(), kKnobDefault);
  // v3 predates the v4 `useReductionLayout` knob; it defaults to the knob
  // default (-1 = heuristic / currently off).
  EXPECT_EQ(attr.getUseReductionLayout(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV3ReserializesAsV6) {
  PerfConfigTestEnv e;
  // A v3 string is accepted on input; serialization always emits the canonical
  // named form with nPerBlockG1=0 inserted and the newer knobs defaulted to -1.
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v3:64,64,32,2,1,2,16,1,1,0,1,1,0,-1,1,0"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 0, 32, 2, 1, 2, 16, 1, 1, 0, 1, 1, 0, -1, 1, 0,
                       -1, -1}));
}

TEST(PerfConfigParsingTest, GemmGemmParamsV3TooManyParams) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v3:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

// --- GemmGemmParamsAttr: v4 (adds the trailing `useReductionLayout` knob) ---

TEST(PerfConfigParsingTest, GemmGemmParamsV4ReadsReductionLayout) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v4:64,64,32,2,1,2,16,1,1,0,1,1,0,-1,1,0,1"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseAsyncCopy(), 1);
  EXPECT_EQ(attr.getUseReductionLayout(), 1);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV4ReserializesAsV6) {
  PerfConfigTestEnv e;
  StringRef original = "attn:v4:64,64,32,2,1,2,16,1,1,0,1,1,0,-1,1,0,1";
  auto attr = GemmGemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 0, 32, 2, 1, 2, 16, 1, 1, 0, 1, 1, 0, -1, 1, 0,
                       1, -1}));
}

TEST(PerfConfigParsingTest,
     GemmGemmParamsV4ExplicitOffReductionLayoutReserializesAsV6) {
  PerfConfigTestEnv e;
  StringRef original = "attn:v4:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,0";
  auto attr = GemmGemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseReductionLayout(), 0);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 0, 32, 2, 1, 2, 16, 1, 1, 0, 1, -1, -1, -1, -1,
                       -1, 0, -1}));
}

TEST(PerfConfigParsingTest, GemmGemmParamsV4RejectsBadReductionLayout) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v4:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,2"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV4ReadsReductionLayoutSentinel) {
  PerfConfigTestEnv e;
  // useReductionLayout is a tri-state gate like the other knobs: the
  // `kKnobDefault` (-1 = heuristic / currently off) sentinel is accepted and
  // defaults the v5 OptimizeEpilogue knob.
  StringRef original = "attn:v4:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1";
  auto attr = GemmGemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseReductionLayout(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 0, 32, 2, 1, 2, 16, 1, 1, 0, 1, -1, -1, -1, -1,
                       -1, -1, -1}));
}

TEST(PerfConfigParsingTest, GemmGemmParamsV4TooFewParams) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v4:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

// --- GemmGemmParamsAttr: v5 (adds `useOptimizeEpilogue`) ---

TEST(PerfConfigParsingTest, GemmGemmParamsV5ReadsOptimizeEpilogue) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v5:64,64,32,2,1,2,16,1,1,0,1,1,0,-1,1,0,1,0"));
  ASSERT_TRUE(attr);
  // v5 predates the v6 `nPerBlockG1` tunable field; it decodes to 0
  // ("untiled").
  EXPECT_EQ(attr.getNPerBlockG1(), 0);
  EXPECT_EQ(attr.getUseReductionLayout(), 1);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), 0);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV5ReserializesToNamedForm) {
  PerfConfigTestEnv e;
  // A v5 string is accepted on input; serialization always emits the canonical
  // named form with an explicit nPerBlockG1=0.
  StringRef original = "attn:v5:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1,1";
  auto attr = GemmGemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), 1);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 0, 32, 2, 1, 2, 16, 1, 1, 0, 1, -1, -1, -1, -1,
                       -1, -1, 1}));
}

TEST(PerfConfigParsingTest, GemmGemmParamsV5RejectsBadOptimizeEpilogue) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v5:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1,2"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV5TooFewParams) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v5:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV5TooManyParams) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v5:64,64,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1,1,99"));
  EXPECT_FALSE(attr);
}

// --- GemmGemmParamsAttr: v6 (adds the nPerBlockG1 second-GEMM head-dim
//     tunable field as the 3rd field) ---

TEST(PerfConfigParsingTest, GemmGemmParamsV6ReadsNPerBlockG1) {
  PerfConfigTestEnv e;
  // v6 carries the second-GEMM head-dim tile nPerBlockG1 as the 3rd field
  // (here 16, a real tile). The v5 knob block trails unchanged.
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v6:64,64,16,32,2,1,2,16,1,1,0,1,1,0,-1,1,0,1,0"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getNPerBlockG1(), 16);
  EXPECT_EQ(attr.getUseAsyncCopy(), 1);
  EXPECT_EQ(attr.getUseReductionLayout(), 1);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), 0);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV6ReserializesToNamedForm) {
  PerfConfigTestEnv e;
  // A non-zero nPerBlockG1 (16) must survive re-serialization unchanged.
  StringRef original = "attn:v6:64,64,16,32,2,1,2,16,1,1,0,1,1,0,-1,1,0,1,-1";
  auto attr = GemmGemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getNPerBlockG1(), 16);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 16, 32, 2, 1, 2, 16, 1, 1, 0, 1, 1, 0, -1, 1, 0,
                       1, -1}));
}

TEST(PerfConfigParsingTest, GemmGemmParamsV6UntiledReserializesToNamedForm) {
  PerfConfigTestEnv e;
  // nPerBlockG1=0 is the "untiled" case (what pre-v6 configs decode to). It is
  // still emitted explicitly.
  StringRef original =
      "attn:v6:64,64,0,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1,-1";
  auto attr = GemmGemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getNPerBlockG1(), 0);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(),
            attnNamed({64, 64, 0, 32, 2, 1, 2, 16, 1, 1, 0, 1, -1, -1, -1, -1,
                       -1, -1, -1}));
}

TEST(PerfConfigParsingTest, GemmGemmParamsV6RejectsBadOptimizeEpilogue) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v6:64,64,16,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1,2"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV6TooFewParams) {
  PerfConfigTestEnv e;
  // v6 expects 19 fields (12 tunables incl. nPerBlockG1 + 7 knobs); one short
  // (18 fields, i.e. the v5 nPerBlockG1-less layout) must be rejected.
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v6:64,64,16,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsV6TooManyParams) {
  PerfConfigTestEnv e;
  // A stray trailing field beyond the 19 v6 fields must be rejected.
  auto attr = GemmGemmParamsAttr::get(
      e.str("attn:v6:64,64,16,32,2,1,2,16,1,1,0,1,-1,-1,-1,-1,-1,-1,1,99"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsUnknownVersionV7) {
  PerfConfigTestEnv e;
  auto attr =
      GemmGemmParamsAttr::get(e.str("attn:v7:64,64,32,2,1,2,16,1,1,0,1"));
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

// --- GemmParamsAttr: named schema (canonical `gemm:key=value,...`) ---

// RockAttrDefs.td generates the emitter, this schema and the parser from one
// field list, so they cannot disagree with each other -- but they can all move
// together, and every tuning DB entry ever written is a string built from these
// exact keys. Pin the wire format so that renaming, reordering or redefaulting
// a field has to be a deliberate edit here, with the compatibility break
// visible in the diff.
TEST(PerfConfigParsingTest, GemmParamsNamedSchemaIsPinned) {
  EXPECT_EQ(GemmParamsAttr::getPerfConfigPrefix(), "gemm");
  EXPECT_EQ(llvm::join(GemmParamsAttr::getPerfConfigKeys(), ","),
            "mPerBlock,nPerBlock,kPerBlock,kpack,numCTAs,numWaves,"
            "matrixInstrNonkdim,splitKFactor,numStages,wavesPerEU,"
            "gridGroupSize,useAsyncCopy,useBlockPingpong,useInThreadTranspose,"
            "useBufferOps,useBufferAtomics,useReductionLayout,"
            "useOptimizeEpilogue,useFastAtomics");
  // The value each field takes when a config string omits it.
  EXPECT_EQ(joinInts(GemmParamsAttr::getPerfConfigDefaults()),
            "32,32,16,1,1,4,0,1,1,0,0,-1,-1,-1,-1,-1,-1,-1,-1");
}

TEST(PerfConfigParsingTest, GemmParamsNamedRoundTrip) {
  PerfConfigTestEnv e;
  std::string original =
      gemmNamed({128, 128, 16, 1, 1, 4, 32, 1, 2, 0, 1, 1, 0, -1, 1, 0, 1, 0});
  auto attr = GemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlock(), 128);
  EXPECT_EQ(attr.getUseReductionLayout(), 1);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), 0);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(), original);
}

TEST(PerfConfigParsingTest, GemmParamsNamedMissingKeysUseDefaults) {
  PerfConfigTestEnv e;
  // A bare `gemm:` (empty body): every field falls back to its default.
  auto attr = GemmParamsAttr::get(e.str("gemm:"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlock(), 32);
  EXPECT_EQ(attr.getNPerBlock(), 32);
  EXPECT_EQ(attr.getKPerBlock(), 16);
  EXPECT_EQ(attr.getKpack(), 1);
  EXPECT_EQ(attr.getNumCTAs(), 1);
  EXPECT_EQ(attr.getNumWaves(), 4);
  EXPECT_EQ(attr.getMatrixInstrNonkdim(), 0);
  EXPECT_EQ(attr.getSplitKFactor(), 1);
  EXPECT_EQ(attr.getNumStages(), 1);
  EXPECT_EQ(attr.getWavesPerEU(), 0);
  EXPECT_EQ(attr.getGridGroupSize(), 0);
  EXPECT_EQ(attr.getUseAsyncCopy(), kKnobDefault);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmParamsNamedPartialOverridesDefaults) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:mPerBlock=256,useBufferOps=0"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlock(), 256);
  EXPECT_EQ(attr.getNPerBlock(), 32);
  EXPECT_EQ(attr.getUseBufferOps(), 0);
  EXPECT_EQ(attr.getUseAsyncCopy(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmParamsNamedIgnoresWhitespace) {
  PerfConfigTestEnv e;
  auto attr =
      GemmParamsAttr::get(e.str("gemm: mPerBlock = 256 , nPerBlock=128 "));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlock(), 256);
  EXPECT_EQ(attr.getNPerBlock(), 128);
}

TEST(PerfConfigParsingTest, GemmParamsNamedRejectsUnknownKey) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:mPerBlock=128,bogus=1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsNamedRejectsGemmGemmKey) {
  PerfConfigTestEnv e;
  // `mPerBlockG0` and `nPerBlockG1` belong to the attn schema; both are unknown
  // to gemm.
  EXPECT_FALSE(GemmParamsAttr::get(e.str("gemm:mPerBlockG0=128")));
  EXPECT_FALSE(GemmParamsAttr::get(e.str("gemm:nPerBlockG1=64")));
}

TEST(PerfConfigParsingTest, GemmParamsNamedAcceptsAndDiscardsScheduleHint) {
  PerfConfigTestEnv e;
  WarningCapture capture(e.ctx);
  auto attr = GemmParamsAttr::get(e.str("gemm:mPerBlock=128,scheduleHint=2"));
  ASSERT_TRUE(attr);
  expectScheduleHintWarning(capture, 2);
  EXPECT_EQ(attr.getMPerBlock(), 128);
}

TEST(PerfConfigParsingTest, GemmParamsNamedRejectsDuplicateKey) {
  PerfConfigTestEnv e;
  EXPECT_FALSE(GemmParamsAttr::get(e.str("gemm:mPerBlock=128,mPerBlock=256")));
  // Rejected even when the two spellings agree, since a config with a repeated
  // key was not produced by the serializer.
  EXPECT_FALSE(GemmParamsAttr::get(e.str("gemm:mPerBlock=128,mPerBlock=128")));
}

TEST(PerfConfigParsingTest, GemmParamsNamedRejectsNonIntegerValue) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:mPerBlock=abc"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsNamedRejectsBadKnob) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:useBufferOps=2"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsNamedRejectsMissingValue) {
  PerfConfigTestEnv e;
  // A `key=value` entry with an empty value is malformed and rejected.
  auto attr = GemmParamsAttr::get(e.str("gemm:mPerBlock="));
  EXPECT_FALSE(attr);
}

// --- GemmGemmParamsAttr: named schema (canonical `attn:key=value,...`) ---

// See `GemmParamsNamedSchemaIsPinned`. The attention schema additionally
// carries `nPerBlockG1` as its third field, whose 0 default means "untiled" and
// is what pre-v6 positional configs decode to.
TEST(PerfConfigParsingTest, GemmGemmParamsNamedSchemaIsPinned) {
  EXPECT_EQ(GemmGemmParamsAttr::getPerfConfigPrefix(), "attn");
  EXPECT_EQ(llvm::join(GemmGemmParamsAttr::getPerfConfigKeys(), ","),
            "mPerBlockG0,nPerBlockG0,nPerBlockG1,kPerBlock,kpack,numCTAs,"
            "numWaves,matrixInstrNonkdim,splitKFactor,numStages,wavesPerEU,"
            "gridGroupSize,useAsyncCopy,useBlockPingpong,useInThreadTranspose,"
            "useBufferOps,useBufferAtomics,useReductionLayout,"
            "useOptimizeEpilogue,useFastAtomics");
  EXPECT_EQ(joinInts(GemmGemmParamsAttr::getPerfConfigDefaults()),
            "32,32,0,16,1,1,4,0,1,1,0,0,-1,-1,-1,-1,-1,-1,-1,-1");
}

TEST(PerfConfigParsingTest, GemmGemmParamsNamedRoundTrip) {
  PerfConfigTestEnv e;
  // A non-zero nPerBlockG1 (16) round-trips like any other tunable field.
  std::string original = attnNamed(
      {64, 64, 16, 32, 2, 1, 2, 16, 1, 1, 0, 1, 1, 0, -1, 1, 0, 1, 0});
  auto attr = GemmGemmParamsAttr::get(e.str(original));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlockG0(), 64);
  EXPECT_EQ(attr.getNPerBlockG1(), 16);
  EXPECT_EQ(attr.getUseReductionLayout(), 1);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), 0);
  EXPECT_EQ(attr.getPerfConfigAttr().strref(), original);
}

TEST(PerfConfigParsingTest, GemmGemmParamsNamedMissingKeysUseDefaults) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(e.str("attn:"));
  ASSERT_TRUE(attr);
  EXPECT_EQ(attr.getMPerBlockG0(), 32);
  EXPECT_EQ(attr.getNPerBlockG0(), 32);
  // An omitted nPerBlockG1 defaults to 0 ("untiled"), matching how pre-v6
  // positional configs decode.
  EXPECT_EQ(attr.getNPerBlockG1(), 0);
  EXPECT_EQ(attr.getKPerBlock(), 16);
  EXPECT_EQ(attr.getUseOptimizeEpilogue(), kKnobDefault);
}

TEST(PerfConfigParsingTest, GemmGemmParamsNamedRejectsGemmKey) {
  PerfConfigTestEnv e;
  // `mPerBlock` belongs to the gemm schema; it is unknown to attn.
  auto attr = GemmGemmParamsAttr::get(e.str("attn:mPerBlock=64"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsNamedRejectsUnknownKey) {
  PerfConfigTestEnv e;
  auto attr = GemmGemmParamsAttr::get(e.str("attn:bogus=1"));
  EXPECT_FALSE(attr);
}
