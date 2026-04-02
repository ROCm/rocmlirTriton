//===- PerfConfigParsingTests.cpp - Tests for GemmParams/GemmGemmParams ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
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

// --- GemmParamsAttr ---

TEST(PerfConfigParsingTest, GemmParamsValidV1) {
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
}

TEST(PerfConfigParsingTest, GemmParamsWrongPrefix) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("attn:v1:128,128,16,1,1,4,32,1,2,0,1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsWrongVersion) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:v2:128,128,16,1,1,4,32,1,2,0,1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsTooFewParams) {
  PerfConfigTestEnv e;
  auto attr = GemmParamsAttr::get(e.str("gemm:v1:128,128,16"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmParamsTooManyParams) {
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

// --- GemmGemmParamsAttr ---

TEST(PerfConfigParsingTest, GemmGemmParamsValidV1) {
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
}

TEST(PerfConfigParsingTest, GemmGemmParamsWrongPrefix) {
  PerfConfigTestEnv e;
  auto attr =
      GemmGemmParamsAttr::get(e.str("gemm:v1:64,64,32,2,1,2,16,1,1,0,1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsWrongVersion) {
  PerfConfigTestEnv e;
  auto attr =
      GemmGemmParamsAttr::get(e.str("attn:v2:64,64,32,2,1,2,16,1,1,0,1"));
  EXPECT_FALSE(attr);
}

TEST(PerfConfigParsingTest, GemmGemmParamsTooFewParams) {
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
