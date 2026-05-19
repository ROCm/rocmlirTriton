//===- PerfConfigOrderingGemmTests.cpp - Tests for single-GEMM ordering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct GemmOrderingTestEnv {
  MLIRContext ctx;
  OpBuilder b;
  Type f16, f32;
  // Block-scaling element types: f8E4M3FN is a typical MXFP A/B element type
  // and f8E8M0FNU is the MXFP scale type used on AMD gfx950 scaled MFMA.
  Type f8E4M3, f8E8M0;

  GemmOrderingTestEnv() : b(&ctx) {
    ctx.getOrLoadDialect<RockDialect>();
    f16 = b.getF16Type();
    f32 = b.getF32Type();
    f8E4M3 = Float8E4M3FNType::get(&ctx);
    f8E8M0 = Float8E8M0FNUType::get(&ctx);
  }

  // Build a GemmParamsAttr from positional fields. Mirrors the perfconfig
  // string format documented on Rock_GemmParamsAttr.
  GemmParamsAttr gemm(int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock,
                      int64_t kpack, int64_t numCTAs, int64_t numWaves,
                      int64_t matrixInstrNonkdim, int64_t splitKFactor,
                      int64_t numStages, int64_t wavesPerEU,
                      int64_t gridGroupSize) {
    return GemmParamsAttr::get(
        &ctx, mPerBlock, nPerBlock, kPerBlock, kpack, numCTAs, numWaves,
        matrixInstrNonkdim, splitKFactor, numStages, wavesPerEU, gridGroupSize);
  }
};
} // namespace

// --- isGemmParamsConservativelyApplicable ---

TEST(PerfConfigOrderingGemmTest, IsApplicableRejectsKpackNot1) {
  GemmOrderingTestEnv e;
  auto p = e.gemm(32, 32, 32, /*kpack=*/2, 1, 4, 0, 1, 1, 0, 0);
  EXPECT_FALSE(isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx942"));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableRejectsSplitKNot1) {
  GemmOrderingTestEnv e;
  auto p = e.gemm(32, 32, 32, 1, 1, 4, 0, /*splitKFactor=*/2, 1, 0, 0);
  EXPECT_FALSE(isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx942"));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableRejectsNumCTAsNot1) {
  GemmOrderingTestEnv e;
  auto p = e.gemm(32, 32, 32, 1, /*numCTAs=*/2, 4, 0, 1, 1, 0, 0);
  EXPECT_FALSE(isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx942"));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableRejectsLDSOverflow) {
  GemmOrderingTestEnv e;
  // 256x256x256 fp32 numStages=2 needs ~1 MiB; gfx942 has 64 KiB LDS. Mirrors
  // the config exercised by lds-overflow-not-applicable.mlir.
  auto p = e.gemm(256, 256, 256, 1, 1, 4, 16, 1, 2, 0, 0);
  EXPECT_FALSE(isGemmParamsConservativelyApplicable(p, e.f32, e.f32, "gfx942"));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableAcceptsConservativeDefault) {
  GemmOrderingTestEnv e;
  auto p = e.gemm(32, 32, 32, 1, 1, 4, 0, 1, 1, 0, 0);
  EXPECT_TRUE(isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx942"));
  EXPECT_TRUE(isGemmParamsConservativelyApplicable(p, e.f32, e.f32, "gfx942"));
  EXPECT_TRUE(isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx1100"));
  EXPECT_TRUE(isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx950"));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableLDSAtBoundary) {
  GemmOrderingTestEnv e;
  // 64x64x64 fp32 numStages=2 = (64*64*32*2)/8*2 = 65536 bytes == gfx942 LDS.
  // `<=` so this just fits.
  auto p = e.gemm(64, 64, 64, 1, 1, 4, 0, 1, 2, 0, 0);
  EXPECT_TRUE(isGemmParamsConservativelyApplicable(p, e.f32, e.f32, "gfx942"));
}

// --- orderParams ---

TEST(PerfConfigOrderingGemmTest, OrderParamsEmpty) {
  GemmOrderingTestEnv e;
  std::vector<GemmParamsAttr> in;
  auto out =
      orderParams<GemmParamsAttr>(in, [](GemmParamsAttr) { return true; });
  EXPECT_TRUE(out.empty());
}

TEST(PerfConfigOrderingGemmTest, OrderParamsFirstAlreadyApplicable) {
  GemmOrderingTestEnv e;
  auto a = e.gemm(32, 32, 32, 1, 1, 4, 0, 1, 1, 0, 0);
  auto b = e.gemm(64, 64, 64, 1, 1, 4, 0, 1, 1, 0, 0);
  std::vector<GemmParamsAttr> in{a, b};
  auto out =
      orderParams<GemmParamsAttr>(in, [](GemmParamsAttr) { return true; });
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], a);
  EXPECT_EQ(out[1], b);
}

TEST(PerfConfigOrderingGemmTest, OrderParamsBumpsSecondToFront) {
  GemmOrderingTestEnv e;
  auto bad = e.gemm(32, 32, 32, /*kpack=*/2, 1, 4, 0, 1, 1, 0, 0);
  auto good = e.gemm(32, 32, 32, 1, 1, 4, 0, 1, 1, 0, 0);
  auto other = e.gemm(64, 64, 64, 1, 1, 4, 0, 1, 1, 0, 0);
  std::vector<GemmParamsAttr> in{bad, good, other};
  auto out = orderParams<GemmParamsAttr>(in, [&](GemmParamsAttr p) {
    return isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx942");
  });
  ASSERT_EQ(out.size(), 3u);
  // `good` is the first applicable; it gets rotated to the front while the
  // relative order of the remaining entries (bad, other) is preserved.
  EXPECT_EQ(out[0], good);
  EXPECT_EQ(out[1], bad);
  EXPECT_EQ(out[2], other);
}

TEST(PerfConfigOrderingGemmTest, OrderParamsNoneApplicableKeepsOrder) {
  GemmOrderingTestEnv e;
  auto a = e.gemm(32, 32, 32, /*kpack=*/2, 1, 4, 0, 1, 1, 0, 0);
  auto b = e.gemm(32, 32, 32, 1, 1, 4, 0, /*splitKFactor=*/2, 1, 0, 0);
  std::vector<GemmParamsAttr> in{a, b};
  auto out = orderParams<GemmParamsAttr>(in, [&](GemmParamsAttr p) {
    return isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx942");
  });
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], a);
  EXPECT_EQ(out[1], b);
}

// --- getConservativeDefaultGemmParams ---

TEST(PerfConfigOrderingGemmTest, ConservativeDefaultGemmParamsFields) {
  GemmOrderingTestEnv e;
  auto p = getConservativeDefaultGemmParams(&e.ctx);
  EXPECT_EQ(p.getMPerBlock(), 32);
  EXPECT_EQ(p.getNPerBlock(), 32);
  EXPECT_EQ(p.getKPerBlock(), 32);
  EXPECT_EQ(p.getKpack(), 1);
  EXPECT_EQ(p.getNumCTAs(), 1);
  EXPECT_EQ(p.getNumWaves(), 4);
  EXPECT_EQ(p.getMatrixInstrNonkdim(), 0);
  EXPECT_EQ(p.getSplitKFactor(), 1);
  EXPECT_EQ(p.getNumStages(), 1);
  EXPECT_EQ(p.getWavesPerEU(), 0);
  EXPECT_EQ(p.getGridGroupSize(), 0);
}

TEST(PerfConfigOrderingGemmTest, ConservativeDefaultGemmParamsPassesPredicate) {
  GemmOrderingTestEnv e;
  auto p = getConservativeDefaultGemmParams(&e.ctx);
  // Same arch matrix as AmdArchDbTests::LDSSize: cover both 64 KiB LDS archs
  // and the larger gfx950 / gfx1250.
  for (StringRef arch : {"gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100",
                         "gfx1200", "gfx1250"}) {
    EXPECT_TRUE(isGemmParamsConservativelyApplicable(p, e.f16, e.f16, arch))
        << "default not applicable on " << arch << " for f16";
    EXPECT_TRUE(isGemmParamsConservativelyApplicable(p, e.f32, e.f32, arch))
        << "default not applicable on " << arch << " for f32";
  }
}

// --- block-scaled (MXFP) GEMM ---

TEST(PerfConfigOrderingGemmTest,
     IsApplicableRejectsKPerBlockNotDivisibleByQuant) {
  GemmOrderingTestEnv e;
  // kPerBlock = 32, quantBlockSize = 64 → 32 % 64 != 0, rejected.
  // This mirrors `GridwiseGemmToBlockwise`'s `markAsNotApplicable` site
  // (kPerBlock is not a multiple of quantBlockSize).
  auto p = e.gemm(32, 32, 32, 1, 1, 4, 0, 1, 1, 0, 0);
  EXPECT_FALSE(isGemmParamsConservativelyApplicable(
      p, e.f8E4M3, e.f8E4M3, "gfx950", /*quantBlockSize=*/64, e.f8E8M0,
      e.f8E8M0));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableAcceptsKPerBlockDivisibleByQuant) {
  GemmOrderingTestEnv e;
  // kPerBlock = 64, quantBlockSize = 32 → 64 % 32 == 0, divisibility OK.
  auto p = e.gemm(64, 64, 64, 1, 1, 4, 0, 1, 2, 0, 0);
  EXPECT_TRUE(isGemmParamsConservativelyApplicable(
      p, e.f8E4M3, e.f8E4M3, "gfx950", /*quantBlockSize=*/32, e.f8E8M0,
      e.f8E8M0));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableChargesScaleTilesToLDS) {
  GemmOrderingTestEnv e;
  // Build a config that just fits A+B tiles but blows the budget once the
  // E8M0 scale tiles are added. Pick gfx942 (64 KiB LDS), fp8 A/B, kPerBlock
  // = 256, quantBlockSize = 1 so the scale tile equals the element tile.
  // A+B bytes = (256·256 + 256·256)·8 / 8 = 128 KiB already over budget; use a
  // smaller tile so the *no-scale* case fits and the *scaled* one doesn't.
  //   mPerBlock=nPerBlock=128, kPerBlock=128, numStages=1, fp8 → 32 KiB; fits.
  //   Add scales at quantBlockSize=1 (1 scale per element) → another 32 KiB,
  //   total 64 KiB; still within 64 KiB. Push numStages=2 → 128 KiB; over.
  auto p = e.gemm(128, 128, 128, 1, 1, 4, 0, 1, /*numStages=*/2, 0, 0);
  // Without scales: 64 KiB; fits exactly on gfx942.
  EXPECT_TRUE(
      isGemmParamsConservativelyApplicable(p, e.f8E4M3, e.f8E4M3, "gfx942"));
  // With scales (quantBlockSize=1): doubles bytes → 128 KiB; over budget.
  EXPECT_FALSE(isGemmParamsConservativelyApplicable(
      p, e.f8E4M3, e.f8E4M3, "gfx942", /*quantBlockSize=*/1, e.f8E8M0,
      e.f8E8M0));
}

TEST(PerfConfigOrderingGemmTest, ConservativeDefaultScaledKPerBlockBumpedUp) {
  GemmOrderingTestEnv e;
  // quantBlockSize <= 32 leaves kPerBlock at the baseline 32.
  EXPECT_EQ(getConservativeDefaultGemmParams(&e.ctx, /*qBS=*/16).getKPerBlock(),
            32);
  EXPECT_EQ(getConservativeDefaultGemmParams(&e.ctx, /*qBS=*/32).getKPerBlock(),
            32);
  // Larger quantBlockSize forces kPerBlock up to the next multiple.
  EXPECT_EQ(getConservativeDefaultGemmParams(&e.ctx, /*qBS=*/64).getKPerBlock(),
            64);
  EXPECT_EQ(
      getConservativeDefaultGemmParams(&e.ctx, /*qBS=*/128).getKPerBlock(),
      128);
}

TEST(PerfConfigOrderingGemmTest,
     ConservativeDefaultScaledPassesPredicateOnGfx950) {
  GemmOrderingTestEnv e;
  // The conservative default with the same quantBlockSize plugged into the
  // predicate must always be applicable — that's the contract that keeps the
  // fallback safe for MIGRAPHX_SKIP_BENCHMARKING on scaled GEMMs.
  for (int64_t qBS : {16, 32, 64, 128}) {
    auto p = getConservativeDefaultGemmParams(&e.ctx, qBS);
    EXPECT_TRUE(isGemmParamsConservativelyApplicable(
        p, e.f8E4M3, e.f8E4M3, "gfx950", qBS, e.f8E8M0, e.f8E8M0))
        << "scaled default not applicable for quantBlockSize=" << qBS;
  }
}
