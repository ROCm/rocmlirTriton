//===- PerfConfigOrderingGemmTests.cpp - Tests for single-GEMM ordering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/MathExtras.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <memory>
#include <set>

using namespace mlir;
using namespace mlir::rock;

namespace {
struct GemmOrderingTestEnv {
  MLIRContext ctx;
  OpBuilder b;
  Type f16, f32, f4E2M1;
  // Block-scaling element types: f8E4M3FN is a typical MXFP A/B element type
  // and f8E8M0FNU is the MXFP scale type used on AMD gfx950 scaled MFMA.
  Type f8E4M3, f8E8M0;

  GemmOrderingTestEnv() : b(&ctx) {
    ctx.getOrLoadDialect<RockDialect>();
    f16 = b.getF16Type();
    f32 = b.getF32Type();
    f4E2M1 = Float4E2M1FNType::get(&ctx);
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
        matrixInstrNonkdim, splitKFactor, numStages, wavesPerEU, gridGroupSize,
        /*useAsyncCopy=*/kKnobDefault,
        /*useBlockPingpong=*/kKnobDefault,
        /*useInThreadTranspose=*/kKnobDefault,
        /*useBufferOps=*/kKnobDefault,
        /*useBufferAtomics=*/kKnobDefault,
        /*useReductionLayout=*/kKnobDefault,
        /*useOptimizeEpilogue=*/kKnobDefault,
        /*useBf16x3ForF32=*/kKnobDefault);
  }
};

// A module holding a single non-scaled `rock.gemm` of shape
// [1,M,K]x[1,K,N] -> [1,M,N], ready to be fed to `createTunableParamSpace`.
// `elemType` picks the A/B/C element type from the freshly created context.
struct TuningSpaceGemmEnv {
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;

  TuningSpaceGemmEnv(llvm::function_ref<Type(OpBuilder &)> elemType, int64_t m,
                     int64_t n, int64_t k, StringRef arch) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    OpBuilder b(&ctx);
    Location loc = b.getUnknownLoc();
    Type elem = elemType(b);
    auto aType = RankedTensorType::get({1, m, k}, elem);
    auto bType = RankedTensorType::get({1, k, n}, elem);
    auto cType = RankedTensorType::get({1, m, n}, elem);

    module = ModuleOp::create(loc);
    b.setInsertionPointToEnd(module->getBody());
    auto func = func::FuncOp::create(
        b, loc, "test", b.getFunctionType({aType, bType}, {cType}));
    Block *body = func.addEntryBlock();
    b.setInsertionPointToStart(body);
    auto gemmOp = GemmOp::create(
        b, loc, /*c=*/cType, /*a=*/body->getArgument(0),
        /*b=*/body->getArgument(1), /*scaleA=*/Value(), /*scaleB=*/Value(),
        /*aTransposed=*/UnitAttr{}, /*bTransposed=*/UnitAttr{},
        /*oTransposed=*/UnitAttr{}, /*aScaleTransposed=*/UnitAttr{},
        /*bScaleTransposed=*/UnitAttr{}, /*quantBlockSize=*/IntegerAttr{},
        /*params=*/nullptr);
    func::ReturnOp::create(b, loc, gemmOp.getResult());
    // `arch` must be readable by `rock::getArchValue`, which walks up looking
    // for the `rock.arch` attribute.
    (*module)->setAttr(rock::ArchAttr::getMnemonic(),
                       b.getStringAttr(Twine("amdgcn-amd-amdhsa:") + arch));
  }

  // Builds the full tuning space and checks that every non-power-of-two
  // kPerBlock candidate obeys the `windowDividingKPerBlock` heuristic. Returns
  // the set of kPerBlock values the space offers. Every op built here is a
  // plain gemm, which never opens the widened range, so the window rules hold
  // for all of them.
  std::set<int64_t> collectAndCheckKPerBlocks(int64_t k) {
    std::unique_ptr<TuningParamSet> space(
        createTunableParamSpace(*module, TuningParamSetKind::Full));
    std::set<int64_t> kValues;
    if (!space || space->tuningRange.empty()) {
      ADD_FAILURE() << "empty tuning space";
      return kValues;
    }
    for (auto param : space->tuningRange) {
      auto gemmParams = cast<GemmParamsAttr>(param);
      int64_t minMN =
          std::min(gemmParams.getMPerBlock(), gemmParams.getNPerBlock());
      int64_t kPerBlock = gemmParams.getKPerBlock();
      kValues.insert(kPerBlock);
      if (llvm::isPowerOf2_64(static_cast<uint64_t>(kPerBlock)))
        continue;
      EXPECT_EQ(k % kPerBlock, 0) << "non-pow2 kPerBlock=" << kPerBlock
                                  << " must evenly divide K=" << k;
      EXPECT_EQ(llvm::popcount(static_cast<uint64_t>(kPerBlock)), 2)
          << "non-pow2 kPerBlock=" << kPerBlock
          << " must peel into two pow2 segments";
      EXPECT_GE(kPerBlock, minMN / 2) << "non-pow2 kPerBlock=" << kPerBlock
                                      << " below window for min(m,n)=" << minMN;
      EXPECT_LT(kPerBlock, minMN)
          << "non-pow2 kPerBlock=" << kPerBlock
          << " must be the smallest tile edge for min(m,n)=" << minMN;
    }
    return kValues;
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
  // 256x256x512 fp32 numStages=3 overflows gfx942 LDS. Mirrors the config
  // exercised by lds-overflow-not-applicable.mlir.
  auto p = e.gemm(256, 256, 512, 1, 1, 4, 16, 1, 3, 0, 0);
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
  EXPECT_EQ(p.getUseOptimizeEpilogue(), kKnobDefault);
  EXPECT_EQ(p.getUseBf16x3ForF32(), kKnobDefault);
}

TEST(PerfConfigOrderingGemmTest, ConservativeDefaultFp4KPerBlock) {
  GemmOrderingTestEnv e;
  EXPECT_EQ(getConservativeDefaultGemmParams(
                &e.ctx, /*quantBlockSize=*/std::nullopt, e.f4E2M1, e.f32)
                .getKPerBlock(),
            64);
  EXPECT_EQ(getConservativeDefaultGemmParams(
                &e.ctx, /*quantBlockSize=*/std::nullopt, e.f32, e.f4E2M1)
                .getKPerBlock(),
            64);
}

// --- FP4 partial-upcast-group rejection ---

TEST(PerfConfigOrderingGemmTest, IsApplicableRejectsFp4PartialUpcastGroup) {
  GemmOrderingTestEnv e;
  // The reproducer from docs/fp4_tuning_crashes.md: kPerBlock = 64 is not a
  // whole `16x16x128` k-tile, so the dot decomposes into an upcast.
  auto p = e.gemm(16, 16, 64, 1, 1, /*numWaves=*/4, /*matrixInstrNonkdim=*/16,
                  1, 1, 0, 0);
  EXPECT_FALSE(
      isGemmParamsConservativelyApplicable(p, e.f4E2M1, e.f4E2M1, "gfx950"));
  // Same tile with a non-4-bit element type has no upcast to get wrong.
  EXPECT_TRUE(isGemmParamsConservativelyApplicable(p, e.f16, e.f16, "gfx950"));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableAcceptsFp4OnNativeScaledMfma) {
  GemmOrderingTestEnv e;
  // kPerBlock is a whole number of k-tiles for the selected instruction, so
  // the GEMM stays native: 128 for the 16x16 form, 64 for the 32x32 one.
  for (auto [nonKDim, kPerBlock] :
       {std::pair<int64_t, int64_t>{16, 128}, {32, 64}}) {
    auto p =
        e.gemm(16, 16, kPerBlock, 1, 1, /*numWaves=*/4, nonKDim, 1, 1, 0, 0);
    EXPECT_TRUE(
        isGemmParamsConservativelyApplicable(p, e.f4E2M1, e.f4E2M1, "gfx950"))
        << "rejected native config with matrixInstrNonkdim=" << nonKDim;
  }
}

TEST(PerfConfigOrderingGemmTest, IsApplicableRejectsFp4OnSingleWave) {
  GemmOrderingTestEnv e;
  // Triton refuses the native path for a single wave, even on a whole k-tile.
  auto p = e.gemm(16, 16, 128, 1, 1, /*numWaves=*/1,
                  /*matrixInstrNonkdim=*/16, 1, 1, 0, 0);
  EXPECT_FALSE(
      isGemmParamsConservativelyApplicable(p, e.f4E2M1, e.f4E2M1, "gfx950"));
}

TEST(PerfConfigOrderingGemmTest, IsApplicableRejectsSafeButDecomposedFp4) {
  GemmOrderingTestEnv e;
  // This tile does give every thread a whole group, but it decomposes, and
  // that is judged by the pass at conversion time rather than here.
  auto p = e.gemm(128, 128, 32, 1, 1, /*numWaves=*/2, /*matrixInstrNonkdim=*/32,
                  1, 1, 0, 0);
  EXPECT_FALSE(
      isGemmParamsConservativelyApplicable(p, e.f4E2M1, e.f4E2M1, "gfx950"));
}

TEST(PerfConfigOrderingGemmTest, ScaledAccelKDimFollowsTileWhenNonKDimUnset) {
  // With no enforced non-k dimension Triton derives it from the smaller tile
  // dimension, and has no scaled intrinsic for a tile below 16.
  EXPECT_EQ(getScaledAccelKDim("gfx950", 32, 64, /*matrixInstrNonkdim=*/0), 64);
  EXPECT_EQ(getScaledAccelKDim("gfx950", 16, 64, /*matrixInstrNonkdim=*/0),
            128);
  EXPECT_EQ(getScaledAccelKDim("gfx950", 8, 64, /*matrixInstrNonkdim=*/0), 0);
  // Only CDNA4 has the scaled MFMA family; elsewhere the dot is decomposed.
  EXPECT_EQ(getScaledAccelKDim("gfx942", 32, 64, /*matrixInstrNonkdim=*/32), 0);
}

TEST(PerfConfigOrderingGemmTest, IsApplicableLeavesFp4AloneWithoutScaledMfma) {
  GemmOrderingTestEnv e;
  // With no native path to stay on the check stands down instead of rejecting
  // every 4-bit config on the target.
  auto p = e.gemm(16, 32, 64, 1, 1, /*numWaves=*/2, /*matrixInstrNonkdim=*/16,
                  1, /*numStages=*/2, 0, 0);
  EXPECT_FALSE(archHasScaledMfma("gfx942"));
  EXPECT_TRUE(
      isGemmParamsConservativelyApplicable(p, e.f4E2M1, e.f4E2M1, "gfx942"));
}

TEST(PerfConfigOrderingGemmTest, ConservativeDefaultFp4PassesPredicate) {
  GemmOrderingTestEnv e;
  // Whatever the quick-tuning table holds, the prepended fallback must be
  // applicable, or `front()` has no guarantee at all.
  auto p = getConservativeDefaultGemmParams(
      &e.ctx, /*quantBlockSize=*/std::nullopt, e.f4E2M1, e.f4E2M1);
  EXPECT_TRUE(
      isGemmParamsConservativelyApplicable(p, e.f4E2M1, e.f4E2M1, "gfx950"));
  for (int64_t qBS : {16, 32, 64, 128}) {
    auto scaled =
        getConservativeDefaultGemmParams(&e.ctx, qBS, e.f4E2M1, e.f4E2M1);
    EXPECT_TRUE(isGemmParamsConservativelyApplicable(
        scaled, e.f4E2M1, e.f4E2M1, "gfx950", qBS, e.f8E8M0, e.f8E8M0))
        << "FP4 default not applicable for quantBlockSize=" << qBS;
  }
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

TEST(PerfConfigOrderingGemmTest, IsApplicableHandlesShapedScaleType) {
  // `RockGemmWrapperInterface::getScale{A,B}Type()`'s default impl returns the
  // tensor type of the scale operand, not the element type. The predicate
  // must normalize so `getIntOrFloatBitWidth` doesn't assert on a shaped type
  // (and so scale LDS isn't undercounted). Pass a shaped tensor type and a
  // bare element type for the other side to lock the behaviour.
  GemmOrderingTestEnv e;
  auto p = e.gemm(128, 128, 128, 1, 1, 4, 0, 1, /*numStages=*/2, 0, 0);
  Type shapedScale = RankedTensorType::get({1, 128, 128}, e.f8E8M0);
  Type shapedA = RankedTensorType::get({1, 128, 128}, e.f8E4M3);
  EXPECT_FALSE(isGemmParamsConservativelyApplicable(
      p, shapedA, e.f8E4M3, "gfx942", /*quantBlockSize=*/1, shapedScale,
      e.f8E8M0))
      << "shaped scale type should be normalized like a bare element type";
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

// Production wiring: build a real scaled rock.gemm and walk through
// `PopulateParamsInfo::fromOp` to make sure the scale element type is what
// reaches the predicate (`getScale{A,B}Type()` hands back a tensor type and
// `fromOp` is responsible for normalizing it).
TEST(PerfConfigOrderingGemmTest, FromOpExtractsScaleElementTypeOnRealGemmOp) {
  MLIRContext ctx;
  DialectRegistry reg;
  reg.insert<rock::RockDialect>();
  reg.insert<func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  Type f8E4M3 = Float8E4M3FNType::get(&ctx);
  Type f8E8M0 = Float8E8M0FNUType::get(&ctx);

  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  b.setInsertionPointToEnd(module->getBody());

  // Shapes: A [G,M,K]=[1,64,64], B [G,K,N]=[1,64,64],
  // scaleA [G,M,K/qBS]=[1,64,2], scaleB [G,N,K/qBS]=[1,64,2], qBS=32.
  auto aT = RankedTensorType::get({1, 64, 64}, f8E4M3);
  auto bT = RankedTensorType::get({1, 64, 64}, f8E4M3);
  auto scaleT = RankedTensorType::get({1, 64, 2}, f8E8M0);
  auto cT = RankedTensorType::get({1, 64, 64}, b.getF32Type());

  auto funcType = b.getFunctionType({aT, bT, scaleT, scaleT}, {cT});
  auto func = func::FuncOp::create(b, loc, "test", funcType);
  Block *body = func.addEntryBlock();
  b.setInsertionPointToStart(body);

  auto gemmOp = GemmOp::create(
      b, loc, /*c=*/cT, /*a=*/body->getArgument(0), /*b=*/body->getArgument(1),
      /*scaleA=*/body->getArgument(2), /*scaleB=*/body->getArgument(3),
      /*aTransposed=*/UnitAttr{}, /*bTransposed=*/UnitAttr{},
      /*oTransposed=*/UnitAttr{}, /*aScaleTransposed=*/UnitAttr{},
      /*bScaleTransposed=*/UnitAttr{},
      /*quantBlockSize=*/b.getI64IntegerAttr(32),
      /*params=*/nullptr);
  // `arch` must be readable by `rock::getArchValue`, which walks up looking
  // for the `rock.arch` attribute.
  (*module)->setAttr(rock::ArchAttr::getMnemonic(),
                     b.getStringAttr("amdgcn-amd-amdhsa:gfx950"));

  auto info = PopulateParamsInfo::fromOp(
      cast<RockGemmWrapperInterface>(gemmOp.getOperation()));

  // Crucial: info must carry the *element* type, not the scale tensor type.
  ASSERT_TRUE(info.aScaleType);
  ASSERT_TRUE(info.bScaleType);
  EXPECT_EQ(info.aScaleType, f8E8M0);
  EXPECT_EQ(info.bScaleType, f8E8M0);
  ASSERT_TRUE(info.quantBlockSize.has_value());
  EXPECT_EQ(*info.quantBlockSize, 32);

  // And feeding the predicate from `info` must not assert and must accept the
  // conservative default for this op.
  auto p = getConservativeDefaultGemmParams(&ctx, info.quantBlockSize);
  EXPECT_TRUE(isGemmParamsConservativelyApplicable(
      p, info.gemmAType, info.gemmBType, info.arch, info.quantBlockSize,
      info.aScaleType, info.bScaleType));
}

// --- non-power-of-two kPerBlock candidate generation ---

// The tuning space for a plain accelerated gemm must offer non-power-of-two
// kPerBlock values that evenly divide K (peeled into pow2 K segments
// downstream), so the tuner can drop K padding on shapes like a conv with
// K = C*fil_h*fil_w. Each such candidate must satisfy the window heuristic in
// `windowDividingKPerBlock` (RockTuningImpl.cpp): it peels into exactly two
// pow2 segments and is the smallest tile edge, i.e. in [min(m,n)/2, min(m,n)).
TEST(PerfConfigOrderingGemmTest, TuningSpaceIncludesNonPow2KDivisors) {
  // K = 576 = 64 * 3 * 3 (a conv's C*fil_h*fil_w). Its non-pow2 divisors in the
  // tunable range include 36, 48, 72, 96, 144. gfx1201 (RDNA4) with f16 drives
  // the WMMA path, whose base kPerBlock list is {32, 64}.
  const int64_t K = 576;
  TuningSpaceGemmEnv e([](OpBuilder &b) { return b.getF16Type(); },
                       /*m=*/256, /*n=*/256, K, "gfx1201");
  std::set<int64_t> kValues = e.collectAndCheckKPerBlocks(K);

  // The target non-pow2 K tile (48 = 576/12) is offered (it lands in the
  // [32,64) window of a min(m,n)=64 tile), ...
  EXPECT_TRUE(kValues.count(48)) << "expected non-pow2 kPerBlock=48 in space";
  // ... and a pow2 divisor from the base list is still present.
  EXPECT_TRUE(kValues.count(64)) << "expected base pow2 kPerBlock=64 in space";
}

// The same heuristic must feed the non-accel (FMA) tuning range, whose pow2
// base kPerBlock list is {1, 4, 8, 16} instead of the accelerated lists.
TEST(PerfConfigOrderingGemmTest, TuningSpaceIncludesNonPow2KDivisorsOnFma) {
  // f32 has no matrix-accelerator instruction on gfx1201, so this gemm takes
  // the FMA path. K = 48 has the non-pow2 divisor 24 (= 16 + 8).
  const int64_t K = 48;
  TuningSpaceGemmEnv e([](OpBuilder &b) { return b.getF32Type(); },
                       /*m=*/128, /*n=*/128, K, "gfx1201");
  std::set<int64_t> kValues = e.collectAndCheckKPerBlocks(K);

  // A small pow2 tile that only the non-accel base list contains, i.e. proof
  // this really is the FMA range and not an accelerated one.
  EXPECT_TRUE(kValues.count(4)) << "expected non-accel base kPerBlock=4 "
                                   "in space; is this the FMA path?";
  // 24 lands in the [16,32) window of a min(m,n)=32 tile.
  EXPECT_TRUE(kValues.count(24)) << "expected non-pow2 kPerBlock=24 in space";
}

// A K that no power-of-two tile divides has no remainder-free kPerBlock in the
// default space, which is what the widened range exists to reach. It stays shut
// on a plain gemm though: without a conv's merge alignment to narrow it, it
// admits several tiles rather than one, and we have no gemm speedup numbers to
// weigh that growth against (see the TODO in computeKPerBlock).
TEST(PerfConfigOrderingGemmTest, TuningSpaceKeepsWidenedRangeOffOnFmaGemm) {
  // K = 4635 = 515 * 3 * 3, from the AIROCMLIR-1182 convs but here as a plain
  // gemm. It is odd, so none of the non-accel pow2 tiles {4, 8, 16} divides it.
  // f32 on gfx1101 (RDNA3) has no matrix-accelerator instruction, so this takes
  // the FMA path.
  const int64_t K = 4635;
  TuningSpaceGemmEnv e([](OpBuilder &b) { return b.getF32Type(); },
                       /*m=*/128, /*n=*/128, K, "gfx1101");
  std::set<int64_t> kValues = e.collectAndCheckKPerBlocks(K);

  // 45 is what the widened range would add: it is measurably the best kPerBlock
  // for the conv this K comes from, yet it falls outside every tile's window
  // and peels into four segments (45 = 32 + 8 + 4 + 1), so both rules reject
  // it.
  EXPECT_FALSE(kValues.count(45))
      << "kPerBlock=45 is reachable only through the widened range";
  // The window still contributes what it always did: 9 divides K, peels into
  // two segments and lands in the [8,16) window of a 16-wide tile.
  EXPECT_TRUE(kValues.count(9)) << "expected windowed kPerBlock=9 in space";
}

// The restriction is on the shape, not the path, so an accelerated gemm with an
// equally awkward K keeps the default rules too.
TEST(PerfConfigOrderingGemmTest, TuningSpaceKeepsWidenedRangeOffOnWmmaGemm) {
  // K = 3024 = 336 * 3 * 3, from the f16 half of the same conv set. Neither of
  // WMMA's pow2 tiles {32, 64} divides it. f16 on gfx1101 (RDNA3) drives WMMA.
  const int64_t K = 3024;
  TuningSpaceGemmEnv e([](OpBuilder &b) { return b.getF16Type(); },
                       /*m=*/256, /*n=*/256, K, "gfx1101");
  std::set<int64_t> kValues = e.collectAndCheckKPerBlocks(K);

  // 112 = 64+32+16 divides K and is a multiple of the 16-wide WMMA instruction,
  // so the widened range would take it, but it peels into three segments and
  // falls outside every window.
  EXPECT_FALSE(kValues.count(112))
      << "kPerBlock=112 is reachable only through the widened range";
  // A pow2 tile from the base list is still present, remainder or not.
  EXPECT_TRUE(kValues.count(32)) << "expected base pow2 kPerBlock=32 in space";
}

// A K that a pow2 tile does divide keeps the default two-segment/window rules,
// so the widened range cannot grow the search space of a well-behaved shape.
TEST(PerfConfigOrderingGemmTest, TuningSpaceKeepsWindowOnFmaForPow2DivisibleK) {
  // K = 432 = 48 * 3 * 3, from the same conv set. 16 divides it, so the gate
  // must not fire.
  const int64_t K = 432;
  TuningSpaceGemmEnv e([](OpBuilder &b) { return b.getF32Type(); },
                       /*m=*/128, /*n=*/128, K, "gfx1101");
  std::set<int64_t> kValues = e.collectAndCheckKPerBlocks(K);

  // 27 and 54 divide K and sit inside the flat range, but both peel into four
  // segments (27 = 16 + 8 + 2 + 1), so only the widened mode would offer them.
  EXPECT_FALSE(kValues.count(27))
      << "kPerBlock=27 must stay out while a pow2 tile divides K";
  EXPECT_FALSE(kValues.count(54))
      << "kPerBlock=54 must stay out while a pow2 tile divides K";
  // The two-segment divisors the window does allow are unaffected.
  EXPECT_TRUE(kValues.count(24)) << "expected non-pow2 kPerBlock=24 in space";
  EXPECT_TRUE(kValues.count(16)) << "expected base pow2 kPerBlock=16 in space";
}

// gfx950 is opted out of the non-pow2 kPerBlock candidates while the LLVM
// backend bug that miscompiles the peeled K loop there is unfixed, so its
// tuning space must stay purely power-of-two.
TEST(PerfConfigOrderingGemmTest, TuningSpaceHasNoNonPow2KDivisorsOnGfx950) {
  // Same K = 576 as the gfx1201 case above, which does offer 36/48/72/...
  const int64_t K = 576;
  TuningSpaceGemmEnv e([](OpBuilder &b) { return b.getF16Type(); },
                       /*m=*/256, /*n=*/256, K, "gfx950");
  std::set<int64_t> kValues = e.collectAndCheckKPerBlocks(K);

  ASSERT_FALSE(kValues.empty());
  for (int64_t kPerBlock : kValues)
    EXPECT_TRUE(llvm::isPowerOf2_64(static_cast<uint64_t>(kPerBlock)))
        << "gfx950 must not offer non-pow2 kPerBlock=" << kPerBlock;
}
