//===- PerfConfigOrderingGemmGemmTests.cpp - Gemm+Gemm ordering tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
/// Stands up a minimal `rock.attention` op so the gemm+gemm predicate can read
/// `op.getCType()` / `op.getTransposedC()`. `headV` controls the gemm1 N dim
/// (i.e. `gemm1NPerBlock`); the other shape dims are arbitrary since the
/// predicate doesn't read them.
struct GemmGemmOrderingTestEnv {
  MLIRContext ctx;
  OpBuilder builder;
  OwningOpRef<ModuleOp> module;
  Type f16, f32;
  AttentionOp attn;

  explicit GemmGemmOrderingTestEnv(int64_t headV = 64, bool useF32 = false)
      : builder(&ctx) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    f16 = builder.getF16Type();
    f32 = builder.getF32Type();
    Location loc = builder.getUnknownLoc();

    module = ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module->getBody());

    // Shapes: Q,O are [G, seq_q, head_qk/head_v]; K is [G, head_qk, seq_k];
    // V is [G, seq_k, head_v]. Only V's `head_v` (the last dim) matters for
    // `gemm1NPerBlock` since the predicate reads `op.getCType()`.
    Type elemType = useF32 ? f32 : f16;
    auto qT = RankedTensorType::get({1, 64, 64}, elemType);
    auto kT = RankedTensorType::get({1, 64, 64}, elemType);
    auto vT = RankedTensorType::get({1, 64, headV}, elemType);
    auto oT = RankedTensorType::get({1, 64, headV}, elemType);

    auto funcType = builder.getFunctionType({qT, kT, vT}, {oT});
    auto func = func::FuncOp::create(builder, loc, "test", funcType);
    Block *body = func.addEntryBlock();
    builder.setInsertionPointToStart(body);

    attn = AttentionOp::create(
        builder, loc, /*result=*/oT, /*lse=*/Type{},
        /*queries=*/body->getArgument(0), /*keys=*/body->getArgument(1),
        /*values=*/body->getArgument(2),
        /*preSoftmaxElemWiseInputs=*/ValueRange{},
        /*currentSeqLen=*/Value{}, /*prefixOffset=*/Value{},
        /*numHeadsQ=*/builder.getI32IntegerAttr(1),
        /*numHeadsKV=*/builder.getI32IntegerAttr(1),
        /*qTransposed=*/UnitAttr{}, /*kTransposed=*/UnitAttr{},
        /*vTransposed=*/UnitAttr{}, /*oTransposed=*/UnitAttr{},
        /*causal=*/UnitAttr{},
        /*splitKV=*/builder.getI32IntegerAttr(1),
        /*softmaxType=*/TypeAttr{},
        /*params0=*/nullptr, /*params1=*/nullptr,
        /*preSoftmaxHasSplitKVTransforms=*/BoolAttr{});
  }

  RockGemmGemmWrapperInterface op() {
    return cast<RockGemmGemmWrapperInterface>(attn.getOperation());
  }

  // Build a GemmGemmParamsAttr from positional fields. Mirrors the perfconfig
  // string format documented on Rock_GemmGemmParamsAttr.
  GemmGemmParamsAttr params(int64_t mPerBlockG0, int64_t nPerBlockG0,
                            int64_t kPerBlock, int64_t kpack, int64_t numCTAs,
                            int64_t numWaves, int64_t matrixInstrNonkdim,
                            int64_t splitKFactor, int64_t numStages,
                            int64_t wavesPerEU, int64_t gridGroupSize) {
    return GemmGemmParamsAttr::get(
        &ctx, mPerBlockG0, nPerBlockG0, kPerBlock, kpack, numCTAs, numWaves,
        matrixInstrNonkdim, splitKFactor, numStages, wavesPerEU, gridGroupSize,
        /*useAsyncCopy=*/kKnobDefault,
        /*useBlockPingpong=*/kKnobDefault,
        /*useInThreadTranspose=*/kKnobDefault,
        /*useBufferOps=*/kKnobDefault,
        /*useBufferAtomics=*/kKnobDefault,
        /*useReductionLayout=*/0);
  }
};
} // namespace

// --- isGemmGemmParamsConservativelyApplicable ---

TEST(PerfConfigOrderingGemmGemmTest, IsApplicableRejectsKpackNot1) {
  GemmGemmOrderingTestEnv e;
  auto p = e.params(32, 32, 32, /*kpack=*/2, 1, 4, 0, 1, 1, 0, 0);
  EXPECT_FALSE(isGemmGemmParamsConservativelyApplicable(
      e.builder, p, e.f16, e.f16, "gfx942", e.op()));
}

TEST(PerfConfigOrderingGemmGemmTest, IsApplicableRejectsSplitKNot1) {
  GemmGemmOrderingTestEnv e;
  auto p = e.params(32, 32, 32, 1, 1, 4, 0, /*splitKFactor=*/2, 1, 0, 0);
  EXPECT_FALSE(isGemmGemmParamsConservativelyApplicable(
      e.builder, p, e.f16, e.f16, "gfx942", e.op()));
}

TEST(PerfConfigOrderingGemmGemmTest, IsApplicableRejectsNumCTAsNot1) {
  GemmGemmOrderingTestEnv e;
  auto p = e.params(32, 32, 32, 1, /*numCTAs=*/2, 4, 0, 1, 1, 0, 0);
  EXPECT_FALSE(isGemmGemmParamsConservativelyApplicable(
      e.builder, p, e.f16, e.f16, "gfx942", e.op()));
}

TEST(PerfConfigOrderingGemmGemmTest, IsApplicableRejectsGemm0LDSOverflow) {
  GemmGemmOrderingTestEnv e;
  // Huge gemm0 tile blows the LDS budget by itself, regardless of headV.
  auto p = e.params(256, 256, 256, 1, 1, 4, 16, 1, 2, 0, 0);
  EXPECT_FALSE(isGemmGemmParamsConservativelyApplicable(
      e.builder, p, e.f32, e.f32, "gfx942", e.op()));
}

TEST(PerfConfigOrderingGemmGemmTest, IsApplicableRejectsGemm1VTileOverflow) {
  // gemm0 tile alone fits, but the V tile (nPerBlockG0 × headV × bits) tips
  // it over. This is what distinguishes the gemm+gemm predicate from the
  // single-gemm one: with the conservative default (32x32x32, fp32,
  // numStages=1) gemm0 alone needs (32*32*32 + 32*32*32)/8 = 8 KiB; adding
  // V tile of 32 × headV × 32 bits for headV = 16384 yields 2 MiB, well over
  // gfx942's 64 KiB LDS.
  GemmGemmOrderingTestEnv e(/*headV=*/16384, /*useF32=*/true);
  auto p = e.params(32, 32, 32, 1, 1, 4, 0, 1, 1, 0, 0);
  EXPECT_FALSE(isGemmGemmParamsConservativelyApplicable(
      e.builder, p, e.f32, e.f32, "gfx942", e.op()));
}

TEST(PerfConfigOrderingGemmGemmTest, IsApplicableAcceptsConservativeDefault) {
  GemmGemmOrderingTestEnv e;
  auto p = e.params(32, 32, 32, 1, 1, 4, 0, 1, 1, 0, 0);
  EXPECT_TRUE(isGemmGemmParamsConservativelyApplicable(
      e.builder, p, e.f16, e.f16, "gfx942", e.op()));
  EXPECT_TRUE(isGemmGemmParamsConservativelyApplicable(
      e.builder, p, e.f32, e.f32, "gfx942", e.op()));
  EXPECT_TRUE(isGemmGemmParamsConservativelyApplicable(
      e.builder, p, e.f16, e.f16, "gfx1100", e.op()));
}

// --- orderParams<GemmGemmParamsAttr> ---

TEST(PerfConfigOrderingGemmGemmTest, OrderParamsBumpsSecondToFront) {
  GemmGemmOrderingTestEnv e;
  auto bad = e.params(32, 32, 32, /*kpack=*/2, 1, 4, 0, 1, 1, 0, 0);
  auto good = e.params(32, 32, 32, 1, 1, 4, 0, 1, 1, 0, 0);
  auto other = e.params(64, 64, 32, 1, 1, 4, 0, 1, 1, 0, 0);
  std::vector<GemmGemmParamsAttr> in{bad, good, other};
  auto out = orderParams<GemmGemmParamsAttr>(in, [&](GemmGemmParamsAttr p) {
    return isGemmGemmParamsConservativelyApplicable(e.builder, p, e.f16, e.f16,
                                                    "gfx942", e.op());
  });
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0], good);
  EXPECT_EQ(out[1], bad);
  EXPECT_EQ(out[2], other);
}

// --- getConservativeDefaultGemmGemmParams ---

TEST(PerfConfigOrderingGemmGemmTest, ConservativeDefaultGemmGemmParamsFields) {
  GemmGemmOrderingTestEnv e;
  auto p = getConservativeDefaultGemmGemmParams(&e.ctx);
  EXPECT_EQ(p.getMPerBlockG0(), 32);
  EXPECT_EQ(p.getNPerBlockG0(), 32);
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

TEST(PerfConfigOrderingGemmGemmTest,
     ConservativeDefaultGemmGemmParamsPassesPredicate) {
  // For a typical attention shape (headV = 64) the conservative default fits
  // LDS on every supported arch. Larger-headV behaviour is covered separately
  // by IsApplicableRejectsGemm1VTileOverflow.
  GemmGemmOrderingTestEnv e;
  auto p = getConservativeDefaultGemmGemmParams(&e.ctx);
  for (StringRef arch : {"gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100",
                         "gfx1200", "gfx1250"}) {
    EXPECT_TRUE(isGemmGemmParamsConservativelyApplicable(e.builder, p, e.f16,
                                                         e.f16, arch, e.op()))
        << "default not applicable on " << arch << " for f16";
    EXPECT_TRUE(isGemmGemmParamsConservativelyApplicable(e.builder, p, e.f32,
                                                         e.f32, arch, e.op()))
        << "default not applicable on " << arch << " for f32";
  }
}
