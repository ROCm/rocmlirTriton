//===- AmdArchDbTests.cpp - Tests for AMD Architecture Database -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct ArchTestEnv {
  MLIRContext ctx;
  OpBuilder b;
  Type f32, f16, bf16, i32, i8;
  Type f8e4m3fn, f8e5m2, f6e3m2fn, f6e2m3fn, f4e2m1fn;

  ArchTestEnv() : b(&ctx) {
    f32 = b.getF32Type();
    f16 = b.getF16Type();
    bf16 = b.getBF16Type();
    i32 = b.getI32Type();
    i8 = b.getIntegerType(8);
    f8e4m3fn = Float8E4M3FNType::get(&ctx);
    f8e5m2 = Float8E5M2Type::get(&ctx);
    f6e3m2fn = Float6E3M2FNType::get(&ctx);
    f6e2m3fn = Float6E2M3FNType::get(&ctx);
    f4e2m1fn = Float4E2M1FNType::get(&ctx);
  }
};
} // namespace

TEST(AmdArchDbTest, ParseArchString) {
  {
    auto [chip, features] = parseArchString("gfx942");
    EXPECT_EQ(chip.str(), "gfx942");
    EXPECT_EQ(features, 0u);
  }
  {
    auto [chip, features] = parseArchString("amdgcn-amd-amdhsa:gfx942:xnack-");
    EXPECT_EQ(chip.str(), "gfx942");
    EXPECT_EQ(features, 0u);
  }
  {
    auto [chip, features] = parseArchString("gfx999");
    EXPECT_EQ(chip.str(), "gfx999");
    EXPECT_EQ(features, 0u);
  }
  {
    auto [chip, features] = parseArchString("badarch");
    EXPECT_TRUE(chip.empty());
    EXPECT_EQ(features, 0u);
  }
}

// --- isFastAtomicMaxSupported ---

TEST(AmdArchDbTest, FastAtomicMaxF32) {
  ArchTestEnv e;
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx906", e.f32)); // GCN5_1
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx908", e.f32)); // CDNA1
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx90a", e.f32)); // CDNA2
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx942", e.f32)); // CDNA3
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx950", e.f32)); // CDNA4
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1010", e.f32)); // RDNA1
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1030", e.f32)); // RDNA2
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1100", e.f32)); // RDNA3
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1170", e.f32)); // GFX1170
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1200", e.f32)); // RDNA4
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1250", e.f32)); // GFX1250
}

TEST(AmdArchDbTest, FastAtomicMaxNonF32Unsupported) {
  ArchTestEnv e;
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx90a", e.f16));  // CDNA2
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx950", e.f16));  // CDNA4
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1010", e.f16)); // RDNA1
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1030", e.f16)); // RDNA2
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1100", e.f16)); // RDNA3
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1170", e.f16)); // GFX1170
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1200", e.f16)); // RDNA4
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1250", e.f16)); // GFX1250
}

// --- getMaxNumChiplets ---

TEST(AmdArchDbTest, MaxNumChiplets) {
  EXPECT_EQ(getMaxNumChiplets("gfx906"), 1);  // GCN5_1
  EXPECT_EQ(getMaxNumChiplets("gfx908"), 1);  // CDNA1
  EXPECT_EQ(getMaxNumChiplets("gfx90a"), 1);  // CDNA2
  EXPECT_EQ(getMaxNumChiplets("gfx942"), 8);  // CDNA3
  EXPECT_EQ(getMaxNumChiplets("gfx950"), 8);  // CDNA4
  EXPECT_EQ(getMaxNumChiplets("gfx1010"), 1); // RDNA1
  EXPECT_EQ(getMaxNumChiplets("gfx1030"), 1); // RDNA2
  EXPECT_EQ(getMaxNumChiplets("gfx1100"), 1); // RDNA3
  EXPECT_EQ(getMaxNumChiplets("gfx1170"), 1); // GFX1170
  EXPECT_EQ(getMaxNumChiplets("gfx1200"), 1); // RDNA4
  EXPECT_EQ(getMaxNumChiplets("gfx1250"), 8); // GFX1250
}

// --- inferNumChiplets ---

TEST(AmdArchDbTest, InferNumChiplets) {
  EXPECT_EQ(inferNumChiplets("gfx942", 304), 8);
  EXPECT_EQ(inferNumChiplets("gfx942", 80), 4);
  EXPECT_EQ(inferNumChiplets("gfx942", 120), 1);
  EXPECT_EQ(inferNumChiplets("gfx950", 256), 8);
  EXPECT_EQ(inferNumChiplets("gfx906", 60), 1);
  EXPECT_EQ(inferNumChiplets("amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", 304),
            8);
}

// MI350X / MI355X compute-partition modes, whose per-partition CU counts are
// SPX=256, DPX=128, QPX=64, CPX=32.
TEST(AmdArchDbTest, InferNumChipletsCdna4Partitions) {
  EXPECT_EQ(inferNumChiplets("gfx950", 256), 8);
  EXPECT_EQ(inferNumChiplets("gfx950", 128), 4);
  EXPECT_EQ(inferNumChiplets("gfx950", 64), 2);
  EXPECT_EQ(inferNumChiplets("gfx950", 32), 1);
  // An unrecognized count must fall back to 1 rather than guess an odd chiplet
  // number: makeGroupedGridLayout and makeGxNGridLayout assert
  // numChiplets % 2 == 0 for any count greater than 1.
  EXPECT_EQ(inferNumChiplets("gfx950", 96), 1);
}

// gfx1250 reports 256 WGPs over 8 XCDs. No partition-mode CU counts
// are published for this family, hence the single recognized configuration.
TEST(AmdArchDbTest, InferNumChipletsGfx1250) {
  EXPECT_EQ(inferNumChiplets("gfx1250", 256), 8);
  EXPECT_EQ(inferNumChiplets("gfx1250", 32), 1);
  // gfx1251 (MI430X) parses into the same ISA family even though it does not
  // share gfx1250's XCDs, so it inherits this mapping. Pinned here so that
  // splitting the two later is a deliberate change rather than a silent one.
  EXPECT_EQ(inferNumChiplets("gfx1251", 256), 8);
}

// --- getMinNumCU ---

TEST(AmdArchDbTest, MinNumCU) {
  EXPECT_EQ(getMinNumCU("gfx906"), 10);   // GCN5_1
  EXPECT_EQ(getMinNumCU("gfx908"), 120);  // CDNA1
  EXPECT_EQ(getMinNumCU("gfx90a"), 104);  // CDNA2
  EXPECT_EQ(getMinNumCU("gfx942"), 20);   // CDNA3
  EXPECT_EQ(getMinNumCU("gfx950"), 256);  // CDNA4
  EXPECT_EQ(getMinNumCU("gfx1010"), 30);  // RDNA1
  EXPECT_EQ(getMinNumCU("gfx1030"), 30);  // RDNA2
  EXPECT_EQ(getMinNumCU("gfx1100"), 2);   // RDNA3
  EXPECT_EQ(getMinNumCU("gfx1170"), 2);   // GFX1170
  EXPECT_EQ(getMinNumCU("gfx1200"), 12);  // RDNA4
  EXPECT_EQ(getMinNumCU("gfx1250"), 256); // GFX1250
}

// --- getMaxWavesPerEU ---

TEST(AmdArchDbTest, MaxWavesPerEU) {
  EXPECT_EQ(getMaxWavesPerEU("gfx906"), 8);   // GCN5_1
  EXPECT_EQ(getMaxWavesPerEU("gfx908"), 8);   // CDNA1
  EXPECT_EQ(getMaxWavesPerEU("gfx90a"), 8);   // CDNA2
  EXPECT_EQ(getMaxWavesPerEU("gfx942"), 8);   // CDNA3
  EXPECT_EQ(getMaxWavesPerEU("gfx950"), 8);   // CDNA4
  EXPECT_EQ(getMaxWavesPerEU("gfx1010"), 16); // RDNA1
  EXPECT_EQ(getMaxWavesPerEU("gfx1030"), 16); // RDNA2
  EXPECT_EQ(getMaxWavesPerEU("gfx1100"), 16); // RDNA3
  EXPECT_EQ(getMaxWavesPerEU("gfx1170"), 16); // GFX1170
  EXPECT_EQ(getMaxWavesPerEU("gfx1200"), 16); // RDNA4
  EXPECT_EQ(getMaxWavesPerEU("gfx1250"), 16); // GFX1250
}

// --- getVGPRsPerEU ---

TEST(AmdArchDbTest, VGPRsPerEU) {
  EXPECT_EQ(getVGPRsPerEU("gfx906"), 256);   // GCN5_1
  EXPECT_EQ(getVGPRsPerEU("gfx908"), 256);   // CDNA1
  EXPECT_EQ(getVGPRsPerEU("gfx90a"), 512);   // CDNA2
  EXPECT_EQ(getVGPRsPerEU("gfx942"), 512);   // CDNA3
  EXPECT_EQ(getVGPRsPerEU("gfx950"), 512);   // CDNA4
  EXPECT_EQ(getVGPRsPerEU("gfx1010"), 1024); // RDNA1
  EXPECT_EQ(getVGPRsPerEU("gfx1030"), 1024); // RDNA2
  EXPECT_EQ(getVGPRsPerEU("gfx1100"), 1536); // RDNA3, 1536 physical VGPRs
  EXPECT_EQ(getVGPRsPerEU("gfx1102"), 1024); // RDNA3, cut-down VGPR file
  EXPECT_EQ(getVGPRsPerEU("gfx1170"), 1024); // GFX1170, no Feature1536VGPRs
  EXPECT_EQ(getVGPRsPerEU("gfx1200"), 1536); // RDNA4
  EXPECT_EQ(getVGPRsPerEU("gfx1250"), 1536); // GFX1250
}

// --- getWaveSize ---

TEST(AmdArchDbTest, WaveSize) {
  EXPECT_EQ(getWaveSize("gfx906"), 64);  // GCN5_1
  EXPECT_EQ(getWaveSize("gfx908"), 64);  // CDNA1
  EXPECT_EQ(getWaveSize("gfx90a"), 64);  // CDNA2
  EXPECT_EQ(getWaveSize("gfx942"), 64);  // CDNA3
  EXPECT_EQ(getWaveSize("gfx950"), 64);  // CDNA4
  EXPECT_EQ(getWaveSize("gfx1030"), 32); // RDNA2
  EXPECT_EQ(getWaveSize("gfx1100"), 32); // RDNA3
  EXPECT_EQ(getWaveSize("gfx1150"), 32); // RDNA3.5
  EXPECT_EQ(getWaveSize("gfx1170"), 32); // GFX1170
  EXPECT_EQ(getWaveSize("gfx1200"), 32); // RDNA4
  EXPECT_EQ(getWaveSize("gfx1250"), 32); // GFX1250
}

// --- getLDSSize ---

TEST(AmdArchDbTest, LDSSize) {
  EXPECT_EQ(getLDSSize("gfx906"), 65536);   // GCN5_1: 64 KB
  EXPECT_EQ(getLDSSize("gfx90a"), 65536);   // CDNA2: 64 KB
  EXPECT_EQ(getLDSSize("gfx942"), 65536);   // CDNA3: 64 KB
  EXPECT_EQ(getLDSSize("gfx950"), 163840);  // CDNA4: 160 KB
  EXPECT_EQ(getLDSSize("gfx1030"), 65536);  // RDNA2: 64 KB
  EXPECT_EQ(getLDSSize("gfx1100"), 65536);  // RDNA3: 64 KB
  EXPECT_EQ(getLDSSize("gfx1170"), 65536);  // GFX1170: 64 KB (gfx11 core)
  EXPECT_EQ(getLDSSize("gfx1200"), 65536);  // RDNA4: 64 KB
  EXPECT_EQ(getLDSSize("gfx1250"), 327680); // GFX1250: 320 KB
}

// --- getLastLevelCacheSize ---

TEST(AmdArchDbTest, LastLevelCacheSize) {
  constexpr int64_t kMiB = 1024 * 1024;
  EXPECT_EQ(getLastLevelCacheSize("gfx906"), 4 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx908"), 8 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx90a"), 8 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx942"), 256 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx950"), 256 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1010"), 4 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1030"), 128 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1100"), 96 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1170"), 1 * kMiB); // APU L2, no MALL
  EXPECT_EQ(getLastLevelCacheSize("gfx1200"), 64 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1250"), 256 * kMiB);
}

TEST(AmdArchDbTest, LastLevelCacheSizeWithTriple) {
  constexpr int64_t kMiB = 1024 * 1024;
  EXPECT_EQ(getLastLevelCacheSize("amdgcn-amd-amdhsa:gfx942"), 256 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("amdgcn-amd-amdhsa:gfx906:xnack-"), 4 * kMiB);
}

// --- getMatrixAccelKind ---

TEST(AmdArchDbTest, MatrixAccelMfma) {
  ArchTestEnv e;
  // gfx906 has v_dot intrinsics but no MFMA/WMMA, so we expect None for every
  // type combination here.
  EXPECT_EQ(getMatrixAccelKind("gfx906", e.f16, e.f16),
            MatrixAccelKind::None); // GCN5_1
  EXPECT_EQ(getMatrixAccelKind("gfx906", e.f32, e.f32),
            MatrixAccelKind::None); // GCN5_1
  EXPECT_EQ(getMatrixAccelKind("gfx906", e.i8, e.i8),
            MatrixAccelKind::None); // GCN5_1
  EXPECT_EQ(getMatrixAccelKind("gfx908", e.f16, e.f16),
            MatrixAccelKind::MFMA); // CDNA1
  EXPECT_EQ(getMatrixAccelKind("gfx90a", e.f16, e.f16),
            MatrixAccelKind::MFMA); // CDNA2
  EXPECT_EQ(getMatrixAccelKind("gfx90a", e.f32, e.f32),
            MatrixAccelKind::MFMA); // CDNA2
  EXPECT_EQ(getMatrixAccelKind("gfx942", e.f16, e.f16),
            MatrixAccelKind::MFMA); // CDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx942", e.f32, e.f32),
            MatrixAccelKind::MFMA); // CDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx942", e.i8, e.i8),
            MatrixAccelKind::MFMA); // CDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx942", e.bf16, e.bf16),
            MatrixAccelKind::MFMA); // CDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f16, e.f16),
            MatrixAccelKind::MFMA); // CDNA4
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f32, e.f32),
            MatrixAccelKind::MFMA); // CDNA4
  EXPECT_EQ(getMatrixAccelKind("gfx90a", e.i8, e.i8),
            MatrixAccelKind::MFMA); // CDNA2
  EXPECT_EQ(getMatrixAccelKind("gfx90a", e.bf16, e.bf16),
            MatrixAccelKind::MFMA); // CDNA2
}

TEST(AmdArchDbTest, MatrixAccelWmma) {
  ArchTestEnv e;
  EXPECT_EQ(getMatrixAccelKind("gfx1100", e.f16, e.f16),
            MatrixAccelKind::WMMA); // RDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx1170", e.f16, e.f16),
            MatrixAccelKind::WMMA); // GFX1170 (WMMA v2)
  EXPECT_EQ(getMatrixAccelKind("gfx1200", e.f16, e.f16),
            MatrixAccelKind::WMMA); // RDNA4
  EXPECT_EQ(getMatrixAccelKind("gfx1250", e.f16, e.f16),
            MatrixAccelKind::WMMA); // GFX1250
  EXPECT_EQ(getMatrixAccelKind("gfx1250", e.f32, e.f32),
            MatrixAccelKind::WMMA); // GFX1250
  EXPECT_EQ(getMatrixAccelKind("gfx1100", e.i8, e.i8),
            MatrixAccelKind::WMMA); // RDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx1170", e.i8, e.i8),
            MatrixAccelKind::WMMA); // GFX1170
  EXPECT_EQ(getMatrixAccelKind("gfx1100", e.f32, e.f32),
            MatrixAccelKind::None); // RDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx1170", e.f32, e.f32),
            MatrixAccelKind::None); // GFX1170 (no f32 WMMA inputs)
  EXPECT_EQ(getMatrixAccelKind("gfx1200", e.f32, e.f32),
            MatrixAccelKind::None); // RDNA4
  // gfx1170 has native fp8 WMMA (WMMA v2); without scales it uses regular WMMA.
  EXPECT_EQ(getMatrixAccelKind("gfx1170", e.f8e4m3fn, e.f8e4m3fn),
            MatrixAccelKind::WMMA); // GFX1170 (fp8 WMMA v2)
  EXPECT_EQ(getMatrixAccelKind("gfx1170", e.f8e4m3fn, e.f8e5m2),
            MatrixAccelKind::WMMA); // GFX1170 (fp8 x bf8)
  EXPECT_EQ(getMatrixAccelKind("gfx1170", e.f8e5m2, e.f8e4m3fn),
            MatrixAccelKind::WMMA); // GFX1170 (bf8 x fp8)
  EXPECT_EQ(getMatrixAccelKind("gfx1170", e.f8e5m2, e.f8e5m2),
            MatrixAccelKind::WMMA); // GFX1170 (bf8 x bf8)
}

// gfx115x (gfx1150/gfx1151) is RDNA3.5: it carries a gfx11 major number just
// like gfx117x, but unlike gfx1170 (GFX1170 / WMMA v2) it is a plain
// RDNA3-class part. This boundary test guards against gfx115x accidentally
// being routed into the gfx117x / RDNA4 matrix path: it must behave as RDNA3
// (WMMA v1, no native fp8 WMMA). The getMatrixAccelKind calls also exercise
// getArch("gfx1150"), which would llvm_unreachable if gfx115x resolved to an
// Unknown family.
TEST(AmdArchDbTest, MatrixAccelRdna35Gfx1150Boundary) {
  ArchTestEnv e;
  // RDNA3-class WMMA v1 for the classic input types (matches gfx1100 above).
  EXPECT_EQ(getMatrixAccelKind("gfx1150", e.f16, e.f16),
            MatrixAccelKind::WMMA); // RDNA3.5
  EXPECT_EQ(getMatrixAccelKind("gfx1150", e.bf16, e.bf16),
            MatrixAccelKind::WMMA); // RDNA3.5
  EXPECT_EQ(getMatrixAccelKind("gfx1150", e.i8, e.i8),
            MatrixAccelKind::WMMA); // RDNA3.5
  EXPECT_EQ(getMatrixAccelKind("gfx1150", e.f32, e.f32),
            MatrixAccelKind::None); // RDNA3.5 (no f32 WMMA inputs)
  // Key boundary vs gfx1170: RDNA3(.5) has no native fp8 WMMA, so fp8 -> None,
  // whereas gfx1170 (WMMA v2) returns WMMA for the same inputs.
  EXPECT_EQ(getMatrixAccelKind("gfx1150", e.f8e4m3fn, e.f8e4m3fn),
            MatrixAccelKind::None); // RDNA3.5 (no fp8 WMMA; gfx1170 has it)
  EXPECT_EQ(getMatrixAccelKind("gfx1150", e.f8e4m3fn, e.f8e5m2),
            MatrixAccelKind::None);
  EXPECT_EQ(getMatrixAccelKind("gfx1150", e.f8e5m2, e.f8e4m3fn),
            MatrixAccelKind::None);
  EXPECT_EQ(getMatrixAccelKind("gfx1150", e.f8e5m2, e.f8e5m2),
            MatrixAccelKind::None);
}

TEST(AmdArchDbTest, MatrixAccelScaledMfma) {
  ArchTestEnv e;
  // ScaledMFMA: gfx950 (CDNA4) with F8/F6/F4 types and scales provided
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f8e4m3fn, e.f8e4m3fn, e.f32, e.f32),
            MatrixAccelKind::ScaledMFMA);
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f8e5m2, e.f8e5m2, e.f32, e.f32),
            MatrixAccelKind::ScaledMFMA);
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f8e4m3fn, e.f8e5m2, e.f32, e.f32),
            MatrixAccelKind::ScaledMFMA); // mixed F8 types
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f6e3m2fn, e.f6e3m2fn, e.f32, e.f32),
            MatrixAccelKind::ScaledMFMA);
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f6e2m3fn, e.f6e2m3fn, e.f32, e.f32),
            MatrixAccelKind::ScaledMFMA);
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f4e2m1fn, e.f4e2m1fn, e.f32, e.f32),
            MatrixAccelKind::ScaledMFMA);

  // Without scales, F8 on gfx950 falls back to regular MFMA
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f8e4m3fn, e.f8e4m3fn),
            MatrixAccelKind::MFMA);

  // ScaledMFMA not available on gfx942 (CDNA3, mfmaVersion < 4)
  EXPECT_NE(getMatrixAccelKind("gfx942", e.f8e4m3fn, e.f8e4m3fn, e.f32, e.f32),
            MatrixAccelKind::ScaledMFMA);

  // Scales with non-F8/F6/F4 types fall back to regular MFMA
  EXPECT_EQ(getMatrixAccelKind("gfx950", e.f16, e.f16, e.f32, e.f32),
            MatrixAccelKind::MFMA);
}

TEST(AmdArchDbTest, MatrixAccelScaledWmma) {
  ArchTestEnv e;
  // ScaledWMMA: gfx1250 (wmmaVersion 3) with supported F8/F4 types + scales
  EXPECT_EQ(getMatrixAccelKind("gfx1250", e.f8e4m3fn, e.f8e4m3fn, e.f32, e.f32),
            MatrixAccelKind::ScaledWMMA);
  EXPECT_EQ(getMatrixAccelKind("gfx1250", e.f8e5m2, e.f8e5m2, e.f32, e.f32),
            MatrixAccelKind::ScaledWMMA);
  // TODO: F4 (E2M1) and F6 (E3M2, E2M3) should also return ScaledWMMA on
  // gfx1250, but the Triton WmmaGroup intrinsic database doesn't have these
  // entries yet. Update these to EXPECT_EQ once upstreamed.
  EXPECT_NE(getMatrixAccelKind("gfx1250", e.f4e2m1fn, e.f4e2m1fn, e.f32, e.f32),
            MatrixAccelKind::ScaledWMMA);
  EXPECT_NE(getMatrixAccelKind("gfx1250", e.f6e3m2fn, e.f6e3m2fn, e.f32, e.f32),
            MatrixAccelKind::ScaledWMMA);
  EXPECT_NE(getMatrixAccelKind("gfx1250", e.f6e2m3fn, e.f6e2m3fn, e.f32, e.f32),
            MatrixAccelKind::ScaledWMMA);

  // ScaledWMMA not available on gfx1100/gfx1200 (wmmaVersion < 3)
  EXPECT_NE(getMatrixAccelKind("gfx1100", e.f8e4m3fn, e.f8e4m3fn, e.f32, e.f32),
            MatrixAccelKind::ScaledWMMA);
  EXPECT_NE(getMatrixAccelKind("gfx1200", e.f8e4m3fn, e.f8e4m3fn, e.f32, e.f32),
            MatrixAccelKind::ScaledWMMA);

  // Without scales, F8 on gfx1250 falls back to regular WMMA
  EXPECT_EQ(getMatrixAccelKind("gfx1250", e.f8e4m3fn, e.f8e4m3fn),
            MatrixAccelKind::WMMA);

  // Scales with non-F8/F4 types fall back to regular WMMA
  EXPECT_EQ(getMatrixAccelKind("gfx1250", e.f16, e.f16, e.f32, e.f32),
            MatrixAccelKind::WMMA);
}

// --- supportsTDM ---

TEST(AmdArchDbTest, SupportsTDM) {
  EXPECT_FALSE(supportsTDM("gfx906"));  // GCN5_1
  EXPECT_FALSE(supportsTDM("gfx908"));  // CDNA1
  EXPECT_FALSE(supportsTDM("gfx90a"));  // CDNA2
  EXPECT_FALSE(supportsTDM("gfx942"));  // CDNA3
  EXPECT_FALSE(supportsTDM("gfx950"));  // CDNA4
  EXPECT_FALSE(supportsTDM("gfx1010")); // RDNA1
  EXPECT_FALSE(supportsTDM("gfx1030")); // RDNA2
  EXPECT_FALSE(supportsTDM("gfx1100")); // RDNA3
  EXPECT_FALSE(supportsTDM("gfx1170")); // GFX1170
  EXPECT_FALSE(supportsTDM("gfx1200")); // RDNA4
  EXPECT_TRUE(supportsTDM("gfx1250"));  // GFX1250
}

// --- getArch (arch-string parsing) ---

TEST(AmdArchDbTest, GetArchChipParsingGfx1170) {
  // The ISAFamily component of getArch() is exercised by every gfx1170 test in
  // this file: if "gfx1170" resolved to Unknown (or the wrong family), those
  // would llvm_unreachable or return wrong values. Here we lock the chip-name
  // parsing for the arch-string forms that appear as rock.arch attributes.
  EXPECT_EQ(std::get<1>(getArch("gfx1170")), "gfx1170");
  EXPECT_EQ(std::get<1>(getArch("amdgcn-amd-amdhsa:gfx1170")), "gfx1170");
  EXPECT_EQ(std::get<1>(getArch("amdgcn-amd-amdhsa:gfx1170:xnack-")),
            "gfx1170");
}

// --- supportsMultiCTALaunch / getMaxNumCTAs ---

TEST(AmdArchDbTest, MultiCTALaunch) {
  // Cluster / multi-CTA launch is gfx1250-only. gfx1170 is a gfx11-based core
  // and does not support it (TargetFeatures::supportsMultiCTALaunch ==
  // isGFX1250).
  EXPECT_FALSE(supportsMultiCTALaunch("gfx1100")); // RDNA3
  EXPECT_FALSE(supportsMultiCTALaunch("gfx1170")); // GFX1170
  EXPECT_FALSE(supportsMultiCTALaunch("gfx1200")); // RDNA4
  EXPECT_TRUE(supportsMultiCTALaunch("gfx1250"));  // GFX1250

  EXPECT_EQ(getMaxNumCTAs("gfx1170"), 1);  // no multi-CTA -> 1
  EXPECT_EQ(getMaxNumCTAs("gfx1250"), 16); // multi-CTA -> 16
}

// --- getMaxKpack ---

TEST(AmdArchDbTest, MaxKpack) {
  // kpack in {1,2} for gfx9<gfx950 and all of gfx10/gfx11/gfx12<gfx1250;
  // kpack==1 only on gfx950+, gfx1250+. gfx1170 (0x1170) is in the gfx11 range.
  EXPECT_EQ(getMaxKpack("gfx908"), 2);  // CDNA1
  EXPECT_EQ(getMaxKpack("gfx950"), 1);  // CDNA4
  EXPECT_EQ(getMaxKpack("gfx1100"), 2); // RDNA3
  EXPECT_EQ(getMaxKpack("gfx1170"), 2); // GFX1170
  EXPECT_EQ(getMaxKpack("gfx1200"), 2); // RDNA4
  EXPECT_EQ(getMaxKpack("gfx1250"), 1); // GFX1250
}

// --- Dtype overload of isFastAtomicMaxSupported ---
//
// A thin adapter over the Type-based overload; the tests below mirror a subset
// of the Type-overload cases above and serve to catch regressions in the
// internal Dtype -> mlir::Type mapping.

TEST(AmdArchDbTest, FastAtomicMaxDtypeF32) {
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx906", Dtype::F32)); // GCN5_1
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx950", Dtype::F32)); // CDNA4
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1100", Dtype::F32)); // RDNA3
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1170", Dtype::F32)); // GFX1170
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1250", Dtype::F32)); // GFX1250
}

TEST(AmdArchDbTest, FastAtomicMaxDtypeNonF32Unsupported) {
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1100", Dtype::F16));  // RDNA3
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1250", Dtype::F16));  // GFX1250
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1100", Dtype::BF16)); // RDNA3
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1250", Dtype::BF16)); // GFX1250
}

// --- archSupportsAccelFp8 ---
//
// Implemented via getMatrixAccelKind, so this reflects what selectFor reports
// for FP8 (FNUZ or OCP) input pairs without scales: any MFMA or WMMA result is
// taken to mean the arch has fp8 matrix-acceleration intrinsics.

TEST(AmdArchDbTest, ArchSupportsAccelFp8) {
  EXPECT_FALSE(archSupportsAccelFp8("gfx906"));  // GCN5_1: no matrix accel
  EXPECT_FALSE(archSupportsAccelFp8("gfx908"));  // CDNA1: MFMA but no FP8 MFMA
  EXPECT_FALSE(archSupportsAccelFp8("gfx90a"));  // CDNA2: MFMA but no FP8 MFMA
  EXPECT_TRUE(archSupportsAccelFp8("gfx942"));   // CDNA3 (native fp8 MFMA)
  EXPECT_TRUE(archSupportsAccelFp8("gfx950"));   // CDNA4 (native fp8 MFMA)
  EXPECT_FALSE(archSupportsAccelFp8("gfx1010")); // RDNA1: no WMMA at all
  EXPECT_FALSE(archSupportsAccelFp8("gfx1030")); // RDNA2: no WMMA at all
  EXPECT_FALSE(archSupportsAccelFp8("gfx1100")); // RDNA3: no fp8 WMMA
  EXPECT_TRUE(archSupportsAccelFp8("gfx1170"));  // GFX1170 (native fp8 WMMA v2)
  EXPECT_TRUE(archSupportsAccelFp8("gfx1200"));  // RDNA4 (native fp8 WMMA)
  EXPECT_TRUE(archSupportsAccelFp8("gfx1250"));  // GFX1250 (native fp8 WMMA)
}

TEST(AmdArchDbTest, ArchSupportsAccelFp8WithTriple) {
  EXPECT_TRUE(archSupportsAccelFp8("amdgcn-amd-amdhsa:gfx942"));
  EXPECT_FALSE(archSupportsAccelFp8("amdgcn-amd-amdhsa:gfx906"));
}

// --- archSupportsScaledGemm ---
//
// Scaled matrix acceleration: ScaledMFMA on CDNA4 (gfx950), ScaledWMMA on
// gfx1250. Implemented via getMatrixAccelKind by probing fp8 input
// pairs together with a non-null scale type.

TEST(AmdArchDbTest, ArchSupportsScaledGemm) {
  EXPECT_FALSE(archSupportsScaledGemm("gfx906"));  // GCN5_1
  EXPECT_FALSE(archSupportsScaledGemm("gfx908"));  // CDNA1
  EXPECT_FALSE(archSupportsScaledGemm("gfx90a"));  // CDNA2
  EXPECT_FALSE(archSupportsScaledGemm("gfx942"));  // CDNA3 (no ScaledMFMA)
  EXPECT_TRUE(archSupportsScaledGemm("gfx950"));   // CDNA4 (ScaledMFMA)
  EXPECT_FALSE(archSupportsScaledGemm("gfx1010")); // RDNA1
  EXPECT_FALSE(archSupportsScaledGemm("gfx1030")); // RDNA2
  EXPECT_FALSE(archSupportsScaledGemm("gfx1100")); // RDNA3
  EXPECT_FALSE(archSupportsScaledGemm("gfx1170")); // GFX1170 (WMMA v2 < 3)
  EXPECT_FALSE(archSupportsScaledGemm("gfx1200")); // RDNA4 (no ScaledWMMA)
  EXPECT_TRUE(archSupportsScaledGemm("gfx1250"));  // GFX1250 (ScaledWMMA)
}

TEST(AmdArchDbTest, ArchSupportsScaledGemmWithTriple) {
  EXPECT_TRUE(archSupportsScaledGemm("amdgcn-amd-amdhsa:gfx950"));
  EXPECT_FALSE(archSupportsScaledGemm("amdgcn-amd-amdhsa:gfx942"));
}

// --- archSupportsNonKPackedScaledInput ---
//
// Only CDNA4 (gfx950) can lower a scaled fp4 dot whose operand is packed along
// the non-K (M/N) dimension, via its transpose-load repack.

TEST(AmdArchDbTest, ArchSupportsNonKPackedScaledInput) {
  EXPECT_FALSE(archSupportsNonKPackedScaledInput("gfx906"));  // GCN5_1
  EXPECT_FALSE(archSupportsNonKPackedScaledInput("gfx908"));  // CDNA1
  EXPECT_FALSE(archSupportsNonKPackedScaledInput("gfx90a"));  // CDNA2
  EXPECT_FALSE(archSupportsNonKPackedScaledInput("gfx942"));  // CDNA3
  EXPECT_TRUE(archSupportsNonKPackedScaledInput("gfx950"));   // CDNA4
  EXPECT_FALSE(archSupportsNonKPackedScaledInput("gfx1100")); // RDNA3
  EXPECT_FALSE(archSupportsNonKPackedScaledInput("gfx1200")); // RDNA4
  EXPECT_FALSE(archSupportsNonKPackedScaledInput("gfx1250")); // GFX1250 (WMMA)
}

TEST(AmdArchDbTest, ArchSupportsNonKPackedScaledInputWithTriple) {
  EXPECT_TRUE(archSupportsNonKPackedScaledInput("amdgcn-amd-amdhsa:gfx950"));
  EXPECT_FALSE(archSupportsNonKPackedScaledInput("amdgcn-amd-amdhsa:gfx1250"));
}

// --- gfx906-specific arch-string forms ---

// Verify that triple- and feature-suffixed forms of the gfx906 arch string
// (the kind that show up as rock.arch attributes) all parse correctly and
// dispatch to the GCN5_1 code paths.
TEST(AmdArchDbTest, Gfx906ArchStringForms) {
  EXPECT_EQ(getMinNumCU("amdgcn-amd-amdhsa:gfx906"), 10);
  EXPECT_EQ(getMinNumCU("amdgcn-amd-amdhsa:gfx906:xnack-"), 10);
  EXPECT_EQ(getMaxWavesPerEU("amdgcn-amd-amdhsa:gfx906"), 8);
  EXPECT_EQ(getWaveSize("amdgcn-amd-amdhsa:gfx906"), 64);
}
