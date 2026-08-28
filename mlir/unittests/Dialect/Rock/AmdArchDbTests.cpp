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
  Type f8e4m3fn, f8e4m3fnuz, f8e5m2, f6e3m2fn, f6e2m3fn, f4e2m1fn;

  ArchTestEnv() : b(&ctx) {
    f32 = b.getF32Type();
    f16 = b.getF16Type();
    bf16 = b.getBF16Type();
    i32 = b.getI32Type();
    i8 = b.getIntegerType(8);
    f8e4m3fn = Float8E4M3FNType::get(&ctx);
    f8e4m3fnuz = Float8E4M3FNUZType::get(&ctx);
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
  EXPECT_EQ(inferNumChiplets("gfx942", 228), 6); // MI300A
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

// This is a hard floor: getNumCUOnFunc rejects a smaller rock.num_cu, so a
// family that spans APUs has to admit its smallest APU.
TEST(AmdArchDbTest, MinNumCU) {
  EXPECT_EQ(getMinNumCU("gfx906"), 10);   // GCN5_1
  EXPECT_EQ(getMinNumCU("gfx908"), 120);  // CDNA1
  EXPECT_EQ(getMinNumCU("gfx90a"), 104);  // CDNA2
  EXPECT_EQ(getMinNumCU("gfx942"), 20);   // CDNA3
  EXPECT_EQ(getMinNumCU("gfx950"), 256);  // CDNA4
  EXPECT_EQ(getMinNumCU("gfx1010"), 20);  // RDNA1
  EXPECT_EQ(getMinNumCU("gfx1030"), 2);   // RDNA2
  EXPECT_EQ(getMinNumCU("gfx1100"), 2);   // RDNA3
  EXPECT_EQ(getMinNumCU("gfx1170"), 2);   // GFX1170
  EXPECT_EQ(getMinNumCU("gfx1200"), 12);  // RDNA4
  EXPECT_EQ(getMinNumCU("gfx1250"), 256); // GFX1250
  // The RDNA2 APUs the old family floor of 30 rejected outright.
  EXPECT_LE(getMinNumCU("gfx1036"), 2); // 2 CUs
  EXPECT_LE(getMinNumCU("gfx1033"), 8); // 8 CUs
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
  // GFX1250 has Feature1024AddressableVGPRs, not Feature1536VGPRs.
  EXPECT_EQ(getVGPRsPerEU("gfx1250"), 1024);
  EXPECT_EQ(getVGPRsPerEU("gfx1251"), 1024);
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
  EXPECT_EQ(getLastLevelCacheSize("gfx1170"), 1 * kMiB); // guess, see below
  EXPECT_EQ(getLastLevelCacheSize("gfx1200"), 32 * kMiB);
  // gfx125x has no Infinity Cache; its last level is 2x96 MiB of L2 on the
  // Fabric Cache Dies.
  EXPECT_EQ(getLastLevelCacheSize("gfx1250"), 192 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1251"), 192 * kMiB);
}

// Within one ISA family the discrete parts differ from each other and, above
// all, from the APUs: an APU's L2 is its last level cache unless it has a MALL.
// Pinned per chip so that a family-wide value can never creep back in.
TEST(AmdArchDbTest, LastLevelCacheSizePerChip) {
  constexpr int64_t kMiB = 1024 * 1024;
  // RDNA1, no MALL in the family.
  EXPECT_EQ(getLastLevelCacheSize("gfx1012"), 2 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1013"), 4 * kMiB);
  // RDNA2: Infinity Cache on the discrete parts, L2 only on the APUs.
  EXPECT_EQ(getLastLevelCacheSize("gfx1031"), 96 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1032"), 32 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1034"), 16 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1033"), 1 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1035"), 2 * kMiB);
  // 2 CU part: 256 KiB, the smallest last level cache of any target.
  EXPECT_EQ(getLastLevelCacheSize("gfx1036"), 256 * 1024);
  // RDNA3 / RDNA3.5. gfx1151 is the only APU here known to have a MALL.
  EXPECT_EQ(getLastLevelCacheSize("gfx1101"), 64 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1102"), 32 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1103"), 2 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1150"), 2 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1151"), 32 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1152"), 1 * kMiB);
  // gfx1153's L2 is unpublished; estimated from gfx1152, which bounds it from
  // above.
  EXPECT_EQ(getLastLevelCacheSize("gfx1153"), 1 * kMiB);
  // RDNA4.
  EXPECT_EQ(getLastLevelCacheSize("gfx1201"), 64 * kMiB);
}

// The gfx117x caches are unpublished, so they fall back to their ISA family,
// whose value is a guess at a bare APU L2. Pinned so that the guess is a
// deliberate choice: if any of these turns out to carry a MALL, as the APU
// gfx1151 does, the real value is much larger. These and gfx1153, pinned
// above, are the only chips not on a published number.
TEST(AmdArchDbTest, LastLevelCacheSizeUnpublishedChips) {
  constexpr int64_t kMiB = 1024 * 1024;
  EXPECT_EQ(getLastLevelCacheSize("gfx1170"), 1 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1171"), 1 * kMiB);
  EXPECT_EQ(getLastLevelCacheSize("gfx1172"), 1 * kMiB);
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

// Tests for getAccelInstrMinKDim

// Unwraps the query so the expected K extents below stay readable. Only for
// cases that must succeed; the failing ones are checked with failed() directly.
static int64_t minKDim(StringRef arch, Type inputTypeA, Type inputTypeB,
                       uint32_t instrNonKDim) {
  FailureOr<int64_t> kDim =
      getAccelInstrMinKDim(arch, inputTypeA, inputTypeB, instrNonKDim);
  EXPECT_TRUE(succeeded(kDim)) << arch << " has no matrix instruction at "
                               << instrNonKDim << "x" << instrNonKDim;
  return succeeded(kDim) ? *kDim : 0;
}

TEST(AmdArchDbTest, AccelInstrMinKDimMfma) {
  ArchTestEnv e;
  // CDNA3 f16: mfma_f32_16x16x16f16 at a 16x16 instruction tile,
  // mfma_f32_32x32x8f16 at 32x32. The tuner sweeps matrixInstrNonkdim over
  // {16,32} and keeps the smaller of the two, so 8 is the bar a kPerBlock has
  // to clear for an f16 GEMM on gfx942.
  EXPECT_EQ(minKDim("gfx942", e.f16, e.f16, 16), 16);
  EXPECT_EQ(minKDim("gfx942", e.f16, e.f16, 32), 8);
  // bf16 selects the _1k intrinsics on CDNA3, which have the same K extents.
  EXPECT_EQ(minKDim("gfx942", e.bf16, e.bf16, 16), 16);
  EXPECT_EQ(minKDim("gfx942", e.bf16, e.bf16, 32), 8);
  // i8 consumes twice the K of f16 at both tiles (mfma_i32_16x16x32_i8 /
  // mfma_i32_32x32x16_i8).
  EXPECT_EQ(minKDim("gfx942", e.i8, e.i8, 16), 32);
  EXPECT_EQ(minKDim("gfx942", e.i8, e.i8, 32), 16);
  // The f32 MFMAs are very narrow (mfma_f32_16x16x4f32 / mfma_f32_32x32x2f32),
  // so they barely constrain kPerBlock at all.
  EXPECT_EQ(minKDim("gfx942", e.f32, e.f32, 16), 4);
  EXPECT_EQ(minKDim("gfx942", e.f32, e.f32, 32), 2);
  // CDNA1 / CDNA2 carry the same f16 intrinsics as CDNA3.
  EXPECT_EQ(minKDim("gfx908", e.f16, e.f16, 16), 16);
  EXPECT_EQ(minKDim("gfx90a", e.f16, e.f16, 32), 8);
  // A triple-prefixed arch string resolves identically.
  EXPECT_EQ(minKDim("amdgcn-amd-amdhsa:gfx942", e.f16, e.f16, 32), 8);
}

// CDNA4 lists two intrinsics per (tile, type) -- a wide one and the CDNA3-era
// narrow one, widest first. The probe asks selectFor() for an input K of zero
// precisely so that it skips every candidate and falls through to the narrow
// end of that list, which is the property the heuristic rests on: a kPerBlock
// only has to be a multiple of the *narrowest* instruction to fill some
// instruction. If selectFor() ever stops falling through, these values flip to
// the wide ones and the tuning space silently loses candidates.
TEST(AmdArchDbTest, AccelInstrMinKDimMfmaFallsThroughToNarrowest) {
  ArchTestEnv e;
  // f16 at 16x16: {mfma_f32_16x16x32_f16 (K=32), mfma_f32_16x16x16f16 (K=16)}.
  EXPECT_EQ(minKDim("gfx950", e.f16, e.f16, 16), 16);
  // f16 at 32x32: {mfma_f32_32x32x16_f16 (K=16), mfma_f32_32x32x8f16 (K=8)}.
  EXPECT_EQ(minKDim("gfx950", e.f16, e.f16, 32), 8);
  // i8 at 16x16: {mfma_i32_16x16x64_i8 (K=64), mfma_i32_16x16x32_i8 (K=32)}.
  EXPECT_EQ(minKDim("gfx950", e.i8, e.i8, 16), 32);
  // i8 at 32x32: {mfma_i32_32x32x32_i8 (K=32), mfma_i32_32x32x16_i8 (K=16)}.
  EXPECT_EQ(minKDim("gfx950", e.i8, e.i8, 32), 16);
}

TEST(AmdArchDbTest, AccelInstrMinKDimWmma) {
  ArchTestEnv e;
  // All WMMA instructions are 16x16, so instrNonKDim makes no difference.
  EXPECT_EQ(minKDim("gfx1100", e.f16, e.f16, 16), 16); // RDNA3
  EXPECT_EQ(minKDim("gfx1100", e.f16, e.f16, 32), 16); // RDNA3
  EXPECT_EQ(minKDim("gfx1170", e.f16, e.f16, 16), 16); // WMMA v2
  EXPECT_EQ(minKDim("gfx1200", e.f16, e.f16, 16), 16); // RDNA4
  // gfx1250 doubles the f16 instruction to wmma_f32_16x16x32_f16.
  EXPECT_EQ(minKDim("gfx1250", e.f16, e.f16, 16), 32);
  EXPECT_EQ(minKDim("gfx1100", e.i8, e.i8, 16), 16); // iu8 16
  EXPECT_EQ(minKDim("gfx1250", e.i8, e.i8, 16), 64); // iu8 64
  // gfx1250 is the only arch with an f32 WMMA input path
  // (wmma_f32_16x16x4_f32).
  EXPECT_EQ(minKDim("gfx1250", e.f32, e.f32, 16), 4);
}

TEST(AmdArchDbTest, AccelInstrMinKDimNoAccel) {
  ArchTestEnv e;
  // No matrix core at all.
  EXPECT_TRUE(failed(getAccelInstrMinKDim("gfx906", e.f16, e.f16, 16)));
  EXPECT_TRUE(failed(getAccelInstrMinKDim("gfx1010", e.f16, e.f16, 16)));
  // A matrix core that has no instruction for these inputs: RDNA3 WMMA takes no
  // f32 operands. This is the FMA path, which the widened kPerBlock range
  // treats as having no instruction-alignment requirement.
  EXPECT_TRUE(failed(getAccelInstrMinKDim("gfx1100", e.f32, e.f32, 16)));
}

// Triton's composeMfmaKeyFor silently rewrites OCP FP8 (E4M3FN / E5M2) inputs
// to f16 on any MFMA v<=3 (see MfmaGroup.cpp, and the note on
// archSupportsAccelFp8 below), so on CDNA3 an OCP-spelled fp8 GEMM is held to
// f16's K extents rather than to those of the native FNUZ intrinsics.
TEST(AmdArchDbTest, AccelInstrMinKDimOcpFp8EmulatedAsF16OnCdna3) {
  ArchTestEnv e;
  // Native FNUZ fp8: mfma_f32_16x16x32_fp8_fp8 / mfma_f32_32x32x16_fp8_fp8.
  EXPECT_EQ(minKDim("gfx942", e.f8e4m3fnuz, e.f8e4m3fnuz, 16), 32);
  EXPECT_EQ(minKDim("gfx942", e.f8e4m3fnuz, e.f8e4m3fnuz, 32), 16);
  // Same arch, OCP spelling: emulated with f16, so it reports f16's extents.
  EXPECT_EQ(minKDim("gfx942", e.f8e4m3fn, e.f8e4m3fn, 16), 16);
  EXPECT_EQ(minKDim("gfx942", e.f8e4m3fn, e.f8e4m3fn, 32), 8);
  // CDNA4 has native OCP fp8, so no substitution happens there.
  EXPECT_EQ(minKDim("gfx950", e.f8e4m3fn, e.f8e4m3fn, 16), 32);
  EXPECT_EQ(minKDim("gfx950", e.f8e4m3fn, e.f8e4m3fn, 32), 16);
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

// --- hasTanhInsts ---
//
// Mirrors LLVM's FeatureTanhInsts (v_tanh_f32/f16/bf16), which only gfx1250
// and later carry.

TEST(AmdArchDbTest, HasTanhInsts) {
  EXPECT_FALSE(hasTanhInsts("gfx906"));  // GCN5_1
  EXPECT_FALSE(hasTanhInsts("gfx908"));  // CDNA1
  EXPECT_FALSE(hasTanhInsts("gfx90a"));  // CDNA2
  EXPECT_FALSE(hasTanhInsts("gfx942"));  // CDNA3
  EXPECT_FALSE(hasTanhInsts("gfx950"));  // CDNA4
  EXPECT_FALSE(hasTanhInsts("gfx1010")); // RDNA1
  EXPECT_FALSE(hasTanhInsts("gfx1030")); // RDNA2
  EXPECT_FALSE(hasTanhInsts("gfx1100")); // RDNA3
  EXPECT_FALSE(hasTanhInsts("gfx1170")); // GFX1170
  EXPECT_FALSE(hasTanhInsts("gfx1200")); // RDNA4
  EXPECT_TRUE(hasTanhInsts("gfx1250"));  // GFX1250
}

TEST(AmdArchDbTest, HasTanhInstsWithTriple) {
  EXPECT_TRUE(hasTanhInsts("amdgcn-amd-amdhsa:gfx1250"));
  EXPECT_FALSE(hasTanhInsts("amdgcn-amd-amdhsa:gfx942"));
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

// --- preferBf16x3ForF32Dot ---

TEST(AmdArchDbTest, PreferBf16x3ForF32Dot) {
  // Only CDNA4 has been measured to win from the 3xBF16 decomposition.
  EXPECT_FALSE(preferBf16x3ForF32Dot("gfx906"));  // GCN5_1
  EXPECT_FALSE(preferBf16x3ForF32Dot("gfx908"));  // CDNA1
  EXPECT_FALSE(preferBf16x3ForF32Dot("gfx90a"));  // CDNA2
  EXPECT_FALSE(preferBf16x3ForF32Dot("gfx942"));  // CDNA3
  EXPECT_TRUE(preferBf16x3ForF32Dot("gfx950"));   // CDNA4
  EXPECT_FALSE(preferBf16x3ForF32Dot("gfx1100")); // RDNA3
  EXPECT_FALSE(preferBf16x3ForF32Dot("gfx1170")); // GFX1170
  EXPECT_FALSE(preferBf16x3ForF32Dot("gfx1200")); // RDNA4
  EXPECT_FALSE(preferBf16x3ForF32Dot("gfx1250")); // GFX1250

  // Full target triples resolve the same way as bare chip names.
  EXPECT_TRUE(
      preferBf16x3ForF32Dot("amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-"));
  EXPECT_FALSE(preferBf16x3ForF32Dot("amdgcn-amd-amdhsa:gfx942"));
}
