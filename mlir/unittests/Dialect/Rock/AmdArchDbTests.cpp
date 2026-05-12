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

// --- isFastAtomicAddSupported ---

TEST(AmdArchDbTest, FastAtomicAddF32) {
  ArchTestEnv e;
  EXPECT_TRUE(isFastAtomicAddSupported("gfx908", e.f32));  // CDNA1
  EXPECT_TRUE(isFastAtomicAddSupported("gfx90a", e.f32));  // CDNA2
  EXPECT_TRUE(isFastAtomicAddSupported("gfx942", e.f32));  // CDNA3
  EXPECT_TRUE(isFastAtomicAddSupported("gfx950", e.f32));  // CDNA4
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1010", e.f32)); // RDNA1
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1030", e.f32)); // RDNA2
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1100", e.f32)); // RDNA3
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1200", e.f32)); // RDNA4
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1250", e.f32)); // GFX1250
}

TEST(AmdArchDbTest, FastAtomicAddF16) {
  ArchTestEnv e;
  EXPECT_TRUE(isFastAtomicAddSupported("gfx908", e.f16));   // CDNA1
  EXPECT_TRUE(isFastAtomicAddSupported("gfx90a", e.f16));   // CDNA2
  EXPECT_TRUE(isFastAtomicAddSupported("gfx942", e.f16));   // CDNA3
  EXPECT_TRUE(isFastAtomicAddSupported("gfx950", e.f16));   // CDNA4
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1010", e.f16)); // RDNA1
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1030", e.f16)); // RDNA2
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1100", e.f16)); // RDNA3
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1200", e.f16));  // RDNA4
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1250", e.f16));  // GFX1250
}

TEST(AmdArchDbTest, FastAtomicAddBf16) {
  ArchTestEnv e;
  EXPECT_FALSE(isFastAtomicAddSupported("gfx90a", e.bf16));  // CDNA2
  EXPECT_FALSE(isFastAtomicAddSupported("gfx942", e.bf16));  // CDNA3
  EXPECT_TRUE(isFastAtomicAddSupported("gfx950", e.bf16));   // CDNA4
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1010", e.bf16)); // RDNA1
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1030", e.bf16)); // RDNA2
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1100", e.bf16)); // RDNA3
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1200", e.bf16));  // RDNA4
  EXPECT_TRUE(isFastAtomicAddSupported("gfx1250", e.bf16));  // GFX1250
}

TEST(AmdArchDbTest, FastAtomicAddIntUnsupported) {
  ArchTestEnv e;
  EXPECT_FALSE(isFastAtomicAddSupported("gfx90a", e.i32));  // CDNA2
  EXPECT_FALSE(isFastAtomicAddSupported("gfx942", e.i32));  // CDNA3
  EXPECT_FALSE(isFastAtomicAddSupported("gfx950", e.i32));  // CDNA4
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1100", e.i32)); // RDNA3
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1200", e.i32)); // RDNA4
  EXPECT_FALSE(isFastAtomicAddSupported("gfx1250", e.i32)); // GFX1250
}

TEST(AmdArchDbTest, FastAtomicAddWithTriple) {
  ArchTestEnv e;
  EXPECT_TRUE(isFastAtomicAddSupported("amdgcn-amd-amdhsa:gfx90a", e.f32));
}

// --- isFastAtomicMaxSupported ---

TEST(AmdArchDbTest, FastAtomicMaxF32) {
  ArchTestEnv e;
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx908", e.f32)); // CDNA1
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx90a", e.f32)); // CDNA2
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx950", e.f32)); // CDNA4
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1010", e.f32)); // RDNA1
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1030", e.f32)); // RDNA2
  EXPECT_TRUE(isFastAtomicMaxSupported("gfx1100", e.f32)); // RDNA3
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
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1200", e.f16)); // RDNA4
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx1250", e.f16)); // GFX1250
}

// --- getMaxNumChiplets ---

TEST(AmdArchDbTest, MaxNumChiplets) {
  EXPECT_EQ(getMaxNumChiplets("gfx908"), 1);  // CDNA1
  EXPECT_EQ(getMaxNumChiplets("gfx90a"), 1);  // CDNA2
  EXPECT_EQ(getMaxNumChiplets("gfx942"), 8);  // CDNA3
  EXPECT_EQ(getMaxNumChiplets("gfx950"), 8);  // CDNA4
  EXPECT_EQ(getMaxNumChiplets("gfx1010"), 1); // RDNA1
  EXPECT_EQ(getMaxNumChiplets("gfx1030"), 1); // RDNA2
  EXPECT_EQ(getMaxNumChiplets("gfx1100"), 1); // RDNA3
  EXPECT_EQ(getMaxNumChiplets("gfx1200"), 1); // RDNA4
  EXPECT_EQ(getMaxNumChiplets("gfx1250"), 8); // GFX1250
}

// --- getMinNumCU ---

TEST(AmdArchDbTest, MinNumCU) {
  EXPECT_EQ(getMinNumCU("gfx908"), 120);  // CDNA1
  EXPECT_EQ(getMinNumCU("gfx90a"), 104);  // CDNA2
  EXPECT_EQ(getMinNumCU("gfx942"), 20);   // CDNA3
  EXPECT_EQ(getMinNumCU("gfx950"), 256);  // CDNA4
  EXPECT_EQ(getMinNumCU("gfx1010"), 30);  // RDNA1
  EXPECT_EQ(getMinNumCU("gfx1030"), 30);  // RDNA2
  EXPECT_EQ(getMinNumCU("gfx1100"), 2);   // RDNA3
  EXPECT_EQ(getMinNumCU("gfx1200"), 12);  // RDNA4
  EXPECT_EQ(getMinNumCU("gfx1250"), 256); // GFX1250
}

// --- getMaxWavesPerEU ---

TEST(AmdArchDbTest, MaxWavesPerEU) {
  EXPECT_EQ(getMaxWavesPerEU("gfx908"), 8);   // CDNA1
  EXPECT_EQ(getMaxWavesPerEU("gfx90a"), 8);   // CDNA2
  EXPECT_EQ(getMaxWavesPerEU("gfx942"), 8);   // CDNA3
  EXPECT_EQ(getMaxWavesPerEU("gfx950"), 8);   // CDNA4
  EXPECT_EQ(getMaxWavesPerEU("gfx1010"), 16); // RDNA1
  EXPECT_EQ(getMaxWavesPerEU("gfx1030"), 16); // RDNA2
  EXPECT_EQ(getMaxWavesPerEU("gfx1100"), 16); // RDNA3
  EXPECT_EQ(getMaxWavesPerEU("gfx1200"), 16); // RDNA4
  EXPECT_EQ(getMaxWavesPerEU("gfx1250"), 16); // GFX1250
}

// --- getWaveSize ---

TEST(AmdArchDbTest, WaveSize) {
  EXPECT_EQ(getWaveSize("gfx908"), 64);  // CDNA1
  EXPECT_EQ(getWaveSize("gfx90a"), 64);  // CDNA2
  EXPECT_EQ(getWaveSize("gfx942"), 64);  // CDNA3
  EXPECT_EQ(getWaveSize("gfx950"), 64);  // CDNA4
  EXPECT_EQ(getWaveSize("gfx1030"), 32); // RDNA2
  EXPECT_EQ(getWaveSize("gfx1100"), 32); // RDNA3
  EXPECT_EQ(getWaveSize("gfx1200"), 32); // RDNA4
  EXPECT_EQ(getWaveSize("gfx1250"), 32); // GFX1250
}

// --- getLDSSize ---

TEST(AmdArchDbTest, LDSSize) {
  EXPECT_EQ(getLDSSize("gfx90a"), 65536);   // CDNA2: 64 KB
  EXPECT_EQ(getLDSSize("gfx942"), 65536);   // CDNA3: 64 KB
  EXPECT_EQ(getLDSSize("gfx950"), 163840);  // CDNA4: 160 KB
  EXPECT_EQ(getLDSSize("gfx1030"), 65536);  // RDNA2: 64 KB
  EXPECT_EQ(getLDSSize("gfx1100"), 65536);  // RDNA3: 64 KB
  EXPECT_EQ(getLDSSize("gfx1200"), 65536);  // RDNA4: 64 KB
  EXPECT_EQ(getLDSSize("gfx1250"), 327680); // GFX1250: 320 KB
}

// --- getMatrixAccelKind ---

TEST(AmdArchDbTest, MatrixAccelMfma) {
  ArchTestEnv e;
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
  EXPECT_EQ(getMatrixAccelKind("gfx1200", e.f16, e.f16),
            MatrixAccelKind::WMMA); // RDNA4
  EXPECT_EQ(getMatrixAccelKind("gfx1250", e.f16, e.f16),
            MatrixAccelKind::WMMA); // GFX1250
  EXPECT_EQ(getMatrixAccelKind("gfx1250", e.f32, e.f32),
            MatrixAccelKind::WMMA); // GFX1250
  EXPECT_EQ(getMatrixAccelKind("gfx1100", e.i8, e.i8),
            MatrixAccelKind::WMMA); // RDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx1100", e.f32, e.f32),
            MatrixAccelKind::None); // RDNA3
  EXPECT_EQ(getMatrixAccelKind("gfx1200", e.f32, e.f32),
            MatrixAccelKind::None); // RDNA4
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

// --- computeLdsBoundWavesPerEU ---
//
// The formula is
//   blocks_per_lds_unit = floor(LDS_per_lds_unit / kernel_LDS_bytes)
//   waves_per_block     = max(1, block_size / wave_size)
//   waves_per_lds_unit  = blocks_per_lds_unit * waves_per_block
//   waves_per_EU        = waves_per_lds_unit / 4 (EUs/SIMDs per LDS unit)
// clamped to [1, getMaxWavesPerEU(arch)].

TEST(AmdArchDbTest, LdsBoundNoLdsPressureReturnsArchMax) {
  // kernelLdsBytes == 0 -> only the per-arch ceiling applies.
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx908", 0, 256), 8);   // CDNA1
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 0, 256), 8);   // CDNA2
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx942", 0, 256), 8);   // CDNA3
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx950", 0, 256), 8);   // CDNA4
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx1030", 0, 256), 16); // RDNA2
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx1100", 0, 256), 16); // RDNA3
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx1200", 0, 256), 16); // RDNA4
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx1250", 0, 256), 16); // GFX1250
  // blockSize doesn't matter when there is no LDS pressure.
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 0, 1), 8);
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 0, 8192), 8);
}

TEST(AmdArchDbTest, LdsBoundLdsEqualsUnitClampsToOne) {
  // kernelLdsBytes == ldsPerUnit -> only a single workgroup fits, so even
  // with a maxed-out workgroup the bound clamps to 1 (the floor of 1/4 = 0
  // before clamping).
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 65536, 64),
            1); // 64 KB on CDNA2
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx942", 65536, 64),
            1); // 64 KB on CDNA3
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx950", 163840, 64),
            1); // 160 KB on CDNA4
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx1100", 65536, 64),
            1); // 64 KB on RDNA3
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx1250", 327680, 64),
            1); // 320 KB on GFX1250
}

TEST(AmdArchDbTest, LdsBoundLdsExceedsUnitClampsToOne) {
  // Defensive: kernelLdsBytes > ldsPerUnit shouldn't happen in practice
  // (caller rejects it earlier) but the helper still returns 1.
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 100000, 64), 1);
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx950", 200000, 64), 1);
}

TEST(AmdArchDbTest, LdsBoundBlockSmallerThanWave) {
  // blockSize < waveSize -> wavesPerBlock = max(1, blockSize/waveSize) = 1.
  // For gfx90a (waveSize=64, LDS=64 KB) with kernelLDS=8 KB:
  //   blocksPerLdsUnit = 65536/8192 = 8
  //   wavesPerBlock    = max(1, 32/64) = 1
  //   wavesPerLdsUnit  = 8 * 1 = 8
  //   ldsBound         = 8 / 4 = 2
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 8192, 32), 2);
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 8192, 1), 2);
}

TEST(AmdArchDbTest, LdsBoundBlockEqualsWave) {
  // The four attention configs that motivated the fix all have block_size=64
  // (1 warp). LDS-bound = floor(LDS_per_unit / kernel_LDS) * 1 / 4.
  // gfx950 (LDS_per_unit = 160 KB = 163840 B):
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx950", 19456, 64), 2);  // cfg1
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx950", 16384, 64), 2);  // cfg2
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx950", 65536, 64), 1);  // cfg3
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx950", 131072, 64), 1); // cfg4
  // gfx90a (LDS_per_unit = 64 KB = 65536 B):
  //   kernelLDS=8192 -> blocks=8, waves=8, EU-waves=2.
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 8192, 64), 2);
  //   kernelLDS=16384 -> blocks=4, waves=4, EU-waves=1.
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 16384, 64), 1);
}

TEST(AmdArchDbTest, LdsBoundLargeBlock) {
  // 4 warps on gfx90a (block_size = 256, wave_size = 64 -> wavesPerBlock=4)
  // with kernelLDS=4096: blocks=16, waves=64, EU-waves=16 -> clamp to 8.
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx90a", 4096, 256), 8);
  // Same kernelLDS on gfx1100 (32-wave, max=16): block_size=256 -> 8 waves;
  // blocks=16; waves=128; EU-waves=32 -> clamp to 16.
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx1100", 4096, 256), 16);
  // gfx1100 with heavier LDS=16384: blocks=4, waves=32, EU-waves=8.
  EXPECT_EQ(computeLdsBoundWavesPerEU("gfx1100", 16384, 256), 8);
}

TEST(AmdArchDbTest, LdsBoundArchTripleAccepted) {
  // The arch string can include the LLVM triple prefix; getArch parses it
  // off, so the result must be identical to the bare gfx string.
  EXPECT_EQ(computeLdsBoundWavesPerEU("amdgcn-amd-amdhsa:gfx950", 16384, 64),
            computeLdsBoundWavesPerEU("gfx950", 16384, 64));
  EXPECT_EQ(computeLdsBoundWavesPerEU(
                "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-", 19456, 64),
            computeLdsBoundWavesPerEU("gfx950", 19456, 64));
}

// --- resolveWavesPerEU ---
//
// Policy:
//   requested > 0     -> effective = min(requested, ldsBound),
//                        overRequested = (requested > ldsBound).
//   requested == 0    -> effective = ldsBound iff kernelLdsBytes > 0 and
//                        ldsBound < maxWavesPerEU; else 0 (leave attribute
//                        absent, the AMDGPU backend default suffices).

TEST(AmdArchDbTest, ResolveRequestAchievablePassesThrough) {
  // gfx950 cfg2 LDS (16 KB), block_size=64 -> ldsBound = 2.
  // Asking for 2 is fine.
  auto r = resolveWavesPerEU("gfx950", 16384, 64, /*requested=*/2);
  EXPECT_EQ(r.ldsBound, 2);
  EXPECT_EQ(r.effective, 2);
  EXPECT_FALSE(r.overRequested);
  // Asking for 1 (below ldsBound) is also fine.
  r = resolveWavesPerEU("gfx950", 16384, 64, /*requested=*/1);
  EXPECT_EQ(r.ldsBound, 2);
  EXPECT_EQ(r.effective, 1);
  EXPECT_FALSE(r.overRequested);
}

TEST(AmdArchDbTest, ResolveRequestOverLdsBoundClampsAndFlagsOverRequested) {
  // gfx950 cfg1 / cfg3 / cfg4: requested=8, but LDS pushes the bound to
  // {2, 1, 1} respectively -> effective clamps and overRequested=true so
  // the tuner-side gate rejects.
  for (auto [ldsBytes, expectedBound] :
       std::initializer_list<std::pair<int64_t, int64_t>>{
           {19456, 2}, {65536, 1}, {131072, 1}}) {
    auto r = resolveWavesPerEU("gfx950", ldsBytes, 64, /*requested=*/8);
    EXPECT_EQ(r.ldsBound, expectedBound);
    EXPECT_EQ(r.effective, expectedBound);
    EXPECT_TRUE(r.overRequested);
  }
}

TEST(AmdArchDbTest, ResolveUnspecifiedFillsInLdsBoundForLdsHeavyKernel) {
  // This is the cfg2 production-path branch: requested=0 + LDS pressure
  // means the LDS-bound is stamped so the AMDGPU backend doesn't default
  // to the per-arch max (which would re-trigger the RegAllocGreedy blowup).
  auto r = resolveWavesPerEU("gfx950", 16384, 64, /*requested=*/0);
  EXPECT_EQ(r.ldsBound, 2);
  EXPECT_EQ(r.effective, 2);
  EXPECT_FALSE(r.overRequested);
}

TEST(AmdArchDbTest, ResolveUnspecifiedNoLdsLeavesAttrAbsent) {
  // requested=0, no LDS -> effective=0 (attribute should not be stamped,
  // AMDGPU default is fine).
  auto r = resolveWavesPerEU("gfx950", 0, 256, /*requested=*/0);
  EXPECT_EQ(r.ldsBound, 8); // == getMaxWavesPerEU(gfx950)
  EXPECT_EQ(r.effective, 0);
  EXPECT_FALSE(r.overRequested);
}

TEST(AmdArchDbTest, ResolveUnspecifiedLdsBoundEqualsArchMaxLeavesAbsent) {
  // requested=0 + LDS so light that LDS-bound matches the per-arch ceiling
  // -> nothing to stamp (the backend's default already targets the max).
  // gfx950, block_size=256, kernelLDS=4096 B (4 KiB): blocks=40, waves=160,
  // EU-waves=40, clamped to maxWavesPerEU=8 == ldsBound.
  auto r = resolveWavesPerEU("gfx950", 4096, 256, /*requested=*/0);
  EXPECT_EQ(r.ldsBound, 8);
  EXPECT_EQ(r.effective, 0);
  EXPECT_FALSE(r.overRequested);
}

TEST(AmdArchDbTest, ResolveRequestEqualToLdsBoundIsAchievable) {
  // Boundary: requested == ldsBound -> achievable, no clamp, no
  // overRequested. Lock this on a non-cfg2 arch (gfx1100, 32-wide
  // wavefront, maxWavesPerEU=16) so it's independent of the cfg2 case
  // covered above. 64 KB LDS, kernelLDS=8 KB, block_size=128:
  //   blocks=8, wavesPerBlock=128/32=4, wavesPerLdsUnit=32, ldsBound=8.
  // Asking for 8 hits the boundary exactly.
  auto r = resolveWavesPerEU("gfx1100", 8192, 128, /*requested=*/8);
  EXPECT_EQ(r.ldsBound, 8);
  EXPECT_EQ(r.effective, 8);
  EXPECT_FALSE(r.overRequested);
}

TEST(AmdArchDbTest, ResolveRdnaArchesUseArchMaxAndWaveSize) {
  // Per-arch coverage for resolveWavesPerEU. RDNA arches have a 32-wide
  // wavefront and a higher per-arch max (16). 64 KB LDS, block_size=128
  // (4 warps on RDNA):
  //   gfx1100: blocks=floor(65536/4096)=16, wavesPerBlock=128/32=4,
  //   wavesPerLdsUnit=64, ldsBound=16, clamped to 16. Requested=8 is fine.
  auto r = resolveWavesPerEU("gfx1100", 4096, 128, /*requested=*/8);
  EXPECT_EQ(r.ldsBound, 16);
  EXPECT_EQ(r.effective, 8);
  EXPECT_FALSE(r.overRequested);
  // ...and with heavier LDS so the bound is lower than the requested 16:
  // gfx1100 kernelLDS=16384, block_size=128: blocks=4, waves=16,
  // EU-waves=4. Requested=16 is over the LDS bound.
  r = resolveWavesPerEU("gfx1100", 16384, 128, /*requested=*/16);
  EXPECT_EQ(r.ldsBound, 4);
  EXPECT_EQ(r.effective, 4);
  EXPECT_TRUE(r.overRequested);
}

TEST(AmdArchDbTest, ResolveGfx1250UsesLargerLdsUnit) {
  // gfx1250 has 320 KB of LDS per LDS-unit and a 32-wide wavefront. With
  // block_size=128, kernelLDS=16 KB: blocks=20, waves=80, EU-waves=20,
  // clamped to maxWavesPerEU=16.
  auto r = resolveWavesPerEU("gfx1250", 16384, 128, /*requested=*/0);
  EXPECT_EQ(r.ldsBound, 16);
  EXPECT_EQ(r.effective, 0); // ldsBound == maxWavesPerEU -> no stamp.
}

// --- supportsTDM ---

TEST(AmdArchDbTest, SupportsTDM) {
  EXPECT_FALSE(supportsTDM("gfx908"));  // CDNA1
  EXPECT_FALSE(supportsTDM("gfx90a"));  // CDNA2
  EXPECT_FALSE(supportsTDM("gfx942"));  // CDNA3
  EXPECT_FALSE(supportsTDM("gfx950"));  // CDNA4
  EXPECT_FALSE(supportsTDM("gfx1010")); // RDNA1
  EXPECT_FALSE(supportsTDM("gfx1030")); // RDNA2
  EXPECT_FALSE(supportsTDM("gfx1100")); // RDNA3
  EXPECT_FALSE(supportsTDM("gfx1200")); // RDNA4
  EXPECT_TRUE(supportsTDM("gfx1250"));  // GFX1250
}
