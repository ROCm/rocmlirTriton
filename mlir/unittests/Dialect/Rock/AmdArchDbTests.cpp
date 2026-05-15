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
  EXPECT_TRUE(isFastAtomicAddSupported("gfx906", e.f32));  // GCN5_1 (Vega20)
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
  EXPECT_FALSE(isFastAtomicAddSupported("gfx906", e.f16));  // GCN5_1 (Vega20)
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
  EXPECT_FALSE(isFastAtomicAddSupported("gfx906", e.bf16));  // GCN5_1 (Vega20)
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
  EXPECT_FALSE(isFastAtomicAddSupported("gfx906", e.i32));  // GCN5_1 (Vega20)
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
  EXPECT_FALSE(isFastAtomicMaxSupported("gfx906", e.f32)); // GCN5_1 (Vega20)
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
  EXPECT_EQ(getMaxNumChiplets("gfx906"), 1);  // GCN5_1 (Vega20)
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
  EXPECT_EQ(getMinNumCU("gfx906"), 10);   // GCN5_1 (Vega20)
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
  EXPECT_EQ(getMaxWavesPerEU("gfx906"), 8);   // GCN5_1 (Vega20)
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
  EXPECT_EQ(getWaveSize("gfx906"), 64);  // GCN5_1 (Vega20)
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
  EXPECT_EQ(getLDSSize("gfx906"), 65536);   // GCN5_1 (Vega20): 64 KB
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
  // gfx906 (Vega20) has v_dot intrinsics but no MFMA/WMMA, so we expect None
  // for every type combination here.
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

// --- supportsTDM ---

TEST(AmdArchDbTest, SupportsTDM) {
  EXPECT_FALSE(supportsTDM("gfx906"));  // GCN5_1 (Vega20)
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

// --- gfx906-specific arch-string forms ---

// Verify that triple- and feature-suffixed forms of the gfx906 arch string
// (the kind that show up as rock.arch attributes) all parse correctly and
// dispatch to the GCN5_1 code paths.
TEST(AmdArchDbTest, Gfx906ArchStringForms) {
  ArchTestEnv e;
  EXPECT_EQ(getMinNumCU("amdgcn-amd-amdhsa:gfx906"), 10);
  EXPECT_EQ(getMinNumCU("amdgcn-amd-amdhsa:gfx906:xnack-"), 10);
  EXPECT_EQ(getMaxWavesPerEU("amdgcn-amd-amdhsa:gfx906"), 8);
  EXPECT_EQ(getWaveSize("amdgcn-amd-amdhsa:gfx906"), 64);
  EXPECT_TRUE(isFastAtomicAddSupported("amdgcn-amd-amdhsa:gfx906", e.f32));
}
