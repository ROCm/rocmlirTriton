//===- ParamLookupTableTests.cpp - Tests for Tuning Params Lookup ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

TEST(FindFallbackTest, ExactMatch) {
  // Exact match should return itself
  EXPECT_EQ("gfx942_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_conv_f16"));
}

TEST(FindFallbackTest, OldestRelative) {
  // gfx906 is supported but has no tuning entries. gfx908 has a conv_f16 entry
  // and is the closest (oldest) available gfx9* relative to gfx906.
  EXPECT_EQ("gfx908_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx906_conv_f16"));
}

TEST(FindFallbackTest, YoungestRelative) {
  // gfx1200 is the youngest available relative for gfx1900
  EXPECT_EQ("gfx1200_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1900_conv_f16"));
}

TEST(FindFallbackTest, OlderRelativeIsCloser) {
  // gfx949 is closer to gfx942 than gfx950
  EXPECT_EQ("gfx942_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx949_conv_f16"));
}

TEST(FindFallbackTest, YoungerRelativeIsCloser) {
  // gfx940 is closer to gfx942 than gfx90a
  EXPECT_EQ("gfx942_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx940_conv_f16"));
}

TEST(FindFallbackTest, PreferYoungerWhenEquidistant) {
  // gfx90a and gfx908 are equidistant to gfx909, prefer younger gfx90a
  EXPECT_EQ("gfx90a_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx909_conv_f16"));
}

TEST(FindFallbackTest, NoRelativesByPrefix) {
  // No relatives with matching prefix
  EXPECT_EQ("",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx800_conv_f16"));
}

TEST(FindFallbackTest, NoRelativesBySuffix) {
  // No relatives with matching suffix
  EXPECT_EQ("",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_op_type"));
}

TEST(FindFallbackTest, UnavailableTuningList) {
  // gfx1201 shares gfx1200 lists whenever both arches were tuned for the same
  // op/dtype; missing gfx1201 keys fall back to the gfx1200 equivalent.
  EXPECT_EQ("gfx1200_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_gemm_f16"));
  EXPECT_EQ("gfx1200_gemm_f32",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_gemm_f32"));
  EXPECT_EQ("gfx1200_gemm_i8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_gemm_i8"));
  EXPECT_EQ("gfx1200_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_conv_f16"));
  EXPECT_EQ("gfx1200_conv_f32",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_conv_f32"));
  EXPECT_EQ("gfx1200_conv_i8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_conv_i8"));
  EXPECT_EQ("gfx1200_attention_f16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1201_attention_f16"));
  EXPECT_EQ("gfx1200_attention_f32",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1201_attention_f32"));
  // gfx906 has no gemm_f16 entry, so it falls back to its closest relative that
  // does, gfx908
  EXPECT_EQ("gfx908_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx906_gemm_f16"));
  EXPECT_EQ("gfx1100_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1000_gemm_f16"));
}

TEST(FindFallbackTest, StrixFallsBackToGfx1151) {
  // The Strix Halo variants should fall back to gfx1151
  EXPECT_EQ("gfx1151_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1150_gemm_f16"));
  EXPECT_EQ("gfx1151_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1152_gemm_f16"));
}

TEST(FindFallbackTest, AttentionStrixFallsBackToGfx1151) {
  // The Strix Halo variants should fall back to gfx1151 for attention too.
  EXPECT_EQ("gfx1151_attention_f16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1150_attention_f16"));
  EXPECT_EQ("gfx1151_attention_f16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1152_attention_f16"));
  EXPECT_EQ("gfx1151_attention_i8",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1150_attention_i8"));
  EXPECT_EQ("gfx1151_attention_i8",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1152_attention_i8"));
}

TEST(FindFallbackTest, Fp8FallsBackToArchRelative) {
  // gfx942 has no fp8 tuning entries, but gfx950 ships a gemm_fp8 list. A
  // same-datatype architecture relative is preferred over a datatype
  // substitution, so gfx942_gemm_fp8 falls back to gfx950_gemm_fp8 rather than
  // to gfx942_gemm_i8.
  EXPECT_EQ("gfx950_gemm_fp8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_gemm_fp8"));
}

TEST(FindFallbackTest, Fp8FallsBackToRdna4) {
  // gfx1100 (RDNA3) has no fp8 tuning entries, but gfx1201 (RDNA4) ships a
  // gemm_fp8 list and is the only same-datatype relative in the gfx11/gfx12
  // family, so gfx1100_gemm_fp8 falls back to gfx1201_gemm_fp8.
  EXPECT_EQ("gfx1201_gemm_fp8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1100_gemm_fp8"));
}

TEST(FindFallbackTest, F4FallsBackToI8) {
  // f4 has neither its own tuning entries nor a same-datatype architecture
  // relative, so it falls back to the closest datatype, i8.
  EXPECT_EQ("gfx942_gemm_i8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_gemm_f4"));
}

TEST(FindFallbackTest, Gfx908ExactMatches) {
  // gfx908 now ships its own gemm/conv quick-tuning lists, so each dtype is an
  // exact match rather than a fallback.
  EXPECT_EQ("gfx908_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx908_gemm_f16"));
  EXPECT_EQ("gfx908_conv_i8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx908_conv_i8"));
  EXPECT_EQ("gfx908_attention_f32",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx908_attention_f32"));
}

TEST(FindFallbackTest, OlderArchFallsBackToGfx908) {
  // gfx906 has no lists of its own; gfx908 is its closest gfx9* relative that
  // does, across gemm, conv, and attention.
  EXPECT_EQ("gfx908_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx906_gemm_f16"));
  EXPECT_EQ("gfx908_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx906_conv_f16"));
  EXPECT_EQ("gfx908_attention_f16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx906_attention_f16"));
}
