//===- ParamLookupTableTests.cpp - Tests for Tuning Params Lookup ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
  // gfx906 is supported but has no tuning entries. The oldest gfx9* relative
  // is gfx908, but its conv_f16 list has only one config, so the fallback
  // search drops it and picks gfx90a.
  EXPECT_EQ("gfx90a_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx906_conv_f16"));
}

TEST(FindFallbackTest, YoungestRelative) {
  // gfx1201 is the youngest available relative for gfx1900
  EXPECT_EQ("gfx1201_conv_f16",
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
  // Fall back for single-config lists
  EXPECT_EQ("gfx1200_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_gemm_f16"));
  EXPECT_EQ("gfx1201_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1200_conv_f16"));
  EXPECT_EQ("gfx1100_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1150_gemm_f16"));
  EXPECT_EQ("gfx90a_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx908_gemm_f16"));
  EXPECT_EQ("gfx1100_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1000_gemm_f16"));
}

TEST(FindFallbackTest, Fp8FallsBackToI8) {
  // fp8 has no tuning entries; fall back to the closest datatype, i8.
  EXPECT_EQ("gfx942_gemm_i8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_gemm_fp8"));
}

TEST(FindFallbackTest, F4FallsBackToI8) {
  // f4 has no 4-bit neighbour, so it also falls back to i8.
  EXPECT_EQ("gfx942_gemm_i8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_gemm_f4"));
}
