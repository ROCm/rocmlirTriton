//===- ParamLookupTableTests.cpp - Tests for Tuning Params Lookup ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

// Architectures shipping a full set of attention quick-tuning lists.
static constexpr StringLiteral kAttentionArchs[] = {
    "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151"};

// Of those, the ones with no gemm+gemm lists of their own, which therefore
// still borrow attention's at every precision.
static constexpr StringLiteral kUntunedGemmGemmArchs[] = {
    "gfx908", "gfx90a", "gfx942", "gfx1151"};

// Architectures that do ship gemm+gemm lists, and the precisions they cover.
static constexpr StringLiteral kTunedGemmGemmArchs[] = {"gfx1100", "gfx950"};
static constexpr StringLiteral kTunedGemmGemmDataTypes[] = {"f16", "f32"};

// Data types the attention lists are tuned for.
static constexpr StringLiteral kAttentionDataTypes[] = {"bf16", "f16", "f32",
                                                        "i8"};

// Data types gfx1100 borrows from gfx1101 for gemm and conv.
static constexpr StringLiteral kNavi3SharedDataTypes[] = {"f16", "f32", "i8"};

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
  // gfx1200 is the youngest available relative for gfx1900 with a conv_f16
  // tuning list; gfx1201's overlapping conv list was removed.
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
  // Fall back for single-config lists
  EXPECT_EQ("gfx1200_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_gemm_f16"));
  // gfx906 has no gemm_f16 entry, so it falls back to its closest relative that
  // does, gfx908
  EXPECT_EQ("gfx908_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx906_gemm_f16"));
  // gfx1100 no longer ships gemm_f16; gfx1101 is the closest gfx11* relative.
  EXPECT_EQ("gfx1101_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1000_gemm_f16"));
}

TEST(FindFallbackTest, Gfx1201UsesGfx1200ForRemovedLists) {
  for (StringRef dataType : {"f16", "f32", "i8"}) {
    std::string convTarget = (Twine("gfx1201_conv_") + dataType).str();
    EXPECT_EQ((Twine("gfx1200_conv_") + dataType).str(),
              ParamLookupTable<GemmParamsAttr>::findFallback(convTarget))
        << "for target " << convTarget;
  }

  for (StringRef dataType : {"f16", "f32"}) {
    std::string attentionTarget =
        (Twine("gfx1201_attention_") + dataType).str();
    EXPECT_EQ((Twine("gfx1200_attention_") + dataType).str(),
              ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                  attentionTarget))
        << "for target " << attentionTarget;
  }
}

TEST(FindFallbackTest, StrixFallsBackToGfx1151) {
  // gfx1150 ships its own tuned tables, so it resolves to itself rather than to
  // a relative. gfx1152 has none and still falls back.
  EXPECT_EQ("gfx1150_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1150_gemm_f16"));
  EXPECT_EQ("gfx1151_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1152_gemm_f16"));
}

TEST(FindFallbackTest, AttentionStrixFallsBackToGfx1151) {
  // gfx1150 has tuned f16 and f32 attention tables so it resolves to itself
  // there. No i8 attention shapes were tuned for it, so i8 still falls back,
  // as does gfx1152, which has no tables of its own.
  EXPECT_EQ("gfx1150_attention_f16",
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

TEST(FindFallbackTest, Gfx1100BorrowsGfx1101GemmAndConvLists) {
  // gfx1100 has no gemm or conv lists, so it falls back to gfx1101.
  for (StringRef kernelType : {"gemm", "conv"}) {
    for (StringRef dataType : kNavi3SharedDataTypes) {
      std::string target =
          (Twine("gfx1100") + "_" + kernelType + "_" + dataType).str();
      EXPECT_EQ((Twine("gfx1101") + "_" + kernelType + "_" + dataType).str(),
                ParamLookupTable<GemmParamsAttr>::findFallback(target))
          << "for target " << target;
    }
  }
}

TEST(FindFallbackTest, Gfx1100KeepsAttentionPrecisionsGfx1101Lacks) {
  // gfx1100 keeps its bf16 and i8 attention lists.
  EXPECT_EQ("gfx1100_attention_bf16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1100_attention_bf16"));
  EXPECT_EQ("gfx1100_attention_i8",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1100_attention_i8"));
  // f16 and f32 fall back to gfx1101.
  EXPECT_EQ("gfx1101_attention_f16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1100_attention_f16"));
  EXPECT_EQ("gfx1101_attention_f32",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1100_attention_f32"));
}

TEST(FindFallbackTest, Gfx1101BorrowsGfx1100AttentionWhereItHasNone) {
  // gfx1101 has no bf16 or i8 attention lists, so it falls back to gfx1100.
  EXPECT_EQ("gfx1100_attention_bf16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1101_attention_bf16"));
  EXPECT_EQ("gfx1100_attention_i8",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1101_attention_i8"));
}

TEST(FindFallbackTest, Gfx1101GemmGemmPrefersOwnAttentionOverGfx1100) {
  // Prefer gfx1101 attention over gfx1100 gemm+gemm.
  for (StringRef dataType : kTunedGemmGemmDataTypes) {
    std::string target =
        (Twine("gfx1101_gemmelementwisegemm_") + dataType).str();
    EXPECT_EQ((Twine("gfx1101_attention_") + dataType).str(),
              ParamLookupTable<GemmGemmParamsAttr>::findFallback(target))
        << "for target " << target;
  }
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

TEST(FindFallbackTest, GemmGemmUsesItsOwnListWhereTuned) {
  // gfx1100 and gfx950 ship gemm+gemm lists for f16 and f32, so those are exact
  // hits rather than fallbacks.
  for (StringRef arch : kTunedGemmGemmArchs) {
    for (StringRef dataType : kTunedGemmGemmDataTypes) {
      std::string target =
          (Twine(arch) + "_gemmelementwisegemm_" + dataType).str();
      EXPECT_EQ(target,
                ParamLookupTable<GemmGemmParamsAttr>::findFallback(target))
          << "for target " << target;
    }
  }
}

TEST(FindFallbackTest, GemmGemmBorrowsAttentionAtSamePrecision) {
  // Where gemm+elementwise+gemm has no list of its own it borrows attention's,
  // and the requested precision has to survive that substitution: the suffix
  // match used to slice candidate keys by a length derived from the target,
  // which underflowed for the much longer "_gemmelementwisegemm_<dt>" suffix
  // and made every key in the architecture family look like a relative. Every
  // dtype then resolved to the lexicographically nearest one, i8.
  for (StringRef arch : kUntunedGemmGemmArchs) {
    for (StringRef dataType : kAttentionDataTypes) {
      std::string target =
          (Twine(arch) + "_gemmelementwisegemm_" + dataType).str();
      EXPECT_EQ((Twine(arch) + "_attention_" + dataType).str(),
                ParamLookupTable<GemmGemmParamsAttr>::findFallback(target))
          << "for target " << target;
    }
  }
  // The tuned architectures only cover f16 and f32; their other precisions
  // still fall back, and must not be captured by the f16/f32 gemm+gemm lists.
  for (StringRef arch : kTunedGemmGemmArchs) {
    for (StringRef dataType : {"bf16", "i8"}) {
      std::string target =
          (Twine(arch) + "_gemmelementwisegemm_" + dataType).str();
      EXPECT_EQ((Twine(arch) + "_attention_" + dataType).str(),
                ParamLookupTable<GemmGemmParamsAttr>::findFallback(target))
          << "for target " << target;
    }
  }
}

TEST(FindFallbackTest, ConvGemmBorrowsAttentionAtSamePrecision) {
  // conv+elementwise+gemm shares the same gridwise path, and the same gap.
  for (StringRef arch : kAttentionArchs) {
    for (StringRef dataType : kAttentionDataTypes) {
      std::string target =
          (Twine(arch) + "_convelementwisegemm_" + dataType).str();
      // gfx1100 only ships attention bf16/i8; f16/f32 borrow gfx1101's lists.
      StringRef attentionArch = arch;
      if (arch == "gfx1100" && (dataType == "f16" || dataType == "f32"))
        attentionArch = "gfx1101";
      EXPECT_EQ((Twine(attentionArch) + "_attention_" + dataType).str(),
                ParamLookupTable<GemmGemmParamsAttr>::findFallback(target))
          << "for target " << target;
    }
  }
}

TEST(FindFallbackTest, GemmGemmKeepsPrecisionWhileArchFallsBack) {
  // gfx1170 and gfx1200 ship only f16 and f32 attention lists, so an i8
  // gemm+gemm has to cross architectures. It must still land on an i8 list:
  // gfx1151 is the closest gfx11/gfx12 relative that has one.
  EXPECT_EQ("gfx1151_attention_i8",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1170_gemmelementwisegemm_i8"));
}

TEST(FindFallbackTest, GemmGemmPrefersOwnArchOverRelativesGemmGemmList) {
  // gfx1170 ships no gemm+gemm list but gfx1100, a relative, does. gfx1170's
  // own attention list still wins: attention shares the gridwise code and the
  // perf-config format, whereas a relative architecture differs in LDS capacity
  // and matrix-instruction shapes, making it the more expensive substitution.
  EXPECT_EQ("gfx1170_attention_f16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1170_gemmelementwisegemm_f16"));
  // Likewise on the CDNA side, where the relative holding a list is gfx950.
  EXPECT_EQ("gfx942_attention_f32",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx942_gemmelementwisegemm_f32"));
}

TEST(FindFallbackTest, GemmGemmSubstitutesKernelTypeBeforeDataType) {
  // fp8 has no attention list anywhere, so both substitutions are needed and
  // the datatype one (fp8 -> i8) is only reached after the kernel-type one.
  EXPECT_EQ("gfx942_attention_i8",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx942_gemmelementwisegemm_fp8"));
  EXPECT_EQ("gfx942_attention_i8",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx942_gemmelementwisegemm_f4"));
}

TEST(FindFallbackTest, LongUnknownKernelTypeHasNoRelatives) {
  // A kernel type with no entries and no fallback must report failure rather
  // than matching an unrelated list. These suffixes are longer than the table's
  // whole keys, which is what used to trigger the underflow.
  EXPECT_EQ("", ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                    "gfx942_someunknownfusedkerneltype_f16"));
  EXPECT_EQ("", ParamLookupTable<GemmParamsAttr>::findFallback(
                    "gfx942_someunknownfusedkerneltype_f16"));
}

TEST(FindFallbackTest, MalformedKeysAreRejected) {
  // Keys must have all three components.
  EXPECT_EQ("", ParamLookupTable<GemmParamsAttr>::findFallback("gfx942"));
  EXPECT_EQ("", ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_f16"));
  EXPECT_EQ("", ParamLookupTable<GemmParamsAttr>::findFallback(""));
}

TEST(LookupTest, GemmGemmResolvesToItsOwnListOnTunedArch) {
  // End-to-end through the public entry point. gfx1100 has tuned gemm+gemm
  // lists, so f16 must get the f16 one -- not f32's, and above all not the i8
  // attention list every precision used to collapse onto.
  MLIRContext ctx;
  Type f16 = Float16Type::get(&ctx);
  Type f32 = Float32Type::get(&ctx);
  Type i8 = IntegerType::get(&ctx, 8);
  StringRef arch = "amdgcn-amd-amdhsa:gfx1100";
  auto get = [&](KernelType kernel, Type t) {
    return ParamLookupTable<GemmGemmParamsAttr>::lookup(arch, kernel, t);
  };

  auto gemmGemmF16 = get(KernelType::GemmElementwiseGemm, f16);
  EXPECT_FALSE(gemmGemmF16.empty());
  EXPECT_FALSE(gemmGemmF16 == get(KernelType::GemmElementwiseGemm, f32));
  EXPECT_FALSE(gemmGemmF16 == get(KernelType::Attention, i8));
  // The merged list keeps attention's entries as a tail, so it is strictly
  // larger than the attention list it was seeded from.
  EXPECT_GT(gemmGemmF16.size(), get(KernelType::Attention, f16).size());
}

TEST(LookupTest, GemmGemmResolvesToAttentionListOfSamePrecision) {
  // gfx942 has no gemm+gemm lists, so it borrows attention's -- at its own
  // precision.
  MLIRContext ctx;
  Type f16 = Float16Type::get(&ctx);
  Type i8 = IntegerType::get(&ctx, 8);
  StringRef arch = "amdgcn-amd-amdhsa:gfx942";
  auto get = [&](KernelType kernel, Type t) {
    return ParamLookupTable<GemmGemmParamsAttr>::lookup(arch, kernel, t);
  };

  auto gemmGemmF16 = get(KernelType::GemmElementwiseGemm, f16);
  auto attentionF16 = get(KernelType::Attention, f16);
  auto attentionI8 = get(KernelType::Attention, i8);

  EXPECT_FALSE(gemmGemmF16.empty());
  EXPECT_TRUE(gemmGemmF16 == attentionF16);
  // The f16 and i8 lists must actually differ, or the assertion above would
  // hold even with the bug present.
  EXPECT_FALSE(attentionF16 == attentionI8);
  EXPECT_FALSE(gemmGemmF16 == attentionI8);
}

TEST(LookupTest, Gfx1100GemmAndConvServeGfx1101Lists) {
  // Verify the public lookup returns gfx1101's lists for gfx1100.
  MLIRContext ctx;
  SmallVector<Type, 3> dataTypes = {Float16Type::get(&ctx),
                                    Float32Type::get(&ctx),
                                    IntegerType::get(&ctx, 8)};
  auto get = [&](StringRef arch, KernelType kernel, Type t) {
    return ParamLookupTable<GemmParamsAttr>::lookup(arch, kernel, t);
  };

  for (KernelType kernel : {KernelType::Gemm, KernelType::Conv}) {
    for (Type dataType : dataTypes) {
      auto navi31 = get("amdgcn-amd-amdhsa:gfx1100", kernel, dataType);
      auto navi32 = get("amdgcn-amd-amdhsa:gfx1101", kernel, dataType);
      EXPECT_FALSE(navi31.empty());
      EXPECT_TRUE(navi31 == navi32) << "for " << stringifyEnum(kernel).lower()
                                    << " at " << getDataTypeString(dataType);
    }
  }

  // Verify different types and operations use different lists.
  auto gemmF16 = get("gfx1101", KernelType::Gemm, dataTypes[0]);
  EXPECT_FALSE(gemmF16 == get("gfx1101", KernelType::Gemm, dataTypes[1]));
  EXPECT_FALSE(gemmF16 == get("gfx1101", KernelType::Conv, dataTypes[0]));
}
