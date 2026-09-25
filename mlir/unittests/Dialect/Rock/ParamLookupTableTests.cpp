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
#include "llvm/ADT/STLExtras.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

// Architectures shipping a full set of attention quick-tuning lists.
static constexpr StringLiteral kAttentionArchs[] = {
    "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201"};

// Of those, the ones with no gemm+gemm lists of their own, which therefore
// still borrow attention's at every precision.
static constexpr StringLiteral kUntunedGemmGemmArchs[] = {"gfx908", "gfx90a",
                                                          "gfx942", "gfx1151"};

// Architectures that do ship gemm+gemm lists, and the precisions they cover.
static constexpr StringLiteral kTunedGemmGemmArchs[] = {"gfx1100", "gfx950"};
static constexpr StringLiteral kTunedGemmGemmDataTypes[] = {"f16", "f32"};

// Data types the attention lists are tuned for.
static constexpr StringLiteral kAttentionDataTypes[] = {"bf16", "f16", "f32",
                                                        "i8"};

// Data types with dedicated gfx1100 and gfx1101 gemm and conv lists.
static constexpr StringLiteral kNavi3TunedDataTypes[] = {"f16", "f32", "i8"};

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
  // gfx1201 is the youngest available relative for gfx1900 with a conv_f16
  // tuning list.
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
  // gfx1201 ships no regular gemm_f16 list but does ship a split-K-free one,
  // which the unconditional pair fallback finds before changing architecture.
  EXPECT_EQ("gfx1201_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_gemm_f16"));
  // gfx906 has no gemm_f16 entry, so it falls back to its closest relative that
  // does, gfx908
  EXPECT_EQ("gfx908_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx906_gemm_f16"));
  // gfx1100 is the closest gfx11* relative with a gemm_f16 list.
  EXPECT_EQ("gfx1100_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1000_gemm_f16"));
}

TEST(FindFallbackTest, Gfx1201UsesItsOwnLists) {
  // gfx1201 ships gemm, conv and attention lists at every precision it is
  // tuned for, so none of them resolve to a gfx12/gfx11 relative.
  for (StringRef dataType : {"f16", "f32", "i8"}) {
    std::string convTarget = (Twine("gfx1201_conv_") + dataType).str();
    EXPECT_EQ(convTarget,
              ParamLookupTable<GemmParamsAttr>::findFallback(convTarget))
        << "for target " << convTarget;
  }

  for (StringRef dataType : {"f16", "f32", "fp8", "i8"}) {
    std::string gemmTarget = (Twine("gfx1201_gemm_") + dataType).str();
    EXPECT_EQ(gemmTarget,
              ParamLookupTable<GemmParamsAttr>::findFallback(gemmTarget))
        << "for target " << gemmTarget;
  }

  for (StringRef dataType : kAttentionDataTypes) {
    std::string attentionTarget =
        (Twine("gfx1201_attention_") + dataType).str();
    EXPECT_EQ(
        attentionTarget,
        ParamLookupTable<GemmGemmParamsAttr>::findFallback(attentionTarget))
        << "for target " << attentionTarget;
  }
}

TEST(FindFallbackTest, ArchitectureFallbackUsesBothLists) {
  // Neither table has an exact gfx1202 key. Once the exact split-K pair misses,
  // gfx1201's split-K-free list is closer than gfx1200's regular list.
  EXPECT_EQ("gfx1201_gemm_f32",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1202_gemm_f32"));
  EXPECT_EQ("gfx1201_attention_f16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1202_attention_f16"));
}

TEST(FindFallbackTest, Gfx1201KeepsItsRemainingLists) {
  // The other half of the same change: dropping only some of an architecture's
  // lists must not disturb the ones it keeps, so these still resolve to
  // themselves rather than to a gfx12/gfx11 relative.
  EXPECT_EQ("gfx1201_attention_bf16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1201_attention_bf16"));
  EXPECT_EQ("gfx1201_attention_i8",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1201_attention_i8"));
  EXPECT_EQ("gfx1201_gemm_fp8",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx1201_gemm_fp8"));
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

TEST(FindFallbackTest, Gfx1100UsesOwnGemmAndConvLists) {
  // gfx1100 ships dedicated gemm and conv lists at each tuned precision.
  for (StringRef kernelType : {"gemm", "conv"}) {
    for (StringRef dataType : kNavi3TunedDataTypes) {
      std::string target =
          (Twine("gfx1100") + "_" + kernelType + "_" + dataType).str();
      EXPECT_EQ(target, ParamLookupTable<GemmParamsAttr>::findFallback(target))
          << "for target " << target;
    }
  }
}

TEST(FindFallbackTest, Gfx1100UsesOwnAttentionLists) {
  // gfx1100 ships dedicated attention lists at each tuned precision.
  for (StringRef dataType : kAttentionDataTypes) {
    std::string target = (Twine("gfx1100_attention_") + dataType).str();
    EXPECT_EQ(target,
              ParamLookupTable<GemmGemmParamsAttr>::findFallback(target))
        << "for target " << target;
  }
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

TEST(FindFallbackTest, Bf16FallsBackToF16WhereNoBf16ListExists) {
  // No gemm or conv list is tuned for bf16 on any architecture, so those have
  // to substitute the datatype. Attention is what keeps the substitution
  // honest: gfx942 has a bf16 list there, and it must win.
  EXPECT_EQ("gfx942_gemm_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_gemm_bf16"));
  EXPECT_EQ("gfx942_conv_f16",
            ParamLookupTable<GemmParamsAttr>::findFallback("gfx942_conv_bf16"));
  EXPECT_EQ("gfx942_attention_bf16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx942_attention_bf16"));
}

TEST(FindFallbackTest, Bf16PrefersRelativeArchOverF16) {
  // Precision is the last axis to give way, exactly as for fp8 above. gfx1150
  // ships no bf16 attention list but gfx1151 does, and borrowing that beats
  // dropping to gfx1150's own f16 one.
  EXPECT_EQ("gfx1151_attention_bf16",
            ParamLookupTable<GemmGemmParamsAttr>::findFallback(
                "gfx1150_attention_bf16"));
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
      EXPECT_EQ((Twine(arch) + "_attention_" + dataType).str(),
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

TEST(LookupTest, RefreshedQuickTuningListsHaveAtMostFortyConfigs) {
  // Legacy lists can exceed the cap. Cover every list regenerated after the
  // generator gained --max-configs, so future refreshes cannot regress it.
  constexpr size_t maxConfigs = 40;
  MLIRContext ctx;
  Type f16 = Float16Type::get(&ctx);
  Type f32 = Float32Type::get(&ctx);
  Type i8 = IntegerType::get(&ctx, 8);

  auto expectGemmListWithinCap = [&](StringRef arch, KernelType kernel,
                                     Type dataType) {
    auto configs = ParamLookupTable<GemmParamsAttr>::lookup(
        arch, kernel, dataType, /*supportsSplitK=*/true);
    EXPECT_LE(configs.size(), maxConfigs)
        << "for " << arch << " " << stringifyEnum(kernel).lower() << " "
        << getDataTypeString(dataType);
  };
  auto expectGemmGemmListWithinCap = [&](StringRef arch, KernelType kernel,
                                         Type dataType) {
    auto configs = ParamLookupTable<GemmGemmParamsAttr>::lookup(
        arch, kernel, dataType, /*supportsSplitK=*/true);
    EXPECT_LE(configs.size(), maxConfigs)
        << "for " << arch << " " << stringifyEnum(kernel).lower() << " "
        << getDataTypeString(dataType);
  };

  for (StringRef arch :
       {"gfx1151", "gfx1170", "gfx1200", "gfx1150", "gfx1101", "gfx1201"}) {
    for (Type dataType : {f16, f32})
      expectGemmListWithinCap(arch, KernelType::Conv, dataType);
  }
  for (StringRef arch : {"gfx1150", "gfx1201"}) {
    for (Type dataType : {f16, f32, i8})
      expectGemmListWithinCap(arch, KernelType::Gemm, dataType);
    expectGemmListWithinCap(arch, KernelType::Conv, i8);
  }
  expectGemmListWithinCap("gfx1101", KernelType::Gemm, f16);

  expectGemmGemmListWithinCap("gfx1150", KernelType::Attention, f32);
  for (Type dataType : {f16, f32})
    expectGemmGemmListWithinCap("gfx1201", KernelType::Attention, dataType);
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
    return ParamLookupTable<GemmGemmParamsAttr>::lookup(
        arch, kernel, t, /*supportsSplitK=*/true);
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
    return ParamLookupTable<GemmGemmParamsAttr>::lookup(
        arch, kernel, t, /*supportsSplitK=*/true);
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

TEST(LookupTest, Gfx1100GemmAndConvUseOwnLists) {
  // Verify the public lookup returns gfx1100's dedicated lists.
  MLIRContext ctx;
  SmallVector<Type, 3> dataTypes = {Float16Type::get(&ctx),
                                    Float32Type::get(&ctx),
                                    IntegerType::get(&ctx, 8)};
  auto get = [&](StringRef arch, KernelType kernel, Type t,
                 bool supportsSplitK) {
    return ParamLookupTable<GemmParamsAttr>::lookup(arch, kernel, t,
                                                    supportsSplitK);
  };

  for (KernelType kernel : {KernelType::Gemm, KernelType::Conv}) {
    for (Type dataType : dataTypes) {
      auto navi31 = get("amdgcn-amd-amdhsa:gfx1100", kernel, dataType,
                        /*supportsSplitK=*/true);
      auto navi32 = get("amdgcn-amd-amdhsa:gfx1101", kernel, dataType,
                        /*supportsSplitK=*/true);
      EXPECT_FALSE(navi31.empty());
      EXPECT_FALSE(navi31 == navi32) << "for " << stringifyEnum(kernel).lower()
                                     << " at " << getDataTypeString(dataType);
    }
  }

  // Verify different types and operations use different lists.
  auto gemmF16 = get("gfx1100", KernelType::Gemm, dataTypes[0], false);
  EXPECT_FALSE(gemmF16 ==
               get("gfx1100", KernelType::Gemm, dataTypes[1], false));
  EXPECT_FALSE(gemmF16 ==
               get("gfx1100", KernelType::Conv, dataTypes[0], false));
}

TEST(DataTypeStringTest, Bf16IsKeyedSeparatelyFromF16) {
  // The two 16-bit floats used to share the "f16" key, which left every
  // *_attention_bf16 list in the table unreachable.
  MLIRContext ctx;
  EXPECT_EQ("bf16", getDataTypeString(BFloat16Type::get(&ctx)));
  EXPECT_EQ("f16", getDataTypeString(Float16Type::get(&ctx)));
}

TEST(LookupTest, Bf16AttentionGetsItsOwnList) {
  // End-to-end through the public entry point, which is where the keying used
  // to go wrong: makeKey spelled bf16 as f16, so these lists could never be
  // served.
  MLIRContext ctx;
  Type bf16 = BFloat16Type::get(&ctx);
  Type f16 = Float16Type::get(&ctx);
  auto get = [&](StringRef arch, Type t) {
    return ParamLookupTable<GemmGemmParamsAttr>::lookup(
        arch, KernelType::Attention, t, /*supportsSplitK=*/true);
  };

  // Each architecture's bf16 list must stay distinct from its f16 list.
  for (StringRef arch : {"gfx942", "gfx1100", "gfx1201"}) {
    auto attentionBf16 = get(arch, bf16);
    EXPECT_FALSE(attentionBf16.empty()) << "for " << arch;
    EXPECT_FALSE(attentionBf16 == get(arch, f16)) << "for " << arch;
  }
}

TEST(LookupTest, Bf16GemmAndConvShareTheF16Lists) {
  // The other half: with no bf16 gemm or conv list anywhere, the datatype
  // fallback has to land bf16 on exactly the f16 list it used to key as, or
  // splitting the key would have cost these kernels their tuning entirely.
  MLIRContext ctx;
  Type bf16 = BFloat16Type::get(&ctx);
  Type f16 = Float16Type::get(&ctx);
  auto get = [&](KernelType kernel, Type t) {
    return ParamLookupTable<GemmParamsAttr>::lookup("gfx942", kernel, t,
                                                    /*supportsSplitK=*/true);
  };

  for (KernelType kernel : {KernelType::Gemm, KernelType::Conv}) {
    auto bf16List = get(kernel, bf16);
    EXPECT_FALSE(bf16List.empty()) << "for " << stringifyEnum(kernel).lower();
    EXPECT_TRUE(bf16List == get(kernel, f16))
        << "for " << stringifyEnum(kernel).lower();
  }
}

TEST(LookupTest, SupportsSplitKSelectsPreferredExactList) {
  // gfx1151's regular gemm f16 list contains split-K configs, while its
  // no-split-K list does not.
  MLIRContext ctx;
  Type f16 = Float16Type::get(&ctx);
  StringRef arch = "amdgcn-amd-amdhsa:gfx1151";
  auto regular = ParamLookupTable<GemmParamsAttr>::lookup(
      arch, KernelType::Gemm, f16, /*supportsSplitK=*/true);
  auto noSplitK = ParamLookupTable<GemmParamsAttr>::lookup(
      arch, KernelType::Gemm, f16, /*supportsSplitK=*/false);

  EXPECT_FALSE(regular.empty());
  EXPECT_FALSE(noSplitK.empty());
  EXPECT_FALSE(regular == noSplitK);
  auto hasSplitK = [](StringRef config) {
    return !config.contains("splitKFactor=1,");
  };
  EXPECT_TRUE(llvm::any_of(regular, hasSplitK));
  EXPECT_FALSE(llvm::any_of(noSplitK, hasSplitK));
}

TEST(LookupTest, MissingNoSplitKListUsesRegularPair) {
  // gfx1100 ships gemm+elementwise+gemm lists in the regular table only. When
  // split-K is unsupported, lookup must borrow that exact regular list rather
  // than a different no-split-K gemm/conv/attention table.
  MLIRContext ctx;
  Type f16 = Float16Type::get(&ctx);
  StringRef arch = "amdgcn-amd-amdhsa:gfx1100";
  auto regular = ParamLookupTable<GemmGemmParamsAttr>::lookup(
      arch, KernelType::GemmElementwiseGemm, f16, /*supportsSplitK=*/true);
  auto noSplitK = ParamLookupTable<GemmGemmParamsAttr>::lookup(
      arch, KernelType::GemmElementwiseGemm, f16, /*supportsSplitK=*/false);

  EXPECT_FALSE(noSplitK.empty());
  EXPECT_TRUE(noSplitK == regular);
}

TEST(LookupTest, ArchitectureFallbackPreservesSplitKPreference) {
  // gfx1202 has no exact key in either table, so both fallbacks use gfx1201
  // while preserving the selected table.
  MLIRContext ctx;
  Type f32 = Float32Type::get(&ctx);
  auto get = [&](StringRef arch, bool supportsSplitK) {
    return ParamLookupTable<GemmParamsAttr>::lookup(arch, KernelType::Gemm, f32,
                                                    supportsSplitK);
  };

  auto regular = get("amdgcn-amd-amdhsa:gfx1202", true);
  auto noSplitK = get("amdgcn-amd-amdhsa:gfx1202", false);
  EXPECT_FALSE(regular.empty());
  EXPECT_FALSE(noSplitK.empty());
  EXPECT_TRUE(regular == get("amdgcn-amd-amdhsa:gfx1201", true));
  EXPECT_TRUE(noSplitK == get("amdgcn-amd-amdhsa:gfx1201", false));
  EXPECT_FALSE(regular == noSplitK);
}

// Problem hash of a gfx942 f32 GEMM 128x512x512, one of the problems the
// shipped Gfx942GemmF32 shard was generated for. rocmlir-gen is the only
// speller of the key (--emit-quick-tuning-problem-key-hash); the value is
// pinned in test/rocmlir-gen/quick-tuning-problem-key-hash.mlir's company and
// exercised end to end by test/rocmlir-gen/quick-tuning-per-problem.mlir.
static constexpr QuickTuningProblemKeyHash kGfx942GemmF32MappedProblem =
    8175943205932196350ULL;
static constexpr QuickTuningTableLookUpKeyVersionHash kGemmKeyVersionHash =
    6791176183107838810ULL;

static SmallVector<StringRef> lookupGfx942GemmF32(
    bool supportsSplitK,
    std::optional<QuickTuningProblemKeyHash> problemKeyHash, MLIRContext &ctx,
    QuickTuningTableLookUpKeyVersionHash keyVersionHash = kGemmKeyVersionHash) {
  std::optional<QuickTuningProblemKey> problemKey;
  if (problemKeyHash)
    problemKey = QuickTuningProblemKey{*problemKeyHash, keyVersionHash};
  return ParamLookupTable<GemmParamsAttr>::lookup(
      "amdgcn-amd-amdhsa:gfx942", KernelType::Gemm, Float32Type::get(&ctx),
      supportsSplitK, problemKey);
}

TEST(LookupTest, PerProblemHashNarrowsTheSetCover) {
  // A mapped problem is served its own ranking instead of the key's set cover.
  // The two lists are disjoint for this problem, which is the point of the
  // layer: the per-problem winner is usually a config the set cover never
  // offered. Regenerating the shards can change the row; keep the assertions
  // by picking another mapped problem rather than dropping them.
  MLIRContext ctx;
  auto setCover =
      lookupGfx942GemmF32(/*supportsSplitK=*/true, std::nullopt, ctx);
  auto perProblem = lookupGfx942GemmF32(
      /*supportsSplitK=*/true, kGfx942GemmF32MappedProblem, ctx);

  EXPECT_FALSE(perProblem.empty());
  EXPECT_LT(perProblem.size(), setCover.size());
  for (StringRef config : perProblem)
    EXPECT_FALSE(llvm::is_contained(setCover, config)) << "for " << config;
}

TEST(LookupTest, UnmappedProblemHashFallsThroughToTheSetCover) {
  // Per-problem lookup has no key fallback of its own: a ranking only holds
  // for the problem it was measured on, so a hash with no row must come back
  // with exactly what the hashless lookup returns rather than a neighbour's
  // ranking.
  MLIRContext ctx;
  auto setCover =
      lookupGfx942GemmF32(/*supportsSplitK=*/true, std::nullopt, ctx);
  testing::internal::CaptureStderr();
  auto unmapped = lookupGfx942GemmF32(
      /*supportsSplitK=*/true, kGfx942GemmF32MappedProblem + 1, ctx);
  std::string warnings = testing::internal::GetCapturedStderr();

  EXPECT_FALSE(setCover.empty());
  EXPECT_TRUE(unmapped == setCover);
  EXPECT_TRUE(warnings.empty());
}

TEST(LookupTest, LookupKeyVersionMismatchWarnsAndUsesSetCover) {
  MLIRContext ctx;
  auto setCover =
      lookupGfx942GemmF32(/*supportsSplitK=*/true, std::nullopt, ctx);

  testing::internal::CaptureStderr();
  auto stale = lookupGfx942GemmF32(
      /*supportsSplitK=*/true, kGfx942GemmF32MappedProblem, ctx,
      kGemmKeyVersionHash + 1);
  std::string warnings = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(stale == setCover);
  EXPECT_NE(warnings.find("table lookup key version hash"), std::string::npos);
  EXPECT_NE(warnings.find("Regenerate the map"), std::string::npos);
}

TEST(LookupTest, UnsupportedProblemFieldsWarnAndUseSetCover) {
  MLIRContext ctx;
  auto setCover =
      lookupGfx942GemmF32(/*supportsSplitK=*/true, std::nullopt, ctx);
  QuickTuningProblemKey problemKey{kGfx942GemmF32MappedProblem,
                                   kGemmKeyVersionHash, "convolution_groups"};

  testing::internal::CaptureStderr();
  auto unsupported = ParamLookupTable<GemmParamsAttr>::lookup(
      "amdgcn-amd-amdhsa:gfx942", KernelType::Gemm, Float32Type::get(&ctx),
      /*supportsSplitK=*/true, problemKey);
  std::string warnings = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(unsupported == setCover);
  EXPECT_NE(warnings.find("does not represent"), std::string::npos);
  EXPECT_NE(warnings.find("convolution_groups"), std::string::npos);
  EXPECT_NE(warnings.find("Exhaustively tune"), std::string::npos);
}

TEST(LookupTest, UntunableProblemFieldsUseSetCoverQuietly) {
  // Retuning cannot add a mode the tuning pipeline cannot express, so such a
  // mode still avoids the mapped ranking but is not worth a warning.
  MLIRContext ctx;
  auto setCover =
      lookupGfx942GemmF32(/*supportsSplitK=*/true, std::nullopt, ctx);
  QuickTuningProblemKey problemKey{kGfx942GemmF32MappedProblem,
                                   kGemmKeyVersionHash, /*unsupportedFields=*/"",
                                   /*untunableFields=*/"asymmetric_padding"};

  testing::internal::CaptureStderr();
  auto untunable = ParamLookupTable<GemmParamsAttr>::lookup(
      "amdgcn-amd-amdhsa:gfx942", KernelType::Gemm, Float32Type::get(&ctx),
      /*supportsSplitK=*/true, problemKey);
  std::string warnings = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(untunable == setCover);
  EXPECT_TRUE(warnings.empty());
}

TEST(LookupTest, UnsupportedProblemFieldsWithoutAMapAreQuiet) {
  // gfx942_gemm_bf16 ships no per-problem map, so there is no ranking for an
  // unsupported mode to miss and nothing worth diagnosing.
  MLIRContext ctx;
  Type bf16 = BFloat16Type::get(&ctx);
  auto setCover = ParamLookupTable<GemmParamsAttr>::lookup(
      "amdgcn-amd-amdhsa:gfx942", KernelType::Gemm, bf16,
      /*supportsSplitK=*/true);
  QuickTuningProblemKey problemKey{kGfx942GemmF32MappedProblem,
                                   kGemmKeyVersionHash, "convolution_groups"};

  testing::internal::CaptureStderr();
  auto unsupported = ParamLookupTable<GemmParamsAttr>::lookup(
      "amdgcn-amd-amdhsa:gfx942", KernelType::Gemm, bf16,
      /*supportsSplitK=*/true, problemKey);
  std::string warnings = testing::internal::GetCapturedStderr();

  EXPECT_TRUE(unsupported == setCover);
  EXPECT_TRUE(warnings.empty());
}

TEST(LookupTest, PerProblemDoesNotDependOnSplitKLegality) {
  // `supportsSplitK` chooses between the two set covers and has no say over
  // the per-problem rankings, which come back whole either way. Dropping the
  // members a split-K-illegal caller cannot run is that caller's job, and it
  // always has something left because select_perfconfigs reserves a
  // splitKFactor=1 slot in every row. Gating the rankings here instead would
  // strand every attention shard, attention never being split-K legal.
  MLIRContext ctx;
  auto splitK = lookupGfx942GemmF32(/*supportsSplitK=*/true,
                                    kGfx942GemmF32MappedProblem, ctx);
  auto noSplitK = lookupGfx942GemmF32(/*supportsSplitK=*/false,
                                      kGfx942GemmF32MappedProblem, ctx);

  EXPECT_FALSE(splitK.empty());
  EXPECT_TRUE(splitK == noSplitK);
  EXPECT_TRUE(llvm::any_of(splitK, [](StringRef config) {
    return config.contains("splitKFactor=1,");
  })) << "every shipped row must keep a split-K-free config";
}
