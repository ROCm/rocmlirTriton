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
  // gfx1100 now has an exact no-split-K pair, which wins before architecture
  // fallback to gfx1101.
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

TEST(FindFallbackTest, Gfx1100UsesItsNoSplitKPairsForGemmAndConv) {
  // gfx1100 has exact no-split-K gemm and conv lists, so pair fallback wins
  // before changing architecture to gfx1101.
  for (StringRef kernelType : {"gemm", "conv"}) {
    for (StringRef dataType : kNavi3SharedDataTypes) {
      std::string target =
          (Twine("gfx1100") + "_" + kernelType + "_" + dataType).str();
      EXPECT_EQ(target, ParamLookupTable<GemmParamsAttr>::findFallback(target))
          << "for target " << target;
    }
  }
}

TEST(FindFallbackTest, Gfx1100UsesItsOwnAttentionLists) {
  // bf16/i8 are regular entries and f16/f32 are exact no-split-K pairs.
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

TEST(LookupTest, Gfx1100GemmAndConvServeTheirNoSplitKLists) {
  // Verify an exact no-split-K pair wins before architecture fallback even
  // when the caller prefers the regular table.
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
      auto preferredRegular =
          get("amdgcn-amd-amdhsa:gfx1100", kernel, dataType, true);
      auto preferredNoSplitK =
          get("amdgcn-amd-amdhsa:gfx1100", kernel, dataType, false);
      EXPECT_FALSE(preferredRegular.empty());
      EXPECT_TRUE(preferredRegular == preferredNoSplitK)
          << "for " << stringifyEnum(kernel).lower() << " at "
          << getDataTypeString(dataType);
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

  // gfx1100 and gfx1201 have regular bf16 lists and no-split-K f16 lists; the
  // two precisions must continue to resolve to distinct measurements.
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
  // gfx950 has no split-K-free lists, so a problem that does not support
  // split-K must still fall back to the exact regular key.
  MLIRContext ctx;
  Type f16 = Float16Type::get(&ctx);
  StringRef arch = "amdgcn-amd-amdhsa:gfx950";
  auto regular = ParamLookupTable<GemmParamsAttr>::lookup(
      arch, KernelType::Gemm, f16, /*supportsSplitK=*/true);
  auto noSplitK = ParamLookupTable<GemmParamsAttr>::lookup(
      arch, KernelType::Gemm, f16, /*supportsSplitK=*/false);

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
