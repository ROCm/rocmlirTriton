//===- QuickTuningProblemMapTests.cpp - Per-problem map lookup ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/QuickTuningProblemMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FormatVariadic.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

namespace {
// A map shaped like a generated shard: interned perfconfigs, a ribbon of
// indices into them, and problems sorted by hash. The rows deliberately differ
// in length, share perfconfigs and appear out of index order, as real shards
// do.
const StringRef perfConfigs[] = {"A", "B", "C"};
constexpr uint16_t perfConfigIndices[] = {0, 1, 2, 2, 0};
constexpr QuickTuningProblemRef problems[] = {
    {10, 0, 3}, {20, 3, 1}, {30, 4, 1}};
constexpr QuickTuningTableLookUpKeyVersionHash keyVersionHash = 42;

QuickTuningProblemMap makeMap() {
  return QuickTuningProblemMap(keyVersionHash, problems, perfConfigIndices,
                               perfConfigs);
}

/// Keys the first gemm-like op of a single-kernel module wrapping `func`.
std::optional<QuickTuningProblemKey> keyFor(MLIRContext &ctx, StringRef func) {
  DialectRegistry reg;
  reg.insert<RockDialect, func::FuncDialect>();
  ctx.appendDialectRegistry(reg);
  ctx.loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(("module {" + func + "}").str(), &ctx);
  if (!module)
    return std::nullopt;
  return getQuickTuningProblemKey(*module);
}

/// A 3x3, 8-channel, stride-2 convolution on an 8x8 image (4x4 output).
std::string conv(StringRef inType, StringRef outType, StringRef padding,
                 int64_t groups = 1) {
  int64_t perGroup = 8 / groups;
  return llvm::formatv(
             R"mlir(
    func.func @k(%f: tensor<{3}x{4}x{4}x3x3x{0}>, %i: tensor<1x{3}x{4}x8x8x{0}>)
        -> tensor<1x{3}x{4}x4x4x{1}>
        attributes {{rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel} {{
      %o = rock.conv(%f, %i) {{
        filter_layout = ["g", "k", "c", "0", "1"],
        input_layout = ["ni", "gi", "ci", "0i", "1i"],
        output_layout = ["no", "go", "ko", "0o", "1o"],
        dilations = [1 : index, 1 : index],
        strides = [2 : index, 2 : index],
        padding = [{2}]
      } : tensor<{3}x{4}x{4}x3x3x{0}>, tensor<1x{3}x{4}x8x8x{0}>
        -> tensor<1x{3}x{4}x4x4x{1}>
      return %o : tensor<1x{3}x{4}x4x4x{1}>
    })mlir",
             inType, outType, padding, groups, perGroup)
      .str();
}

std::string gemm(StringRef inType, StringRef outType) {
  return llvm::formatv(R"mlir(
    func.func @k(%a: tensor<1x16x16x{0}>, %b: tensor<1x16x16x{0}>)
        -> tensor<1x16x16x{1}>
        attributes {{rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel} {{
      %c = rock.gemm %a * %b
        : tensor<1x16x16x{0}> * tensor<1x16x16x{0}> -> tensor<1x16x16x{1}>
      return %c : tensor<1x16x16x{1}>
    })mlir",
                       inType, outType)
      .str();
}

constexpr StringLiteral kSymmetricPadding =
    "1 : index, 1 : index, 1 : index, 1 : index";
} // namespace

TEST(QuickTuningProblemMapTest, HitReturnsRowInOrder) {
  EXPECT_EQ(makeMap().lookup(10), SmallVector<StringRef>({"A", "B", "C"}));
}

TEST(QuickTuningProblemMapTest, HitReturnsOnlyItsOwnRow) {
  EXPECT_EQ(makeMap().lookup(20), SmallVector<StringRef>({"C"}));
  EXPECT_EQ(makeMap().lookup(30), SmallVector<StringRef>({"A"}));
}

TEST(QuickTuningProblemMapTest, MissReturnsEmpty) {
  QuickTuningProblemMap map = makeMap();
  // Below, between and above the stored hashes: the binary search must not
  // round to a neighbour, because a ranking only holds for the problem it was
  // measured on.
  EXPECT_TRUE(map.lookup(0).empty());
  EXPECT_TRUE(map.lookup(15).empty());
  EXPECT_TRUE(map.lookup(30 + 1).empty());
}

TEST(QuickTuningProblemMapTest, EmptyMapMisses) {
  QuickTuningProblemMap map(keyVersionHash, {}, {}, {});
  EXPECT_TRUE(map.lookup(10).empty());
}

TEST(QuickTuningProblemMapTest, RecordsLookupKeyVersionHash) {
  EXPECT_EQ(makeMap().getKeyVersionHash(), keyVersionHash);
}

TEST(QuickTuningProblemMapDeathTest, UnsortedProblemsTripTheAssert) {
  constexpr QuickTuningProblemRef unsorted[] = {{20, 0, 1}, {10, 1, 1}};
  EXPECT_DEBUG_DEATH(
      {
        QuickTuningProblemMap map(keyVersionHash, unsorted, perfConfigIndices,
                                  perfConfigs);
        (void)map;
      },
      "sorted by hash");
}

// The accumulator type `-t` implies (i8 -> i32) is the problem the int8 shards
// were measured on, not a different output type.
TEST(QuickTuningProblemKeyTest, ImpliedOutputTypeIsTheBaseProblem) {
  MLIRContext ctx;
  for (const std::string &func :
       {gemm("i8", "i32"), conv("i8", "i32", kSymmetricPadding)}) {
    auto key = keyFor(ctx, func);
    ASSERT_TRUE(key);
    EXPECT_FALSE(key->hasUnrepresentedFields())
        << key->unsupportedFields << " / " << key->untunableFields;
  }
}

// The gemm problem string records -out_datatype, so retuning can cover another
// output type; the conv one cannot, so the conv mode falls back silently.
TEST(QuickTuningProblemKeyTest, OutputTypeIsTunableOnlyWhereRecorded) {
  MLIRContext ctx;
  auto gemmKey = keyFor(ctx, gemm("f16", "f32"));
  ASSERT_TRUE(gemmKey);
  EXPECT_EQ(gemmKey->unsupportedFields, "output_data_type");
  EXPECT_EQ(gemmKey->untunableFields, "");

  auto convKey = keyFor(ctx, conv("f16", "f32", kSymmetricPadding));
  ASSERT_TRUE(convKey);
  EXPECT_EQ(convKey->unsupportedFields, "");
  EXPECT_EQ(convKey->untunableFields, "output_data_type");
}

// Groups are recorded (-g) and so worth retuning; the high pads are not.
TEST(QuickTuningProblemKeyTest, ConvGroupsWarnButAsymmetricPaddingIsSilent) {
  MLIRContext ctx;
  auto grouped = keyFor(ctx, conv("f16", "f16", kSymmetricPadding, 2));
  ASSERT_TRUE(grouped);
  EXPECT_EQ(grouped->unsupportedFields, "convolution_groups");
  EXPECT_EQ(grouped->untunableFields, "");

  auto trimmed = keyFor(ctx, conv("f16", "f16",
                                  "1 : index, 0 : index, 1 : index, "
                                  "0 : index"));
  ASSERT_TRUE(trimmed);
  EXPECT_EQ(trimmed->unsupportedFields, "");
  EXPECT_EQ(trimmed->untunableFields, "asymmetric_padding");
}
