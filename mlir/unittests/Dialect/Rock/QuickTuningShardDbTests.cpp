//===- QuickTuningShardDbTests.cpp - quick-tuning shard tests -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShardDb.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

namespace {

constexpr size_t kNumShards = 9;

std::optional<KernelType> parseKernelType(StringRef value) {
  return llvm::StringSwitch<std::optional<KernelType>>(value)
      .Case("gemm", KernelType::Gemm)
      .Case("conv", KernelType::Conv)
      .Case("attention", KernelType::Attention)
      .Default(std::nullopt);
}

Type parseDataType(MLIRContext &ctx, StringRef value) {
  return llvm::StringSwitch<Type>(value)
      .Case("f32", Float32Type::get(&ctx))
      .Case("f16", Float16Type::get(&ctx))
      .Case("i8", IntegerType::get(&ctx, 8))
      .Default(Type());
}

SmallVector<StringRef> lookup(StringRef arch, KernelType kernel, Type type,
                              uint64_t hash) {
  if (kernel == KernelType::Gemm || kernel == KernelType::Conv)
    return ParamLookupTable<GemmParamsAttr>::lookup(arch, kernel, type, hash);
  return ParamLookupTable<GemmGemmParamsAttr>::lookup(arch, kernel, type, hash);
}

bool isNonSplitK(StringRef config) {
  if (config.contains("splitKFactor=1"))
    return true;
  auto [prefix, rest] = config.split(':');
  (void)prefix;
  auto [version, fieldsText] = rest.split(':');
  if (!version.starts_with("v") || fieldsText.empty())
    return true;
  SmallVector<StringRef> fields;
  fieldsText.split(fields, ',');
  return fields.size() > 7 && fields[7] == "1";
}

TEST(QuickTuningShardDbTest, ShardsAreWellFormed) {
  llvm::StringSet<> keys;
  for (const QuickTuningShard &shard : getQuickTuningShards()) {
    EXPECT_TRUE(keys.insert(shard.key).second) << shard.key;
    EXPECT_GT(shard.numConfigs, 0u) << shard.key;
    EXPECT_GT(shard.numProblems, 0u) << shard.key;
    EXPECT_GT(shard.numTopN, 0u) << shard.key;
    ASSERT_NE(shard.problems, nullptr) << shard.key;
    ASSERT_NE(shard.problemConfigs, nullptr) << shard.key;

    for (size_t i = 0; i < shard.numProblems; ++i) {
      EXPECT_NE(shard.problems[i], kQuickTuningNoProblem) << shard.key;
      if (i)
        EXPECT_LT(shard.problems[i - 1], shard.problems[i]) << shard.key;
      bool hasNonSplitK = false;
      for (uint16_t index : shard.getProblemConfigs(i)) {
        if (index == kQuickTuningNoConfig)
          continue;
        ASSERT_LT(index, shard.numConfigs) << shard.key;
        hasNonSplitK |= isNonSplitK(shard.getConfigs()[index]);
      }
      EXPECT_TRUE(hasNonSplitK) << shard.key << " problem " << i;
    }
  }
  EXPECT_EQ(keys.size(), kNumShards);
}

TEST(QuickTuningShardDbTest, HitReturnsExactlyRecordedTopN) {
  MLIRContext ctx;
  size_t checked = 0;
  for (const QuickTuningShard &shard : getQuickTuningShards()) {
    auto [arch, rest] = shard.key.split('_');
    auto [kernelText, typeText] = rest.rsplit('_');
    std::optional<KernelType> kernel = parseKernelType(kernelText);
    Type type = parseDataType(ctx, typeText);
    if (!kernel || !type)
      continue;

    for (size_t i = 0; i < shard.numProblems; ++i) {
      SmallVector<StringRef> expected;
      for (uint16_t index : shard.getProblemConfigs(i))
        if (index != kQuickTuningNoConfig)
          expected.push_back(shard.getConfigs()[index]);
      SmallVector<StringRef> actual =
          lookup(arch, *kernel, type, shard.problems[i]);
      EXPECT_EQ(llvm::join(actual, "\n"), llvm::join(expected, "\n"))
          << shard.key << " problem " << i;
      EXPECT_EQ(actual.size(), shard.numTopN) << shard.key << " problem " << i;
      ++checked;
    }
  }
  EXPECT_GT(checked, 0u);
}

TEST(QuickTuningShardDbTest, MissAndNoProblemReturnTheCover) {
  MLIRContext ctx;
  for (const QuickTuningShard &shard : getQuickTuningShards()) {
    auto [arch, rest] = shard.key.split('_');
    auto [kernelText, typeText] = rest.rsplit('_');
    std::optional<KernelType> kernel = parseKernelType(kernelText);
    Type type = parseDataType(ctx, typeText);
    if (!kernel || !type)
      continue;
    SmallVector<StringRef> cover =
        lookup(arch, *kernel, type, kQuickTuningNoProblem);
    EXPECT_EQ(llvm::join(cover, "\n"),
              llvm::join(lookup(arch, *kernel, type, uint64_t{1}), "\n"));
    EXPECT_EQ(llvm::join(cover, "\n"),
              llvm::join(lookup(arch, *kernel, type, ~uint64_t{0}), "\n"));
  }
}

TEST(QuickTuningShardDbTest, FallbackDoesNotProbeProblemMap) {
  MLIRContext ctx;
  const QuickTuningShard *shard = nullptr;
  for (const QuickTuningShard &candidate : getQuickTuningShards())
    if (candidate.key == "gfx908_gemm_i8")
      shard = &candidate;
  ASSERT_NE(shard, nullptr);
  Type i8 = IntegerType::get(&ctx, 8);
  EXPECT_EQ(
      llvm::join(lookup("gfx906", KernelType::Gemm, i8, shard->problems[0]),
                 "\n"),
      llvm::join(lookup("gfx906", KernelType::Gemm, i8, kQuickTuningNoProblem),
                 "\n"));
}

} // namespace
