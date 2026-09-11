//===- QuickTuningShardDbTests.cpp - Tests for the quick-tuning shards ----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests over the whole compiled-in quick-tuning database: that the generated
// shards are well formed, that an unknown problem still sweeps the untouched
// set cover for every key, and that a known problem's list leads with the bests
// recorded for it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/Tuning/QuickTuningProblemKey.h"
#include "mlir/Dialect/Rock/Tuning/QuickTuningShardDb.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::rock;

namespace {

// The shard whose recorded problems the narrowing tests below exercise. It has
// to be a key `lookup` can reach exactly, so that those tests observe the shard
// they name.
constexpr StringLiteral kProblemShardKey = "gfx908_gemm_i8";

// The number of keys the database holds. Pinned so that adding or dropping a
// shard is a deliberate edit to the index and to this number, rather than
// something a stale generated file can do quietly.
constexpr size_t kNumShards = 92;

// Every key the pre-sharding table held is still spelled with one of these, so
// a shard whose key names something else was written against a different
// `makeKey` than the one lookups use.
constexpr StringLiteral kKernelTypes[] = {
    "gemm", "conv", "attention", "gemmelementwisegemm", "convelementwisegemm"};

// The keys that deliberately ship no shard of their own, each paired with the
// key it is expected to reach instead. gfx1100 and gfx1201 stopped carrying
// their own lists for these operations when gfx1101 and gfx1200 were tuned;
// rather than duplicating the winner's data under a second key, they are left
// to `findFallback`. See ShardlessKeysReachTheirFallbacksList.
struct ShardlessKey {
  StringLiteral key;
  StringLiteral fallback;
};
constexpr ShardlessKey kShardlessKeys[] = {
    {"gfx1100_attention_f16", "gfx1101_attention_f16"},
    {"gfx1100_attention_f32", "gfx1101_attention_f32"},
    {"gfx1100_conv_f16", "gfx1101_conv_f16"},
    {"gfx1100_conv_f32", "gfx1101_conv_f32"},
    {"gfx1100_conv_i8", "gfx1101_conv_i8"},
    {"gfx1100_gemm_f16", "gfx1101_gemm_f16"},
    {"gfx1100_gemm_f32", "gfx1101_gemm_f32"},
    {"gfx1100_gemm_i8", "gfx1101_gemm_i8"},
    {"gfx1201_attention_f16", "gfx1200_attention_f16"},
    {"gfx1201_attention_f32", "gfx1200_attention_f32"},
    {"gfx1201_conv_f16", "gfx1200_conv_f16"},
    {"gfx1201_conv_f32", "gfx1200_conv_f32"},
    {"gfx1201_conv_i8", "gfx1200_conv_i8"},
};

// Lists join into one string for comparison: gtest prints the mismatching lines
// that way, where a raw container diff would print bytes.
std::string joinConfigs(ArrayRef<StringRef> configs) {
  return llvm::join(configs, "\n");
}

// The shard answering to `key`, or null when no shard does.
const QuickTuningShard *findShard(StringRef key) {
  for (const QuickTuningShard &shard : getQuickTuningShards())
    if (shard.key == key)
      return &shard;
  return nullptr;
}

// The set cover `shard` records, which is what an unknown problem sweeps.
SmallVector<StringRef> getCover(const QuickTuningShard &shard) {
  SmallVector<StringRef> cover;
  for (uint16_t configIdx : shard.getCover())
    cover.push_back(shard.getConfigs()[configIdx]);
  return cover;
}

// The list `shard` is specified to hand out for `problemHash`, computed from
// the shard's arrays independently of the lookup being tested: the recorded
// non-split-K best, then the recorded split-K one, then the set cover, dropping
// sentinels and repeats, capped once a problem was hit.
SmallVector<StringRef> expectedList(const QuickTuningShard &shard,
                                    uint64_t problemHash) {
  ArrayRef<uint64_t> problems = shard.getProblems();
  auto *hit = llvm::find(problems, problemHash);
  if (hit == problems.end())
    return getCover(shard);

  size_t problemIdx = hit - problems.begin();
  SmallVector<StringRef> expected;
  for (uint16_t configIdx :
       {shard.getBestNonSplitK(problemIdx), shard.getBestSplitK(problemIdx)}) {
    if (expected.size() >= kQuickTuningListMaxDefault)
      break;
    if (configIdx != kQuickTuningNoConfig)
      expected.push_back(shard.getConfigs()[configIdx]);
  }
  for (StringRef config : getCover(shard)) {
    if (expected.size() >= kQuickTuningListMaxDefault)
      break;
    if (!llvm::is_contained(expected, config))
      expected.push_back(config);
  }
  return expected;
}

// A key's kernel type as a `KernelType`, or nullopt for one no `makeKey` emits.
std::optional<KernelType> parseKernelType(StringRef kernelType) {
  return llvm::StringSwitch<std::optional<KernelType>>(kernelType)
      .Case("gemm", KernelType::Gemm)
      // Both convolution directions share the "conv" spelling.
      .Case("conv", KernelType::Conv)
      .Case("attention", KernelType::Attention)
      .Case("gemmelementwisegemm", KernelType::GemmElementwiseGemm)
      .Case("convelementwisegemm", KernelType::ConvElementwiseGemm)
      .Default(std::nullopt);
}

// A type `getDataTypeString` spells as `dataType`, or null when none does. The
// `_bf16` keys are the ones that come back null: bf16 and f16 share the "f16"
// spelling, so no lookup can reach a `_bf16` key. They are dead entries carried
// over unchanged from the pre-sharding table.
Type parseDataType(MLIRContext &ctx, StringRef dataType) {
  return llvm::StringSwitch<Type>(dataType)
      .Case("f32", Float32Type::get(&ctx))
      .Case("f16", Float16Type::get(&ctx))
      .Case("i8", IntegerType::get(&ctx, 8))
      .Case("fp8", Float8E4M3FNType::get(&ctx))
      .Case("f4", Float4E2M1FNType::get(&ctx))
      .Default(Type());
}

// `lookup` through whichever instantiation owns `kernelType`. They share one
// table, so this only keeps each key going through the class that would really
// ask for it.
SmallVector<StringRef> lookup(StringRef arch, KernelType kernelType,
                              Type dataType, uint64_t problemHash) {
  if (kernelType == KernelType::Gemm || kernelType == KernelType::Conv)
    return ParamLookupTable<GemmParamsAttr>::lookup(arch, kernelType, dataType,
                                                    problemHash);
  return ParamLookupTable<GemmGemmParamsAttr>::lookup(arch, kernelType,
                                                      dataType, problemHash);
}

// Runs `body` over every shard whose key a lookup can reach exactly, passing
// the arguments that reach it. Reports how many keys it skipped so that a
// mistake in the key-to-argument mapping cannot quietly empty out a sweep.
void forEachReachableKey(MLIRContext &ctx,
                         llvm::function_ref<void(const QuickTuningShard &,
                                                 StringRef, KernelType, Type)>
                             body) {
  size_t skipped = 0;
  for (const QuickTuningShard &shard : getQuickTuningShards()) {
    auto [arch, rest] = shard.key.split('_');
    auto [kernelType, dataType] = rest.rsplit('_');
    std::optional<KernelType> kernel = parseKernelType(kernelType);
    Type type = parseDataType(ctx, dataType);
    if (!kernel || !type) {
      ++skipped;
      continue;
    }
    body(shard, arch, *kernel, type);
  }
  // The `_bf16` keys, and nothing else.
  EXPECT_EQ(skipped, 7u);
}

class QuickTuningShardDbTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Every expectation here is written against the default cap.
    if (llvm::sys::Process::GetEnv(kQuickTuningListMaxEnvVar))
      GTEST_SKIP() << kQuickTuningListMaxEnvVar << " is set";
  }

  // The shard the narrowing tests exercise.
  const QuickTuningShard &getProblemShard() {
    for (const QuickTuningShard &shard : getQuickTuningShards())
      if (shard.key == kProblemShardKey)
        return shard;
    llvm::report_fatal_error(Twine("no ") + kProblemShardKey + " shard");
  }

  MLIRContext ctx;
};

TEST_F(QuickTuningShardDbTest, ShardsAreWellFormed) {
  llvm::StringSet<> keys;
  for (const QuickTuningShard &shard : getQuickTuningShards()) {
    EXPECT_TRUE(keys.insert(shard.key).second) << "duplicate key " << shard.key;

    auto [arch, rest] = shard.key.split('_');
    auto [kernelType, dataType] = rest.rsplit('_');
    EXPECT_TRUE(arch.starts_with("gfx")) << "for key " << shard.key;
    EXPECT_FALSE(dataType.empty()) << "for key " << shard.key;
    EXPECT_TRUE(llvm::is_contained(kKernelTypes, kernelType))
        << "for key " << shard.key;

    // A key with an empty cover would make `lookup` hand out nothing at all.
    EXPECT_GT(shard.numCover, 0u) << "for key " << shard.key;
    for (uint16_t configIdx : shard.getCover())
      EXPECT_LT(configIdx, shard.numConfigs) << "for key " << shard.key;

    // Both problem arrays are indexed off `numProblems`, so a null one has to
    // mean a zero count and the reverse.
    EXPECT_EQ(shard.numProblems == 0, shard.problems == nullptr)
        << "for key " << shard.key;
    EXPECT_EQ(shard.numProblems == 0, shard.problemConfigs == nullptr)
        << "for key " << shard.key;

    for (size_t i = 0; i < shard.numProblems; ++i) {
      // Zero is reserved for "the caller named no problem", so a shard must not
      // record it: see kQuickTuningNoProblem.
      EXPECT_NE(shard.problems[i], kQuickTuningNoProblem)
          << "for key " << shard.key;
      // `lookup` binary-searches the hashes.
      if (i > 0)
        EXPECT_LT(shard.problems[i - 1], shard.problems[i])
            << "for key " << shard.key;
      for (uint16_t configIdx :
           {shard.getBestNonSplitK(i), shard.getBestSplitK(i)})
        EXPECT_TRUE(configIdx == kQuickTuningNoConfig ||
                    configIdx < shard.numConfigs)
            << "for key " << shard.key;
      // Every problem has to name a non-split-K best. A row naming neither
      // config reorders nothing, and one naming only a split-K config would put
      // a split-K config at `front()`, where a skip-benchmarking consumer needs
      // one that is legal in every fusion context. Data with such a problem in
      // it has to be resolved -- by leading with the set cover for it, say --
      // rather than shipped.
      EXPECT_NE(shard.getBestNonSplitK(i), kQuickTuningNoConfig)
          << "for key " << shard.key;
    }
  }
  EXPECT_EQ(keys.size(), kNumShards);
}

TEST_F(QuickTuningShardDbTest, UnknownProblemSweepsTheWholeCover) {
  // The regression that matters for everything that already worked: splitting
  // the table into shards, and adding per-problem data to them, must leave the
  // list an unknown problem sweeps exactly as it was -- the whole set cover, in
  // its recorded order, uncapped. `problemHash == 0` is the same case, and is
  // what every caller with no problem to name passes.
  forEachReachableKey(ctx, [&](const QuickTuningShard &shard, StringRef arch,
                               KernelType kernelType, Type dataType) {
    SmallVector<StringRef> cover = getCover(shard);
    for (uint64_t problemHash :
         {kQuickTuningNoProblem, uint64_t{1}, ~uint64_t{0}})
      EXPECT_EQ(joinConfigs(cover),
                joinConfigs(lookup(arch, kernelType, dataType, problemHash)))
          << "for key " << shard.key << " and hash " << problemHash;
  });
}

TEST_F(QuickTuningShardDbTest, ShardlessKeysReachTheirFallbacksList) {
  // An architecture that stopped carrying its own list for an operation gets no
  // shard at all, not a copy of the winner's data under a second key: shards
  // are written whole per key, so a duplicate would have to be rewritten twice
  // every time the original is retuned, and would silently rot when it wasn't.
  // What makes that safe is that `findFallback` already resolves these keys to
  // the architecture whose measurements superseded them, and key resolution is
  // problem-agnostic, so the fallback's shard is probed exactly as an exact hit
  // would be. Pinned here so that a change to the fallback order cannot quietly
  // send one of these architectures to an unrelated list.
  for (const ShardlessKey &entry : kShardlessKeys) {
    EXPECT_EQ(findShard(entry.key), nullptr)
        << entry.key << " ships a shard again; drop it from kShardlessKeys or "
        << "stop generating it";
    const QuickTuningShard *fallback = findShard(entry.fallback);
    ASSERT_NE(fallback, nullptr)
        << "no " << entry.fallback << " shard to back " << entry.key;

    auto [arch, rest] = entry.key.split('_');
    auto [kernelType, dataType] = rest.rsplit('_');
    std::optional<KernelType> kernel = parseKernelType(kernelType);
    Type type = parseDataType(ctx, dataType);
    ASSERT_TRUE(kernel && type) << "for key " << entry.key;

    // The whole set cover, uncapped, just as the pre-sharding table handed out
    // for these keys before their lists were removed.
    EXPECT_EQ(joinConfigs(getCover(*fallback)),
              joinConfigs(lookup(arch, *kernel, type, kQuickTuningNoProblem)))
        << "for key " << entry.key;
  }
}

TEST_F(QuickTuningShardDbTest, KnownProblemLeadsWithItsRecordedBests) {
  // Every recorded problem, of every key: the list leads with the best
  // non-split-K config and then the best split-K one, skips whichever of the
  // two is absent, and is backfilled from the set cover without repeats up to
  // the cap.
  size_t problemsChecked = 0;
  forEachReachableKey(ctx, [&](const QuickTuningShard &shard, StringRef arch,
                               KernelType kernelType, Type dataType) {
    for (uint64_t problemHash : shard.getProblems()) {
      EXPECT_EQ(joinConfigs(expectedList(shard, problemHash)),
                joinConfigs(lookup(arch, kernelType, dataType, problemHash)))
          << "for key " << shard.key << " and hash " << problemHash;
      ++problemsChecked;
    }
  });
  // The sweep above is vacuous on a database with no measurements in it, which
  // is a state a bad regeneration can leave behind.
  EXPECT_GT(problemsChecked, 0u);
}

TEST_F(QuickTuningShardDbTest, SplitKBestFollowsTheNonSplitKOne) {
  // The two slots per problem, spelled out on the shard that has them. The
  // order is load-bearing: `front()` is what a MIGRAPHX_SKIP_BENCHMARKING
  // consumer runs and what a perf-database miss falls back to, so it has to be
  // a config that is legal in every fusion context -- which a split-K one is
  // not.
  const QuickTuningShard &shard = getProblemShard();
  ASSERT_GT(shard.numProblems, 0u)
      << kProblemShardKey << " no longer records any problem; point "
      << "kProblemShardKey at a shard that does";

  Type i8 = IntegerType::get(&ctx, 8);
  SmallVector<StringRef> cover = getCover(shard);
  size_t withBothSlots = 0, withSentinel = 0;
  for (size_t i = 0; i < shard.numProblems; ++i) {
    SmallVector<StringRef> list =
        lookup("gfx908", KernelType::Gemm, i8, shard.problems[i]);
    uint16_t nonSplitK = shard.getBestNonSplitK(i);
    uint16_t splitK = shard.getBestSplitK(i);
    ASSERT_NE(nonSplitK, kQuickTuningNoConfig) << "for problem " << i;
    ASSERT_FALSE(list.empty()) << "for problem " << i;
    EXPECT_EQ(list[0], shard.getConfigs()[nonSplitK]) << "for problem " << i;
    if (splitK == kQuickTuningNoConfig) {
      // With no split-K best recorded the backfill starts right after the
      // non-split-K one, rather than leaving a hole for the missing slot.
      ASSERT_GT(list.size(), 1u) << "for problem " << i;
      EXPECT_TRUE(llvm::is_contained(cover, list[1])) << "for problem " << i;
      ++withSentinel;
      continue;
    }
    ASSERT_GT(list.size(), 1u) << "for problem " << i;
    EXPECT_EQ(list[1], shard.getConfigs()[splitK]) << "for problem " << i;
    ++withBothSlots;
  }
  // Both slot shapes have to occur, or one of the two branches above is never
  // taken. Split-K is the rare one: 83 of 2,479 shipped configs use it.
  EXPECT_GT(withBothSlots, 0u);
  EXPECT_GT(withSentinel, 0u);
}

TEST_F(QuickTuningShardDbTest, BackfillDoesNotRepeatARecordedBest) {
  // A recorded best is usually in the set cover as well, and must not be swept
  // twice.
  const QuickTuningShard &shard = getProblemShard();
  ASSERT_GT(shard.numProblems, 0u);

  Type i8 = IntegerType::get(&ctx, 8);
  size_t deduped = 0;
  for (size_t i = 0; i < shard.numProblems; ++i) {
    SmallVector<StringRef> list =
        lookup("gfx908", KernelType::Gemm, i8, shard.problems[i]);
    llvm::StringSet<> seen;
    for (StringRef config : list)
      EXPECT_TRUE(seen.insert(config).second)
          << config << " repeats for problem " << i;

    // The dedup is only exercised where a recorded best is in the cover too,
    // which is what makes the list shorter than bests plus cover.
    size_t bests = (shard.getBestNonSplitK(i) != kQuickTuningNoConfig) +
                   (shard.getBestSplitK(i) != kQuickTuningNoConfig);
    if (list.size() < bests + shard.numCover)
      ++deduped;
  }
  EXPECT_GT(deduped, 0u);
}

TEST_F(QuickTuningShardDbTest, FallbackResolvedKeyStillFindsItsProblem) {
  // Key resolution is deliberately problem-agnostic: `findFallback` is
  // untouched and the problem array is probed on whatever shard it lands on. So
  // a problem measured for gfx908_gemm_i8 is still found by a lookup that only
  // reached that key by substitution -- f4 borrowing i8's list, or gfx906
  // borrowing gfx908's. This is what dropping the data type from the hashed key
  // buys.
  const QuickTuningShard &shard = getProblemShard();
  ASSERT_GT(shard.numProblems, 0u);
  uint64_t problemHash = shard.problems[0];

  SmallVector<StringRef> exact = lookup("gfx908", KernelType::Gemm,
                                        IntegerType::get(&ctx, 8), problemHash);
  // Narrowing happened at all, or the comparisons below hold vacuously.
  EXPECT_NE(joinConfigs(getCover(shard)), joinConfigs(exact));

  // f4 has no list of its own anywhere and no same-precision relative, so it
  // substitutes the data type: gfx908_gemm_f4 -> gfx908_gemm_i8.
  EXPECT_EQ(joinConfigs(exact),
            joinConfigs(lookup("gfx908", KernelType::Gemm,
                               Float4E2M1FNType::get(&ctx), problemHash)));
  // gfx906 has no lists of its own, so it substitutes the architecture:
  // gfx906_gemm_i8 -> gfx908_gemm_i8.
  EXPECT_EQ(joinConfigs(exact),
            joinConfigs(lookup("gfx906", KernelType::Gemm,
                               IntegerType::get(&ctx, 8), problemHash)));
}

TEST_F(QuickTuningShardDbTest, CrossOpFallbackCannotFalseHit) {
  // The other half of leaving key resolution problem-agnostic: a substitution
  // that crossed operations must not report a hit. It cannot, because the
  // kernel type is in the hashed key, so the hash of a problem of one operation
  // is simply absent from another operation's shard and the set cover comes
  // back.
  //
  // Spelled here on the one shard that has problems to false-hit: it answers to
  // a gemm key, and probing it with the same shape as a gemm+gemm, a conv or an
  // attention problem misses.
  const QuickTuningShard &shard = getProblemShard();
  ASSERT_GT(shard.numProblems, 0u);

  Type i8 = IntegerType::get(&ctx, 8);
  StringRef fields =
      " -transA false -transB false -transO false -g 1 -m 1024 -n 1024 -k 1024";
  uint64_t gemmHash = hashQuickTuningProblemKey((Twine("Gemm") + fields).str());
  ASSERT_TRUE(llvm::is_contained(shard.getProblems(), gemmHash));
  EXPECT_NE(joinConfigs(getCover(shard)),
            joinConfigs(lookup("gfx908", KernelType::Gemm, i8, gemmHash)));

  for (StringRef kernelType :
       {"Conv", "Attention", "GemmElementwiseGemm", "ConvElementwiseGemm"}) {
    uint64_t otherHash =
        hashQuickTuningProblemKey((Twine(kernelType) + fields).str());
    EXPECT_EQ(joinConfigs(getCover(shard)),
              joinConfigs(lookup("gfx908", KernelType::Gemm, i8, otherHash)))
        << "for kernel type " << kernelType;
  }

  // And from the other direction: a gemm+gemm lookup on gfx908 resolves to the
  // attention shard, which holds no problems, so even the gemm problem's own
  // hash leaves it with attention's set cover.
  SmallVector<StringRef> gemmGemm =
      lookup("gfx908", KernelType::GemmElementwiseGemm, i8, gemmHash);
  EXPECT_EQ(joinConfigs(gemmGemm),
            joinConfigs(lookup("gfx908", KernelType::Attention, i8,
                               kQuickTuningNoProblem)));
}

TEST_F(QuickTuningShardDbTest, UntunedKeysBehaveAsBeforeSharding) {
  // The keys that carry no measurements at all -- attention_i8, attention_bf16
  // and gemm_fp8 in the shipped data -- have to be indistinguishable from their
  // pre-sharding selves, whatever problem asks. `UnknownProblemSweepsTheWhole`
  // `Cover` already pins that for arbitrary hashes; this adds the hashes the
  // database does record elsewhere, which are the ones that could cross over.
  const QuickTuningShard &shard = getProblemShard();
  ASSERT_GT(shard.numProblems, 0u);

  size_t untunedKeys = 0;
  forEachReachableKey(ctx, [&](const QuickTuningShard &candidate,
                               StringRef arch, KernelType kernelType,
                               Type dataType) {
    if (candidate.numProblems > 0)
      return;
    ++untunedKeys;
    for (uint64_t problemHash : shard.getProblems())
      EXPECT_EQ(joinConfigs(getCover(candidate)),
                joinConfigs(lookup(arch, kernelType, dataType, problemHash)))
          << "for key " << candidate.key;
  });
  EXPECT_EQ(untunedKeys, kNumShards - 7 - 1);
}

} // namespace
