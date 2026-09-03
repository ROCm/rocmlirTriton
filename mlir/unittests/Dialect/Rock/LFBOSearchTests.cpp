//===- LFBOSearchTests.cpp - Driving the LFBO search without a GPU --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The LFBO search only ever asks for the timings of the batch it just proposed,
// so it can be driven here with made-up ones. That is enough to check what it
// does with the space it is given, which is the part a tuning run cannot be
// asked about cheaply.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/TuningSearch.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/xxhash.h"

#include "LFBOSearch.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>

using namespace mlir;
using namespace mlir::rock;

namespace {

struct GemmModule {
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;

  GemmModule(int64_t m, int64_t n, int64_t k, StringRef arch) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    OpBuilder b(&ctx);
    Location loc = b.getUnknownLoc();
    Type elem = b.getF16Type();
    auto aType = RankedTensorType::get({1, m, k}, elem);
    auto bType = RankedTensorType::get({1, k, n}, elem);
    auto cType = RankedTensorType::get({1, m, n}, elem);

    module = ModuleOp::create(loc);
    b.setInsertionPointToEnd(module->getBody());
    auto func = func::FuncOp::create(
        b, loc, "test", b.getFunctionType({aType, bType}, {cType}));
    Block *body = func.addEntryBlock();
    b.setInsertionPointToStart(body);
    auto gemmOp = GemmOp::create(
        b, loc, /*c=*/cType, /*a=*/body->getArgument(0),
        /*b=*/body->getArgument(1), /*scaleA=*/Value(), /*scaleB=*/Value(),
        /*aTransposed=*/UnitAttr{}, /*bTransposed=*/UnitAttr{},
        /*oTransposed=*/UnitAttr{}, /*aScaleTransposed=*/UnitAttr{},
        /*bScaleTransposed=*/UnitAttr{}, /*quantBlockSize=*/IntegerAttr{},
        /*params=*/nullptr);
    func::ReturnOp::create(b, loc, gemmOp.getResult());
    (*module)->setAttr(rock::ArchAttr::getMnemonic(),
                       b.getStringAttr(Twine("amdgcn-amd-amdhsa:") + arch));
  }
};

// Times a config so that the search has something to prefer. The value is
// arbitrary, but it has to be the config's own: a copy carries the time of the
// config it stands on from one generation to the next and calls a neighbour an
// improvement by comparing against it (`advanceSearchCopies`), so a timing that
// depended on which batch a config arrived in would make every comparison
// meaningless and every copy run out of patience at once.
//
// Spread over a decade so that any two configs differ by far more than
// `minImprovementDelta`, which a search reads as no improvement at all.
//
// Hashed with xxh3 rather than `llvm::hash_value`, whose execution seed is an
// address under `LLVM_ENABLE_ABI_BREAKING_CHECKS` and so differs from run to
// run. A search is deliberately deterministic given its seed (`LFBOOptions`),
// and these tests are only worth anything while they inherit that.
double fakeTime(StringRef perfConfig) {
  return 1000.0 + static_cast<double>(llvm::xxh3_64bits(perfConfig) % 9000);
}

std::vector<BenchmarkResult> fakeTimings(ArrayRef<PerfConfigString> batch) {
  std::vector<BenchmarkResult> results;
  for (const PerfConfigString &perfConfig : batch)
    results.push_back(
        {perfConfig, fakeTime(perfConfig), BenchmarkResult::Status::Success});
  return results;
}

// Runs the search for `generations` batches, answering each with made-up
// timings, and returns the batches as it proposed them. They are kept apart
// because the first is not like the others: it is the initial population, which
// includes the quick list rather than anything the search chose.
std::vector<std::vector<PerfConfigString>>
runSearch(ModuleOp mod, unsigned generations, unsigned seed = 0) {
  LFBOOptions options;
  // Enough of a first batch to fit a model on, but small enough to stay quick.
  options.initialPopulation = 40;
  options.numNeighbors = 40;
  if (seed)
    options.seed = seed;
  std::unique_ptr<TuningSearchStrategy> search =
      createLFBOSearchStrategy(mod, options);

  std::vector<std::vector<PerfConfigString>> batches;
  std::vector<BenchmarkResult> results;
  for (unsigned generation = 0; generation < generations; ++generation) {
    std::vector<PerfConfigString> batch = search->getPerfConfigBatch(results);
    if (batch.empty())
      break;
    results = fakeTimings(batch);
    batches.push_back(std::move(batch));
  }
  return batches;
}

std::vector<PerfConfigString>
flatten(ArrayRef<std::vector<PerfConfigString>> batches) {
  std::vector<PerfConfigString> all;
  for (const std::vector<PerfConfigString> &batch : batches)
    all.insert(all.end(), batch.begin(), batch.end());
  return all;
}

// Every value `proposed` was seen to carry for one parameter.
std::set<int64_t> valuesTried(MLIRContext &ctx,
                              ArrayRef<PerfConfigString> proposed,
                              StringRef param) {
  std::set<int64_t> tried;
  for (const PerfConfigString &perfConfig : proposed) {
    auto params = GemmParamsAttr::get(StringAttr::get(&ctx, perfConfig));
    if (!params) {
      ADD_FAILURE() << "the search proposed an unparseable config: "
                    << std::string(perfConfig);
      continue;
    }
    SmallVector<StringRef> names;
    SmallVector<int64_t> values;
    params.getParamNames(names);
    params.getParamValues(values);
    for (auto [name, value] : llvm::zip_equal(names, values))
      if (name == param)
        tried.insert(value);
  }
  return tried;
}

// The knobs are the parameters the enumerated space pins, so a search that
// never sets one is a search leaving them on the table. It has to move them off
// the default it is seeded with, which is also the one case where a copy starts
// at a value that is not on the axis it walks.
//
// `useBf16x3ForF32` is not among them: this GEMM is f16, which is a problem the
// axes pin the knob on, see `Bf16x3ForF32IsExploredOnlyOnAnF32Dot`
// (TuningParamAxesTests.cpp).
//
// A few seeds are asked together rather than one run alone: a generation
// benchmarks only a tenth of the neighbours it builds, and which tenth is up to
// a model fitted on the invented timings above, so whether one run happens to
// spend that tenth on a given knob says nothing. A knob no seed ever moves is
// the one worth reporting, being pinned by the axes or out of the moves' reach.
TEST(LFBOSearchTest, ExploresTheKnobs) {
  const StringRef knobs[] = {"useAsyncCopy",         "useBlockPingpong",
                             "useInThreadTranspose", "useBufferOps",
                             "useBufferAtomics",     "useReductionLayout",
                             "useOptimizeEpilogue"};

  std::set<StringRef> moved;
  for (unsigned seed : {1u, 2u, 3u}) {
    GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
    std::vector<PerfConfigString> proposed =
        flatten(runSearch(*e.module, 4, seed));
    ASSERT_FALSE(proposed.empty()) << "the search proposed nothing at all";
    for (StringRef knob : knobs)
      for (int64_t value : valuesTried(e.ctx, proposed, knob))
        if (value != kKnobDefault)
          moved.insert(knob);
  }

  for (StringRef knob : knobs)
    EXPECT_TRUE(moved.count(knob))
        << knob << " was never tried off its default";
}

// `kpack` and `wavesPerEU` are pinned by the enumerated space for the same
// reason the knobs are, and a search that leaves them where it found them gives
// up the one advantage it has over brute force, which is that a wider space
// costs it nothing. Both are legal beyond the pin: `kpack=2` is what the quick
// configs and the gemm+gemm space use, and a nonzero `wavesPerEU` is an
// occupancy hint the backend honours.
TEST(LFBOSearchTest, ExploresWhatTheEnumeratedSpacePins) {
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::vector<PerfConfigString> proposed = flatten(runSearch(*e.module, 4));
  ASSERT_FALSE(proposed.empty()) << "the search proposed nothing at all";

  EXPECT_TRUE(valuesTried(e.ctx, proposed, "kpack").count(2))
      << "kpack was never tried above one";
  std::set<int64_t> wavesPerEU = valuesTried(e.ctx, proposed, "wavesPerEU");
  EXPECT_TRUE(llvm::any_of(wavesPerEU, [](int64_t value) { return value > 0; }))
      << "wavesPerEU was left at zero, so no occupancy hint was ever asked for";
}

// Every config the search itself builds has to be one the space admits, since
// that is the only reason to believe it can be compiled. The initial population
// is exempt: it carries the quick list, whose configs come from the heuristic
// and need not be in the space being searched.
TEST(LFBOSearchTest, ProposesOnlyAdmissibleConfigs) {
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  std::vector<std::vector<PerfConfigString>> batches = runSearch(*e.module, 4);
  ASSERT_GT(batches.size(), 1u) << "the search never got past its first batch";
  std::vector<PerfConfigString> proposed =
      flatten(ArrayRef(batches).drop_front());
  ASSERT_FALSE(proposed.empty());

  for (const PerfConfigString &perfConfig : proposed) {
    auto params = GemmParamsAttr::get(StringAttr::get(&e.ctx, perfConfig));
    ASSERT_TRUE(params);
    SmallVector<int64_t> values;
    params.getParamValues(values);
    EXPECT_TRUE(axes->isFeasible(values))
        << "proposed a config the space disowns: " << std::string(perfConfig);
  }
}

// The effort is the one budget a tuning run gets to set, so the cheaper one has
// to both ask for less and be what stops the search.
TEST(LFBOSearchTest, QuickEffortSpendsLessThanFull) {
  LFBOOptions full;
  full.setEffort(SearchEffort::Full);
  LFBOOptions quick;
  quick.setEffort(SearchEffort::Quick);
  EXPECT_LT(quick.initialPopulation, full.initialPopulation);
  EXPECT_LT(quick.copies, full.copies);
  EXPECT_LT(quick.maxGenerations, full.maxGenerations);

  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  // Narrower than the effort asks for, so that the test stays quick. It is the
  // number of generations that is under test here, not the size of one.
  quick.numNeighbors = 40;
  std::unique_ptr<TuningSearchStrategy> search =
      createLFBOSearchStrategy(*e.module, quick);

  // The initial population, and then at most one batch per generation. The
  // search should stop on its own well before the loop's own bound.
  const unsigned allowed = quick.maxGenerations + 1;
  unsigned batches = 0;
  size_t firstBatchSize = 0;
  std::vector<BenchmarkResult> results;
  for (; batches < 2 * allowed; ++batches) {
    std::vector<PerfConfigString> batch = search->getPerfConfigBatch(results);
    if (batch.empty())
      break;
    if (batches == 0)
      firstBatchSize = batch.size();
    results = fakeTimings(batch);
  }
  EXPECT_GT(batches, 1u) << "the search never got past its first batch";
  EXPECT_LE(batches, allowed)
      << "the quick effort ran for more generations than it allows";
  // The quick list is a floor on the first batch, so this is not the initial
  // population itself, only far short of what the full effort would benchmark.
  EXPECT_LT(firstBatchSize, full.initialPopulation)
      << "the quick effort opened with a full effort's initial population";
}

// A trace has to be readable on its own and has to agree with what the search
// did, since it is all a finished tuning run can be asked about afterwards.
TEST(LFBOSearchTest, TracesEachIteration) {
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  SmallString<128> tracePath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("lfbo-trace", "jsonl", tracePath));

  LFBOOptions options;
  options.initialPopulation = 40;
  options.numNeighbors = 40;
  options.trace = openTrace(tracePath);
  std::unique_ptr<TuningSearchStrategy> search =
      createLFBOSearchStrategy(*e.module, options);

  std::vector<size_t> batchSizes;
  std::vector<BenchmarkResult> results;
  for (unsigned iteration = 0; iteration < 4; ++iteration) {
    std::vector<PerfConfigString> batch = search->getPerfConfigBatch(results);
    batchSizes.push_back(batch.size());
    if (batch.empty())
      break;
    results = fakeTimings(batch);
  }
  // The trace is flushed as it is written, so it can be read while the search
  // is still alive; destroying it first would work as well.
  auto buffer = llvm::MemoryBuffer::getFile(tracePath);
  ASSERT_TRUE(buffer) << "the search wrote no trace to " << tracePath;

  SmallVector<StringRef> lines;
  (*buffer)->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/false);
  llvm::sys::fs::remove(tracePath);
  // One line describing the run, then one per call the search answered.
  ASSERT_EQ(lines.size(), batchSizes.size() + 1);

  llvm::Expected<llvm::json::Value> header = llvm::json::parse(lines.front());
  ASSERT_TRUE(!!header) << "the header is not JSON: " << lines.front().str();
  llvm::json::Object *headerObj = header->getAsObject();
  ASSERT_TRUE(headerObj);
  EXPECT_EQ(headerObj->getString("kind"), "header");
  EXPECT_EQ(headerObj->getInteger("copies"),
            static_cast<int64_t>(options.copies));
  // The parameters have to be named, or nothing downstream can tell which axis
  // is which.
  llvm::json::Array *params = headerObj->getArray("params");
  ASSERT_TRUE(params);
  EXPECT_FALSE(params->empty());

  double lastBest = std::numeric_limits<double>::infinity();
  unsigned totalSucceeded = 0;
  for (auto [idx, line] : llvm::enumerate(ArrayRef(lines).drop_front())) {
    llvm::Expected<llvm::json::Value> record = llvm::json::parse(line);
    ASSERT_TRUE(!!record) << "iteration " << idx
                          << " is not JSON: " << line.str();
    llvm::json::Object *obj = record->getAsObject();
    ASSERT_TRUE(obj);
    EXPECT_EQ(obj->getString("kind"), "iteration");
    EXPECT_EQ(obj->getInteger("proposed"),
              static_cast<int64_t>(batchSizes[idx]));

    std::optional<int64_t> copies = obj->getInteger("copies");
    std::optional<int64_t> alive = obj->getInteger("alive");
    ASSERT_TRUE(copies && alive);
    EXPECT_LE(*alive, *copies) << "more copies alive than exist";

    // The best time so far is a running minimum, so it can only fall.
    if (std::optional<double> best = obj->getNumber("bestNs")) {
      EXPECT_LE(*best, lastBest);
      lastBest = *best;
      EXPECT_FALSE(obj->getString("bestOrigin")->empty())
          << "a config was measured but nothing says where it came from";
    }

    if (std::optional<double> precision = obj->getNumber("pickPrecision")) {
      EXPECT_GE(*precision, 0.0);
      EXPECT_LE(*precision, 1.0);
    }
    totalSucceeded += *obj->getInteger("succeeded");
  }

  // Every proposed config was answered with a timing, and the last batch is
  // still in flight when the search is asked to stop.
  size_t answered = 0;
  for (size_t size : ArrayRef(batchSizes).drop_back())
    answered += size;
  EXPECT_EQ(totalSucceeded, answered);
}

std::vector<int64_t> coarseMoves(ArrayRef<int64_t> ladder, int64_t value,
                                 unsigned radius) {
  SmallVector<int64_t> moves;
  radiusMoves(ladder, value, radius, moves);
  return std::vector<int64_t>(moves.begin(), moves.end());
}

std::vector<int64_t> fineMoves(ArrayRef<int64_t> ladder, int64_t value) {
  SmallVector<int64_t> moves;
  stepMoves(ladder, value, moves);
  return std::vector<int64_t>(moves.begin(), moves.end());
}

// The axis of a named parameter, so a test can state one without restating the
// order the parameters come in.
std::vector<int64_t> axisOf(const TuningParamAxes &axes, StringRef name) {
  SmallVector<StringRef> names;
  axes.getParamNames(names);
  auto param = llvm::find(names, name);
  if (param == names.end()) {
    ADD_FAILURE() << name << " is not a parameter";
    return {};
  }
  return axes.getAxes()[param - names.begin()];
}

// Helion's radius is a distance in log2 space, and it has to stay one whatever
// an axis's spacing is. A plain GEMM lists every multiple of 16 for its tiles
// (`tileValues`, TuningSearch.cpp), so counting places along that axis would
// mean a move of 16 where a power-of-two axis moves by a factor of four - and
// it would do so on the tiles, which every trial sets out to move.
TEST(LFBOSearchTest, CoarseMovesTravelInDoublings) {
  // A power-of-two axis reads the same either way: `numWaves` at four reaches a
  // quarter of it and four times it.
  EXPECT_EQ(coarseMoves({1, 2, 4, 8, 16}, 4, 2),
            (std::vector<int64_t>{1, 2, 8, 16}));

  // The same reach on an axis of every multiple of 16, where counting places
  // would have stopped at 96 below and 160 above.
  std::vector<int64_t> tiles = {1, 2, 4, 8};
  for (int64_t tile = 16; tile <= 256; tile += 16)
    tiles.push_back(tile);
  std::vector<int64_t> fromMidway = coarseMoves(tiles, 128, 2);
  EXPECT_EQ(fromMidway.front(), 32);
  EXPECT_EQ(fromMidway.back(), 256);
  EXPECT_FALSE(llvm::is_contained(fromMidway, 128))
      << "a move that stays where it is, is not a move";

  // An axis carrying a value with no ratio to take, such as the zero that
  // stands for an occupancy hint nobody asked for, moves by places instead.
  EXPECT_EQ(coarseMoves({0, 1, 2, 3, 4}, 4, 2), (std::vector<int64_t>{2, 3}));
}

// The fine move is what an axis of every multiple of 16 is for: a copy sitting
// on a good tile gets to try the tiles beside it.
TEST(LFBOSearchTest, FineMovesStepToTheNextValue) {
  EXPECT_EQ(fineMoves({16, 32, 48, 64}, 48), (std::vector<int64_t>{32, 64}));
  // A copy can start from a config the search would not have proposed, and then
  // the value the axis would take on next is a move like any other.
  EXPECT_EQ(fineMoves({16, 32, 48, 64}, 40),
            (std::vector<int64_t>{32, 48, 64}));
}

// Both rules on the M tile of a real GEMM, at the tile a quick config is likely
// to start from. The moves are the ladder's own values, so the axis being every
// multiple of 16 is what decides how many there are.
TEST(LFBOSearchTest, MovesFromATileTheSearchStartsOn) {
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);
  std::vector<int64_t> axis = axisOf(*axes, "mPerBlock");
  ASSERT_TRUE(llvm::is_contained(axis, 128));

  EXPECT_EQ(fineMoves(axis, 128), (std::vector<int64_t>{112, 144}));

  // A factor of four either way, cut short above by the axis: 512 is past
  // `MAX_MN_PER_BLOCK`, and a reach past the top of a ladder is the top.
  std::vector<int64_t> coarse = coarseMoves(axis, 128, /*radius=*/2);
  EXPECT_EQ(coarse.front(), 32);
  EXPECT_EQ(coarse.back(), 256);
  EXPECT_FALSE(llvm::is_contained(coarse, 16))
      << "16 is a factor of eight away, which a radius of two does not reach";
  // Every multiple of 16 from 32 through 256, less the tile it started on.
  EXPECT_EQ(coarse.size(), 14u);
  // Helion's power-of-two fragment would offer these three and stop; the tiles
  // between them are tiles a kernel holds just as well, so they are moves too.
  for (int64_t tile : {32, 64, 256})
    EXPECT_TRUE(llvm::is_contained(coarse, tile)) << tile << " is not offered";
  EXPECT_TRUE(llvm::is_contained(coarse, 208));
}

// A knob's axis is off and on, but every config the tuning space and the quick
// list hand out spells it `kKnobDefault`, which is deliberately neither. A copy
// seeded from such a config has to be offered both settings, or the knob it
// starts on is the knob it dies on. Which of the two the space then admits is
// `isFeasible`'s question, see `BufferAtomicsNeedBufferOps`
// (TuningParamAxesTests.cpp).
TEST(LFBOSearchTest, KnobMovesFromTheDefaultReachBothSettings) {
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);
  std::vector<int64_t> axis = axisOf(*axes, "useAsyncCopy");
  ASSERT_EQ(axis, (std::vector<int64_t>{0, 1}));

  const std::vector<int64_t> bothSettings = {0, 1};
  EXPECT_EQ(fineMoves(axis, kKnobDefault), bothSettings);
  // A coarse move asks for a wider window and gets the same one: an axis
  // carrying a zero counts places, and the clamp keeps both ends on the axis.
  EXPECT_EQ(coarseMoves(axis, kKnobDefault, /*radius=*/2), bothSettings);

  // Once on the axis, the setting it is not on is the only move left.
  EXPECT_EQ(fineMoves(axis, 0), (std::vector<int64_t>{1}));
  EXPECT_EQ(fineMoves(axis, 1), (std::vector<int64_t>{0}));
}

// The same config twice is a benchmark spent learning nothing.
TEST(LFBOSearchTest, ProposesEachConfigOnce) {
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::vector<PerfConfigString> proposed = flatten(runSearch(*e.module, 4));
  ASSERT_FALSE(proposed.empty());

  std::set<StringRef> seen;
  for (const PerfConfigString &perfConfig : proposed)
    EXPECT_TRUE(seen.insert(StringRef(perfConfig)).second)
        << "proposed twice: " << std::string(perfConfig);
}

} // namespace
