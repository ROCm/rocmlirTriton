//===- LLMSearchTests.cpp - Driving the LLM search without a model --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The search reaches a model through one subprocess that speaks JSON, so a
// script standing in for that subprocess is all it takes to drive the whole
// round machine here: what it asks for, what it does with the answer, and what
// it does with an answer it cannot use. None of it needs a network, an API key
// or a GPU.
//
// The fixture scripts are written out by the tests themselves rather than kept
// as data files, so that what a round is answered with sits next to the
// expectation about it.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/TuningSearch.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

#include "LLMProposer.h"
#include "LLMSearch.h"

#include "gtest/gtest.h"

#include <memory>
#include <set>

using namespace mlir;
using namespace mlir::rock;

namespace {

//===----------------------------------------------------------------------===//
// Standing in for the model
//===----------------------------------------------------------------------===//

/// A helper script, written somewhere it can be run from and removed when the
/// test is done with it.
class FixtureProposer {
public:
  /// `body` is Python, run with `request` already parsed into a dict of that
  /// name, `args` holding the parsed command line, and `render` in scope. It
  /// is responsible for writing `args.response`.
  explicit FixtureProposer(StringRef body) {
    if (llvm::sys::fs::createTemporaryFile("rocmlir-llm-fixture", "py", path)) {
      ADD_FAILURE() << "could not create a fixture script";
      return;
    }
    std::error_code ec;
    llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      ADD_FAILURE() << "could not write " << path << ": " << ec.message();
      return;
    }
    // `render` is the fixture's stand-in for `configs.render_perf_config`:
    // the exemplar the request carried, with the named fields replaced. Open
    // coded rather than imported so that a test states the config it wants
    // without the real helper being in the picture at all.
    out << "import argparse, json, sys\n"
           "parser = argparse.ArgumentParser()\n"
           "parser.add_argument('--request')\n"
           "parser.add_argument('--response')\n"
           "parser.add_argument('--session', default='')\n"
           "parser.add_argument('--transcript', default='')\n"
           "args = parser.parse_args()\n"
           "request = json.load(open(args.request))\n"
           "def render(**changes):\n"
           "    prefix, _, body = request['defaultPerfConfig'].partition(':')\n"
           "    fields = []\n"
           "    for piece in body.split(','):\n"
           "        key, _, value = piece.partition('=')\n"
           "        fields.append('%s=%s' % (key, changes.get(key, value)))\n"
           "    return prefix + ':' + ','.join(fields)\n"
        << body << "\n";
    out.close();

    // Run through the interpreter rather than a shebang: the test has no say
    // over which Python is on $PATH, and one that cannot be executed would
    // look like the search failing rather than the fixture.
    if (auto found = llvm::sys::findProgramByName("python3"))
      interpreter = *found;
    else
      ADD_FAILURE() << "python3 is not on $PATH";
  }

  ~FixtureProposer() {
    if (!path.empty())
      llvm::sys::fs::remove(path);
  }

  /// What to hand `LLMSearchOptions::proposerPath`. Empty when the fixture
  /// could not be set up, which a test should have failed on already.
  std::string program() const {
    if (interpreter.empty() || path.empty())
      return {};
    // The proposer runs one program with the arguments it is given, so the
    // interpreter and the script have to arrive as one word. A wrapper script
    // is the portable way to say that.
    return wrap();
  }

private:
  /// A one-line shell script that runs the fixture under the interpreter and
  /// forwards the proposer's arguments.
  std::string wrap() const {
    if (!wrapper.empty())
      return wrapper;
    SmallString<128> wrapperPath;
    if (llvm::sys::fs::createTemporaryFile("rocmlir-llm-wrap", "sh",
                                           wrapperPath)) {
      ADD_FAILURE() << "could not create the fixture's wrapper";
      return {};
    }
    std::error_code ec;
    {
      llvm::raw_fd_ostream out(wrapperPath, ec, llvm::sys::fs::OF_Text);
      if (ec) {
        ADD_FAILURE() << "could not write " << wrapperPath << ": "
                      << ec.message();
        return {};
      }
      out << "#!/bin/sh\nexec \"" << interpreter << "\" \"" << path
          << "\" \"$@\"\n";
    }
    llvm::sys::fs::setPermissions(wrapperPath, llvm::sys::fs::all_read |
                                                   llvm::sys::fs::all_exe |
                                                   llvm::sys::fs::owner_write);
    wrapper = std::string(wrapperPath);
    return wrapper;
  }

  SmallString<128> path;
  std::string interpreter;
  mutable std::string wrapper;
};

/// Answers every round with `configs`, a Python expression for a list of
/// perf-config strings -- usually `render()` calls.
std::string alwaysAnswers(StringRef configs) {
  return ("json.dump({'configs': " + configs + "}, open(args.response, 'w'))")
      .str();
}

/// The request a fixture was given, kept for a test to look at: a scratch path
/// to write to, and the parsed JSON once it has been.
///
/// The parse is owned here rather than handed back by value, because what a
/// test wants is a pointer into it and one into a temporary would dangle for
/// every assertion after the first.
class RecordedRequest {
public:
  RecordedRequest() {
    if (llvm::sys::fs::createTemporaryFile("rocmlir-llm-seen", "json", path))
      ADD_FAILURE() << "could not create a scratch file";
  }

  ~RecordedRequest() {
    if (!path.empty())
      llvm::sys::fs::remove(path);
  }

  StringRef str() const { return path; }

  /// The whole request, or nullptr when the fixture left nothing usable.
  const llvm::json::Object *request() {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
      ADD_FAILURE() << "the fixture wrote nothing to " << path;
      return nullptr;
    }
    llvm::Expected<llvm::json::Value> read =
        llvm::json::parse((*buffer)->getBuffer());
    if (!read) {
      ADD_FAILURE() << "the fixture wrote something unparseable: "
                    << llvm::toString(read.takeError());
      return nullptr;
    }
    parsed = std::move(*read);
    const llvm::json::Object *object = parsed->getAsObject();
    if (!object)
      ADD_FAILURE() << "the request is not a JSON object";
    return object;
  }

  /// Just its description of the problem, which is what most of this asks
  /// about.
  const llvm::json::Object *problem() {
    const llvm::json::Object *object = request();
    if (!object)
      return nullptr;
    const llvm::json::Object *described = object->getObject("problem");
    if (!described)
      ADD_FAILURE() << "the request carries no problem description";
    return described;
  }

private:
  SmallString<128> path;
  std::optional<llvm::json::Value> parsed;
};

/// Answers as `alwaysAnswers` does, and keeps a copy of the request, so that a
/// test can look at what the model would have been told about the problem.
std::string recordsTheRequest(const RecordedRequest &seen, StringRef configs) {
  return ("json.dump(request, open('" + seen.str() + "', 'w'))\n" +
          alwaysAnswers(configs))
      .str();
}

//===----------------------------------------------------------------------===//
// The module under test
//===----------------------------------------------------------------------===//

struct GemmModule {
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;

  /// `aTransposed` stores A as [G] x K x M instead of [G] x M x K, which is a
  /// fact about the problem the search is expected to pass on.
  GemmModule(int64_t m, int64_t n, int64_t k, StringRef arch,
             bool aTransposed = false) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    OpBuilder b(&ctx);
    Location loc = b.getUnknownLoc();
    Type elem = b.getF16Type();
    auto aType = aTransposed ? RankedTensorType::get({1, k, m}, elem)
                             : RankedTensorType::get({1, m, k}, elem);
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
        /*aTransposed=*/aTransposed ? b.getUnitAttr() : UnitAttr{},
        /*bTransposed=*/UnitAttr{},
        /*oTransposed=*/UnitAttr{}, /*aScaleTransposed=*/UnitAttr{},
        /*bScaleTransposed=*/UnitAttr{}, /*quantBlockSize=*/IntegerAttr{},
        /*params=*/nullptr);
    func::ReturnOp::create(b, loc, gemmOp.getResult());
    (*module)->setAttr(rock::ArchAttr::getMnemonic(),
                       b.getStringAttr(Twine("amdgcn-amd-amdhsa:") + arch));
  }
};

/// A module holding a forward `rock.conv` over a 3x3 filter whose input is
/// channels-first, which is the layout that earns a `kPerBlock` alignment: its
/// gemmK is Merge(c, y, x), so only a tile that is a multiple of the 3x3
/// footprint advances K without moving the input window.
struct ConvModule {
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;

  explicit ConvModule(StringRef arch) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    static constexpr StringLiteral kSource = R"MLIR(
module attributes {rock.arch = "amdgcn-amd-amdhsa:ARCH"} {
  func.func @rock_conv(%filter: tensor<1x128x8x3x3xf16>,
                       %input: tensor<128x1x8x32x32xf16>)
      -> tensor<128x1x128x30x30xf16>
      attributes {rock.arch = "amdgcn-amd-amdhsa:ARCH", rock.kernel} {
    %result = rock.conv(%filter, %input) {
      filter_layout = ["g", "k", "c", "0", "1"],
      input_layout = ["ni", "gi", "ci", "0i", "1i"],
      output_layout = ["no", "go", "ko", "0o", "1o"],
      dilations = [1 : index, 1 : index],
      strides = [1 : index, 1 : index],
      padding = [0 : index, 0 : index, 0 : index, 0 : index]
    } : tensor<1x128x8x3x3xf16>, tensor<128x1x8x32x32xf16>
      -> tensor<128x1x128x30x30xf16>
    return %result : tensor<128x1x128x30x30xf16>
  }
}
)MLIR";
    std::string source = kSource.str();
    for (size_t at = source.find("ARCH"); at != std::string::npos;
         at = source.find("ARCH", at))
      source.replace(at, 4, arch.str());
    module = parseSourceString<ModuleOp>(source, &ctx);
  }
};

/// A module holding a single grouped-query `rock.attention` with causal
/// masking: eight query heads over two key/value heads, so the K and V batch
/// dimensions are a quarter of Q's.
struct AttentionModule {
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;

  explicit AttentionModule(StringRef arch) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    static constexpr StringLiteral kSource = R"MLIR(
module attributes {rock.arch = "amdgcn-amd-amdhsa:ARCH"} {
  func.func @rock_attention(%q: tensor<4x1024x128xf16>,
                            %k: tensor<1x128x1024xf16>,
                            %v: tensor<1x1024x512xf16>)
      -> tensor<4x1024x512xf16>
      attributes {rock.arch = "amdgcn-amd-amdhsa:ARCH", rock.kernel} {
    %result = rock.attention{
     qk = %q * %k : tensor<4x1024x128xf16>, tensor<1x128x1024xf16>
     causal
     softmax(qk) * %v : tensor<1x1024x512xf16>
    } {numHeadsKV = 2 : i32, numHeadsQ = 8 : i32, softmaxType = f32,
       splitKV = 1 : i32} -> tensor<4x1024x512xf16>
    return %result : tensor<4x1024x512xf16>
  }
}
)MLIR";
    std::string source = kSource.str();
    for (size_t at = source.find("ARCH"); at != std::string::npos;
         at = source.find("ARCH", at))
      source.replace(at, 4, arch.str());
    module = parseSourceString<ModuleOp>(source, &ctx);
  }
};

/// A timing for every config in `batch`. The values do not matter to any test
/// here -- nothing in this file asks the search to prefer one config over
/// another -- so they only have to be finite and distinct enough to order.
std::vector<BenchmarkResult> fakeTimings(ArrayRef<PerfConfigString> batch) {
  std::vector<BenchmarkResult> results;
  double time = 1000.0;
  for (const PerfConfigString &perfConfig : batch)
    results.push_back({perfConfig, time++, BenchmarkResult::Status::Success});
  return results;
}

/// Options that keep a test quick: a small first batch, and the session the
/// search makes for itself, since nothing here reads one back.
LLMOptions testOptions(const FixtureProposer &fixture) {
  LLMOptions options;
  options.search.proposerPath = fixture.program();
  options.search.configsPerRound = 4;
  options.search.maxRounds = 3;
  options.search.initialRandomConfigs = 3;
  // Every round in this file answers at once, so a short deadline only makes a
  // hung fixture fail quickly instead of holding the suite for two minutes.
  options.search.requestTimeoutSec = 60;
  return options;
}

/// Runs the search to exhaustion, answering each batch with made-up timings,
/// and returns the batches as it proposed them.
std::vector<std::vector<PerfConfigString>>
runSearch(ModuleOp mod, const LLMOptions &options, unsigned maxBatches = 8) {
  std::unique_ptr<TuningSearchStrategy> search =
      createLLMSearchStrategy(mod, options);
  std::vector<std::vector<PerfConfigString>> batches;
  std::vector<BenchmarkResult> results;
  for (unsigned batch = 0; batch < maxBatches; ++batch) {
    std::vector<PerfConfigString> proposed =
        search->getPerfConfigBatch(results);
    if (proposed.empty())
      break;
    results = fakeTimings(proposed);
    batches.push_back(std::move(proposed));
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

/// The value `param` carries in `perfConfig`.
std::optional<int64_t> valueOf(MLIRContext &ctx, StringRef perfConfig,
                               StringRef param) {
  auto params = GemmParamsAttr::get(StringAttr::get(&ctx, perfConfig));
  if (!params)
    return std::nullopt;
  SmallVector<StringRef> names;
  SmallVector<int64_t> values;
  params.getParamNames(names);
  params.getParamValues(values);
  for (auto [name, value] : llvm::zip_equal(names, values))
    if (name == param)
      return value;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// What the model is told about the problem
//
// A prompt that leaves out a fact the perf config has to answer to is asking
// the model to guess: the same M/N/K with A stored transposed wants a
// different `kPerBlock`, and a causal attention wants a different tile from a
// dense one. So these hold the description to the op it was taken from.
//===----------------------------------------------------------------------===//

TEST(LLMSearchTest, DescribesAGemmsLayoutTypesAndFusion) {
  RecordedRequest seen;
  FixtureProposer fixture(recordsTheRequest(seen, "[]"));
  GemmModule e(/*m=*/1024, /*n=*/512, /*k=*/256, "gfx942",
               /*aTransposed=*/true);

  runSearch(*e.module, testOptions(fixture));
  const llvm::json::Object *problem = seen.problem();
  ASSERT_TRUE(problem);

  EXPECT_EQ(problem->getString("kernelType"), "Gemm");
  const llvm::json::Object *size = problem->getObject("gemmSize");
  ASSERT_TRUE(size);
  EXPECT_EQ(size->getInteger("m"), 1024);
  EXPECT_EQ(size->getInteger("n"), 512);
  EXPECT_EQ(size->getInteger("k"), 256);
  EXPECT_FALSE(size->get("o")) << "a single GEMM has no second free dimension";

  // The transpose the module was built with, and only that one.
  EXPECT_EQ(problem->getBoolean("transposedA"), true);
  EXPECT_EQ(problem->getBoolean("transposedB"), false);
  EXPECT_EQ(problem->getBoolean("transposedOut"), false);
  EXPECT_FALSE(problem->get("transposedC"))
      << "a single GEMM has no second-GEMM operand to transpose";

  // The output type, which is what the epilogue has to store.
  EXPECT_EQ(problem->getString("cType"), "f16");
  EXPECT_FALSE(problem->get("outType"))
      << "a single GEMM's result is its C, so there is no separate output";

  EXPECT_EQ(problem->getBoolean("hasFusedReduction"), false);
  // Not a scaled GEMM and not a convolution, so neither block shows up.
  EXPECT_FALSE(problem->get("quantBlockSize"));
  EXPECT_FALSE(problem->get("filterLayout"));
  EXPECT_FALSE(problem->get("kPerBlockAlignment"))
      << "a plain GEMM's K carries no structure to align a tile to";
  // Attention's questions are not asked of a GEMM: `causal = false` would be a
  // claim about masking that a GEMM does not have.
  EXPECT_FALSE(problem->get("causal"));
  EXPECT_FALSE(problem->get("numHeadsQ"));
}

TEST(LLMSearchTest, DescribesTheMachineTheHarnessMeasuredNotTheArchDefault) {
  RecordedRequest seen;
  FixtureProposer fixture(recordsTheRequest(seen, "[]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  // What `tuningRunner.py` reads off the live GPU -- HIP's
  // `multiProcessorCount` and `inferNumChiplets` of it -- and stamps through
  // rocmlir-gen's `--num_cu` / `--num_chiplets`. An MI300A, whose 228 CUs over
  // 6 XCDs differ from both arch-database fallbacks, so a description that
  // took those instead would fail here rather than quietly advise a tile
  // count for the wrong machine.
  OpBuilder b(&e.ctx);
  (*e.module)->setAttr(rock::NumCUAttr::getMnemonic(),
                       b.getI64IntegerAttr(228));
  (*e.module)->setAttr(rock::NumChipletsAttr::getMnemonic(),
                       b.getI64IntegerAttr(6));

  runSearch(*e.module, testOptions(fixture));
  const llvm::json::Object *request = seen.request();
  ASSERT_TRUE(request);
  const llvm::json::Object *hardware = request->getObject("hardware");
  ASSERT_TRUE(hardware);

  EXPECT_EQ(hardware->getInteger("numCUs"), 228);
  EXPECT_EQ(hardware->getInteger("numChiplets"), 6);
  // Against the fallbacks themselves, so that this keeps testing what it means
  // to if the database changes.
  StringRef arch = "amdgcn-amd-amdhsa:gfx942";
  EXPECT_NE(hardware->getInteger("numCUs"), rock::getMinNumCU(arch));
  EXPECT_NE(hardware->getInteger("numChiplets"), rock::getMaxNumChiplets(arch));
}

TEST(LLMSearchTest, SaysWhichWayEachKnobsDefaultGoesOnThisArch) {
  RecordedRequest seen;
  FixtureProposer fixture(recordsTheRequest(seen, "[]"));
  // gfx942 is the arch that separates the three defaults from one another and
  // from what the hardware can do: it defaults async copy off while supporting
  // it, and defaults both of the other two on.
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  runSearch(*e.module, testOptions(fixture));
  const llvm::json::Object *hardware = seen.request()->getObject("hardware");
  ASSERT_TRUE(hardware);

  StringRef arch = "amdgcn-amd-amdhsa:gfx942";
  EXPECT_EQ(hardware->getBoolean("defaultAsyncCopy"),
            rock::defaultUseAsyncCopy(arch));
  EXPECT_EQ(hardware->getBoolean("defaultInThreadTranspose"),
            rock::defaultUseInThreadTranspose(arch));
  // Resolved against the resolved async-copy decision, the way
  // `isPingpongScheduleEnabled` resolves it, and not against the knob.
  EXPECT_EQ(
      hardware->getBoolean("defaultBlockPingpong"),
      rock::defaultUseBlockPingpong(arch, rock::defaultUseAsyncCopy(arch)));
  // The point of reporting the default apart from the capability: here the two
  // disagree, and a config that read the capability alone would take an
  // explicit useAsyncCopy=1 for a config that changes nothing.
  EXPECT_TRUE(rock::supportsAsyncCopy(arch));
  EXPECT_FALSE(rock::defaultUseAsyncCopy(arch));
}

TEST(LLMSearchTest, SaysWhatAGridGroupSizeOfZeroWorksOutTo) {
  RecordedRequest seen;
  FixtureProposer fixture(recordsTheRequest(seen, "[]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  runSearch(*e.module, testOptions(fixture));
  const llvm::json::Object *hardware = seen.request()->getObject("hardware");
  ASSERT_TRUE(hardware);

  // Same f32 operands and result, so the width ratio is 1 and the group size
  // is the square root of the CUs on one chiplet. Asserted against the shared
  // helper `makeGroupedGridLayout` itself calls, so the request cannot drift
  // from the layout it claims to describe.
  int64_t numCUs = *hardware->getInteger("numCUs");
  int64_t numChiplets = *hardware->getInteger("numChiplets");
  EXPECT_EQ(hardware->getInteger("defaultGridGroupSize"),
            rock::defaultGridGroupSize(numCUs, numChiplets,
                                       /*inputBitWidth=*/32,
                                       /*outputBitWidth=*/32));
}

TEST(LLMSearchTest, DescribesTheKTileAChannelsFirstConvolutionWants) {
  RecordedRequest seen;
  FixtureProposer fixture(recordsTheRequest(seen, "[]"));
  ConvModule e("gfx942");
  ASSERT_TRUE(e.module) << "the convolution module did not parse";

  runSearch(*e.module, testOptions(fixture));
  const llvm::json::Object *problem = seen.problem();
  ASSERT_TRUE(problem);

  EXPECT_EQ(problem->getString("kernelType"), "Conv");
  const llvm::json::Array *inputLayout = problem->getArray("inputLayout");
  ASSERT_TRUE(inputLayout);
  ASSERT_EQ(inputLayout->size(), 5u);
  // The channel dim ahead of both spatial ones. This is the whole of what
  // earns the alignment, so it is worth pinning next to it.
  EXPECT_EQ((*inputLayout)[2].getAsString(), "ci");

  // The 3x3 filter's footprint: gemmK is Merge(c, 3, 3), and a tile that is
  // not a multiple of 9 lands mid-filter.
  EXPECT_EQ(problem->getInteger("kPerBlockAlignment"), 9);
}

TEST(LLMSearchTest, DescribesAttentionsHeadsMaskingAndSecondGemm) {
  RecordedRequest seen;
  FixtureProposer fixture(recordsTheRequest(seen, "[]"));
  AttentionModule e("gfx942");
  ASSERT_TRUE(e.module) << "the attention module did not parse";

  runSearch(*e.module, testOptions(fixture));
  const llvm::json::Object *problem = seen.problem();
  ASSERT_TRUE(problem);

  EXPECT_EQ(problem->getString("kernelType"), "Attention");

  // Two sequence lengths and two head dimensions, in GEMM terms: M is seq_q,
  // K is head_qk, N is seq_k and O is head_v.
  const llvm::json::Object *size = problem->getObject("gemmSize");
  ASSERT_TRUE(size);
  EXPECT_EQ(size->getInteger("g"), 4);
  EXPECT_EQ(size->getInteger("m"), 1024);
  EXPECT_EQ(size->getInteger("k"), 128);
  EXPECT_EQ(size->getInteger("n"), 1024);
  EXPECT_EQ(size->getInteger("o"), 512);

  // Grouped-query attention, which decides how much K/V a query tile shares.
  EXPECT_EQ(problem->getInteger("numHeadsQ"), 8);
  EXPECT_EQ(problem->getInteger("numHeadsKV"), 2);
  // Causal masking makes half the tiles of a square block do no work, so a
  // model that is not told about it cannot reason about the tile shape.
  EXPECT_EQ(problem->getBoolean("causal"), true);
  EXPECT_EQ(problem->getInteger("splitKV"), 1);
  EXPECT_EQ(problem->getBoolean("hasLse"), false);
  EXPECT_EQ(problem->getBoolean("hasLastValidKVIndex"), false);
  EXPECT_EQ(problem->getBoolean("hasPrefixOffset"), false);
  EXPECT_FALSE(problem->get("slidingWindowLookBack"))
      << "this attention is not a sliding window";
  EXPECT_EQ(problem->getString("softmaxType"), "f32");

  // The second GEMM: its operand's type and layout, and the output's, which a
  // single GEMM has nothing to say about.
  EXPECT_EQ(problem->getString("cType"), "f16");
  EXPECT_EQ(problem->getString("outType"), "f16");
  EXPECT_EQ(problem->getBoolean("transposedC"), false);
  EXPECT_EQ(problem->getBoolean("transposedOut"), false);

  EXPECT_EQ(problem->getBoolean("hasPreSecondGemmFusion"), false);
  EXPECT_EQ(problem->getInteger("numElemwiseInputs"), 0);
  EXPECT_FALSE(problem->get("hasFusedReduction"))
      << "a gemm+gemm op cannot carry a reduction, so the question is not "
         "asked";
}

//===----------------------------------------------------------------------===//
// What the search does with an answer
//===----------------------------------------------------------------------===//

// The helper answers with whole configs, so what the search owes them is a
// round trip through the attribute that owns the format: read the string,
// check the values, spell it again. A field the helper set has to survive it.
TEST(LLMSearchTest, ProposesTheConfigTheHelperNamed) {
  FixtureProposer fixture(alwaysAnswers("[render(mPerBlock=256)]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  options.search.maxRounds = 1;
  std::vector<std::vector<PerfConfigString>> batches =
      runSearch(*e.module, options);
  ASSERT_GT(batches.size(), 1u) << "the search never used the model's answer";

  // The first batch is the seeds; the second is the round that was answered.
  ASSERT_EQ(batches[1].size(), 1u);
  StringRef proposed = batches[1].front();
  EXPECT_EQ(valueOf(e.ctx, proposed, "mPerBlock"), 256)
      << "the named field did not survive: " << proposed.str();

  // Everything the helper left alone came from the exemplar, which for a knob
  // is the default it is spelled with rather than a zero meaning "off".
  EXPECT_EQ(valueOf(e.ctx, proposed, "useAsyncCopy"), kKnobDefault)
      << "an untouched knob was not the exemplar's: " << proposed.str();
  EXPECT_GT(*valueOf(e.ctx, proposed, "nPerBlock"), 0)
      << "an untouched tile came out as zero: " << proposed.str();
}

// A reply that is well-formed JSON but not a perf config means the helper and
// this side disagree about something. Worth saying out loud, but not worth
// ending a tuning run over, and not worth losing the round's other configs to.
TEST(LLMSearchTest, DropsSomethingThatIsNotAPerfConfigForThisKernel) {
  FixtureProposer fixture(alwaysAnswers(
      // Not a perf config at all; the gemm+gemm form, which parses but has the
      // wrong fields for a plain GEMM; and one real config.
      "['mPerBlock=256', 'attn:mPerBlockG0=128', render(mPerBlock=256)]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  options.search.maxRounds = 1;
  std::vector<std::vector<PerfConfigString>> batches =
      runSearch(*e.module, options);
  ASSERT_GT(batches.size(), 1u) << "one unusable config ended the round";

  ASSERT_EQ(batches[1].size(), 1u);
  EXPECT_EQ(valueOf(e.ctx, batches[1].front(), "mPerBlock"), 256);
}

// The seeds are what the first round is judged against, and they are the quick
// tuning list rather than draws from the space, so they have to arrive before
// the model is asked anything.
TEST(LLMSearchTest, OpensWithTheQuickListAndSomeDraws) {
  FixtureProposer fixture(alwaysAnswers("[]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  std::vector<std::vector<PerfConfigString>> batches =
      runSearch(*e.module, options);
  ASSERT_FALSE(batches.empty()) << "the search proposed nothing at all";

  // More than the draws alone, which is what says the quick list is in there.
  EXPECT_GT(batches.front().size(), options.search.initialRandomConfigs)
      << "the first batch is no bigger than its random padding, so the quick "
         "list did not reach it";
}

// An empty answer is the model saying it has nothing left, which Helion treats
// as the end of the search rather than as a failure. A round that proposed
// nothing and then asked again would spend the whole budget on empty replies.
TEST(LLMSearchTest, StopsWhenTheModelProposesNothing) {
  FixtureProposer fixture(alwaysAnswers("[]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  std::vector<std::vector<PerfConfigString>> batches =
      runSearch(*e.module, testOptions(fixture));
  EXPECT_EQ(batches.size(), 1u)
      << "the search kept going after the model ran out of configs";
}

// The rounds are the budget, so the search has to stop at them even when the
// model would happily keep proposing. Each round here answers with a config
// nothing has tried, so nothing but the bound can end it.
TEST(LLMSearchTest, StopsAfterTheRoundsItWasGiven) {
  // Keyed off the round number so that no two rounds repeat themselves, which
  // the dedupe would otherwise swallow into an empty batch.
  FixtureProposer fixture(
      alwaysAnswers("[render(mPerBlock=32 + 16 * request['round'])]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  options.search.maxRounds = 2;
  // Out of the way, so that it is the round count under test and not this.
  options.maxStagnantRounds = 100;

  std::vector<std::vector<PerfConfigString>> batches =
      runSearch(*e.module, options);
  // The seed batch, then one batch per round.
  EXPECT_EQ(batches.size(), options.search.maxRounds + 1)
      << "the search did not run for exactly the rounds it was given";
}

// One conversation per search, so that the model a later round is talking to
// is the one that proposed the earlier rounds. The helper is what holds the
// conversation; what the search owes it is a file to keep it in, the same one
// every round, and nowhere for it to outlive the search.
TEST(LLMSearchTest, GivesEveryRoundTheSameSessionToContinue) {
  RecordedRequest seen;
  FixtureProposer fixture(
      ("open('" + seen.str() + "', 'a').write(args.session + '\\n')\n" +
       alwaysAnswers("[render(mPerBlock=32 + 16 * request['round'])]"))
          .str());
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  options.search.maxRounds = 2;
  options.maxStagnantRounds = 100;
  ASSERT_GT(runSearch(*e.module, options).size(), 2u);

  auto written = llvm::MemoryBuffer::getFile(seen.str());
  ASSERT_TRUE(!!written);
  SmallVector<StringRef> paths;
  StringRef((*written)->getBuffer()).trim().split(paths, '\n');
  ASSERT_EQ(paths.size(), 2u) << "not one line per round";
  EXPECT_FALSE(paths[0].empty())
      << "the search left its rounds nothing to resume";
  EXPECT_EQ(paths[0], paths[1]) << "each round was given a session of its own";
  // The search has been destroyed by now, and an unnamed session is the
  // search's own temporary file rather than something left on the machine.
  EXPECT_FALSE(llvm::sys::fs::exists(paths[0]))
      << "the temporary session outlived the search: " << paths[0];
}

// A named session is somebody asking to keep it, so it stays.
TEST(LLMSearchTest, KeepsASessionItWasToldWhereToPut) {
  SmallString<128> sessionPath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("rocmlir-llm-kept", "json",
                                                  sessionPath));
  llvm::FileRemover removeSession(sessionPath);

  RecordedRequest seen;
  // A knob, as `DropsProposalsTheSpaceRefuses` uses one: every quick config
  // spells all eight at their default, so no seed can have taken this config
  // already and the round cannot come out empty.
  FixtureProposer fixture(("open('" + seen.str() +
                           "', 'w').write(args.session)\n" +
                           alwaysAnswers("[render(useOptimizeEpilogue=0)]"))
                              .str());
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  options.search.initialRandomConfigs = 0;
  options.search.sessionPath = std::string(sessionPath);
  ASSERT_GT(runSearch(*e.module, options).size(), 1u);

  auto written = llvm::MemoryBuffer::getFile(seen.str());
  ASSERT_TRUE(!!written);
  EXPECT_EQ((*written)->getBuffer(), StringRef(sessionPath));
  EXPECT_TRUE(llvm::sys::fs::exists(sessionPath));
}

// Where a run is read back from afterwards. The search only carries the path:
// what a person wants to read is the prompts and the replies, and the helper
// is what has those (see llm/transcript.py). So what is under test here is
// that every round reaches the file, since a transcript missing the round that
// went wrong is worse than none.
TEST(LLMSearchTest, GivesEveryRoundTheTranscriptToWriteTo) {
  SmallString<128> transcriptPath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("rocmlir-llm-transcript",
                                                  "log", transcriptPath));
  llvm::FileRemover removeTranscript(transcriptPath);

  // Keyed off the round number, as `StopsAfterTheRoundsItWasGiven` is, so that
  // the dedupe cannot end the search before the second round.
  FixtureProposer fixture(
      "open(args.transcript, 'a').write('round %d happened\\n' % "
      "request['round'])\n" +
      alwaysAnswers("[render(mPerBlock=32 + 16 * request['round'])]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  options.search.maxRounds = 2;
  options.maxStagnantRounds = 100;
  options.search.transcriptPath = std::string(transcriptPath);

  ASSERT_GT(runSearch(*e.module, options).size(), 2u);
  auto written = llvm::MemoryBuffer::getFile(transcriptPath);
  ASSERT_TRUE(!!written) << "the helper was given no transcript to write to";
  EXPECT_EQ((*written)->getBuffer(), "round 0 happened\nround 1 happened\n");
}

// The space refuses configs the ladders alone do not rule out, and a model
// cannot see those rules. Refusing has to mean dropping the config, not
// proposing it anyway and not abandoning the round it came in.
TEST(LLMSearchTest, DropsProposalsTheSpaceRefuses) {
  // 33 is not on any tile ladder: the tiles are 1, 2, 4, 8 and then multiples
  // of 16, so it is refused before anything is compiled. The knob beside it is
  // admissible and, being a knob, is one no seed can collide with: every quick
  // config spells all eight of them `kKnobDefault`.
  FixtureProposer fixture(
      alwaysAnswers("[render(mPerBlock=33), render(useOptimizeEpilogue=0)]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  options.search.maxRounds = 1;
  // No random padding, so that the seed batch is exactly the quick list and a
  // draw cannot reach the knob first and turn the proposal into a duplicate.
  options.search.initialRandomConfigs = 0;

  std::vector<std::vector<PerfConfigString>> batches =
      runSearch(*e.module, options);
  ASSERT_GT(batches.size(), 1u) << "a refused config ended the round";

  EXPECT_EQ(batches[1].size(), 1u) << "the refused config was proposed anyway";
  EXPECT_EQ(valueOf(e.ctx, batches[1].front(), "useOptimizeEpilogue"), 0);
}

// Everything the search hands out has to be a config the space admits, since
// that is the only reason to believe it can be compiled. Unlike the LFBO
// search, this holds for the first batch too: the quick list is in the space
// being searched here.
TEST(LLMSearchTest, ProposesOnlyAdmissibleConfigs) {
  FixtureProposer fixture(
      alwaysAnswers("[render(mPerBlock=32 + 16 * request['round']), "
                    "render(nPerBlock=32 + 16 * request['round'])]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  std::vector<PerfConfigString> proposed =
      flatten(runSearch(*e.module, testOptions(fixture)));
  ASSERT_FALSE(proposed.empty());

  for (const PerfConfigString &perfConfig : proposed) {
    auto params = GemmParamsAttr::get(StringAttr::get(&e.ctx, perfConfig));
    ASSERT_TRUE(params) << "proposed an unparseable config: "
                        << std::string(perfConfig);
    SmallVector<int64_t> values;
    params.getParamValues(values);
    EXPECT_TRUE(axes->isFeasible(values))
        << "proposed a config the space disowns: " << std::string(perfConfig);
  }
}

// A benchmark spent on a config an earlier round already measured is a
// benchmark spent learning nothing, and a model asked the same question tends
// to give the same answer.
TEST(LLMSearchTest, ProposesEachConfigOnce) {
  // Answers with the same config every round, which is the case the dedupe
  // exists for.
  FixtureProposer fixture(alwaysAnswers("[render(mPerBlock=64)]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  std::vector<PerfConfigString> proposed =
      flatten(runSearch(*e.module, testOptions(fixture)));
  ASSERT_FALSE(proposed.empty());

  std::set<StringRef> seen;
  for (const PerfConfigString &perfConfig : proposed)
    EXPECT_TRUE(seen.insert(StringRef(perfConfig)).second)
        << "proposed twice: " << std::string(perfConfig);
}

//===----------------------------------------------------------------------===//
// What the search tells the model
//===----------------------------------------------------------------------===//

// The refinement rounds only earn their latency if they say what the previous
// one measured, and the rejections only help if they name the config that was
// turned down. Both are checked by having the fixture write the request back
// out, which is the only way to see a prompt's input from here.
TEST(LLMSearchTest, TellsTheModelWhatHappenedToItsLastRound) {
  SmallString<128> requestLog;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("rocmlir-llm-req", "jsonl",
                                                  requestLog));

  FixtureProposer fixture(
      ("open(r'" + requestLog +
       "', 'a').write(json.dumps(request) + '\\n')\n"
       "json.dump({'configs': [render(mPerBlock=33), "
       "render(useOptimizeEpilogue=0)]}, open(args.response, 'w'))")
          .str());
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  LLMOptions options = testOptions(fixture);
  options.search.maxRounds = 2;
  // The knob keeps round 0 from coming out empty; see
  // `DropsProposalsTheSpaceRefuses`.
  options.search.initialRandomConfigs = 0;
  // The first request has to be built after the seeds were measured, or there
  // is nothing for the second round to have been told about.
  options.search.waitForSeeds = true;
  runSearch(*e.module, options);

  auto buffer = llvm::MemoryBuffer::getFile(requestLog);
  ASSERT_TRUE(buffer) << "the fixture logged no requests";
  SmallVector<StringRef> lines;
  (*buffer)->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/false);
  llvm::sys::fs::remove(requestLog);
  ASSERT_GE(lines.size(), 2u) << "the search asked for only one round";

  // Round 0 is asked before anything the model said has been measured, but
  // after the seeds were, since `waitForSeeds` is on.
  llvm::Expected<llvm::json::Value> first = llvm::json::parse(lines[0]);
  ASSERT_TRUE(!!first);
  llvm::json::Object *firstObj = first->getAsObject();
  ASSERT_TRUE(firstObj);
  EXPECT_EQ(firstObj->getInteger("round"), 0);
  EXPECT_FALSE(firstObj->getArray("seedConfigs")->empty())
      << "the quick list was not offered to the model";
  // The space itself, so that the model can be held to it.
  llvm::json::Object *space = firstObj->getObject("space");
  ASSERT_TRUE(space);
  EXPECT_FALSE(space->empty());
  ASSERT_TRUE(firstObj->getObject("defaultConfig"));

  // The exemplar, serialized, which is what the helper completes a sparse
  // proposal against; without it nothing it sends back can be spelled.
  std::optional<StringRef> exemplar = firstObj->getString("defaultPerfConfig");
  ASSERT_TRUE(exemplar);
  EXPECT_TRUE(GemmParamsAttr::get(StringAttr::get(&e.ctx, *exemplar)))
      << "the exemplar is not a perf config: " << exemplar->str();

  llvm::Expected<llvm::json::Value> second = llvm::json::parse(lines[1]);
  ASSERT_TRUE(!!second);
  llvm::json::Object *secondObj = second->getAsObject();
  ASSERT_TRUE(secondObj);
  EXPECT_EQ(secondObj->getInteger("round"), 1);

  llvm::json::Array *results = secondObj->getArray("results");
  ASSERT_TRUE(results);
  EXPECT_FALSE(results->empty())
      << "the second round was told nothing about what the first measured";

  // The tile that is not on any ladder was refused, and saying so is the only
  // way the model learns a rule it cannot read off the ladders.
  llvm::json::Array *rejected = secondObj->getArray("rejected");
  ASSERT_TRUE(rejected);
  ASSERT_FALSE(rejected->empty()) << "the refusal was not reported back";
  llvm::json::Object *refusal = rejected->front().getAsObject();
  ASSERT_TRUE(refusal);
  EXPECT_FALSE(refusal->getString("reason")->empty())
      << "a config was refused without saying why";
  EXPECT_EQ(refusal->getObject("config")->getInteger("mPerBlock"), 33);
}

// A trace is all a finished tuning run can be asked about afterwards, so it has
// to be readable on its own and to agree with what the search did.
TEST(LLMSearchTest, TracesEachRound) {
  FixtureProposer fixture(
      alwaysAnswers("[render(mPerBlock=32 + 16 * request['round'])]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  SmallString<128> tracePath;
  ASSERT_FALSE(
      llvm::sys::fs::createTemporaryFile("llm-trace", "jsonl", tracePath));

  LLMOptions options = testOptions(fixture);
  options.search.maxRounds = 2;
  options.trace = openTrace(tracePath);
  runSearch(*e.module, options);

  auto buffer = llvm::MemoryBuffer::getFile(tracePath);
  ASSERT_TRUE(buffer) << "the search wrote no trace to " << tracePath;
  SmallVector<StringRef> lines;
  (*buffer)->getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/false);
  llvm::sys::fs::remove(tracePath);
  ASSERT_GE(lines.size(), 2u);

  llvm::Expected<llvm::json::Value> header = llvm::json::parse(lines.front());
  ASSERT_TRUE(!!header) << "the header is not JSON: " << lines.front().str();
  llvm::json::Object *headerObj = header->getAsObject();
  ASSERT_TRUE(headerObj);
  EXPECT_EQ(headerObj->getString("kind"), "llm-header");
  EXPECT_EQ(headerObj->getInteger("maxRounds"),
            static_cast<int64_t>(options.search.maxRounds));
  // The plotter tells the two stages apart by this, so a hybrid run's records
  // have to be distinguishable from LFBO's.
  EXPECT_NE(headerObj->getString("kind"), "header");

  for (StringRef line : ArrayRef(lines).drop_front()) {
    llvm::Expected<llvm::json::Value> record = llvm::json::parse(line);
    ASSERT_TRUE(!!record) << "a round is not JSON: " << line.str();
    llvm::json::Object *obj = record->getAsObject();
    ASSERT_TRUE(obj);
    EXPECT_EQ(obj->getString("kind"), "llm-round");
    ASSERT_TRUE(obj->getInteger("proposed"));
    ASSERT_TRUE(obj->getInteger("measured"));
    ASSERT_TRUE(obj->getNumber("totalMs"));
  }
}

// The effort is the one budget a tuning run gets to set, so the cheaper one has
// to ask for strictly less of the expensive thing, which here is rounds: every
// round is a model call that a tuning run waits out.
TEST(LLMSearchTest, QuickEffortAsksForFewerRounds) {
  LLMOptions full;
  full.setEffort(SearchEffort::Full);
  LLMOptions quick;
  quick.setEffort(SearchEffort::Quick);
  EXPECT_LT(quick.search.maxRounds, full.search.maxRounds);
  EXPECT_GT(quick.search.maxRounds, 0u)
      << "the quick effort asks for no rounds at all, so it never asks a model";
}

//===----------------------------------------------------------------------===//
// Walking away mid-round
//===----------------------------------------------------------------------===//

TEST(LLMSearchTest, SurvivesBeingAbandonedWithARequestInFlight) {
  // Round 0's request is launched as soon as the seed batch goes out, so that
  // the model's latency is spent alongside a batch of compiles. A client is
  // free never to come back for it -- the tuning driver returns a failure
  // without another call to `getPerfConfigBatch` as soon as one config in the
  // batch fails to compile -- which destroys the search while its helper is
  // still running. The proposer that helper is reading its paths from has to
  // outlive it, so the search drains the request before it goes.
  //
  // The fixture sleeps to keep the request in flight for the destructor to
  // find. There is nothing to assert beyond arriving here: the failure this
  // pins is a use-after-free, so it takes a sanitizer to see it, and the older
  // secondary symptom -- teardown sitting out the whole request deadline --
  // would only show up as a slow test.
  FixtureProposer fixture(std::string("import time\ntime.sleep(0.5)\n") +
                          alwaysAnswers("[render()]"));
  GemmModule e(/*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");

  std::unique_ptr<TuningSearchStrategy> search =
      createLLMSearchStrategy(*e.module, testOptions(fixture));
  EXPECT_FALSE(search->getPerfConfigBatch({}).empty())
      << "no seed batch, so no round was ever launched to abandon";
  search.reset();
}

//===----------------------------------------------------------------------===//
// The transport
//
// Driven directly, because the search makes a failed round fatal on purpose
// and a fatal error cannot be caught and inspected from a test.
//===----------------------------------------------------------------------===//

llvm::json::Object minimalRequest() {
  return llvm::json::Object{{"round", 0}, {"configsRequested", 2}};
}

TEST(LLMProposerTest, ReadsTheConfigsAHelperWrote) {
  // Handed on verbatim: what a perf config means is the search's question,
  // and asking it here would be a second place that knows the format.
  FixtureProposer fixture(
      alwaysAnswers("['gemm:mPerBlock=64', 'gemm:nPerBlock=128']"));
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/60);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_TRUE(!!configs) << llvm::toString(configs.takeError());
  EXPECT_EQ(*configs, (std::vector<std::string>{"gemm:mPerBlock=64",
                                                "gemm:nPerBlock=128"}));
}

TEST(LLMProposerTest, TrimsMoreConfigsThanTheRequestAskedFor) {
  // A model that answered with one config too many used to end the tuning run.
  FixtureProposer fixture(alwaysAnswers(
      "['gemm:mPerBlock=64', 'gemm:nPerBlock=128', 'gemm:kPerBlock=16']"));
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/60);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_TRUE(!!configs);
  EXPECT_EQ(*configs, (std::vector<std::string>{"gemm:mPerBlock=64",
                                                "gemm:nPerBlock=128"}));
}

TEST(LLMProposerTest, RefusesAReplyNestedTooDeeplyToParseSafely) {
  FixtureProposer fixture(
      "open(args.response, 'w').write('[' * 65 + '0' + ']' * 65)");
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/60);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_FALSE(!!configs);
  EXPECT_NE(llvm::toString(configs.takeError()).find("nested too deeply"),
            std::string::npos);
}

TEST(LLMProposerTest, ZeroTimeoutLeavesTheHelperUnlimited) {
  FixtureProposer fixture(alwaysAnswers("['gemm:mPerBlock=64']"));
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/0);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_TRUE(!!configs) << llvm::toString(configs.takeError());
  EXPECT_EQ(configs->size(), 1u);
}

// A helper that cannot do its job says so in the reply, and the message it
// wrote is the whole value of that: "$CURSOR_API_KEY is not set" is actionable
// and "the helper exited 1" is not.
TEST(LLMProposerTest, ReportsAFailureTheHelperExplained) {
  FixtureProposer fixture(
      "json.dump({'error': 'no API key, so no model'}, open(args.response, "
      "'w'))");
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/60);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_FALSE(!!configs);
  EXPECT_EQ(llvm::toString(configs.takeError()), "no API key, so no model");
}

TEST(LLMProposerTest, ReportsAHelperThatFailed) {
  FixtureProposer fixture("sys.exit(3)");
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/60);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_FALSE(!!configs);
  EXPECT_NE(llvm::toString(configs.takeError()).find("exit code 3"),
            std::string::npos);
}

// A model that never answers must not hold a tuning run open indefinitely.
TEST(LLMProposerTest, KillsAHelperThatOutlivesItsDeadline) {
  FixtureProposer fixture("import time; time.sleep(120)");
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/1);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_FALSE(!!configs);
  EXPECT_NE(llvm::toString(configs.takeError()).find("did not answer"),
            std::string::npos);
}

// Everything the model got wrong has been dealt with on the Python side by the
// time a reply arrives, so what is left to check here is that the helper kept
// its own end of the contract. A reply that is not a config list is a broken
// helper, not a bad round.
TEST(LLMProposerTest, RefusesAReplyThatIsNotAConfigList) {
  // The sparse objects the model answers with, sent on unchanged: a helper
  // that forgot to complete them against the exemplar.
  FixtureProposer fixture(alwaysAnswers("[{'mPerBlock': 64}]"));
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/60);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_FALSE(!!configs);
  EXPECT_NE(llvm::toString(configs.takeError()).find("not a string"),
            std::string::npos);
}

// The request has to arrive as the helper's own argument, since that is the
// contract llm/proposer.py is written against.
TEST(LLMProposerTest, PassesTheRequestToTheHelper) {
  FixtureProposer fixture(
      alwaysAnswers("['gemm:mPerBlock=%d' % request['configsRequested']]"));
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"",
                       /*transcriptPath=*/"", /*timeoutSec=*/60);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_TRUE(!!configs) << llvm::toString(configs.takeError());
  ASSERT_EQ(configs->size(), 1u);
  EXPECT_EQ((*configs)[0], "gemm:mPerBlock=2")
      << "the helper did not see the request it was sent";
}

// Where the helper writes down what it asked the model and what it was told.
// Passed on as a path rather than opened here: the file is prose for a person
// (see llm/transcript.py), and the helper is what has the prose.
TEST(LLMProposerTest, PassesTheTranscriptPathToTheHelper) {
  SmallString<128> transcriptPath;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("rocmlir-llm-transcript",
                                                  "log", transcriptPath));
  llvm::FileRemover removeTranscript(transcriptPath);

  FixtureProposer fixture(
      "open(args.transcript, 'a').write('a round happened\\n')\n" +
      alwaysAnswers("['gemm:mPerBlock=64']"));
  LLMProposer proposer(fixture.program(), /*sessionPath=*/"", transcriptPath,
                       /*timeoutSec=*/60);

  llvm::Expected<std::vector<std::string>> configs =
      proposer.propose(minimalRequest());
  ASSERT_TRUE(!!configs) << llvm::toString(configs.takeError());
  auto written = llvm::MemoryBuffer::getFile(transcriptPath);
  ASSERT_TRUE(!!written) << "the helper was given no transcript to write to";
  EXPECT_EQ((*written)->getBuffer(), "a round happened\n");
}

// An explicit path wins, and with nothing to go on the helper is looked for
// beside the running binary -- where `ci-performance-scripts` puts it -- and
// nowhere else. The source tree is not a fallback: this test runs from the
// build tree with the sources right there, and finding them would mean a
// driver can silently run a helper that does not match its build.
TEST(LLMProposerTest, ResolvesTheHelperBesideTheBinary) {
  EXPECT_EQ(LLMProposer::resolveProgram("/some/where/mine.py"),
            "/some/where/mine.py");

  if (std::getenv("ROCMLIR_LLM_PROPOSER"))
    GTEST_SKIP() << "the override this asks about is set in the environment";
  std::string resolved = LLMProposer::resolveProgram("");
  EXPECT_TRUE(StringRef(resolved).ends_with("llm/proposer.py")) << resolved;
  EXPECT_EQ(StringRef(resolved).find("utils/performance"), StringRef::npos)
      << "resolved to the source tree: " << resolved;
}

} // namespace
