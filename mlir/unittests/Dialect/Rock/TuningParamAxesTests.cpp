//===- TuningParamAxesTests.cpp - Axes vs. enumeration equivalence --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `createTunableParamAxes` describes the space `createTunableParamSpace`
// enumerates without enumerating it, which only holds while its `isFeasible`
// keeps saying what the enumerator's loop nest says. Nothing in the compiler
// forces the two to agree, so these tests do: the axes must accept every config
// the enumerator emits and, walking their own product, must accept those and
// nothing else beyond what a widening deliberately opens.
//
// The axes are wider than the enumeration where it narrowed itself only to keep
// its product affordable, so the equality above is checked over the values the
// enumeration does use, and modulo the one widening that opens combinations of
// those values rather than new ones (which K a tile may take). The rest of the
// tests cover the widening: which pins are lifted, and what a lifted parameter
// is still held to.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include <memory>
#include <random>
#include <set>

using namespace mlir;
using namespace mlir::rock;

namespace {

// A module holding a single non-scaled `rock.gemm` of shape
// [1,M,K]x[1,K,N] -> [1,M,N], stored to an output argument as a kernel written
// by `rocmlir-gen` is. The store is what lets the output element type be found
// from the gemm, which is how split-K legality is decided (see
// `traceRootOutputToArgs`).
struct GemmModule {
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;

  GemmModule(llvm::function_ref<Type(OpBuilder &)> elemType, int64_t m,
             int64_t n, int64_t k, StringRef arch,
             bool enableSplitKForTuning = false) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    OpBuilder b(&ctx);
    Location loc = b.getUnknownLoc();
    Type elem = elemType(b);
    auto aType = RankedTensorType::get({1, m, k}, elem);
    auto bType = RankedTensorType::get({1, k, n}, elem);
    auto cType = RankedTensorType::get({1, m, n}, elem);

    module = ModuleOp::create(loc);
    b.setInsertionPointToEnd(module->getBody());
    auto func = func::FuncOp::create(
        b, loc, "test", b.getFunctionType({aType, bType, cType}, {cType}));
    if (enableSplitKForTuning)
      func->setAttr(rock::EnableSplitKForTuningAttr::getMnemonic(),
                    b.getUnitAttr());
    Block *body = func.addEntryBlock();
    b.setInsertionPointToStart(body);
    auto gemmOp = GemmOp::create(
        b, loc, /*c=*/cType, /*a=*/body->getArgument(0),
        /*b=*/body->getArgument(1), /*scaleA=*/Value(), /*scaleB=*/Value(),
        /*aTransposed=*/UnitAttr{}, /*bTransposed=*/UnitAttr{},
        /*oTransposed=*/UnitAttr{}, /*aScaleTransposed=*/UnitAttr{},
        /*bScaleTransposed=*/UnitAttr{}, /*quantBlockSize=*/IntegerAttr{},
        /*params=*/nullptr);
    Value stored = StoreOp::create(b, loc, /*result=*/cType,
                                   /*source=*/gemmOp.getResult(),
                                   /*dest=*/body->getArgument(2),
                                   /*resultAlias=*/Value(), StoreMethod::Set);
    func::ReturnOp::create(b, loc, stored);
    (*module)->setAttr(rock::ArchAttr::getMnemonic(),
                       b.getStringAttr(Twine("amdgcn-amd-amdhsa:") + arch));
  }
};

// A module holding a single forward `rock.conv` over a channels-first input, as
// `rocmlir-gen --operation conv --fil_layout gkcyx --in_layout ngchw
// --out_layout ngkhw` writes it. That layout is the one whose gemmK merges the
// channel dim outermost, which is what leaves a K tile something to align to
// (see `kPerBlockAlignmentFactor`); `filterSpatial` of 1 leaves it nothing.
struct ConvModule {
  MLIRContext ctx;
  OwningOpRef<ModuleOp> module;

  ConvModule(llvm::function_ref<Type(OpBuilder &)> elemType,
             int64_t filterSpatial, StringRef arch, int64_t channels = 64,
             int64_t outChannels = 256, int64_t inSpatial = 14,
             int64_t batch = 64) {
    DialectRegistry reg;
    reg.insert<rock::RockDialect>();
    reg.insert<func::FuncDialect>();
    ctx.appendDialectRegistry(reg);
    ctx.loadAllAvailableDialects();
    OpBuilder b(&ctx);
    Location loc = b.getUnknownLoc();
    Type elem = elemType(b);
    int64_t outSpatial = inSpatial - filterSpatial + 1;
    auto filType = RankedTensorType::get(
        {1, outChannels, channels, filterSpatial, filterSpatial}, elem);
    auto inType =
        RankedTensorType::get({batch, 1, channels, inSpatial, inSpatial}, elem);
    auto outType = RankedTensorType::get(
        {batch, 1, outChannels, outSpatial, outSpatial}, elem);

    module = ModuleOp::create(loc);
    b.setInsertionPointToEnd(module->getBody());
    auto func = func::FuncOp::create(
        b, loc, "test",
        b.getFunctionType({filType, inType, outType}, {outType}));
    Block *body = func.addEntryBlock();
    b.setInsertionPointToStart(body);
    SmallVector<NamedAttribute> attrs = {
        b.getNamedAttr("filter_layout",
                       b.getStrArrayAttr({"g", "k", "c", "0", "1"})),
        b.getNamedAttr("input_layout",
                       b.getStrArrayAttr({"ni", "gi", "ci", "0i", "1i"})),
        b.getNamedAttr("output_layout",
                       b.getStrArrayAttr({"no", "go", "ko", "0o", "1o"})),
        b.getNamedAttr("dilations", b.getIndexArrayAttr({1, 1})),
        b.getNamedAttr("strides", b.getIndexArrayAttr({1, 1})),
        b.getNamedAttr("padding", b.getIndexArrayAttr({0, 0, 0, 0}))};
    auto convOp = ConvOp::create(
        b, loc, outType, ValueRange{body->getArgument(0), body->getArgument(1)},
        attrs);
    Value stored = StoreOp::create(b, loc, /*result=*/outType,
                                   /*source=*/convOp.getResult(),
                                   /*dest=*/body->getArgument(2),
                                   /*resultAlias=*/Value(), StoreMethod::Set);
    func::ReturnOp::create(b, loc, stored);
    (*module)->setAttr(rock::ArchAttr::getMnemonic(),
                       b.getStringAttr(Twine("amdgcn-amd-amdhsa:") + arch));
  }
};

// A module holding a single `rock.attention`, as `rocmlir-gen --operation
// attention -t f16 -g 1 -seq_len_q 1024 -seq_len_k 1024 -head_dim_qk 128
// -head_dim_v 512` writes it.
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
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 1024 + d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 512 + d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{1024, 128} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 128] -> [131072]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{128, 1024} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 1024] -> [131072]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{1024, 512} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:ARCH"} {
  func.func @rock_attention(%arg0: tensor<131072xf16>, %arg1: tensor<131072xf16>, %arg2: tensor<524288xf16>) -> tensor<1x1024x512xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:ARCH", rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<131072xf16> to tensor<1x1024x128xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<131072xf16> to tensor<1x128x1024xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<524288xf16> to tensor<1x1024x512xf16>
    %result = rock.attention{
     qk = %0 * %1 : tensor<1x1024x128xf16>, tensor<1x128x1024xf16>
     softmax(qk) * %2 : tensor<1x1024x512xf16>
    } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<1x1024x512xf16>
    return %result : tensor<1x1024x512xf16>
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

// The configs of one tuning space, as the values a search would see.
std::set<std::vector<int64_t>> configsOf(MLIRContext &ctx,
                                         const TuningParamSet &space) {
  std::set<std::vector<int64_t>> configs;
  for (const PerfConfigString &perfConfig : space.tuningRange) {
    StringAttr asAttr = StringAttr::get(&ctx, StringRef(perfConfig));
    RockTuningParamAttrInterface params = GemmParamsAttr::get(asAttr);
    if (!params)
      params = GemmGemmParamsAttr::get(asAttr);
    if (!params) {
      ADD_FAILURE() << "unparseable perf config '" << std::string(perfConfig)
                    << "'";
      continue;
    }
    SmallVector<int64_t> values;
    params.getParamValues(values);
    configs.insert(std::vector<int64_t>(values.begin(), values.end()));
  }
  return configs;
}

// The configs `mod`'s space enumerates, as the values a search would see.
//
// Full and exhaustive tuning also hand out the quick list, so that neither can
// come out behind quick tuning, and those configs are a table recorded from
// real tuning runs rather than a product of the ranges: they carry K tiles well
// past `MAX_K_PER_BLOCK` and an unset `matrixInstrNonkdim`, so no axis built
// from those ranges holds them, by design. `TuningParamAxes` describes the
// space a search walks, which is the enumerated one, so the quick configs come
// back out here and the comparisons below stay about the two views of the same
// ranges. That a search can still start from a quick config is a separate
// property, and `valuesAreAdmissible` rather than the axes is what carries it.
std::set<std::vector<int64_t>> enumerateSpace(MLIRContext &ctx, ModuleOp mod,
                                              TuningParamSetKind kind) {
  std::unique_ptr<TuningParamSet> space(createTunableParamSpace(mod, kind));
  std::set<std::vector<int64_t>> configs = configsOf(ctx, *space);
  if (kind == TuningParamSetKind::Quick)
    return configs;
  std::unique_ptr<TuningParamSet> quick(
      createTunableParamSpace(mod, TuningParamSetKind::Quick));
  for (const std::vector<int64_t> &config : configsOf(ctx, *quick))
    configs.erase(config);
  return configs;
}

// Every combination of `lists` that `isFeasible` accepts. Only called on boxes
// small enough to walk.
std::set<std::vector<int64_t>>
feasibleProduct(const TuningParamAxes &axes,
                ArrayRef<std::vector<int64_t>> lists) {
  std::set<std::vector<int64_t>> configs;
  std::vector<int64_t> values(lists.size());
  std::function<void(size_t)> recurse = [&](size_t param) {
    if (param == lists.size()) {
      if (axes.isFeasible(values))
        configs.insert(values);
      return;
    }
    for (int64_t value : lists[param]) {
      values[param] = value;
      recurse(param + 1);
    }
  };
  recurse(0);
  return configs;
}

size_t productSize(ArrayRef<std::vector<int64_t>> lists) {
  size_t size = 1;
  for (const std::vector<int64_t> &axis : lists)
    size *= axis.size();
  return size;
}

// The values each parameter is seen to take across `configs`. Used to walk the
// part of the axes the enumerator covers: the axes are deliberately wider, so
// this is the box in which the two can be compared.
std::vector<std::vector<int64_t>>
observedValues(const std::set<std::vector<int64_t>> &configs,
               size_t numParams) {
  std::vector<std::set<int64_t>> distinct(numParams);
  for (const std::vector<int64_t> &config : configs)
    for (auto [values, value] : llvm::zip_equal(distinct, config))
      values.insert(value);

  std::vector<std::vector<int64_t>> observed;
  for (const std::set<int64_t> &values : distinct)
    observed.emplace_back(values.begin(), values.end());
  return observed;
}

// The index of a named parameter, so a test can speak of one without restating
// the order they come in.
size_t paramIndex(const TuningParamAxes &axes, StringRef name) {
  SmallVector<StringRef> names;
  axes.getParamNames(names);
  auto param = llvm::find(names, name);
  EXPECT_NE(param, names.end()) << name << " is not a parameter";
  return param - names.begin();
}

// The axes must contain the enumerated space. This is the half that catches an
// axis built from the wrong list, or a membership test that rejects a config
// the space really does offer.
void expectAxesAcceptEnumeration(MLIRContext &ctx, ModuleOp mod,
                                 TuningParamSetKind kind) {
  std::unique_ptr<TuningParamAxes> axes = createTunableParamAxes(mod, kind);
  ASSERT_TRUE(axes) << "no axes for a module with a tuning space";
  std::set<std::vector<int64_t>> enumerated = enumerateSpace(ctx, mod, kind);
  ASSERT_FALSE(enumerated.empty());

  SmallVector<StringRef> names;
  axes->getParamNames(names);
  EXPECT_EQ(names.size(), axes->getAxes().size())
      << "one axis per parameter, so that both are indexed alike";

  for (const std::vector<int64_t> &config : enumerated) {
    if (axes->isFeasible(config))
      continue;
    PerfConfigString spelled;
    axes->serialize(config, spelled);
    ADD_FAILURE() << "axes reject an enumerated config: "
                  << std::string(spelled);
  }
}

// f16 on gfx942 drives the MFMA range, the widest of the three.
TEST(TuningParamAxesTest, AxesAcceptEnumerationMfma) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  expectAxesAcceptEnumeration(e.ctx, *e.module, TuningParamSetKind::Exhaustive);
  expectAxesAcceptEnumeration(e.ctx, *e.module, TuningParamSetKind::Full);
}

// f16 on gfx1201 drives the WMMA range, whose kpack and matrixInstrNonkdim
// axes are pinned where MFMA's are not.
TEST(TuningParamAxesTest, AxesAcceptEnumerationWmma) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/256, /*n=*/256, /*k=*/576, "gfx1201");
  expectAxesAcceptEnumeration(e.ctx, *e.module, TuningParamSetKind::Exhaustive);
  expectAxesAcceptEnumeration(e.ctx, *e.module, TuningParamSetKind::Full);
}

// f32 on gfx1201 has no matrix-accel instruction, so this is the FMA range.
TEST(TuningParamAxesTest, AxesAcceptEnumerationNonAccel) {
  GemmModule e([](OpBuilder &b) { return b.getF32Type(); },
               /*m=*/128, /*n=*/128, /*k=*/48, "gfx1201");
  expectAxesAcceptEnumeration(e.ctx, *e.module, TuningParamSetKind::Exhaustive);
}

// The gemm+gemm space, whose parameters are the gemm ones plus the second
// gemm's N tile. The axes read `getRangeGemmGemm`'s lists positionally, so this
// is what holds them to the list that function actually returns.
TEST(TuningParamAxesTest, AxesAcceptEnumerationGemmGemm) {
  AttentionModule e("gfx942");
  ASSERT_TRUE(e.module);
  expectAxesAcceptEnumeration(e.ctx, *e.module, TuningParamSetKind::Full);
}

// The other half: over the values the enumerator itself varies, walking the
// axes must reproduce its space. A membership test that is merely permissive
// passes the test above and fails this one, since it would admit combinations
// of those values that the space omits.
//
// The comparison is restricted to the values the enumeration uses because the
// axes are deliberately wider: they re-open what the enumerators pinned to keep
// their product affordable. Taking the box from the enumerated configs rather
// than naming the parameters keeps this true whenever an axis is widened again.
//
// One widening reaches inside that box, so the equality is stated modulo
// `kPerBlock`: a tile may take any tile-sized step up to its ceiling
// (`tileValues`, TuningSearch.cpp) where `computeKPerBlock` hands it a ladder,
// so a K the enumerator pairs with one tile may now pair with another, and both
// K values are already in the box. Every other parameter still has to match
// exactly, and every enumerated config still has to be accepted.
void expectFeasibleProductEqualsEnumeration(MLIRContext &ctx, ModuleOp mod,
                                            TuningParamSetKind kind,
                                            bool expectRejections = false) {
  std::unique_ptr<TuningParamAxes> axes = createTunableParamAxes(mod, kind);
  ASSERT_TRUE(axes);
  std::set<std::vector<int64_t>> enumerated = enumerateSpace(ctx, mod, kind);
  ASSERT_FALSE(enumerated.empty());

  std::vector<std::vector<int64_t>> box =
      observedValues(enumerated, axes->getAxes().size());
  ASSERT_LT(productSize(box), size_t(4000000))
      << "box too large to walk; pick a smaller problem";

  std::set<std::vector<int64_t>> feasible = feasibleProduct(*axes, box);
  if (expectRejections)
    ASSERT_LT(feasible.size(), productSize(box))
        << "no combination was rejected, so this shape does not exercise the "
           "constraints it was picked for";

  const size_t kIdx = paramIndex(*axes, "kPerBlock");
  auto withoutKPerBlock =
      [kIdx](const std::set<std::vector<int64_t>> &configs) {
        std::set<std::vector<int64_t>> projected;
        for (std::vector<int64_t> config : configs) {
          config.erase(config.begin() + kIdx);
          projected.insert(std::move(config));
        }
        return projected;
      };
  EXPECT_EQ(withoutKPerBlock(feasible), withoutKPerBlock(enumerated));
  for (const std::vector<int64_t> &config : enumerated) {
    if (feasible.count(config))
      continue;
    PerfConfigString spelled;
    axes->serialize(config, spelled);
    ADD_FAILURE() << "walking the axes missed an enumerated config: "
                  << std::string(spelled);
  }
}

TEST(TuningParamAxesTest, FeasibleProductEqualsEnumerationNonAccel) {
  // Small enough that the product of the axes can be walked in full.
  GemmModule e([](OpBuilder &b) { return b.getF32Type(); },
               /*m=*/64, /*n=*/64, /*k=*/64, "gfx1201");
  expectFeasibleProductEqualsEnumeration(e.ctx, *e.module,
                                         TuningParamSetKind::Exhaustive);
}

// K = 576 has non-power-of-two divisors that only some tiles may use, so here
// the kPerBlock axis really is wider than what any one tile allows and the
// product holds combinations the space must reject. This is the case that tells
// a correct membership test from a merely permissive one.
TEST(TuningParamAxesTest, FeasibleProductEqualsEnumerationTileDependentK) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/256, /*n=*/256, /*k=*/576, "gfx1201");
  expectFeasibleProductEqualsEnumeration(e.ctx, *e.module,
                                         TuningParamSetKind::Exhaustive,
                                         /*expectRejections=*/true);
}

// LFBO seeds itself by drawing from the axes and keeping the draws that are
// configs of the space (`LFBOSearch::sampleFeasibleConfigs`), which only works
// while a fair share of draws survive. If a constraint ever makes the space a
// thin slice of its own axes, seeding starves and this test says so before a
// tuning run does.
TEST(TuningParamAxesTest, RandomDrawsFromAxesAreOftenFeasible) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/4096, /*n=*/4096, /*k=*/4096, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  std::mt19937_64 rng(42);
  const unsigned draws = 20000;
  unsigned feasible = 0;
  std::vector<int64_t> config(axes->getAxes().size());
  for (unsigned draw = 0; draw < draws; ++draw) {
    for (auto [value, axis] : llvm::zip_equal(config, axes->getAxes()))
      value =
          axis[std::uniform_int_distribution<size_t>(0, axis.size() - 1)(rng)];
    feasible += axes->isFeasible(config);
  }

  // LFBO allows 100 attempts per config it wants, so anything above a percent
  // leaves it a wide margin.
  EXPECT_GT(feasible * 100.0 / draws, 1.0)
      << feasible << " of " << draws << " draws feasible";
}

// A parameter whose axis holds one value is one a search can never touch, and
// the knobs are the parameters most likely to be left that way, since the
// enumerators pin all eight and the axes are what re-open them. Both spellings
// of a config carry the same knobs.
//
// On an arch and a GEMM where these seven can act; the two schedule knobs are
// pinned where they cannot, see `ScheduleKnobsAreExploredOnlyWhereTheyCanAct`,
// and `useBf16x3ForF32` is left out because this GEMM is f16, see
// `Bf16x3ForF32IsExploredOnlyOnAnF32Dot`.
TEST(TuningParamAxesTest, KnobsCanBeMoved) {
  const StringRef knobs[] = {"useAsyncCopy",         "useBlockPingpong",
                             "useInThreadTranspose", "useBufferOps",
                             "useBufferAtomics",     "useReductionLayout",
                             "useOptimizeEpilogue"};
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  SmallVector<StringRef> names;
  axes->getParamNames(names);
  for (StringRef knob : knobs) {
    auto param = llvm::find(names, knob);
    ASSERT_NE(param, names.end()) << knob << " is not a parameter";
    // Off and on. Not `kKnobDefault`, which is legal but duplicates whichever
    // of the two it resolves to, so it is not worth exploring.
    std::set<int64_t> axis(axes->getAxes()[param - names.begin()].begin(),
                           axes->getAxes()[param - names.begin()].end());
    EXPECT_EQ(axis, (std::set<int64_t>{0, 1})) << knob << " cannot be moved";
  }
}

// What is legal and what is worth exploring are different questions, and the
// knobs are where they part company: `kKnobDefault` is off the axes but must
// still be admitted, since it is how every config the tuning space and the
// quick list hand out spells its knobs. Were `isFeasible` to answer the
// exploration question instead, a search could not start from any of them.
TEST(TuningParamAxesTest, DefaultKnobsAreAdmissibleButNotExplored) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  const TuningParamSetKind kind = TuningParamSetKind::Exhaustive;
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, kind);
  ASSERT_TRUE(axes);

  SmallVector<StringRef> names;
  axes->getParamNames(names);
  std::set<std::vector<int64_t>> enumerated =
      enumerateSpace(e.ctx, *e.module, kind);
  ASSERT_FALSE(enumerated.empty());

  const std::vector<int64_t> &config = *enumerated.begin();
  EXPECT_TRUE(axes->isFeasible(config))
      << "a config of the tuning space is not admitted";

  // That config is off the axes precisely because of its knobs, which is what
  // makes this more than a restatement of the containment tests.
  bool anyOffAxis = false;
  for (auto [name, axis, value] :
       llvm::zip_equal(names, axes->getAxes(), config))
    if (!llvm::is_contained(axis, value)) {
      anyOffAxis = true;
      EXPECT_TRUE(name.starts_with("use"))
          << name << " is off its axis but is not a knob";
    }
  EXPECT_TRUE(anyOffAxis) << "nothing is off the axes, so the two questions "
                             "cannot be told apart here";
}

// The axis of a named parameter, so a test can state one without restating the
// order the parameters come in.
std::vector<int64_t> axisOf(const TuningParamAxes &axes, StringRef name) {
  return axes.getAxes()[paramIndex(axes, name)];
}

// The parameters the enumerators pin to keep the product they write out
// affordable, rather than because a kernel may hold nothing else. Each of these
// reaches something: `kpack=2` is all over the quick perfconfigs and is what
// `getRangeGemmGemm` enumerates, a nonzero `wavesPerEU` is what
// `setKernelAttributes` (TritonToHsaco.cpp) turns into `amdgpu-waves-per-eu`, a
// nonzero `matrixInstrNonkdim` picks the MFMA tile, `numStages` past three is
// how `ChainedDotSchedule` is reached at all, and `gridGroupSize` overrides the
// group size `makeGroupedGridLayout` would compute. A search pays per config
// benchmarked rather than per config in the space, so its axes carry the range
// each parameter may hold. Were any of these to collapse back to one value, a
// run would silently stop trying the parameter at all.
TEST(TuningParamAxesTest, PinsKeptForEnumerationCostAreLifted) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  // The powers of two up to gfx942's `getMaxKpack`, which is 2.
  EXPECT_EQ(axisOf(*axes, "kpack"), (std::vector<int64_t>{1, 2}));
  // Zero, meaning the backend chooses, plus every occupancy target up to
  // gfx942's `getMaxWavesPerEU`, which is 8. The ones between the powers of two
  // are targets of their own, since the backend is asked to hit the number.
  EXPECT_EQ(axisOf(*axes, "wavesPerEU"),
            (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
  // Zero, meaning Triton's heuristic chooses, plus the square MFMA tiles
  // `MfmaIntrinsic::selectFor` has.
  EXPECT_EQ(axisOf(*axes, "matrixInstrNonkdim"),
            (std::vector<int64_t>{0, 16, 32}));
  EXPECT_EQ(axisOf(*axes, "numStages"),
            (std::vector<int64_t>{1, 2, 3, 4, 5, 6}));
  // Zero, meaning `makeGroupedGridLayout` computes the group size, plus every
  // group size up to where a larger one stops describing a different grid.
  EXPECT_EQ(axisOf(*axes, "gridGroupSize"),
            (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                  14, 15, 16}));
  // Every workgroup width `validateNumWaves` allows at wave64, which is where
  // the quick range stops at three of the five.
  EXPECT_EQ(axisOf(*axes, "numWaves"), (std::vector<int64_t>{1, 2, 4, 8, 16}));
}

// Every tile a kernel can hold up to `ceiling`, which is what an axis carries
// where the tile need not be a power of two: the powers of two below 16, then
// every multiple of 16 (`tileValues`, TuningSearch.cpp).
std::vector<int64_t> tileStepsUpTo(int64_t ceiling) {
  std::vector<int64_t> steps = {1, 2, 4, 8};
  for (int64_t tile = 16; tile <= ceiling; tile += 16)
    steps.push_back(tile);
  return steps;
}

// A plain GEMM's tiles need not be powers of two (`pow2TilesRequired`), so all
// three carry every tile rather than only the ladder the enumerators double
// their way up. The ceiling is the ladder's own: `MAX_K_PER_BLOCK` for K and
// `MAX_MN_PER_BLOCK` for M/N.
TEST(TuningParamAxesTest, TilesOfferEveryStepWhereTheyNeedNotBePow2) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  EXPECT_EQ(axisOf(*axes, "kPerBlock"), tileStepsUpTo(512));
  EXPECT_EQ(axisOf(*axes, "mPerBlock"), tileStepsUpTo(256));
  EXPECT_EQ(axisOf(*axes, "nPerBlock"), tileStepsUpTo(256));
}

// The M/N and K requirements are separate, and gfx950 is where they part
// company: the peeled K loop miscompiles there, so the K tile is held to a
// power of two, while `rock-decompose-nonpow2-tiles` still takes any M/N tile
// apart.
TEST(TuningParamAxesTest, Pow2KTileLeavesTheMNTilesAlone) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx950");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  EXPECT_EQ(axisOf(*axes, "kPerBlock"),
            (std::vector<int64_t>{16, 32, 64, 128, 256, 512}));
  EXPECT_EQ(axisOf(*axes, "mPerBlock"), tileStepsUpTo(256));
}

// A conv merges the filter's spatial dims into gemmK, and a K tile that is a
// multiple of their product advances K without moving the padded input window
// (`kPerBlockAlignmentFactor`). Those are the tiles the ladders miss: a 3x3
// filter wants multiples of 9, and only every sixteenth of them is a multiple
// of 16 as well. A search offers all of them up to `kMaxKPerBlock` and finds
// out by measuring which one a shape wants, where `computeKPerBlock` can afford
// only the few that divide K and fit a matrix instruction's K extent as well.
TEST(TuningParamAxesTest, ConvKTilesFollowTheFilterAlignment) {
  ConvModule e([](OpBuilder &b) { return b.getF32Type(); },
               /*filterSpatial=*/3, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Full);
  ASSERT_TRUE(axes);

  std::vector<int64_t> kTiles = axisOf(*axes, "kPerBlock");
  for (int64_t tile = 9; tile <= 512; tile += 9)
    EXPECT_TRUE(llvm::is_contained(kTiles, tile))
        << "the 3x3 filter's alignment asks for a K tile of " << tile;
}

// A 1x1 filter merges only the channel dim into gemmK, which makes it a GEMM as
// far as the K index computation goes, and a GEMM has no alignment to offer
// tiles for. The odd multiples of 9 are what tells the two apart: nothing else
// on this axis reaches them.
TEST(TuningParamAxesTest, A1x1ConvHasNoFilterAlignmentToFollow) {
  ConvModule e([](OpBuilder &b) { return b.getF32Type(); },
               /*filterSpatial=*/1, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Full);
  ASSERT_TRUE(axes);

  std::vector<int64_t> kTiles = axisOf(*axes, "kPerBlock");
  EXPECT_FALSE(llvm::is_contained(kTiles, 9));
  EXPECT_FALSE(llvm::is_contained(kTiles, 27));
}

// Where the K tile has to be a power of two there is no honouring the filter's
// alignment at all, since no multiple of 9 is one: gfx950 miscompiles the
// peeled K loop, so a conv's K axis stays the pow2 ladder just as a GEMM's
// does.
TEST(TuningParamAxesTest, ConvKTilesStayPow2WhereTheyMust) {
  ConvModule e([](OpBuilder &b) { return b.getF32Type(); },
               /*filterSpatial=*/3, "gfx950");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Full);
  ASSERT_TRUE(axes);

  EXPECT_EQ(axisOf(*axes, "kPerBlock"),
            (std::vector<int64_t>{16, 32, 64, 128}));
}

// The wave counts are the ones that fit `maxHardwareWorkgroupSize`, so a wave32
// arch reaches one more than a wave64 one does rather than being held to the
// same count.
TEST(TuningParamAxesTest, NumWavesFollowsTheWaveSize) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx1100");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  EXPECT_EQ(axisOf(*axes, "numWaves"),
            (std::vector<int64_t>{1, 2, 4, 8, 16, 32}));
}

// `kpack` packs two K elements into one MFMA operand, and where that is
// unavailable the parameter is not a choice but a way to spend a benchmark on a
// kernel already timed. `getMaxKpack` is the same bound `validateKpack` holds a
// perf config to, so an axis past it would offer configs that fail to compile.
TEST(TuningParamAxesTest, KpackFollowsWhatTheArchSupports) {
  auto kpackAxisFor = [](StringRef arch,
                         llvm::function_ref<Type(OpBuilder &)> elemType) {
    GemmModule e(elemType, /*m=*/1024, /*n=*/1024, /*k=*/1024, arch);
    std::unique_ptr<TuningParamAxes> axes =
        createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
    EXPECT_TRUE(axes);
    return axes ? axisOf(*axes, "kpack") : std::vector<int64_t>{};
  };
  auto f16 = [](OpBuilder &b) { return b.getF16Type(); };
  auto f32 = [](OpBuilder &b) { return b.getF32Type(); };

  EXPECT_EQ(kpackAxisFor("gfx942", f16), (std::vector<int64_t>{1, 2}));
  // Deprecated from gfx950 on, which Triton's own `HIPOptions` enforces by
  // overwriting any other value with one.
  EXPECT_EQ(kpackAxisFor("gfx950", f16), (std::vector<int64_t>{1}));
  // WMMA takes its operands whole, and the non-accel path has no MFMA operand
  // to pack for.
  EXPECT_EQ(kpackAxisFor("gfx1201", f16), (std::vector<int64_t>{1}));
  EXPECT_EQ(kpackAxisFor("gfx1201", f32), (std::vector<int64_t>{1}));
}

// Splitting K needs a caller that arranged for the partial results to be
// accumulated, and `rock.enable_splitk_for_tuning` is how it says it has. Given
// the opt-in, what differs from the enumerators is only the range: they ask
// `computeOptimalSplitKFactors`, whose work-imbalance model offers 3 and 4 and
// only when it expects a gain, which is a way of keeping brute force short.
// Both of those are in this range, so every config they can hand out is
// reachable here.
TEST(TuningParamAxesTest, SplitKIsExploredOnlyWhenTheCallerAsksForIt) {
  auto splitKAxisFor = [](bool enableSplitKForTuning) {
    GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
                 /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942",
                 enableSplitKForTuning);
    std::unique_ptr<TuningParamAxes> axes =
        createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
    EXPECT_TRUE(axes);
    return axes ? axisOf(*axes, "splitKFactor") : std::vector<int64_t>{};
  };

  EXPECT_EQ(splitKAxisFor(false), (std::vector<int64_t>{1}));
  EXPECT_EQ(splitKAxisFor(true),
            (std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

// The opt-in is the whole of it, and in particular the output element type does
// not come into it. The split's partial results are accumulated atomically, and
// an atomic add the hardware lacks lowers to a CAS loop rather than failing to
// build (AIROCMLIR-967), so the factors above one describe kernels every arch
// can hold. gfx1100 is where that is visible: it has the f32 atomic add but not
// the packed f16 one, and both of its axes are open all the same.
TEST(TuningParamAxesTest, SplitKDoesNotAskForANativeAtomicAdd) {
  auto splitKAxisFor = [](StringRef arch,
                          llvm::function_ref<Type(OpBuilder &)> elemType) {
    GemmModule e(elemType, /*m=*/1024, /*n=*/1024, /*k=*/1024, arch,
                 /*enableSplitKForTuning=*/true);
    std::unique_ptr<TuningParamAxes> axes =
        createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
    EXPECT_TRUE(axes);
    return axes ? axisOf(*axes, "splitKFactor") : std::vector<int64_t>{};
  };
  auto f16 = [](OpBuilder &b) { return b.getF16Type(); };
  auto f32 = [](OpBuilder &b) { return b.getF32Type(); };
  const std::vector<int64_t> split{1, 2, 3, 4, 5, 6, 7, 8, 9};

  EXPECT_EQ(splitKAxisFor("gfx1100", f16), split);
  EXPECT_EQ(splitKAxisFor("gfx1100", f32), split);
  // RDNA4 has the f16 one natively, and CDNA has had it since gfx908.
  EXPECT_EQ(splitKAxisFor("gfx1200", f16), split);
  EXPECT_EQ(splitKAxisFor("gfx942", f16), split);
}

// Re-opening `wavesPerEU` brings back a constraint the enumerators never had to
// state: asking for N waves per EU caps each thread at `vgprsPerEU / N`
// registers, and a C tile whose accumulator overruns that cap spills, which
// costs minutes of register allocation to compile a config that was never going
// to be fast. Pinning the parameter to zero made that vacuous. So the widened
// values have to be held to the budget, and held to it exactly: too strict and
// the widening buys nothing, too lax and a tuning run stalls in the compiler.
TEST(TuningParamAxesTest, WidenedWavesPerEUIsHeldToTheRegisterBudget) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  const TuningParamSetKind kind = TuningParamSetKind::Exhaustive;
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, kind);
  ASSERT_TRUE(axes);
  std::set<std::vector<int64_t>> enumerated =
      enumerateSpace(e.ctx, *e.module, kind);
  ASSERT_FALSE(enumerated.empty());

  const size_t mIdx = paramIndex(*axes, "mPerBlock");
  const size_t nIdx = paramIndex(*axes, "nPerBlock");
  const size_t wavesIdx = paramIndex(*axes, "numWaves");
  const size_t wavesPerEUIdx = paramIndex(*axes, "wavesPerEU");
  // gfx942: `getMaxWavesPerEU` = 8, `getVGPRsPerEU` = 512, `getWaveSize` = 64.
  const int64_t wavesPerEU = 8, vgprsPerEU = 512, waveSize = 64;

  unsigned inBudget = 0, overBudget = 0;
  for (const std::vector<int64_t> &config : enumerated) {
    std::vector<int64_t> raised = config;
    raised[wavesPerEUIdx] = wavesPerEU;
    // One register per accumulator element, spread over the block's threads.
    bool fits = config[mIdx] * config[nIdx] * wavesPerEU <=
                vgprsPerEU * config[wavesIdx] * waveSize;
    fits ? ++inBudget : ++overBudget;
    if (axes->isFeasible(raised) == fits)
      continue;
    PerfConfigString spelled;
    axes->serialize(raised, spelled);
    ADD_FAILURE() << (fits ? "rejected an in-budget config: "
                           : "admitted an over-budget config: ")
                  << std::string(spelled);
  }
  EXPECT_GT(inBudget, 0u) << "no tile here can use the widest wavesPerEU, so "
                             "this shape cannot tell the budget from a ban";
  EXPECT_GT(overBudget, 0u) << "nothing here is over budget, so the guard "
                               "would pass this test unwritten";
}

// `useAsyncCopy` and `useBlockPingpong` ask for a loop schedule rather than
// gate a rewrite, so where the target cannot carry the schedule out they build
// the kernel that asking for nothing builds, and exploring them costs a
// benchmark to learn what the run already timed. Each is unavailable for its
// own reason, and the two do not line up on any one family:
//
//  - an async copy needs the load to have a direct-to-LDS width, which
//    `supportsDirectToLdsLoadBitWidth` grants only on CDNA3, CDNA4 and gfx1250;
//    everywhere else `createStreamOps` falls back to a plain copy per load;
//  - pingpong on a single dot reads that dot's MFMA layout and returns as soon
//    as `BlockPingpong` does not find one, so it needs the GEMM to have been
//    accelerated to MFMA rather than to WMMA or not at all.
TEST(TuningParamAxesTest, ScheduleKnobsAreExploredOnlyWhereTheyCanAct) {
  const std::vector<int64_t> pinned = {kKnobDefault}, explored = {0, 1};
  struct Case {
    StringRef arch;
    bool f16;
    std::vector<int64_t> asyncCopy, pingpong;
  } cases[] = {
      // CDNA3 with MFMA: both act.
      {"gfx942", true, explored, explored},
      // CDNA2 has MFMA but no direct-to-LDS load.
      {"gfx90a", true, pinned, explored},
      // RDNA4 has neither.
      {"gfx1201", true, pinned, pinned},
      // gfx1250 is the WMMA arch that does async copy, and is why this cannot
      // be
      // one predicate serving both knobs.
      {"gfx1250", true, explored, pinned},
      // f32 on RDNA is the non-accel path, which has no dot to pingpong at all.
      {"gfx1201", false, pinned, pinned},
  };

  for (const Case &c : cases) {
    GemmModule e(
        [&](OpBuilder &b) -> Type {
          return c.f16 ? b.getF16Type() : b.getF32Type();
        },
        /*m=*/1024, /*n=*/1024, /*k=*/1024, c.arch);
    std::unique_ptr<TuningParamAxes> axes =
        createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
    ASSERT_TRUE(axes);
    SCOPED_TRACE(c.arch.str() + (c.f16 ? " f16" : " f32"));
    EXPECT_EQ(axisOf(*axes, "useAsyncCopy"), c.asyncCopy);
    EXPECT_EQ(axisOf(*axes, "useBlockPingpong"), c.pingpong);

    // Pinning has to constrain feasibility as well as exploration. The LLM
    // proposes values without walking the axes, so merely exposing a
    // single-value axis would not stop it from requesting an unsupported
    // schedule.
    std::vector<int64_t> config;
    for (const std::vector<int64_t> &axis : axes->getAxes())
      config.push_back(axis.front());
    auto expectPinnedValueRefused = [&](StringRef name,
                                        const std::vector<int64_t> &expected) {
      if (expected != pinned)
        return;
      size_t index = paramIndex(*axes, name);
      int64_t old = config[index];
      config[index] = 1;
      FeasibilityCheck refusedOn = FeasibilityCheck::MalformedConfig;
      EXPECT_FALSE(axes->isFeasible(config, &refusedOn));
      EXPECT_EQ(refusedOn, FeasibilityCheck::NotOnAxis);
      config[index] = old;
    };
    expectPinnedValueRefused("useAsyncCopy", c.asyncCopy);
    expectPinnedValueRefused("useBlockPingpong", c.pingpong);

    // The pin is those two and not the knobs in general.
    for (StringRef knob :
         {"useInThreadTranspose", "useBufferOps", "useBufferAtomics",
          "useReductionLayout", "useOptimizeEpilogue"})
      EXPECT_EQ(axisOf(*axes, knob), explored) << knob << " cannot be moved";
  }
}

// `useBf16x3ForF32` names a dot precision instead of gating a rewrite, so on
// anything but an f32 dot it reaches nothing:
// `RockBlockwiseGemmOpRewritePattern` asks for `InputPrecision::BF16x3` only
// when both of the dot's operands are f32, and issues the one IEEE dot for
// either value of the knob otherwise. Unlike the schedule knobs this turns on
// the problem rather than the target, so it is the element type that moves it.
TEST(TuningParamAxesTest, Bf16x3ForF32IsExploredOnlyOnAnF32Dot) {
  const std::vector<int64_t> pinned = {kKnobDefault}, explored = {0, 1};
  for (bool f32 : {true, false}) {
    // CDNA4, the family whose f32 dots the decomposition was measured to help,
    // i.e. where `preferBf16x3ForF32Dot` answers yes. The axis follows the
    // operand type rather than the arch, so the pin holds on the f16 GEMM here
    // as well.
    GemmModule e(
        [&](OpBuilder &b) -> Type {
          return f32 ? b.getF32Type() : b.getF16Type();
        },
        /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx950");
    std::unique_ptr<TuningParamAxes> axes =
        createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
    ASSERT_TRUE(axes);
    SCOPED_TRACE(f32 ? "f32" : "f16");
    EXPECT_EQ(axisOf(*axes, "useBf16x3ForF32"), f32 ? explored : pinned);
  }
}

// `addTritonPasses` reads `useBufferAtomics` only inside `if (useBufferOps)`,
// so asking for buffer atomics with the pass cluster explicitly off builds the
// kernel that asking for neither builds. Two spellings of one kernel cost a
// search a benchmark to learn what it already timed, so only one of them is a
// config. `kKnobDefault` leaves the cluster on, which is when atomics have
// something to apply to, so it does not collide.
TEST(TuningParamAxesTest, BufferAtomicsNeedBufferOps) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  const TuningParamSetKind kind = TuningParamSetKind::Exhaustive;
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, kind);
  ASSERT_TRUE(axes);
  std::set<std::vector<int64_t>> enumerated =
      enumerateSpace(e.ctx, *e.module, kind);
  ASSERT_FALSE(enumerated.empty());

  const size_t opsIdx = paramIndex(*axes, "useBufferOps");
  const size_t atomicsIdx = paramIndex(*axes, "useBufferAtomics");
  std::vector<int64_t> config = *enumerated.begin();
  auto isFeasibleWith = [&](int64_t useBufferOps, int64_t useBufferAtomics) {
    config[opsIdx] = useBufferOps;
    config[atomicsIdx] = useBufferAtomics;
    return axes->isFeasible(config);
  };

  EXPECT_FALSE(isFeasibleWith(0, 1));
  EXPECT_TRUE(isFeasibleWith(0, 0));
  EXPECT_TRUE(isFeasibleWith(1, 1));
  EXPECT_TRUE(isFeasibleWith(1, 0));
  EXPECT_TRUE(isFeasibleWith(kKnobDefault, 1));

  // Which check refused it, and not merely that something did. A search that
  // reports this to a language model is telling it which rule to stop walking
  // into, so naming the wrong one is worse than naming none.
  config[opsIdx] = 0;
  config[atomicsIdx] = 1;
  FeasibilityCheck refusedOn = FeasibilityCheck::NotOnAxis;
  EXPECT_FALSE(axes->isFeasible(config, &refusedOn));
  EXPECT_EQ(refusedOn, FeasibilityCheck::BufferKnobsDisagree);
  EXPECT_EQ(getFeasibilityCheckName(refusedOn), "bufferKnobsDisagree");
}

// A config off the axes has to be refused as such, not silently rolled into
// whichever check happens to notice it next: the two mean different things to
// whoever is being told, since one is a value that does not exist and the
// other is a combination that does not work.
TEST(TuningParamAxesTest, ReportsAValueThatIsNotOnItsAxis) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  const size_t mIdx = paramIndex(*axes, "mPerBlock");
  std::vector<int64_t> config(axes->getAxes().size());
  for (auto [value, axis] : llvm::zip_equal(config, axes->getAxes()))
    value = axis.front();
  // 33 is on no tile ladder: they run 1, 2, 4, 8 and then multiples of 16.
  config[mIdx] = 33;

  FeasibilityCheck refusedOn = FeasibilityCheck::NotPerformant;
  EXPECT_FALSE(axes->isFeasible(config, &refusedOn));
  EXPECT_EQ(refusedOn, FeasibilityCheck::NotOnAxis);
}

// The enumerated FMA space drops M/N pairs this wide because they account for
// most of its compile time and have never won a tuned shape. The searched view
// of the same space must not reintroduce them.
TEST(TuningParamAxesTest, RefusesOverwideNonAccelMNPair) {
  GemmModule e([](OpBuilder &b) { return b.getF32Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx1201");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  std::vector<int64_t> config;
  for (const std::vector<int64_t> &axis : axes->getAxes())
    config.push_back(axis.front());
  config[paramIndex(*axes, "mPerBlock")] = 256;
  config[paramIndex(*axes, "nPerBlock")] = 256;

  FeasibilityCheck refusedOn = FeasibilityCheck::NotOnAxis;
  EXPECT_FALSE(axes->isFeasible(config, &refusedOn));
  EXPECT_EQ(refusedOn, FeasibilityCheck::OverwideNonAccelMNPair);
  EXPECT_EQ(getFeasibilityCheckName(refusedOn), "overwideNonAccelMNPair");
}

// The axes only help a search if they leave it something to move: a parameter
// whose axis holds one value is one the search can never touch.
TEST(TuningParamAxesTest, ExhaustiveMovesMoreThanOneParameter) {
  GemmModule e([](OpBuilder &b) { return b.getF16Type(); },
               /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  SmallVector<StringRef> names;
  axes->getParamNames(names);
  unsigned tunable = 0;
  for (auto [name, axis] : llvm::zip_equal(names, axes->getAxes()))
    if (axis.size() > 1)
      ++tunable;
  EXPECT_GT(tunable, 1u) << "the exhaustive space pins all but one parameter";
}

// A search reads an axis as a ladder: `feasibleNeighbors` (LFBOSearch.cpp)
// finds the current value with a binary search and treats the values either
// side of it as the ones a step away. An axis that is unordered would make a
// step land anywhere, and a repeated value would waste a benchmark on the
// config the search came from, so the axes are held to being ascending.
TEST(TuningParamAxesTest, AxesAreLaddersASearchCanWalk) {
  auto expectAscending = [](const TuningParamAxes &axes, StringRef what) {
    SmallVector<StringRef> names;
    axes.getParamNames(names);
    for (auto [name, axis] : llvm::zip_equal(names, axes.getAxes()))
      for (size_t idx = 1; idx < axis.size(); ++idx)
        EXPECT_LT(axis[idx - 1], axis[idx])
            << what << "'s " << name << " is not an ascending ladder";
  };

  GemmModule gemm([](OpBuilder &b) { return b.getF16Type(); },
                  /*m=*/1024, /*n=*/1024, /*k=*/1024, "gfx942");
  std::unique_ptr<TuningParamAxes> gemmAxes =
      createTunableParamAxes(*gemm.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(gemmAxes);
  expectAscending(*gemmAxes, "gemm");

  AttentionModule attention("gfx942");
  ASSERT_TRUE(attention.module);
  std::unique_ptr<TuningParamAxes> attentionAxes =
      createTunableParamAxes(*attention.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(attentionAxes);
  expectAscending(*attentionAxes, "attention");
}

// The second gemm's N tile, which only attention has. Zero leaves the head dim
// whole and a tile splits it into chunks, so every power of two below the head
// dim is a tile worth offering. One as wide as it would describe the untiled
// kernel zero already gives, and a wider one would leave `gemm1NChunks` at
// zero, so the ladder stops below it.
TEST(TuningParamAxesTest, SecondGemmNTileStaysBelowTheHeadDim) {
  AttentionModule e("gfx942");
  ASSERT_TRUE(e.module);
  std::unique_ptr<TuningParamAxes> axes =
      createTunableParamAxes(*e.module, TuningParamSetKind::Exhaustive);
  ASSERT_TRUE(axes);

  // The module's head dim of V is 512.
  EXPECT_EQ(axisOf(*axes, "nPerBlockG1"),
            (std::vector<int64_t>{0, 16, 32, 64, 128, 256}));
}

} // namespace
