//===- LLMSearch.cpp - Tuning search guided by a language model -----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLMSearch.h"
#include "LLMProposer.h"

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <random>
#include <set>

#define DEBUG_TYPE "rock-llm-search"

using namespace mlir;
using namespace mlir::rock;

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

//===----------------------------------------------------------------------===//
// Describing the problem
//===----------------------------------------------------------------------===//

/// What the model is being asked to tune, gathered once. The counterpart of
/// Helion's `describe_kernel` and `_gpu_hardware_lines` (llm/workload.py), and
/// a good deal richer than either, because a Triton config as Helion spells it
/// has a handful of hardware-sensitive knobs and a Rock perf config has
/// seventeen.
///
/// Everything here is a fact about the problem or the chip, never advice about
/// it: what a knob is worth reaching for lives in the prompt, on the Python
/// side, where it can be read and argued with.
struct Workload {
  //===--------------------------------------------------------------------===//
  // The problem
  //===--------------------------------------------------------------------===//

  /// Which kernel this is, spelled by `getNameForKernelType` -- the name the
  /// `KernelType` enum gives itself (RockAttrDefs.td), so that what reaches a
  /// prompt or a trace reads the same as the IR. The single-GEMM ops
  /// (`RockGemmWrapperInterface`) are "Gemm", "Conv" or "ConvBwdData", and the
  /// fused ops (`RockGemmGemmWrapperInterface`) are "Attention",
  /// "GemmElementwiseGemm" or "ConvElementwiseGemm".
  ///
  /// Worth carrying rather than inferring from the shapes, because everything
  /// downstream is in GEMM terms and the same M/K/N mean different things
  /// depending on how they arose: a convolution's dimensions come from an
  /// implicit-GEMM mapping, so its N is a batch times a spatial extent and its
  /// K a filter footprint, while attention's are two sequence lengths and two
  /// head dimensions. llm/workload.py keys its explanation off this.
  StringRef kernelType;
  /// The matrix multiply the kernel performs. `o` is the second GEMM's free
  /// dimension, and so is set for the fused ops and only for them.
  int64_t g = 0, m = 0, k = 0, n = 0;
  std::optional<int64_t> o;
  /// The element types of the matrices, named as the ops name them: `a` times
  /// `b` for the first GEMM, and for a fused op `c` is the second GEMM's other
  /// operand (attention's V) while `out` is the result. A single GEMM's result
  /// *is* its C, so `outType` is set for the fused ops only.
  ///
  /// The output type is worth carrying and not only the inputs: it is what the
  /// accumulator has to be converted to and stored as, so it prices the
  /// epilogue that a tile's registers have to survive.
  std::string aType, bType, cType, outType;
  /// Which operands are stored transposed, i.e. with the reduction dimension
  /// contiguous rather than strided. This decides whether a tile's loads
  /// coalesce, and so is a large part of what makes one `kPerBlock`, `kpack`
  /// or `useAsyncCopy` better than another.
  ///
  /// Unset where the question does not apply: a convolution's operand layout
  /// is spelled by `filterLayout` and friends instead, and `transposedC` is
  /// about a second GEMM that a single-GEMM op does not have.
  std::optional<bool> transposedA, transposedB, transposedC, transposedOut;
  /// Block-scaled (MXFP-style) GEMMs only.
  std::optional<int64_t> quantBlockSize;
  std::string aScaleType, bScaleType;
  std::optional<bool> transposedAScale, transposedBScale;
  /// Whether the kernel this is part of also reduces the GEMM's output, which
  /// is what makes `splitKFactor` and the atomics knobs interact: a reduction
  /// already writes with atomics, so a split-K on top of it is priced
  /// differently. Single-GEMM ops only, the only ones that can carry one.
  std::optional<bool> hasFusedReduction;

  //===--------------------------------------------------------------------===//
  // The convolution, where the GEMM came from one
  //
  // The shapes above are the implicit-GEMM image of these, which is what gets
  // tiled; these say how the data is actually laid out in memory underneath,
  // and so what a tile of that image costs to gather.
  //===--------------------------------------------------------------------===//

  std::vector<std::string> filterLayout, inputLayout, outputLayout;
  std::vector<int64_t> padding, strides, dilations;
  /// The multiple a `kPerBlock` wants to be for the conv's K index to stay
  /// cheap to advance, i.e. `kPerBlockAlignmentFactor`. Worth stating and not
  /// leaving to be read off the layouts: it exists only for a channels-first
  /// input, the multiples of it are neither powers of two nor multiples of 16
  /// so nothing else in the space's K axis looks like them, and the reasoning
  /// behind it is about a merged dimension the model is never shown.
  ///
  /// Set only where it is above one, which is the only case that says
  /// anything.
  std::optional<int64_t> kPerBlockAlignment;

  //===--------------------------------------------------------------------===//
  // The fusion, and attention's masking
  //
  // Both cost registers and bound how large a tile can usefully be, so they
  // belong to the problem as much as the shapes do.
  //===--------------------------------------------------------------------===//

  /// Whether anything runs between the two GEMMs -- attention's pre-softmax
  /// bias or scale, say -- and how many extra operands it reads. Fused ops
  /// only.
  std::optional<bool> hasPreSecondGemmFusion;
  std::optional<int64_t> numElemwiseInputs;
  /// Attention only. Grouped-query attention when `numHeadsQ` exceeds
  /// `numHeadsKV`; `splitKV` above 1 is flash decoding, which multiplies the
  /// grid and so changes what tile count keeps the machine busy.
  std::optional<int64_t> numHeadsQ, numHeadsKV, splitKV;
  std::optional<bool> causal;
  /// Set when the kernel masks against a KV extent known only at run time, and
  /// when that masking is a sliding window of the given look-back.
  std::optional<bool> hasLastValidKVIndex, hasPrefixOffset;
  std::optional<int64_t> slidingWindowLookBack;
  /// The accumulator the softmax runs in, where the op pins one.
  std::string softmaxType;
  /// Whether the log-sum-exp output is asked for, which is another tile of
  /// results the kernel has to keep and write.
  std::optional<bool> hasLse;

  //===--------------------------------------------------------------------===//
  // The chip
  //
  // Naming it explicitly is the point: a model told "gfx950" reasons about a
  // particular machine, and one told "an AMD GPU" pattern-matches.
  //===--------------------------------------------------------------------===//

  std::string arch;
  std::string chip;
  bool isCDNA = false, isRDNA = false;
  /// The grid the kernel is launched over, and so what a tile count has to
  /// divide into for the machine to be busy. These are the live `num_cu` and
  /// `num_chiplets` module attributes that `--target-num-cu` sets, not the
  /// arch's nominal figures.
  int64_t numCUs = 0, numChiplets = 0;
  /// What the matrix instructions are, which is what bounds
  /// `matrixInstrNonkdim` and `kpack`.
  StringRef accelKind;
  int64_t waveSize = 0;
  /// The occupancy budget a tile is spent out of, and so why a large tile
  /// fails rather than merely running slowly.
  int64_t ldsSize = 0, vgprsPerEU = 0, maxWavesPerEU = 0;
  /// Infinity Cache / MALL where the chip has one, otherwise the L2. What
  /// `gridGroupSize` is trading in.
  int64_t lastLevelCacheSize = 0;
  /// Which knobs this chip can actually act on. Partly redundant with the
  /// ladders -- `GemmParamAxes` already pins `useAsyncCopy` to its default
  /// when the arch cannot issue one -- but a ladder says only what is allowed,
  /// and the model is being asked to reason rather than to pattern-match.
  int64_t maxKpack = 0, maxNumCTAs = 0;
  bool supportsAsyncCopy = false, supportsTDM = false;
  bool supportsNonPow2KPerBlock = false, supportsScaledGemm = false;
  bool preferBf16x3ForF32Dot = false;
};

std::string typeName(Type type) {
  if (!type)
    return {};
  std::string text;
  llvm::raw_string_ostream os(text);
  getElementTypeOrSelf(type).print(os);
  return text;
}

/// The dimension names a convolution's layout attribute spells, e.g.
/// ["gi", "ni", "ci", "0i", "1i"]. Empty for anything that is not a
/// convolution, and for a convolution whose layouts have not been attached
/// yet -- they arrive with the lowering, not with the op.
std::vector<std::string> layoutNames(Operation *op, StringRef attrName) {
  std::vector<std::string> names;
  auto layout = op->getAttrOfType<ArrayAttr>(attrName);
  if (!layout)
    return names;
  for (Attribute dim : layout)
    if (auto name = dyn_cast<StringAttr>(dim))
      names.push_back(name.getValue().str());
  return names;
}

std::vector<int64_t> intsOf(ArrayAttr values) {
  std::vector<int64_t> result;
  if (!values)
    return result;
  for (Attribute value : values)
    if (auto number = dyn_cast<IntegerAttr>(value))
      result.push_back(number.getInt());
  return result;
}

/// How the operands of a convolution are laid out, and how its window walks
/// them. The tuning space is over the implicit GEMM, but these are what that
/// GEMM's loads actually turn into.
void gatherConvolution(Workload &workload, RockConvInterface conv) {
  Operation *op = conv;
  workload.filterLayout = layoutNames(op, "filter_layout");
  workload.inputLayout = layoutNames(op, "input_layout");
  workload.outputLayout = layoutNames(op, "output_layout");
  workload.padding = intsOf(conv.getPadding());
  workload.strides = intsOf(conv.getStrides());
  workload.dilations = intsOf(conv.getDilations());
}

/// Attention's masking, its head counts and its optional outputs: everything
/// about the problem that `GemmGemmSize` does not carry.
void gatherAttention(Workload &workload, AttentionOp attention) {
  workload.numHeadsQ = attention.getNumHeadsQ();
  workload.numHeadsKV = attention.getNumHeadsKV();
  workload.causal = attention.getCausal();
  workload.splitKV = attention.getSplitKV();
  workload.hasLastValidKVIndex = attention.getLastValidKVIndex() != nullptr;
  workload.hasPrefixOffset = attention.getPrefixOffset() != nullptr;
  if (std::optional<uint32_t> lookBack = attention.getSlidingWindowLookBack())
    workload.slidingWindowLookBack = *lookBack;
  if (TypeAttr softmaxType = attention.getSoftmaxTypeAttr())
    workload.softmaxType = typeName(softmaxType.getValue());
  workload.hasLse = attention.getLse() != nullptr;
}

/// Everything in `Workload` that depends only on the architecture and on the
/// op's place in the module, which is everything except the shapes, the types
/// and the matrix-instruction kind.
void gatherHardware(Workload &workload, Operation *op, StringRef arch) {
  workload.arch = arch.str();
  workload.chip = std::get<0>(rock::parseArchString(arch)).str();
  workload.isCDNA = rock::isCDNA(arch);
  workload.isRDNA = rock::isRDNA(arch);
  workload.numCUs = rock::getNumCUValue(op);
  workload.numChiplets = rock::getNumChipletsValue(op);
  workload.waveSize = rock::getWaveSize(arch);
  workload.ldsSize = rock::getLDSSize(arch);
  workload.vgprsPerEU = rock::getVGPRsPerEU(arch);
  workload.maxWavesPerEU = rock::getMaxWavesPerEU(arch);
  workload.lastLevelCacheSize = rock::getLastLevelCacheSize(arch);
  workload.maxKpack = rock::getMaxKpack(arch);
  workload.maxNumCTAs = rock::getMaxNumCTAs(arch);
  workload.supportsAsyncCopy = rock::supportsAsyncCopy(arch);
  workload.supportsTDM = rock::supportsTDM(arch);
  workload.supportsNonPow2KPerBlock = rock::supportsNonPow2KPerBlock(arch);
  workload.supportsScaledGemm = rock::archSupportsScaledGemm(arch);
  workload.preferBf16x3ForF32Dot = rock::preferBf16x3ForF32Dot(arch);
}

/// Finds the op whose tuning space is being searched and describes it. Walks
/// the module the same way, and in the same order, as
/// `createTunableParamAxes`, so that the description and the space are of the
/// same op. Returns nullopt when the module holds no such op, which is the one
/// case in which there is nothing to tune.
std::optional<Workload> describeWorkload(ModuleOp mod) {
  Workload workload;
  bool found = false;

  mod->walk([&](rock::RockGemmWrapperInterface op) {
    PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);
    workload.kernelType = getNameForKernelType(info.kernelType);
    workload.g = info.gemmSize.g;
    workload.m = info.gemmSize.m;
    workload.k = info.gemmSize.k;
    workload.n = info.gemmSize.n;
    workload.aType = typeName(info.gemmAType);
    workload.bType = typeName(info.gemmBType);
    workload.cType = typeName(op.getCType());
    workload.hasFusedReduction = info.hasFusedReduction;
    workload.quantBlockSize = info.quantBlockSize;
    workload.aScaleType = typeName(info.aScaleType);
    workload.bScaleType = typeName(info.bScaleType);
    // Only `rock.gemm` states its operand layouts as transposes. A
    // convolution's are a dimension order, gathered below.
    if (auto gemm = dyn_cast<rock::GemmOp>(op.getOperation())) {
      workload.transposedA = gemm.getATransposed();
      workload.transposedB = gemm.getBTransposed();
      workload.transposedOut = gemm.getOTransposed();
      if (info.quantBlockSize) {
        workload.transposedAScale = gemm.getAScaleTransposed();
        workload.transposedBScale = gemm.getBScaleTransposed();
      }
    }
    if (auto conv = dyn_cast<rock::RockConvInterface>(op.getOperation())) {
      gatherConvolution(workload, conv);
      // Asked of the GEMM wrapper rather than the conv, since it is a fact
      // about how this conv's gemmK is built.
      if (int64_t alignment = rock::kPerBlockAlignmentFactor(op); alignment > 1)
        workload.kPerBlockAlignment = alignment;
    }
    gatherHardware(workload, op, info.arch);
    workload.accelKind =
        getNameForMatrixAccelKind(rock::getMatrixAccelKind(info.arch, op));
    found = true;
    return WalkResult::interrupt();
  });
  if (found)
    return workload;

  mod->walk([&](rock::RockGemmGemmWrapperInterface op) {
    GemmGemmSize size = op.getGemmGemmSize();
    StringRef arch = rock::getArchValue(op);
    workload.kernelType = getNameForKernelType(op.getKernelType());
    workload.g = size.g;
    workload.m = size.m;
    workload.k = size.k;
    workload.n = size.n;
    workload.o = size.o;
    workload.aType = typeName(op.getAType());
    workload.bType = typeName(op.getBType());
    workload.cType = typeName(op.getCType());
    workload.outType = typeName(op.getOutType());
    workload.transposedA = op.getTransposedA();
    workload.transposedB = op.getTransposedB();
    workload.transposedC = op.getTransposedC();
    workload.transposedOut = op.getTransposedOut();
    workload.hasPreSecondGemmFusion = rock::gemmGemmHasPreSecondGemmFusion(op);
    workload.numElemwiseInputs =
        op.getPreSecondGemmElemwiseInputsMutable().size();
    if (auto attention = dyn_cast<rock::AttentionOp>(op.getOperation()))
      gatherAttention(workload, attention);
    if (auto conv = dyn_cast<rock::RockConvInterface>(op.getOperation()))
      gatherConvolution(workload, conv);
    gatherHardware(workload, op, arch);
    // Both GEMMs run on the same instructions; the second is what the perf
    // config's `G1` parameters are about.
    workload.accelKind =
        getNameForMatrixAccelKind(rock::getMatrixAccelKind(arch, op).first);
    found = true;
    return WalkResult::interrupt();
  });
  return found ? std::optional<Workload>(std::move(workload)) : std::nullopt;
}

/// Emits `value` under `key` only when it is set, so that the Python side can
/// tell "does not apply to this kernel" from any particular value: a
/// convolution has no `transposedA`, and saying `false` would be a claim about
/// its layout rather than an absence of one.
template <typename T>
void addIfSet(llvm::json::Object &object, StringRef key,
              const std::optional<T> &value) {
  if (value)
    object[key] = *value;
}

llvm::json::Array arrayOf(ArrayRef<int64_t> values) {
  llvm::json::Array result;
  for (int64_t value : values)
    result.push_back(value);
  return result;
}

llvm::json::Array arrayOf(ArrayRef<std::string> values) {
  llvm::json::Array result;
  for (const std::string &value : values)
    result.push_back(value);
  return result;
}

llvm::json::Object problemOf(const Workload &workload) {
  llvm::json::Object gemmSize{
      {"g", workload.g},
      {"m", workload.m},
      {"k", workload.k},
      {"n", workload.n},
  };
  if (workload.o)
    gemmSize["o"] = *workload.o;

  llvm::json::Object problem{
      {"kernelType", workload.kernelType}, {"gemmSize", std::move(gemmSize)},
      {"aType", workload.aType},           {"bType", workload.bType},
      {"cType", workload.cType},
  };
  if (!workload.outType.empty())
    problem["outType"] = workload.outType;
  addIfSet(problem, "transposedA", workload.transposedA);
  addIfSet(problem, "transposedB", workload.transposedB);
  addIfSet(problem, "transposedC", workload.transposedC);
  addIfSet(problem, "transposedOut", workload.transposedOut);
  addIfSet(problem, "hasFusedReduction", workload.hasFusedReduction);
  if (workload.quantBlockSize) {
    problem["quantBlockSize"] = *workload.quantBlockSize;
    problem["aScaleType"] = workload.aScaleType;
    problem["bScaleType"] = workload.bScaleType;
    addIfSet(problem, "transposedAScale", workload.transposedAScale);
    addIfSet(problem, "transposedBScale", workload.transposedBScale);
  }
  if (!workload.filterLayout.empty())
    problem["filterLayout"] = arrayOf(workload.filterLayout);
  if (!workload.inputLayout.empty())
    problem["inputLayout"] = arrayOf(workload.inputLayout);
  if (!workload.outputLayout.empty())
    problem["outputLayout"] = arrayOf(workload.outputLayout);
  if (!workload.padding.empty())
    problem["padding"] = arrayOf(workload.padding);
  if (!workload.strides.empty())
    problem["strides"] = arrayOf(workload.strides);
  if (!workload.dilations.empty())
    problem["dilations"] = arrayOf(workload.dilations);
  addIfSet(problem, "kPerBlockAlignment", workload.kPerBlockAlignment);
  addIfSet(problem, "hasPreSecondGemmFusion", workload.hasPreSecondGemmFusion);
  addIfSet(problem, "numElemwiseInputs", workload.numElemwiseInputs);
  addIfSet(problem, "numHeadsQ", workload.numHeadsQ);
  addIfSet(problem, "numHeadsKV", workload.numHeadsKV);
  addIfSet(problem, "causal", workload.causal);
  addIfSet(problem, "splitKV", workload.splitKV);
  addIfSet(problem, "hasLastValidKVIndex", workload.hasLastValidKVIndex);
  addIfSet(problem, "hasPrefixOffset", workload.hasPrefixOffset);
  addIfSet(problem, "slidingWindowLookBack", workload.slidingWindowLookBack);
  if (!workload.softmaxType.empty())
    problem["softmaxType"] = workload.softmaxType;
  addIfSet(problem, "hasLse", workload.hasLse);
  return problem;
}

llvm::json::Object hardwareOf(const Workload &workload) {
  return llvm::json::Object{
      {"arch", workload.arch},
      {"chip", workload.chip},
      {"isCDNA", workload.isCDNA},
      {"isRDNA", workload.isRDNA},
      {"numCUs", workload.numCUs},
      {"numChiplets", workload.numChiplets},
      {"accelKind", workload.accelKind},
      {"waveSize", workload.waveSize},
      {"ldsSize", workload.ldsSize},
      {"vgprsPerEU", workload.vgprsPerEU},
      {"maxWavesPerEU", workload.maxWavesPerEU},
      {"lastLevelCacheSize", workload.lastLevelCacheSize},
      {"maxKpack", workload.maxKpack},
      {"maxNumCTAs", workload.maxNumCTAs},
      {"supportsAsyncCopy", workload.supportsAsyncCopy},
      {"supportsTDM", workload.supportsTDM},
      {"supportsNonPow2KPerBlock", workload.supportsNonPow2KPerBlock},
      {"supportsScaledGemm", workload.supportsScaledGemm},
      {"preferBf16x3ForF32Dot", workload.preferBf16x3ForF32Dot},
  };
}

//===----------------------------------------------------------------------===//
// LLMSearch
//===----------------------------------------------------------------------===//

class LLMSearch : public TuningSearchStrategy {
public:
  LLMSearch(ModuleOp mod, const LLMOptions &options)
      : mod(mod), options(options), rng(options.seed), trace(options.trace),
        proposer(options.search.proposerPath, options.sessionPath,
                 options.search.requestTimeoutSec) {}

  std::vector<PerfConfigString>
  getPerfConfigBatch(ArrayRef<BenchmarkResult> prevResults) override;

  bool isIterative() const override { return true; }

private:
  bool buildSearchSpace();

  /// The batch the search opens with: the quick tuning list, padded with draws
  /// from the axes. Helion pads its default config with random draws for the
  /// same reason -- a prompt written against one heuristic's output alone has
  /// seen no variety at all.
  std::vector<PerfConfigString> buildSeedBatch();

  /// Folds a benchmarked batch into what the next prompt will say.
  void recordResults(ArrayRef<BenchmarkResult> prevResults);

  /// Turns the helper's proposed perf configs into a batch to benchmark:
  /// drops the ones the space will not accept and the ones already tried, and
  /// notes why for the next prompt.
  std::vector<PerfConfigString> acceptProposals(ArrayRef<std::string> configs);

  /// The state of the search as the helper is given it; see the request schema
  /// in LLMProposer.h.
  llvm::json::Object buildRequest(unsigned round) const;

  /// Starts a request for the next round. `policy` is what makes round 0's
  /// overlap possible: asynchronously there, and deferred -- so that `get()`
  /// runs it here and now -- in the synchronous rounds that follow.
  void launchProposal(std::launch policy);

  /// The configs of the round in flight, launching it first if nothing is.
  ///
  /// A failure ends the process. Helion is explicit about why: "LLM failures
  /// are intentionally fatal: silently falling back to plain LFBO when the
  /// user opted into the LLM autotuner masks real config or connectivity bugs
  /// (e.g. wrong API key, missing mTLS cert)". A tuning run that quietly
  /// stopped consulting the model would look like a model giving bad advice.
  std::vector<std::string> awaitProposals();

  /// Whether the rounds have stopped paying for themselves. Helion's
  /// `_update_early_stop_state`.
  bool improvementHasStalled();

  void traceHeader();
  void traceRound(double elapsedMs, ArrayRef<PerfConfigString> batch);

  /// Whether `values` is a config to try -- one the space accepts and nobody
  /// has proposed yet -- and, if so, how it is spelled. Records the rejection
  /// when it is not.
  bool accept(const ConfigValues &values, PerfConfigString &out);

  size_t pickIndex(size_t count) {
    assert(count > 0 && "picking from an empty range");
    return std::uniform_int_distribution<size_t>(0, count - 1)(rng);
  }

  ModuleOp mod;
  LLMOptions options;
  std::mt19937_64 rng;

  std::unique_ptr<TuningParamAxes> axes;
  Workload workload;
  /// The values of each parameter the model may choose from: the axes widened
  /// by `kKnobDefault` wherever the parameter is a knob, since that is legal,
  /// is what every quick-list config spells, and is the answer worth having
  /// where the model has no opinion. See `TuningParamAxes::getKnobParams`.
  std::vector<std::vector<int64_t>> ladders;
  SmallVector<StringRef> paramNames;
  SmallVector<bool> knobParams;
  /// What a parameter the model does not mention is taken to mean.
  ConfigValues defaultValues;
  std::vector<ConfigValues> quickSeeds;

  //===--------------------------------------------------------------------===//
  // What the next prompt will say
  //===--------------------------------------------------------------------===//

  /// Every config proposed so far, kept as values so that two spellings of one
  /// config are one config, and no round spends itself re-proposing what an
  /// earlier one already tried. Helion's `seen_config_keys`.
  std::set<ConfigValues> seen;
  /// Every measurement, in the order it arrived, and the values behind each so
  /// that a config never has to be parsed twice.
  std::vector<BenchmarkResult> results;
  llvm::StringMap<ConfigValues> valuesByConfig;
  /// Proposals the space turned down, with the check that did it. Helion has
  /// no counterpart: it normalizes a config into range rather than rejecting
  /// it, so it never has to explain one away.
  std::vector<std::pair<ConfigValues, FeasibilityCheck>> rejected;
  /// How the round just past was received, for the trace. A model that keeps
  /// proposing configs the space refuses is being told something wrong, and
  /// that is worth being able to see.
  unsigned rejectedThisRound = 0;
  unsigned duplicatesThisRound = 0;

  //===--------------------------------------------------------------------===//
  // Round state machine
  //===--------------------------------------------------------------------===//

  bool started = false;
  bool done = false;
  /// Requests started so far, which is also the round number the next one
  /// carries. Bounded by `LLMSearchOptions::maxRounds`.
  unsigned roundsRequested = 0;
  /// Proposal batches handed to the client, as distinct from the seed batch.
  /// What decides whether there is a round to have improved on.
  unsigned proposalsHandedOut = 0;
  /// The request in flight. Round 0's is launched before the seed batch is
  /// benchmarked so that the model's latency is spent alongside a batch of
  /// compiles rather than after it; Helion's `_call_llm_async` future.
  std::optional<std::future<llvm::Expected<std::vector<std::string>>>> pending;

  double bestSoFar = kInfinity;
  PerfConfigString bestConfig;
  /// The best time as of the end of the previous round, and how many rounds
  /// have failed to improve on their predecessor since.
  double prevBest = kInfinity;
  unsigned stagnantRounds = 0;

  SharedTrace trace;
  LLMProposer proposer;
  std::chrono::steady_clock::time_point handedOut;
  double totalMs = 0.0;
  unsigned numSucceeded = 0;
  unsigned numNotApplicable = 0;
  unsigned numFailed = 0;
};

bool LLMSearch::buildSearchSpace() {
  axes = createTunableParamAxes(mod, options.candidateSpace);
  if (!axes)
    return false;
  std::optional<Workload> described = describeWorkload(mod);
  if (!described)
    return false;
  workload = std::move(*described);

  axes->getParamNames(paramNames);
  axes->getKnobParams(knobParams);
  for (auto [axis, isKnob] : llvm::zip_equal(axes->getAxes(), knobParams)) {
    std::vector<int64_t> ladder = axis;
    if (isKnob && !llvm::is_contained(ladder, kKnobDefault))
      ladder.insert(ladder.begin(), kKnobDefault);
    ladders.push_back(std::move(ladder));
  }

  // The quick list is the best guess available before anything is measured. It
  // reaches the model twice over: as the batch this search opens with, and, in
  // round 0's prompt, as the unmeasured priors that fill the slot Helion's
  // `build_compiler_analysis_section` fills with its own fired heuristics.
  MLIRContext *ctx = mod.getContext();
  std::unique_ptr<TuningParamSet> quickSpace(
      createTunableParamSpace(mod, TuningParamSetKind::Quick));
  for (const PerfConfigString &perfConfig : quickSpace->tuningRange) {
    RockTuningParamAttrInterface params = parsePerfConfig(ctx, perfConfig);
    assert(params && "a tuning space spelled a config its attribute rejects");
    quickSeeds.emplace_back();
    params.getParamValues(quickSeeds.back());
    assert(quickSeeds.back().size() == ladders.size() &&
           "the quick and searched spaces disagree on what a config is");
  }

  // The exemplar every prompt is written around, and which the helper
  // completes a sparse proposal against: the quick list's own first choice, a
  // config the heuristic believes in, which passes the feasibility checks, and
  // which stays put for the whole run. A base that moved between rounds would
  // make the same proposal mean different things in different ones.
  //
  // Not the axes' exemplar, which is every parameter at the smallest value it
  // can take and so no basis for anything.
  if (!quickSeeds.empty()) {
    defaultValues = quickSeeds.front();
  } else {
    // A space with no quick list at all. A poor config, but a legal one, and
    // the request spells it out, so the model is not left guessing.
    for (const std::vector<int64_t> &ladder : ladders)
      defaultValues.push_back(ladder.front());
  }

  LLVM_DEBUG(llvm::dbgs() << "LLM: proposing over " << ladders.size()
                          << " parameters from " << quickSeeds.size()
                          << " quick seeds, on " << workload.chip << " with "
                          << workload.numCUs << " CUs\n");
  return true;
}

bool LLMSearch::accept(const ConfigValues &values, PerfConfigString &out) {
  FeasibilityCheck refusedOn;
  if (!axes->isFeasible(values, &refusedOn)) {
    ++rejectedThisRound;
    rejected.emplace_back(values, refusedOn);
    return false;
  }
  if (!seen.insert(values).second) {
    ++duplicatesThisRound;
    return false;
  }
  axes->serialize(values, out);
  return true;
}

std::vector<PerfConfigString> LLMSearch::buildSeedBatch() {
  std::vector<PerfConfigString> batch;
  PerfConfigString perfConfig;
  for (const ConfigValues &seed : quickSeeds)
    if (accept(seed, perfConfig))
      batch.push_back(perfConfig);

  // A draw from the axes need not be a config the space contains, and how
  // often it is depends on the space, so cap the attempts rather than the
  // failures: padding the first batch is worth some work but must not become
  // the search.
  const unsigned wanted = options.search.initialRandomConfigs;
  const unsigned maxAttempts = 100 * wanted;
  ConfigValues candidate(ladders.size());
  for (unsigned attempt = 0, kept = 0; attempt < maxAttempts && kept < wanted;
       ++attempt) {
    for (auto [value, ladder] : llvm::zip_equal(candidate, ladders))
      value = ladder[pickIndex(ladder.size())];
    if (accept(candidate, perfConfig)) {
      batch.push_back(perfConfig);
      ++kept;
    }
  }

  // Nothing the seeds ran into is the model's business. Its own rejections are
  // worth reporting because it chose them and can choose otherwise; a pad that
  // drew a few hundred infeasible configs at random has only measured how
  // sparse the space is, and reporting that as advice would bury the prompt.
  rejected.clear();
  rejectedThisRound = 0;
  return batch;
}

void LLMSearch::recordResults(ArrayRef<BenchmarkResult> prevResults) {
  MLIRContext *ctx = mod.getContext();
  for (const BenchmarkResult &result : prevResults) {
    switch (result.status) {
    case BenchmarkResult::Status::Success:
      numSucceeded += result.measured();
      numFailed += !result.measured();
      break;
    case BenchmarkResult::Status::NotApplicable:
      ++numNotApplicable;
      break;
    case BenchmarkResult::Status::Failed:
      ++numFailed;
      break;
    }

    RockTuningParamAttrInterface params =
        parsePerfConfig(ctx, result.perfConfig);
    if (!params)
      continue;
    auto [entry, isNew] = valuesByConfig.try_emplace(result.perfConfig);
    if (!isNew)
      continue;
    params.getParamValues(entry->second);
    results.push_back(result);

    if (result.measured() && result.timeNs < bestSoFar) {
      bestSoFar = result.timeNs;
      bestConfig = result.perfConfig;
    }
  }
}

std::vector<PerfConfigString>
LLMSearch::acceptProposals(ArrayRef<std::string> configs) {
  MLIRContext *ctx = mod.getContext();
  std::vector<PerfConfigString> batch;
  PerfConfigString perfConfig;
  ConfigValues values;
  for (const std::string &proposed : configs) {
    // A whole config in the named form, completed by the helper against the
    // exemplar the request carried; see `LLMProposer::propose`. Read back
    // through the attribute that owns the format, so that the search never
    // has an opinion about how a config is spelled.
    //
    // A string that does not parse, or that parses as the other kernel's perf
    // config, is a broken helper rather than an unlucky model: the helper
    // builds it from the space and the exemplar this request handed it. Not
    // worth ending a tuning run over, but not worth telling the model about
    // either -- it is not advice the model can act on -- so it is counted and
    // said out loud rather than added to `rejected`.
    RockTuningParamAttrInterface params = parsePerfConfig(ctx, proposed);
    values.clear();
    if (params)
      params.getParamValues(values);
    if (values.size() != ladders.size()) {
      llvm::errs() << "warning: the LLM tuning helper proposed something that "
                      "is not a perf config for this kernel: "
                   << proposed << "\n";
      ++rejectedThisRound;
      continue;
    }
    if (accept(values, perfConfig))
      batch.push_back(perfConfig);
  }
  return batch;
}

llvm::json::Object LLMSearch::buildRequest(unsigned round) const {
  PerfConfigString exemplar;
  axes->serialize(defaultValues, exemplar);

  llvm::json::Object space, defaultConfig;
  for (auto [name, ladder, value] :
       llvm::zip_equal(paramNames, ladders, defaultValues)) {
    space[name] = llvm::json::Array(ladder);
    defaultConfig[name] = value;
  }

  // How a config reaches the helper: every parameter, by name. Nothing here is
  // abbreviated, because deciding that a parameter is not worth mentioning is
  // a question about the prompt, and the prompt is the helper's business.
  auto configOf = [&](ArrayRef<int64_t> values) {
    llvm::json::Object config;
    for (auto [name, value] : llvm::zip_equal(paramNames, values))
      config[name] = value;
    return config;
  };

  llvm::json::Array measured;
  for (const BenchmarkResult &result : results) {
    auto values = valuesByConfig.find(result.perfConfig);
    if (values == valuesByConfig.end())
      continue;
    measured.push_back(llvm::json::Object{
        {"config", configOf(values->second)},
        {"status", getNameForBenchmarkStatus(result.effectiveStatus())},
        {"timeNs", finiteOrNull(result.timeNs)},
    });
  }

  llvm::json::Array refused;
  for (const auto &[values, refusedOn] : rejected)
    refused.push_back(llvm::json::Object{
        {"config", configOf(values)},
        {"reason", getFeasibilityCheckName(refusedOn)},
    });

  llvm::json::Array seeds;
  for (const ConfigValues &seed : quickSeeds)
    seeds.push_back(configOf(seed));

  return llvm::json::Object{
      {"round", round},
      {"maxRounds", options.search.maxRounds},
      {"configsRequested", options.search.configsPerRound},
      {"model", options.search.model},
      {"problem", problemOf(workload)},
      {"hardware", hardwareOf(workload)},
      {"space", std::move(space)},
      {"defaultConfig", std::move(defaultConfig)},
      // The same config again, serialized. This is what the helper completes
      // a sparse proposal against and returns a whole one in, so it never has
      // to know the prefix or the field order -- it edits a string this side
      // wrote. See `LLMProposer::propose`.
      {"defaultPerfConfig", std::string(exemplar)},
      {"seedConfigs", std::move(seeds)},
      {"results", std::move(measured)},
      {"rejected", std::move(refused)},
  };
}

void LLMSearch::launchProposal(std::launch policy) {
  // Built here rather than in the task, so that the state it describes is the
  // state as of the launch and not whatever it has become by the time a
  // worker thread gets to it.
  llvm::json::Object request = buildRequest(roundsRequested++);
  pending = std::async(policy, [this, request = std::move(request)]() mutable {
    return proposer.propose(std::move(request));
  });
}

std::vector<std::string> LLMSearch::awaitProposals() {
  if (!pending)
    launchProposal(std::launch::deferred);
  llvm::Expected<std::vector<std::string>> proposals = pending->get();
  pending.reset();

  if (!proposals) {
    // Fatal by design; see the declaration.
    std::string message = ("the LLM tuning search could not get configs: " +
                           llvm::toString(proposals.takeError()));
    llvm::report_fatal_error(message.c_str(), /*GenCrashDiag=*/false);
  }
  return std::move(*proposals);
}

bool LLMSearch::improvementHasStalled() {
  if (std::isfinite(bestSoFar) && std::isfinite(prevBest) && prevBest > 0.0) {
    double improvement = (prevBest - bestSoFar) / prevBest;
    if (improvement < options.minImprovementDelta)
      ++stagnantRounds;
    else
      stagnantRounds = 0;
  }
  prevBest = bestSoFar;
  return stagnantRounds >= options.maxStagnantRounds;
}

void LLMSearch::traceHeader() {
  if (!traceEnabled(trace))
    return;

  llvm::json::Array params;
  for (auto [name, ladder] : llvm::zip_equal(paramNames, ladders))
    params.push_back(
        llvm::json::Object{{"name", name}, {"values", ladder.size()}});

  traceWrite(trace,
             llvm::json::Object{
                 {"kind", "llm-header"},
                 {"arch", workload.arch},
                 {"chip", workload.chip},
                 {"numCUs", workload.numCUs},
                 {"kernelType", workload.kernelType},
                 {"model", options.search.model},
                 {"maxRounds", options.search.maxRounds},
                 {"configsPerRound", options.search.configsPerRound},
                 {"initialRandomConfigs", options.search.initialRandomConfigs},
                 {"waitForSeeds", options.search.waitForSeeds},
                 {"minImprovementDelta", options.minImprovementDelta},
                 {"maxStagnantRounds", options.maxStagnantRounds},
                 {"quickSeeds", quickSeeds.size()},
                 {"params", std::move(params)},
             });
}

void LLMSearch::traceRound(double elapsedMs, ArrayRef<PerfConfigString> batch) {
  if (!traceEnabled(trace))
    return;
  totalMs += elapsedMs;
  traceWrite(trace, llvm::json::Object{
                        {"kind", "llm-round"},
                        {"round", roundsRequested},
                        {"proposals", proposalsHandedOut},
                        {"proposed", batch.size()},
                        {"done", done},
                        {"elapsedMs", elapsedMs},
                        {"totalMs", totalMs},
                        {"measured", results.size()},
                        {"succeeded", numSucceeded},
                        {"notApplicable", numNotApplicable},
                        {"failed", numFailed},
                        {"rejected", rejectedThisRound},
                        {"duplicates", duplicatesThisRound},
                        {"bestNs", finiteOrNull(bestSoFar)},
                        {"bestConfig", bestConfig.str()},
                        {"stagnantRounds", stagnantRounds},
                    });
}

std::vector<PerfConfigString>
LLMSearch::getPerfConfigBatch(ArrayRef<BenchmarkResult> prevResults) {
  auto now = std::chrono::steady_clock::now();
  double elapsedMs =
      started
          ? std::chrono::duration<double, std::milli>(now - handedOut).count()
          : 0.0;
  numSucceeded = numNotApplicable = numFailed = 0;
  rejectedThisRound = duplicatesThisRound = 0;

  std::vector<PerfConfigString> batch;
  if (!started) {
    started = true;
    if (!buildSearchSpace()) {
      done = true;
    } else {
      traceHeader();
      batch = buildSeedBatch();
      done = batch.empty();
      // Helion launches its first request before benchmarking the seeds,
      // "because round 0 does not need any prior search feedback to build its
      // prompt", which spends the model's latency alongside a batch of
      // compiles instead of after it. Waiting instead buys a first proposal
      // that has seen real timings.
      if (!done && !options.search.waitForSeeds)
        launchProposal(std::launch::async);
    }
  } else if (!done) {
    recordResults(prevResults);
    // A round that twice failed to beat its predecessor is one more round of
    // the same, so stop paying for it. Not asked until a proposal has been
    // measured: the seed batch has nothing to have improved on, and the first
    // proposal's results are what Helion's round 0 ends with.
    bool stalled = proposalsHandedOut > 0 && improvementHasStalled();
    bool haveBudget = pending || roundsRequested < options.search.maxRounds;
    if (!stalled && haveBudget) {
      batch = acceptProposals(awaitProposals());
      ++proposalsHandedOut;
    }
    // An empty batch means every config the model named was one the space
    // refused or one already tried, and no reason to expect the next round to
    // differ. Helion stops here too.
    done = batch.empty();
  }

  traceRound(elapsedMs, batch);
  handedOut = std::chrono::steady_clock::now();
  return batch;
}

} // namespace

void mlir::rock::LLMOptions::setEffort(SearchEffort effort) {
  switch (effort) {
  case SearchEffort::Full:
    // What the defaults already are, and what `LLM_SEARCH_DEFAULTS` says.
    return;
  case SearchEffort::Quick:
    // Helion's `QUICK_LLM_SEARCH_DEFAULTS`: one round, so the model gets a
    // single look at the problem and the run costs a single request.
    search.maxRounds = 1;
    return;
  }
  llvm_unreachable("unhandled search effort");
}

std::unique_ptr<TuningSearchStrategy>
mlir::rock::createLLMSearchStrategy(ModuleOp mod, const LLMOptions &options) {
  return std::make_unique<LLMSearch>(mod, options);
}
