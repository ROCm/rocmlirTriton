//===- TuningSearch.cpp - Perf config search strategies -------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the search strategy factory and the brute-force
// strategies, which hand out a fixed tuning space as a single batch, plus the
// view of a tuning space an adaptive search needs: its axes rather than its
// configs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/TuningSearch.h"

#include "LFBOSearch.h"
#include "LLMSeededLFBOSearch.h"
#include "SearchTrace.h"
#include "TuningRanges.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/LdsBlacklist.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::rock;

//===----------------------------------------------------------------------===//
// Tuning space as axes
//
// A tuning space described by the values of each parameter plus a membership
// test instead of by its configs, so that a search need not hold the product in
// memory. Tile shapes and the conditions relating them come from the same
// helpers `createGemm*TuningRangeBF` (RockTuningImpl.cpp) uses, shared through
// TuningRanges.h.
//
// This is not that function's space, and is not meant to be. Writing out a
// product costs a multiple of the whole enumeration per value added, so brute
// force can only afford to sweep a few parameters and pins or coarsens the
// rest. Those are budget decisions, and a search that samples pays per config
// benchmarked rather than per config in the space, so it inherits none of
// them: what it explores is what a kernel may hold, which is what
// `validatePerfConfig` (AffixTuningParameters.cpp) admits, since a config
// outside those bounds fails to compile.
//
// Being free of the enumeration also means being free of its accidents, so a
// parameter is deliberately left pinned where moving it cannot pay:
//
//  - a value the hardware or the pipeline ignores, which would spend a
//    benchmark re-measuring a kernel already timed, e.g. `matrixInstrNonkdim`
//    outside MFMA (`BlockedToWMMA` takes it and drops it) or `useBufferAtomics`
//    while `useBufferOps` is off (only read under it, see `addTritonPasses`);
//  - a value that needs an opt-in the caller has not given or an accumulation
//    the target cannot perform, i.e. `splitKFactor` above one without
//    `rock.enable_splitk_for_tuning` or without a fast atomic add for the
//    output type;
//  - a knob asking for a schedule the target cannot carry out, which the
//    pipeline answers by building the kernel it would have built anyway, see
//    `ScheduleKnobSupport`.
//
// Widening also brings back constraints the enumerators never had to state
// because pinning made them vacuous; `isFeasible` carries those, see
// `wavesPerEUFitsRegisterBudget`.
//===----------------------------------------------------------------------===//

TuningParamAxes::~TuningParamAxes() = default;

namespace {

// The Triton knobs, which `createGemm*TuningRangeBF` pins to `kKnobDefault`
// since enumerating them would double the space it writes out per knob. A
// search over the axes writes nothing out, so it can afford them, and leaving
// them pinned means a run never tries any of the eight.
constexpr StringRef kTunableKnobKeys[] = {
    "useAsyncCopy",        "useBlockPingpong", "useInThreadTranspose",
    "useBufferOps",        "useBufferAtomics", "useReductionLayout",
    "useOptimizeEpilogue", "useBf16x3ForF32"};

// Whether each of the knobs whose reach depends on the target or the problem
// can change `op`'s kernel at all. The two schedule knobs ask for a loop
// schedule instead of gating a rewrite, and a schedule the target cannot carry
// out leaves the same kernel behind as asking for no schedule at all;
// `bf16x3ForF32` names a dot precision, so it reaches nothing unless the dot it
// would relax is an f32 one.
struct KnobSupport {
  bool asyncCopy;
  bool pingpong;
  bool bf16x3ForF32;
};

// Off and on for each knob. `kKnobDefault` is legal, and `isFeasible` accepts
// it, but it is not worth a benchmark of its own: a default resolves to off or
// to on, so trying it as well would only re-measure whichever one it means on
// this arch.
//
// A knob the target or the problem leaves no room for is pinned to that default
// instead, since both of its values would build the one kernel and a search
// would pay a benchmark to learn what it had already timed.
void addKnobAxes(KnobSupport support,
                 llvm::StringMap<std::vector<int64_t>> &byKey) {
  llvm::StringMap<bool> pinned = {{"useAsyncCopy", !support.asyncCopy},
                                  {"useBlockPingpong", !support.pingpong},
                                  {"useBf16x3ForF32", !support.bf16x3ForF32}};
  for (StringRef knob : kTunableKnobKeys) {
    auto it = pinned.find(knob);
    byKey[knob] = (it != pinned.end() && it->second)
                      ? std::vector<int64_t>{kKnobDefault}
                      : std::vector<int64_t>{0, 1};
  }
}

// Whether the dot the `useBf16x3ForF32` knob would relax is an f32 one, which
// is one of the two things the knob needs to change anything at all:
// `RockBlockwiseGemmOpRewritePattern` asks for `InputPrecision::BF16x3` only
// when both of the dot's operands are f32 and issues an IEEE dot otherwise.
// Takes the types the wrapper interfaces hand out, which is an element type for
// a GEMM and a tensor type for a gemm+gemm.
//
// The other thing it needs is fast math, since `resolveUseBf16x3ForF32` pins
// the IEEE path under `-disable-fast-math` whatever the knob says. That one is
// not checked here because it cannot be: it is a pipeline option that never
// reaches the IR, so a space built from a module alone cannot read it. Nothing
// is lost by assuming it, as the tuning driver leaves
// `KernelOptions::disableFastMath` at its default and so tunes with fast math
// on; a caller that turns it off pays one wasted axis.
bool relaxableF32Dot(Type aType, Type bType) {
  return getElementTypeOrSelf(aType).isF32() &&
         getElementTypeOrSelf(bType).isF32();
}

// The powers of two in [1, max].
std::vector<int64_t> powersOfTwoUpTo(int64_t max) {
  std::vector<int64_t> values;
  for (int64_t value = 1; value <= max; value *= 2)
    values.push_back(value);
  return values;
}

// What `kpack` may be set to, which `validateKpack` (AffixTuningParameters.cpp)
// caps per arch. Only MFMA takes a kpack above one.
std::vector<int64_t> kpackValues(StringRef arch, bool isMfma) {
  if (!isMfma)
    return {1};
  return powersOfTwoUpTo(rock::getMaxKpack(arch));
}

// What `wavesPerEU` may be set to: zero, which leaves the choice to the
// backend, plus every count `validateWavesPerEU` allows. Not only the powers of
// two, since the backend is asked to hit this number exactly
// (`setKernelAttributes` writes it as `amdgpu-waves-per-eu N,N`) and the
// register budget it leaves a wave shrinks one wave at a time, so a count
// between two powers of two asks for something neither of them does. The
// enumeration pins this to zero.
std::vector<int64_t> wavesPerEUValues(Operation *op) {
  std::vector<int64_t> values;
  for (int64_t wavesPerEU = 0,
               maxWavesPerEU = rock::getMaxWavesPerEU(rock::getArchValue(op));
       wavesPerEU <= maxWavesPerEU; ++wavesPerEU)
    values.push_back(wavesPerEU);
  return values;
}

// What `numWaves` may be set to: the powers of two whose blockSize
// (`numWaves * waveSize`) fits `maxHardwareWorkgroupSize`, which is what
// `validateNumWaves` accepts. That is 16 waves on the wave64 arches and 32 on
// the wave32 ones.
std::vector<int64_t> numWavesValues(Operation *op) {
  int64_t waveSize = rock::getWaveSize(rock::getArchValue(op));
  return powersOfTwoUpTo(rock::maxHardwareWorkgroupSize / waveSize);
}

// What `matrixInstrNonkdim` may be set to: zero, which leaves the choice to
// Triton's own tile-size heuristic, plus the square MFMA tiles. A nonzero value
// forces `mDim = nDim = matrixInstrNonkdim` in `chooseMfmaInstruction`
// (AccelerateAMDMatmul.cpp), so only sizes with a square intrinsic can be asked
// for, and `MfmaIntrinsic::selectFor` has 32x32 and 16x16.
//
// Zero for anything else: WMMA takes the value and ignores it (`BlockedToWMMA`
// drops the argument), and the non-accel path has no matrix instruction to
// pick, so a second value would only re-measure the same kernel.
std::vector<int64_t> matrixInstrNonkdimValues(bool isMfma) {
  if (!isMfma)
    return {0};
  return {0, 16, 32};
}

// What `numStages` may be set to: the depth of the software pipeline
// `TritonAMDGPUScheduleLoops` builds, which `validateNumStages` asks only to be
// one or more. One turns pipelining off, and the depths above it are not
// interchangeable, since a schedule can ask for a particular one. Each stage
// costs another buffer of the K tile in LDS, and a config that overruns LDS is
// rejected as inapplicable rather than mistimed, so overshooting is cheap.
std::vector<int64_t> numStagesValues() { return {1, 2, 3, 4, 5, 6}; }

// What `gridGroupSize` may be set to: zero, which leaves the choice to
// `makeGroupedGridLayout` (GridLayoutEmitter.cpp), plus every group size up to
// 16. Larger groups trade scheduling freedom for L2 locality across the M
// blocks they cover, and a group of three covers three of them, so the sizes
// between the powers of two are grids of their own and not roundings of a
// nearby one. `validateGridGroupSize` only asks for a non-negative value, since
// the ceiling is the runtime `mBlocks`, past which the layout clamps and every
// larger value describes the same grid; 16 is where this stops buying distinct
// kernels rather than where it becomes illegal.
std::vector<int64_t> gridGroupSizeValues() {
  constexpr int64_t kMaxGridGroupSize = 16;
  std::vector<int64_t> values;
  for (int64_t groupSize = 0; groupSize <= kMaxGridGroupSize; ++groupSize)
    values.push_back(groupSize);
  return values;
}

// What `splitKFactor` may be set to. Splitting K spreads one output tile over
// several workgroups that accumulate into it atomically, which takes a caller
// that arranged for it (`rock.enable_splitk_for_tuning`); without that the
// space is one.
//
// Otherwise every factor through nine. A split multiplies the grid by its
// factor to fill the CUs a single-tile grid leaves idle, so the useful factors
// are the small ones and they are not power-of-two shaped. The enumerators are
// narrower still, offering only what a work-imbalance model expects to pay off,
// which is a way to keep brute force short rather than a bound on what a kernel
// may hold.
//
// The K in question on a gemm+gemm is the second GEMM's, which is also the
// first GEMM's N: `arrangeGemmGemmSplitKTransform` splits `gemmN` in both
// operands and the factor is read from `params1`. Not the first GEMM's K,
// which on such a kernel is a head dimension with nothing worth splitting.
std::vector<int64_t> splitKFactorValues(Operation *op) {
  // `validateSplitKFactor` rejects any factor but one for attention.
  if (isa<AttentionOp>(op))
    return {1};
  auto func = cast<func::FuncOp>(op->getParentOp());
  if (!func->hasAttr(rock::EnableSplitKForTuningAttr::getMnemonic()))
    return {1};
  if (failed(testFusionLegalitySplitK(func)))
    return {1};
  return {1, 2, 3, 4, 5, 6, 7, 8, 9};
}

// What `nPerBlockG1`, the second gemm's N tile, may be set to. Zero leaves
// gemm1 untiled, one tile spanning the whole (power-of-two padded) head dim; a
// tile instead splits the head dim into chunks that each stage their own V tile
// in LDS (see `GridwiseAttentionRewritePattern`).
//
// `validateNPerBlockG1` asks for zero or a positive power of two, so the tiles
// worth offering are the powers of two below the padded head dim: one at or
// above it describes the untiled kernel that zero already stands for. The floor
// is the smallest M/N tile the rest of the space uses and the ceiling the
// largest.
std::vector<int64_t> nPerBlockG1Values(RockGemmGemmWrapperInterface op) {
  std::vector<int64_t> values = {0};
  int64_t gemm1N = llvm::PowerOf2Ceil(op.getGemmGemmSize().o);
  for (int64_t tile = 16; tile <= 256; tile *= 2)
    if (tile < gemm1N)
      values.push_back(tile);
  return values;
}

// The tile sizes a `ladder` from the enumerators stands for. The ladders double
// their way up to keep the product brute force writes out small; where a tile
// need not be a power of two, everything in between is a tile a kernel can hold
// just as well, so the axis is every multiple of 16, the size the ladders start
// at, and the powers of two below it, the narrowest tiles a perf config may
// name. The ceiling stays the ladder's own, which is not a budget decision:
// `computeDPerBlock` and `capKPerBlockByK` cap tiles by the problem's
// dimensions, past which a tile only pads.
std::vector<int64_t> tileValues(ArrayRef<uint32_t> ladder, bool requirePow2) {
  std::vector<int64_t> values(ladder.begin(), ladder.end());
  if (requirePow2 || values.empty())
    return values;
  constexpr int64_t kTileStep = 16;
  int64_t ceiling = *llvm::max_element(values);
  for (int64_t tile = 1; tile < kTileStep && tile <= ceiling; tile *= 2)
    values.push_back(tile);
  for (int64_t tile = kTileStep; tile <= ceiling; tile += kTileStep)
    values.push_back(tile);
  return values;
}

// The K tiles worth trying for one (m, n) tile, which is `computeKPerBlock`
// plus a 16 where its list leaves one out, as the WMMA one does by starting
// higher. Nothing downstream asks for that larger floor: a WMMA instruction
// contracts 16 at a time, and `validatePerfConfig` wants only a power of two
// that a tile can be padded to.
std::vector<uint32_t> kPerBlockValues(RockGemmWrapperInterface gemmOp,
                                      TuningParamSetKind kind,
                                      uint32_t mPerBlock, uint32_t nPerBlock) {
  std::vector<uint32_t> values =
      computeKPerBlock(gemmOp, kind, mPerBlock, nPerBlock);
  constexpr uint32_t kSmallestAccelKTile = 16;
  int64_t k = gemmOp.getGemmSize().k;
  if (kSmallestAccelKTile <= llvm::PowerOf2Ceil(k) &&
      !llvm::is_contained(values, kSmallestAccelKTile)) {
    values.push_back(kSmallestAccelKTile);
    llvm::sort(values);
  }
  return values;
}

// The K tiles a conv wants on top of the ones the ladders reach: the multiples
// of `kPerBlockAlignmentFactor`, the extent a K tile has to be a multiple of
// for the conv's K index computation to stay cheap to advance. For a 3x3 filter
// that is every multiple of 9, which the ladders miss except where a multiple
// of 9 happens to be one of 16 as well.
//
// The whole ladder, without the two conditions `computeKPerBlock` attaches to
// its own aligned candidates. It asks a candidate to divide K and to be a
// multiple of the matrix instruction's K extent as well, and offers them only
// where no pow2 tile already tiles K cleanly. Those keep the *enumerated*
// product small, which is a cost a search does not pay: it pays per benchmark,
// picks the K tile as one value among many, and finds out by measuring which
// one a shape wants. So the axis carries them all, and `MAX_K_PER_BLOCK`, the
// bound every K ladder here answers to, is the only one it keeps.
std::vector<int64_t>
convAlignedKPerBlockValues(RockGemmWrapperInterface gemmOp) {
  int64_t alignment = kPerBlockAlignmentFactor(gemmOp);
  if (alignment <= 1)
    return {};
  std::vector<int64_t> values;
  for (int64_t tile = alignment; tile <= kMaxKPerBlock; tile += alignment)
    values.push_back(tile);
  return values;
}

// Whether the C tile's accumulator fits the register budget a nonzero
// `wavesPerEU` implies. Stamping `amdgpu-waves-per-eu=N,N`, which
// `setKernelAttributes` (TritonToHsaco.cpp) does for every nonzero value, caps
// each thread at `vgprsPerEU / N` registers. A tile whose accumulator overruns
// that cap cannot avoid spilling, and the register allocator then spends
// minutes walking live ranges to compile a config that was never going to be
// fast, so a search is better off not proposing it at all. `attentionSweeps.py`
// rejects these for the same reason and in the same cross-multiplied form, one
// register per accumulator element because every supported type accumulates
// into f32 or i32.
//
// Asked only of nonzero values, which are the ones no enumeration reaches: they
// pin `wavesPerEU` to zero, which stamps no attribute and leaves the budget to
// the hardware, and a space that admits their configs has to keep admitting
// them.
bool wavesPerEUFitsRegisterBudget(Operation *op, int64_t mPerBlock,
                                  int64_t nPerBlock, int64_t numWaves,
                                  int64_t wavesPerEU) {
  if (wavesPerEU == 0)
    return true;
  StringRef arch = rock::getArchValue(op);
  int64_t threads = std::max<int64_t>(1, numWaves * rock::getWaveSize(arch));
  return mPerBlock * nPerBlock * wavesPerEU <=
         rock::getVGPRsPerEU(arch) * threads;
}

// Multiplier on the elements a thread holds, for the dtypes whose conversion
// Triton expands into many LLVM ops. RDNA has no packed hardware conversion
// for fp8, so `tt.fp_to_fp` lowers to ~25 scalar ops per element, which
// inflates the K-loop body specifically. CDNA3 keeps the conversion packed
// (v_cvt_pk_f32_fp8) and CDNA4 needs none at all, since `tt.dot_scaled` takes
// fp8 operands natively, so neither is amplified.
int64_t dtypeAmplifier(StringRef arch, Type elementType) {
  bool isFp8 =
      isa<FloatType>(elementType) && elementType.getIntOrFloatBitWidth() == 8;
  return isFp8 && rock::isRDNA(arch) ? 10 : 1;
}

// How much of the AMDGPU backend's time this config's kernel is expected to
// take, in elements held per thread.
//
// Codegen, the post-RA machine scheduler above all, costs what the instruction
// count of a single basic block costs rather than what the whole kernel's does
// (the Triton IR explosion issue,
// https://github.com/ROCm/triton/issues/940). A GEMM has two basic blocks worth
// counting, each measured in the elements a thread holds:
//
//   1. The K-loop body, which reads a slice of A and of B into registers:
//        (mPerBlock + nPerBlock) * kPerBlock / (threads * kpack)
//   2. The C epilogue, which holds the accumulator and stores it in one go:
//        mPerBlock * nPerBlock / threads
//
// The cost is roughly the larger of the two rather than their sum: halving
// `kPerBlock` shortens the K-loop body and leaves the epilogue as it was, and
// the other way round. So the score is that maximum, times the dtype amplifier.
//
// Ported from `_compile_cost_score` in
// mlir/utils/performance/parameterSweeps.py, which derives it at length. Both
// sides compute in doubles in the same order, so they agree about a config that
// lands on the budget exactly.
double compileCostScore(StringRef arch, Type elementType, int64_t mPerBlock,
                        int64_t nPerBlock, int64_t kPerBlock, int64_t kpack,
                        int64_t numWaves) {
  int64_t threads = std::max<int64_t>(1, numWaves * rock::getWaveSize(arch));
  int64_t elementsPerLoad = std::max<int64_t>(1, kpack);
  double kLoopBody = static_cast<double>(mPerBlock + nPerBlock) * kPerBlock /
                     (threads * elementsPerLoad);
  double cEpilogue = static_cast<double>(mPerBlock * nPerBlock) / threads;
  return std::max(kLoopBody, cEpilogue) * dtypeAmplifier(arch, elementType);
}

// Ported from `_compile_cost_budget` in parameterSweeps.py.
int64_t compileCostBudget(StringRef arch) {
  // 8000 is where RDNA build times start to go wild. Everything else gets
  // through the same work faster, with wider waves and native fp8 paths, and
  // has no measured number of its own.
  return rock::isRDNA(arch) ? 8000 : 12000;
}

// Whether this config's kernel is one the backend can be expected to compile in
// a reasonable time. Above the budget it takes minutes, which a search cannot
// afford to spend on a config whose per-thread state that large makes it a
// poor bet anyway. `parameterSweeps.py` declines to compile these at all
// (`_perf_within_budget`), for the same reason and off the same score.
bool fitsCompileCostBudget(StringRef arch, Type elementType, int64_t mPerBlock,
                           int64_t nPerBlock, int64_t kPerBlock, int64_t kpack,
                           int64_t numWaves) {
  return compileCostScore(arch, elementType, mPerBlock, nPerBlock, kPerBlock,
                          kpack, numWaves) <= compileCostBudget(arch);
}

// Whether the two buffer knobs agree. `addTritonPasses` (Pipelines.cpp) reads
// `useBufferAtomics` only inside `if (useBufferOps)`, so asking for buffer
// atomics with the pass cluster explicitly off builds the same kernel as asking
// for neither, and timing it again teaches a search nothing. Only an explicit
// zero collides: `kKnobDefault` leaves the cluster on, which is when atomics
// have something to apply to.
bool bufferKnobsAgree(int64_t useBufferOps, int64_t useBufferAtomics) {
  return !(useBufferOps == 0 && useBufferAtomics == 1);
}

// Flags which parameters are knobs, so that `valuesAreAdmissible` can hold them
// to what is legal rather than to what is worth exploring.
SmallVector<bool> findKnobParams(RockTuningParamAttrInterface exemplar) {
  SmallVector<StringRef> names;
  exemplar.getParamNames(names);
  SmallVector<bool> found;
  for (StringRef name : names)
    found.push_back(llvm::is_contained(kTunableKnobKeys, name));
  assert(llvm::count(found, true) ==
             std::distance(std::begin(kTunableKnobKeys),
                           std::end(kTunableKnobKeys)) &&
         "a knob was renamed or dropped from the perf config");
  return found;
}

// Orders `byKey` the way `exemplar` lists its parameters, which is what lets
// the axes be indexed the same way a config's values are without anyone
// restating that order. A parameter absent from `byKey` is one this space does
// not tune, and gets the single value the exemplar gives it.
//
// Each axis comes out ascending and without repeats, which is how a search
// reads it: `feasibleNeighbors` (LFBOSearch.cpp) finds the current value with a
// binary search and calls the values either side of it the neighbouring ones,
// so a list in any other order would send a step somewhere other than next
// door. The lists these are built from are written in whichever order their
// enumerator happens to loop in, e.g. `getRangeGemmGemm` puts the problem's own
// K tile ahead of the tiles it adds around it.
std::vector<std::vector<int64_t>>
orderAxes(RockTuningParamAttrInterface exemplar,
          const llvm::StringMap<std::vector<int64_t>> &byKey) {
  SmallVector<StringRef> names;
  exemplar.getParamNames(names);
  SmallVector<int64_t> untunedValues;
  exemplar.getParamValues(untunedValues);

  std::vector<std::vector<int64_t>> axes;
  axes.reserve(names.size());
  for (auto [name, untuned] : llvm::zip_equal(names, untunedValues)) {
    auto it = byKey.find(name);
    axes.push_back(it == byKey.end() ? std::vector<int64_t>{untuned}
                                     : it->second);
    std::vector<int64_t> &axis = axes.back();
    llvm::sort(axis);
    axis.erase(llvm::unique(axis), axis.end());
  }
  // An axis given for a key that names no parameter would silently do nothing,
  // so catch the rename or the typo behind it here and not in a tuning run.
  assert(llvm::all_of(
             byKey.keys(),
             [&](StringRef key) { return llvm::is_contained(names, key); }) &&
         "an axis was given for something that is not a parameter");
  return axes;
}

// Sets `failed`, if the caller asked for it, and says the config is out. Both
// `isFeasible` implementations open with one of these.
auto refuseWith(FeasibilityCheck *refusedOn) {
  return [refusedOn](FeasibilityCheck check) {
    if (refusedOn)
      *refusedOn = check;
    return false;
  };
}

// Whether every value is one its parameter may legally hold. Deliberately
// weaker than "on the axis the search explores": a knob may also be
// `kKnobDefault`, which is how every config the tuning space and the quick list
// hand out spells its knobs, and a search has to recognise such a config to be
// able to start from it and walk onto the axes. Necessary for membership but
// not sufficient, since the axes of the interdependent parameters hold the
// union of their values over the space; each subclass narrows this further.
bool valuesAreAdmissible(ArrayRef<std::vector<int64_t>> axes,
                         ArrayRef<bool> knobParams, ArrayRef<int64_t> values) {
  if (values.size() != axes.size())
    return false;
  for (auto [axis, isKnob, value] : llvm::zip_equal(axes, knobParams, values)) {
    bool legal =
        isKnob ? isValidKnobBoolean(value) : llvm::is_contained(axis, value);
    if (!legal)
      return false;
  }
  return true;
}

class GemmParamAxes : public TuningParamAxes {
public:
  GemmParamAxes(RockGemmWrapperInterface gemmOp, TuningParamSetKind kind)
      : gemmOp(gemmOp), kind(kind), info(PopulateParamsInfo::fromOp(gemmOp)),
        ldsBlacklist(LdsBlacklist::lookupGemm(rock::getArchValue(gemmOp),
                                              info.gemmAType)) {
    StringRef arch = rock::getArchValue(gemmOp);
    int64_t waveSize = rock::getWaveSize(arch);
    int64_t maxWavesPerEU = rock::getMaxWavesPerEU(arch);
    const std::vector<std::vector<uint32_t>> ranges =
        getRangeGemm(gemmOp, waveSize, maxWavesPerEU, kind);
    MatrixAccelKind accelKind = rock::getMatrixAccelKind(arch, gemmOp);
    isMfma = accelKind == MatrixAccelKind::MFMA ||
             accelKind == MatrixAccelKind::ScaledMFMA;

    requirePow2 = pow2TilesRequired(gemmOp);

    llvm::StringMap<std::vector<int64_t>> byKey;
    byKey["mPerBlock"] = tileValues(ranges[0], requirePow2.mn);
    byKey["nPerBlock"] = tileValues(ranges[1], requirePow2.mn);
    byKey["kpack"] = kpackValues(arch, isMfma);
    byKey["numWaves"] = numWavesValues(gemmOp);
    byKey["matrixInstrNonkdim"] = matrixInstrNonkdimValues(isMfma);
    byKey["numStages"] = numStagesValues();
    byKey["wavesPerEU"] = wavesPerEUValues(gemmOp);
    byKey["gridGroupSize"] = gridGroupSizeValues();
    byKey["numCTAs"] = widen(ranges[8]);
    byKey["splitKFactor"] = splitKFactorValues(gemmOp);

    // Every K tile any tile may be paired with. `computeKPerBlock` hands a
    // non-power-of-two K only to the tiles at its own scale, which keeps the
    // enumerated product small rather than answering what a kernel can hold, so
    // a search takes the union and pairs it with every tile.
    SmallVector<int64_t> kPerBlocks;
    for (int64_t mPerBlock : byKey["mPerBlock"]) {
      for (int64_t nPerBlock : byKey["nPerBlock"]) {
        for (int64_t kPerBlock : kTilesFor(mPerBlock, nPerBlock)) {
          if (exceedsTritonTensorCap(mPerBlock, nPerBlock, kPerBlock))
            continue;
          kPerBlocks.push_back(kPerBlock);
        }
      }
    }
    // A conv also wants the K tiles its own im2col alignment asks for, which do
    // not depend on the (m, n) tile. Only where a K tile need not be a power of
    // two at all, since every one of them is one that isn't.
    if (!requirePow2.k)
      llvm::append_range(kPerBlocks, convAlignedKPerBlockValues(gemmOp));
    llvm::sort(kPerBlocks);
    kPerBlocks.erase(llvm::unique(kPerBlocks), kPerBlocks.end());
    byKey["kPerBlock"].assign(kPerBlocks.begin(), kPerBlocks.end());
    // One dot per loop here, so pingpong takes the path through
    // `Pingponger::getDotPingponged` that reads the dot's MFMA layout and
    // returns as soon as it does not find one.
    addKnobAxes({rock::supportsAsyncCopy(rock::getArchValue(gemmOp)),
                 /*pingpong=*/isMfma,
                 /*bf16x3ForF32=*/
                 relaxableF32Dot(gemmOp.getAType(), gemmOp.getBType())},
                byKey);

    exemplar = makeExemplar(gemmOp.getContext(), byKey);
    axes = orderAxes(exemplar, byKey);
    knobParams = findKnobParams(exemplar);
  }

  ArrayRef<std::vector<int64_t>> getAxes() const override { return axes; }

  void getParamNames(SmallVectorImpl<StringRef> &names) const override {
    exemplar.getParamNames(names);
  }

  void getKnobParams(SmallVectorImpl<bool> &isKnob) const override {
    isKnob.assign(knobParams.begin(), knobParams.end());
  }

  void serialize(ArrayRef<int64_t> values,
                 PerfConfigString &out) const override {
    out.clear();
    exemplar.cloneWithParamValues(values).getPerfConfigStr(out);
  }

  bool isFeasible(ArrayRef<int64_t> values,
                  FeasibilityCheck *refusedOn) const override {
    auto refuse = refuseWith(refusedOn);

    if (!valuesAreAdmissible(axes, knobParams, values))
      return refuse(FeasibilityCheck::NotOnAxis);
    auto params =
        dyn_cast_or_null<GemmParamsAttr>(exemplar.cloneWithParamValues(values));
    if (!params)
      return refuse(FeasibilityCheck::MalformedConfig);

    int64_t mPerBlock = params.getMPerBlock();
    int64_t nPerBlock = params.getNPerBlock();
    int64_t kPerBlock = params.getKPerBlock();
    if (exceedsTritonTensorCap(mPerBlock, nPerBlock, kPerBlock))
      return refuse(FeasibilityCheck::TritonTensorCap);
    // Same filter `createGemmTuningRangeBF` applies to the enumerated space, so
    // that a search over these axes does not spend trials on tile shapes known
    // not to fit in LDS. Field order must match GemmLdsKey.
    if (isBlacklisted(ldsBlacklist,
                      {mPerBlock, nPerBlock, kPerBlock, params.getNumWaves(),
                       params.getMatrixInstrNonkdim(), params.getNumStages()}))
      return refuse(FeasibilityCheck::LdsBlacklist);
    // A search chooses what to spend its trials on, and a config that takes the
    // backend minutes to compile costs it many trials' worth of time, so it is
    // worth passing over here even though the enumerated space keeps it: that
    // space is a fixed list a client asked to have benchmarked in full.
    if (!fitsCompileCostBudget(info.arch, info.gemmAType, mPerBlock, nPerBlock,
                               kPerBlock, params.getKpack(),
                               params.getNumWaves()))
      return refuse(FeasibilityCheck::CompileCostBudget);
    if (!wavesPerEUFitsRegisterBudget(gemmOp, mPerBlock, nPerBlock,
                                      params.getNumWaves(),
                                      params.getWavesPerEU()))
      return refuse(FeasibilityCheck::RegisterBudget);
    if (!bufferKnobsAgree(params.getUseBufferOps(),
                          params.getUseBufferAtomics()))
      return refuse(FeasibilityCheck::BufferKnobsDisagree);
    if (kind == TuningParamSetKind::Full) {
      PopulateParams tuningInfo;
      if (failed(tuningInfo.couldBePerformant(info, params)))
        return refuse(FeasibilityCheck::NotPerformant);
    }
    return true;
  }

private:
  static std::vector<int64_t> widen(ArrayRef<uint32_t> values) {
    return std::vector<int64_t>(values.begin(), values.end());
  }

  // The K tiles `computeKPerBlock` offers for `mPerBlock x nPerBlock`, which
  // the axis takes the union of. Only used while building it.
  std::vector<int64_t> kTilesFor(int64_t mPerBlock, int64_t nPerBlock) const {
    return tileValues(kPerBlockValues(gemmOp, kind,
                                      static_cast<uint32_t>(mPerBlock),
                                      static_cast<uint32_t>(nPerBlock)),
                      requirePow2.k);
  }

  // Any config of the space, used both to reconstruct configs from values and
  // to supply the values of the parameters the space leaves alone.
  static RockTuningParamAttrInterface
  makeExemplar(MLIRContext *ctx,
               const llvm::StringMap<std::vector<int64_t>> &byKey) {
    auto first = [&](StringRef key) { return byKey.find(key)->second.front(); };
    return GemmParamsAttr::get(
        ctx, first("mPerBlock"), first("nPerBlock"), first("kPerBlock"),
        first("kpack"), first("numCTAs"), first("numWaves"),
        first("matrixInstrNonkdim"), first("splitKFactor"), first("numStages"),
        first("wavesPerEU"), first("gridGroupSize"),
        /*useAsyncCopy=*/kKnobDefault,
        /*useBlockPingpong=*/kKnobDefault,
        /*useInThreadTranspose=*/kKnobDefault,
        /*useBufferOps=*/kKnobDefault,
        /*useBufferAtomics=*/kKnobDefault,
        /*useReductionLayout=*/kKnobDefault,
        /*useOptimizeEpilogue=*/kKnobDefault,
        /*useBf16x3ForF32=*/kKnobDefault);
  }

  RockGemmWrapperInterface gemmOp;
  TuningParamSetKind kind;
  PopulateParamsInfo info;
  // Tile shapes known to overflow LDS, the same set `createGemmTuningRangeBF`
  // filters the enumerated space with.
  GemmLdsKeySet ldsBlacklist;
  Pow2TileRequirement requirePow2;
  bool isMfma;
  RockTuningParamAttrInterface exemplar;
  // What to explore, and which of its parameters are knobs, whose legal values
  // are wider than that; see `valuesAreAdmissible`.
  std::vector<std::vector<int64_t>> axes;
  SmallVector<bool> knobParams;
};

class GemmGemmParamAxes : public TuningParamAxes {
public:
  GemmGemmParamAxes(RockGemmGemmWrapperInterface gemmGemmOp,
                    TuningParamSetKind kind)
      : gemmGemmOp(gemmGemmOp),
        gemm1NUntiled(static_cast<uint32_t>(
            llvm::PowerOf2Ceil(gemmGemmOp.getGemmGemmSize().o))) {
    StringRef arch = rock::getArchValue(gemmGemmOp);
    int64_t waveSize = rock::getWaveSize(arch);
    const std::vector<std::vector<uint32_t>> ranges =
        getRangeGemmGemm(gemmGemmOp, waveSize, kind);
    // The two gemms of an attention always take the same path, so gemm0's kind
    // decides for both (as `getRangeGemmGemm` also assumes).
    MatrixAccelKind accelKind =
        rock::getMatrixAccelKind(arch, gemmGemmOp).first;
    bool isMfma = accelKind == MatrixAccelKind::MFMA ||
                  accelKind == MatrixAccelKind::ScaledMFMA;

    llvm::StringMap<std::vector<int64_t>> byKey;
    byKey["mPerBlockG0"] = widen(ranges[0]);
    byKey["nPerBlockG0"] = widen(ranges[1]);
    byKey["nPerBlockG1"] = nPerBlockG1Values(gemmGemmOp);
    byKey["kPerBlock"] = widen(ranges[3]);
    byKey["kpack"] = kpackValues(arch, isMfma);
    byKey["numWaves"] = numWavesValues(gemmGemmOp);
    byKey["matrixInstrNonkdim"] = matrixInstrNonkdimValues(isMfma);
    byKey["numStages"] = numStagesValues();
    byKey["wavesPerEU"] = wavesPerEUValues(gemmGemmOp);
    // Attention lays out its grid with `makeGxNGridLayout`, which groups by the
    // head dim and takes no group size, so the parameter reaches nothing here.
    byKey["gridGroupSize"] = {0};
    byKey["numCTAs"] = widen(ranges[10]);
    byKey["splitKFactor"] = splitKFactorValues(gemmGemmOp);
    // Two dots per loop here, which takes pingpong down
    // `transformChainedDotSchedule` instead. That one only rearranges the two
    // dots and the memory operations between them, so it applies whatever the
    // layout is, MFMA or not; what it does want is `numStages` of exactly 4.
    addKnobAxes({rock::supportsAsyncCopy(rock::getArchValue(gemmGemmOp)),
                 /*pingpong=*/true,
                 /*bf16x3ForF32=*/
                 relaxableF32Dot(gemmGemmOp.getAType(), gemmGemmOp.getBType())},
                byKey);

    exemplar = makeExemplar(gemmGemmOp.getContext(), byKey);
    axes = orderAxes(exemplar, byKey);
    knobParams = findKnobParams(exemplar);
  }

  ArrayRef<std::vector<int64_t>> getAxes() const override { return axes; }

  void getParamNames(SmallVectorImpl<StringRef> &names) const override {
    exemplar.getParamNames(names);
  }

  void getKnobParams(SmallVectorImpl<bool> &isKnob) const override {
    isKnob.assign(knobParams.begin(), knobParams.end());
  }

  void serialize(ArrayRef<int64_t> values,
                 PerfConfigString &out) const override {
    out.clear();
    exemplar.cloneWithParamValues(values).getPerfConfigStr(out);
  }

  bool isFeasible(ArrayRef<int64_t> values,
                  FeasibilityCheck *refusedOn) const override {
    auto refuse = refuseWith(refusedOn);

    if (!valuesAreAdmissible(axes, knobParams, values))
      return refuse(FeasibilityCheck::NotOnAxis);
    auto params = dyn_cast_or_null<GemmGemmParamsAttr>(
        exemplar.cloneWithParamValues(values));
    if (!params)
      return refuse(FeasibilityCheck::MalformedConfig);

    int64_t mPerBlock = params.getMPerBlockG0();
    int64_t nPerBlock = params.getNPerBlockG0();
    // gemm1's index/mask tensor is as wide as the head dim one of its chunks
    // covers, which a tile narrows and zero leaves whole; its contraction tile
    // is gemm0's N. Guarded the way `createGemmGemmTuningRangeBF` guards it.
    int64_t nPerBlockG1 = params.getNPerBlockG1();
    if (exceedsTritonTensorCap(mPerBlock,
                               nPerBlockG1 == 0 ? gemm1NUntiled : nPerBlockG1,
                               nPerBlock))
      return refuse(FeasibilityCheck::TritonTensorCap);
    if (exceedsTritonTensorCap(mPerBlock, nPerBlock, params.getKPerBlock()))
      return refuse(FeasibilityCheck::TritonTensorCap);
    if (!wavesPerEUFitsRegisterBudget(gemmGemmOp, mPerBlock, nPerBlock,
                                      params.getNumWaves(),
                                      params.getWavesPerEU()))
      return refuse(FeasibilityCheck::RegisterBudget);
    if (!bufferKnobsAgree(params.getUseBufferOps(),
                          params.getUseBufferAtomics()))
      return refuse(FeasibilityCheck::BufferKnobsDisagree);
    return true;
  }

private:
  static std::vector<int64_t> widen(ArrayRef<uint32_t> values) {
    return std::vector<int64_t>(values.begin(), values.end());
  }

  static RockTuningParamAttrInterface
  makeExemplar(MLIRContext *ctx,
               const llvm::StringMap<std::vector<int64_t>> &byKey) {
    auto first = [&](StringRef key) { return byKey.find(key)->second.front(); };
    return GemmGemmParamsAttr::get(
        ctx, first("mPerBlockG0"), first("nPerBlockG0"), first("nPerBlockG1"),
        first("kPerBlock"), first("kpack"), first("numCTAs"), first("numWaves"),
        first("matrixInstrNonkdim"), first("splitKFactor"), first("numStages"),
        first("wavesPerEU"), first("gridGroupSize"),
        /*useAsyncCopy=*/kKnobDefault,
        /*useBlockPingpong=*/kKnobDefault,
        /*useInThreadTranspose=*/kKnobDefault,
        /*useBufferOps=*/kKnobDefault,
        /*useBufferAtomics=*/kKnobDefault,
        /*useReductionLayout=*/kKnobDefault,
        /*useOptimizeEpilogue=*/kKnobDefault,
        /*useBf16x3ForF32=*/kKnobDefault);
  }

  RockGemmGemmWrapperInterface gemmGemmOp;
  // The head-dim width gemm1 processes when `nPerBlockG1` is zero.
  uint32_t gemm1NUntiled;
  RockTuningParamAttrInterface exemplar;
  // What to explore, and which of its parameters are knobs, whose legal values
  // are wider than that; see `valuesAreAdmissible`.
  std::vector<std::vector<int64_t>> axes;
  SmallVector<bool> knobParams;
};

} // namespace

std::unique_ptr<TuningParamAxes>
mlir::rock::createTunableParamAxes(ModuleOp mod, TuningParamSetKind kind) {
  std::unique_ptr<TuningParamAxes> axes;
  mod->walk([&](rock::RockGemmWrapperInterface op) {
    axes = std::make_unique<GemmParamAxes>(op, kind);
    return WalkResult::interrupt();
  });
  if (axes)
    return axes;
  mod->walk([&](rock::RockGemmGemmWrapperInterface op) {
    axes = std::make_unique<GemmGemmParamAxes>(op, kind);
    return WalkResult::interrupt();
  });
  return axes;
}

namespace {
/// Hands out a fixed list of perf configs as one batch and then stops. The
/// timings are of no interest: the list is settled before the search starts, so
/// the client benchmarks all of it and picks the winner. This covers the
/// brute-force spaces (see `createTunableParamSpace`) as well as a client that
/// has already decided what it wants to try.
class FixedBatchStrategy : public TuningSearchStrategy {
public:
  explicit FixedBatchStrategy(std::vector<PerfConfigString> configs)
      : configs(std::move(configs)) {}

  std::vector<PerfConfigString>
  getPerfConfigBatch(ArrayRef<BenchmarkResult> prevResults) override {
    std::vector<PerfConfigString> batch = std::move(configs);
    // A moved-from vector is only guaranteed to be valid, not empty, and the
    // empty batch is what ends the search.
    configs.clear();
    return batch;
  }

  bool isIterative() const override { return false; }

private:
  std::vector<PerfConfigString> configs;
};

std::unique_ptr<TuningSearchStrategy>
wholeSpaceStrategy(ModuleOp mod, TuningParamSetKind kind) {
  std::unique_ptr<TuningParamSet> space(createTunableParamSpace(mod, kind));
  return std::make_unique<FixedBatchStrategy>(std::move(space->tuningRange));
}
} // namespace

std::optional<SearchStrategyKind>
mlir::rock::parseSearchStrategyKind(StringRef name) {
  return llvm::StringSwitch<std::optional<SearchStrategyKind>>(name)
      .Case("quick", SearchStrategyKind::Quick)
      .Case("full", SearchStrategyKind::Full)
      .Case("exhaustive", SearchStrategyKind::Exhaustive)
      .Case("lfbo", SearchStrategyKind::LFBO)
      .Case("llm", SearchStrategyKind::LLM)
      .Case("llm-lfbo", SearchStrategyKind::LLMSeededLFBO)
      .Default(std::nullopt);
}

StringRef mlir::rock::getSearchStrategyKindName(SearchStrategyKind kind) {
  switch (kind) {
  case SearchStrategyKind::Quick:
    return "quick";
  case SearchStrategyKind::Full:
    return "full";
  case SearchStrategyKind::Exhaustive:
    return "exhaustive";
  case SearchStrategyKind::LFBO:
    return "lfbo";
  case SearchStrategyKind::LLM:
    return "llm";
  case SearchStrategyKind::LLMSeededLFBO:
    return "llm-lfbo";
  }
  llvm_unreachable("unhandled search strategy kind");
}

StringRef
mlir::rock::getNameForBenchmarkStatus(BenchmarkResult::Status status) {
  switch (status) {
  case BenchmarkResult::Status::Success:
    return "success";
  case BenchmarkResult::Status::NotApplicable:
    return "notApplicable";
  case BenchmarkResult::Status::Failed:
    return "failed";
  }
  llvm_unreachable("unhandled benchmark status");
}

llvm::raw_ostream &mlir::rock::operator<<(llvm::raw_ostream &os,
                                          BenchmarkResult::Status status) {
  return os << getNameForBenchmarkStatus(status);
}

StringRef mlir::rock::getFeasibilityCheckName(FeasibilityCheck check) {
  switch (check) {
  case FeasibilityCheck::NotOnAxis:
    return "notOnAxis";
  case FeasibilityCheck::MalformedConfig:
    return "malformedConfig";
  case FeasibilityCheck::TritonTensorCap:
    return "exceedsTritonTensorCap";
  case FeasibilityCheck::LdsBlacklist:
    return "ldsBlacklist";
  case FeasibilityCheck::CompileCostBudget:
    return "compileCostBudget";
  case FeasibilityCheck::RegisterBudget:
    return "wavesPerEURegisterBudget";
  case FeasibilityCheck::BufferKnobsDisagree:
    return "bufferKnobsDisagree";
  case FeasibilityCheck::NotPerformant:
    return "notPerformant";
  }
  llvm_unreachable("unhandled feasibility check");
}

std::unique_ptr<TuningSearchStrategy>
mlir::rock::createTuningSearchStrategy(ModuleOp mod, SearchStrategyKind kind,
                                       const SearchOptions &options) {
  // One writer for the whole run, however many searches it is made of: the
  // stages of a composite share a trace rather than each opening the path and
  // truncating whatever the last one wrote.
  SharedTrace trace = openTrace(options.tracePath);

  switch (kind) {
  case SearchStrategyKind::Quick:
    return wholeSpaceStrategy(mod, TuningParamSetKind::Quick);
  case SearchStrategyKind::Full:
    return wholeSpaceStrategy(mod, TuningParamSetKind::Full);
  case SearchStrategyKind::Exhaustive:
    return wholeSpaceStrategy(mod, TuningParamSetKind::Exhaustive);
  case SearchStrategyKind::LFBO: {
    LFBOOptions lfbo;
    lfbo.setEffort(options.effort);
    lfbo.trace = trace;
    return createLFBOSearchStrategy(mod, lfbo);
  }
  case SearchStrategyKind::LLM: {
    LLMOptions llm;
    llm.search = options.llm;
    llm.setEffort(options.effort);
    llm.trace = trace;
    return createLLMSearchStrategy(mod, llm);
  }
  case SearchStrategyKind::LLMSeededLFBO: {
    LLMOptions llm;
    llm.search = options.llm;
    // The seed stage is a quick one however much the run as a whole may
    // spend, since what follows it is a full search of the same space:
    // Helion's hybrid takes `QUICK_LLM_SEARCH_DEFAULTS` even under full
    // effort (`llm_seeded_lfbo.py`, `get_kwargs_from_profile`).
    llm.setEffort(SearchEffort::Quick);
    llm.trace = trace;

    LFBOOptions lfbo;
    lfbo.setEffort(options.effort);
    lfbo.trace = trace;
    return createLLMSeededLFBOSearchStrategy(mod, llm, lfbo);
  }
  }
  llvm_unreachable("unhandled search strategy kind");
}

std::unique_ptr<TuningSearchStrategy>
mlir::rock::createFixedBatchSearchStrategy(
    std::vector<PerfConfigString> configs) {
  return std::make_unique<FixedBatchStrategy>(std::move(configs));
}
