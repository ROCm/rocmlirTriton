//===- TuningRanges.cpp - Values a perf config parameter may take ---------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The value ladders each tunable perf config parameter is drawn from, and the
// constraints that rule out combinations of them. See TuningRanges.h.
//
//===----------------------------------------------------------------------===//

#include "TuningRanges.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

namespace mlir {
namespace rock {

// Which GEMM output dimension a per-block tile list is being built for. Unlike
// `GemmDimension` (GridwiseGemmParams.h), which collapses M and N into a single
// `MorN`, this distinguishes the two so a tile list can be tailored per axis.
enum class GemmMNDim { M, N };

// Largest per-block M/N tile size we tune for. Also acts as the threshold below
// which a small dimension is covered by a single tightly-fitting tile.
#define MAX_MN_PER_BLOCK 256

// Whether this arch should use the wider CDNA tuning space. Keyed on the CDNA
// product line, not on the matrix instruction kind: gfx1250 issues WMMA but is
// classified as CDNA and shares the wide K/block lists and full numWaves range
// with MFMA parts. MFMA-only knobs such as matrixInstrNonkdim and kpack > 1 are
// chosen separately. Mirrors `rock::isCDNA`.
static bool prefersWideTuningSpace(StringRef arch) {
  return rock::isCDNA(arch);
}

// The matrixInstrNonkdim values the tuner tries, i.e. the MFMA instruction tile
// sizes. Ignored on WMMA, whose instructions are all 16x16.
static constexpr uint32_t kMatrixInstrNonkdims[] = {16, 32};

// Smallest tile covering `d` with few pow2 sub-tiles (rock-decompose-nonpow2-
// tiles splits a tile into one sub-tile per set bit): keep the top pow2 bit and
// round the remainder up to the next pow2. e.g. 77 = 64 + 13 -> 64 + 16 = 80.
static uint32_t tileReducingPartitions(uint32_t d) {
  if (d == 0 || llvm::isPowerOf2_64(d))
    return d;
  uint32_t top = llvm::PowerOf2Ceil(d) / 2; // largest power of two < d
  uint32_t rem = d - top;
  return top + llvm::PowerOf2Ceil(rem);
}

static std::vector<uint32_t>
computeDPerBlock(Operation *op, TuningParamSetKind tuningKind, GemmMNDim dim) {
  // M/N per-block tiles are the same for the accel and non-accel paths
  // ({16, 32, 64, 128, 256}), except that the non-accel path drops M/N *pairs*
  // that are too slow to compile (see isOverwideNonAccelMNPair). The attention
  // (gemm+gemm) non-accel path keeps the {32, 64, 128} space and is handled
  // separately in getRangeGemmGemm.
  std::vector<uint32_t> dPerBlockList;
  for (uint32_t dPerBlock = 16; dPerBlock <= MAX_MN_PER_BLOCK; dPerBlock *= 2)
    dPerBlockList.push_back(dPerBlock);

  // For a plain GEMM with a small (< MAX_MN_PER_BLOCK) M/N, cap the list with a
  // tile that covers the dimension tightly and drop the now-oversized larger
  // tiles (e.g. tile=80 -> {16, 32, 64, 80}). gemm+gemm isn't lowered through
  // rock-decompose-nonpow2-tiles, so it's skipped. Scaled GEMMs are also
  // skipped since the decomposition pass doesn't support them yet, so they must
  // keep power-of-two M/N tiles.
  if (auto gemmOp = dyn_cast<RockGemmWrapperInterface>(op)) {
    bool isScaledGemm = gemmOp.getScaleA() || gemmOp.getScaleB();
    GemmSize size = gemmOp.getGemmSize();
    int64_t d = dim == GemmMNDim::M ? size.m : size.n;
    if (!isScaledGemm && d > 0 && d < MAX_MN_PER_BLOCK) {
      uint32_t tile = tileReducingPartitions(static_cast<uint32_t>(d));
      llvm::erase_if(dPerBlockList, [&](uint32_t v) { return v >= tile; });
      dPerBlockList.push_back(tile);
    }
  }

  return dPerBlockList;
}

// Drop kPerBlock candidates larger than the K dimension rounded up to the next
// power of two: a tile bigger than PowerOf2Ceil(K) would only pad K and waste
// work. Adds the capping tile so K stays covered, but only when it does not
// exceed `kMaxKPerBlock`.
static void capKPerBlockByK(std::vector<uint32_t> &kPerBlockList, int64_t k) {
  assert(k > 0 && !kPerBlockList.empty() &&
         "capKPerBlockByK expects a positive K and a non-empty candidate list");
  uint32_t cap = static_cast<uint32_t>(llvm::PowerOf2Ceil(k));
  llvm::erase_if(kPerBlockList, [&](uint32_t v) { return v > cap; });
  if (!llvm::is_contained(kPerBlockList, cap) && k <= kMaxKPerBlock)
    kPerBlockList.push_back(cap);
  assert(llvm::all_of(kPerBlockList,
                      [](uint32_t v) { return v <= kMaxKPerBlock; }) &&
         "kPerBlock candidates must not exceed kMaxKPerBlock");
}

// Whether no pow2 kPerBlock candidate tiles K cleanly, which is when widening
// the range is worth it. "Cleanly" is two requirements at once:
//
//   - the candidate divides K evenly, so no iteration masks a K remainder;
//   - the candidate is a multiple of `alignTo`, so advancing K moves no inner
//     coordinate (see `kPerBlockAlignmentFactor`).
//
// We skip kPerBlock of 1, since it always divides K, but degenerates the K loop
// into one iteration per K element, so it does not count as a usable tiling.
static bool needsWidenedKPerBlockRange(ArrayRef<uint32_t> kPerBlockList,
                                       int64_t k, int64_t alignTo) {
  return llvm::none_of(kPerBlockList, [&](uint32_t v) {
    return v > 1 && k % v == 0 && v % alignTo == 0;
  });
}

// Triton caps every tensor at 2^20 elements and enforces it via
// OpTrait::impl::verifyTensorSize. Hand-mirrored from `maxTensorNumElements` in
// external/triton/include/triton/Dialect/Triton/IR/Traits.h; re-audited on
// every Triton bump (see docs/bump_triton_version.md section 5.4.1).
static constexpr int64_t kTritonMaxTensorNumElements = 1048576;

bool exceedsTritonTensorCap(uint32_t mPerBlock, uint32_t nPerBlock,
                            uint32_t kPerBlock) {
  int64_t maxMN = std::max(mPerBlock, nPerBlock);
  return static_cast<int64_t>(kPerBlock) * maxMN > kTritonMaxTensorNumElements;
}

bool isOverwideNonAccelMNPair(uint32_t mPerBlock, uint32_t nPerBlock) {
  constexpr uint32_t kWideMN = 256;
  constexpr uint32_t kNarrowMN = 128;
  return std::max(mPerBlock, nPerBlock) >= kWideMN &&
         std::min(mPerBlock, nPerBlock) >= kNarrowMN;
}

static std::vector<uint32_t> computeNumWaves(TuningParamSetKind tuningKind,
                                             int64_t waveSize) {
  if (tuningKind == TuningParamSetKind::Exhaustive) {
    std::vector<uint32_t> numWavesList;

    uint32_t maxNumWaves = maxHardwareWorkgroupSize / waveSize;
    for (uint32_t numWaves = 1; numWaves <= maxNumWaves; numWaves *= 2) {
      numWavesList.push_back(numWaves);
    }
    assert(!numWavesList.empty() && "numWavesList can't be empty");
    return numWavesList;
  }

  return {2, 4, 8};
}

// Bounds of the flat range the widened range adds on top of rule (3)'s window.
//
// The low bound is the largest non-accel pow2 kPerBlock, and it is exclusive:
// at or below it the pow2 list {1,4,8,16} already samples K densely, and the
// window itself reaches the small two-segment divisors (9, 12) on a 16- or
// 32-wide tile. It only ever bites on the non-accel path, since the caller
// clamps the range from below by the path's own smallest pow2 tile, which is
// already above it on an accel path: 17 for MFMA and 32 for WMMA.
static constexpr uint32_t kWidenedMinExclusiveKPerBlock = 16;

// The high bound depends on accel / non-accel, because the resource that
// caps kPerBlock is different for each.
//
// - Non-accel: The dot operands live in registers, so kPerBlock affects
// VGPR pressure directly. Higher values become inefficient very quickly.
//
// - Accel path: It holds its operands through LDS, so it is not bound by
// that ceiling, and it needs a higher one to be useful at all.
//
// This is where the range stops as long as the alignment fits under it; the
// caller extends it when the alignment's smallest multiple lies past it.
static constexpr uint32_t kNonAccelWidenedRangeEnd = 64;
static constexpr uint32_t kAccelWidenedRangeEnd = 128;

// The narrowest K any matrix instruction reachable from this tuning space can
// consume, or 1 if the op has no accel path.
static int64_t accelInstrKAlignment(StringRef arch,
                                    RockGemmWrapperInterface gemmOp) {
  std::optional<int64_t> narrowest;
  for (uint32_t instrNonKDim : kMatrixInstrNonkdims) {
    FailureOr<int64_t> kDim =
        rock::getAccelInstrMinKDim(arch, gemmOp, instrNonKDim);
    if (succeeded(kDim))
      narrowest = narrowest ? std::min(*narrowest, *kDim) : *kDim;
  }
  return narrowest.value_or(1);
}

// Non-power-of-two kPerBlock heuristic. 3 ideas:
//
//   (1) It must divide K evenly
//   (2) It must generate exactly two power-of-two K segments
//   (3) kPerBlock must be within [min(mPerBlock,nPerBlock)/2,
//                                 min(mPerBlock,nPerBlock))
//
// Empirically, this gives a small amount of candidates and always
// captures the best kPerBlock candidate (i.e., we dont need to consider all
// possible kPerBlock candidates) for the convolutions reported in
// https://github.com/ROCm/rocmlirTriton/pull/340. We might want to retune this
// heuristic in the future.
//
// Intuition behind each rule:
//   (1) We avoid a remainder iteration and K padding/masking.
//
//   (2) Having more than 2 usually involves having really small segments (i.e.,
//   8, 4, 2), which must be compiled to FMA (non-accel), so they will very
//   likely perform worse compared to 2 segment decompositions.
//
//   (3) LDS used per K-iteration is kPerBlock*(mPerBlock+nPerBlock), so
//   kPerBlock trades loop/sync overhead (small K -> many iterations) against
//   LDS pressure/occupancy (large K -> fewer resident workgroups). The useful
//   range is at the block's own scale: after min(mPerBlock,nPerBlock) the K
//   tile dominates LDS and occupancy drops, so a bigger K stops helping, hence
//   the upper bound. The lower bound min(mPerBlock,nPerBlock)/2 is just the
//   next pow2 down: the only non-pow2 K that is worth adding are the ones
//   between the largest pow2 gap; smaller K is already sampled by the pow2
//   list.
//
// Its real strength is that it barely grows the search space. It is evaluated
// per (mPerBlock,nPerBlock) tile and returns only the two-segment divisors of
// K that fall in that tile's window (in the worst case, 2 values per tile).
// A power-of-two K is valid for every tile, but a non-pow2 K attaches only
// to the tiles whose window it lands in, so it does not increase search space
// significantly.
//
// For example, for K=1728:
//
// Each enumerated tile gains exactly 2 candidates, and the union across all
// tiles is {18,24,36,48,72,96,144,192}. I.e.,
// {18,24} attach to min(mPerBlock,nPerBlock)=32
// {36,48} attach to min(mPerBlock,nPerBlock)=64
// {72,96} attach to min(mPerBlock,nPerBlock)=128
// {144,192} attach to min(mPerBlock,nPerBlock)=256
//
// The space grows from 4500 to 5460 configs (+21%) instead of 5x (4500 ->
// 22500).
static SmallVector<uint32_t, 2>
windowDividingKPerBlock(int64_t gemmK, uint32_t mPerBlock, uint32_t nPerBlock,
                        uint32_t minBaseK, uint32_t maxK) {
  SmallVector<uint32_t, 2> candidates;
  assert(gemmK > 0 && "gemmK must be greater than 0");

  uint32_t minMN = std::min(mPerBlock, nPerBlock);
  uint32_t lo = std::max(minBaseK, minMN / 2);
  uint32_t hi = std::min<uint32_t>(maxK, minMN);
  for (uint32_t d = lo; d < hi && static_cast<int64_t>(d) <= gemmK; ++d) {
    if (gemmK % d == 0 && llvm::popcount(d) == 2)
      candidates.push_back(d);
  }
  return candidates;
}

// The widened range is bounded differently. Unlike rule (3) it does not use
// min(mPerBlock,nPerBlock).
// What keeps it affordable is the alignment: a conv admits only multiples of
// lcm(trailing product, instruction K), usually a single value inside the
// range, so the space grows about as much as rule (3) does. For example:
//
//   i8 3x3 conv over C=256, WMMA:  372 -> 462 configs (+24%), adding 144
//   f32 3x3 conv over C=387, FMA:  756 -> 927 configs (+23%), adding 27
//
// Note that it only affects convs (we skip 1x1 convs, since they are
// essentially GEMMs).
static SmallVector<uint32_t, 8>
alignedDividingKPerBlock(int64_t gemmK, uint32_t minBaseK, uint32_t maxK,
                         uint32_t maxWidenedK, int64_t multipleOf) {
  SmallVector<uint32_t, 8> candidates;
  assert(gemmK > 0 && "gemmK must be greater than 0");

  uint32_t lo = std::max(minBaseK, kWidenedMinExclusiveKPerBlock + 1);
  uint32_t hi = std::min<uint32_t>(maxK, maxWidenedK) + 1;
  for (uint32_t d = lo; d < hi && static_cast<int64_t>(d) <= gemmK; ++d) {
    if (d % multipleOf == 0 && gemmK % d == 0)
      candidates.push_back(d);
  }
  return candidates;
}

std::vector<uint32_t> computeKPerBlock(RockGemmWrapperInterface gemmOp,
                                       TuningParamSetKind kind,
                                       uint32_t mPerBlock, uint32_t nPerBlock) {
  auto arch = rock::getArchValue(gemmOp);
  bool isAccel = rock::hasAccel(arch, gemmOp);
  bool isWideAccel = isAccel && prefersWideTuningSpace(arch);

  int64_t k = gemmOp.getGemmSize().k;

  std::vector<uint32_t> kList;
  if (isWideAccel)
    kList = kind == TuningParamSetKind::Exhaustive
                ? std::vector<uint32_t>{16, 32, 64, 128, 256, 512}
                : std::vector<uint32_t>{16, 32, 64, 128};
  else if (isAccel)
    kList = kind == TuningParamSetKind::Exhaustive
                ? std::vector<uint32_t>{32, 64, 128, 256}
                : std::vector<uint32_t>{32, 64};
  else {
    kList = {1, 4, 8, 16};
  }

  // Cap the pow2 K tiles by the actual K dimension so we don't tune (and pad)
  // K tiles far larger than the problem's K.
  capKPerBlockByK(kList, k);

  // Also tune non-pow2 K tiles, on non-scaled GEMMs and on the arches where the
  // peeled K loop is known to compile correctly.
  if (!gemmOp.getScaleA() && !gemmOp.getScaleB() &&
      rock::supportsNonPow2KPerBlock(arch)) {
    // An integer GEMM keeps its i32 accumulator exact only while every K
    // segment is at least 4 wide, which rock-gridwise-gemm-to-blockwise
    // enforces; a tile is fully covered by that rule iff it is a multiple of 4.
    bool isIntegerGemm = isa<IntegerType>(gemmOp.getAType()) &&
                         isa<IntegerType>(gemmOp.getBType());
    uint32_t baseMinK = *llvm::min_element(kList);

    constexpr uint32_t maxK = 256;

    auto addCandidates = [&](ArrayRef<uint32_t> candidates) {
      for (uint32_t d : candidates) {
        // Drop tiles the integer GEMM rule would reject.
        if (isIntegerGemm && d % kMinIntegerKSegment != 0)
          continue;
        if (!llvm::is_contained(kList, d))
          kList.push_back(d);
      }
    };

    // Tiling the loop with this value will generate efficient im2col code,
    // so we want to tune for a kPerBlock that is multiple of this value. It is
    // 1 when there is no such requirement, i.e. on a plain GEMM or a 1x1 conv.
    int64_t trailingProduct = kPerBlockAlignmentFactor(gemmOp);

    // We only tune for a kPerBlock that is multiple of the K dimension of the
    // MFMA/WMMA instruction. For example, a 3x3 conv on WMMA wants a multiple
    // of 16. So: lcm(9, 16) = 144. A non-accel (FMA) op contributes 1, leaving
    // the conv requirement on its own.
    int64_t alignTo =
        std::lcm(trailingProduct, accelInstrKAlignment(arch, gemmOp));

    // A K that the default space cannot tile cleanly has no good kPerBlock
    // there, so widen the search for it. trailingProduct > 1 leaves out GEMMs
    // and 1x1 convs, where widening has not proven any speedup yet; it is the
    // conv part of `alignTo` on its own, since `alignTo` also carries the
    // instruction K, which every op has.
    //
    // Answered against the pow2 kList, i.e. before either heuristic contributes
    // to it: the question is whether the *default* space serves this K.
    if (trailingProduct > 1 && needsWidenedKPerBlockRange(kList, k, alignTo)) {
      // Widen at least as far as `alignTo`: every candidate is a multiple of
      // it, so a bound below it would admit nothing.
      uint32_t maxWidenedK = std::max<uint32_t>(
          isAccel ? kAccelWidenedRangeEnd : kNonAccelWidenedRangeEnd, alignTo);
      addCandidates(
          alignedDividingKPerBlock(k, baseMinK, maxK, maxWidenedK, alignTo));
    }

    addCandidates(
        windowDividingKPerBlock(k, mPerBlock, nPerBlock, baseMinK, maxK));

    llvm::sort(kList);
  }
  return kList;
}

std::vector<std::vector<uint32_t>> getRangeGemm(RockGemmWrapperInterface gemmOp,
                                                int64_t waveSize,
                                                int64_t maxWavesPerEU,
                                                TuningParamSetKind kind) {
  auto mPerBlock = computeDPerBlock(gemmOp, kind, GemmMNDim::M);
  auto nPerBlock = computeDPerBlock(gemmOp, kind, GemmMNDim::N);
  std::vector<uint32_t> numWavesRange = computeNumWaves(kind, waveSize);

  std::vector<uint32_t> wavesPerEUList = {0};
  std::vector<uint32_t> gridGroupSizeList = {0};
  // std::vector<uint32_t> wavesPerEUList;
  // wavesPerEUList.push_back(0); // use heuristic
  // for (uint32_t wavesPerEU = 1; wavesPerEU <= maxWavesPerEU; wavesPerEU *= 2)
  // {
  //   wavesPerEUList.push_back(wavesPerEU);
  // }

  auto arch = rock::getArchValue(gemmOp);
  auto accelKind = rock::getMatrixAccelKind(arch, gemmOp);
  bool isMfma = accelKind == MatrixAccelKind::MFMA ||
                accelKind == MatrixAccelKind::ScaledMFMA;
  bool isAccel = accelKind != MatrixAccelKind::None;
  bool isWideAccel = isAccel && prefersWideTuningSpace(arch);

  std::vector<uint32_t> matrixInstrNonkdimList =
      isMfma ? std::vector<uint32_t>(std::begin(kMatrixInstrNonkdims),
                                     std::end(kMatrixInstrNonkdims))
             : std::vector<uint32_t>{0};

  // The K/block axis is not returned here: it depends on the (m,n) tile and is
  // built per tile by computeKPerBlock in createGemmTuningRangeBF.
  std::vector<uint32_t> numCTAsList;
  for (uint32_t n = 1; n <= rock::getMaxNumCTAs(arch); n *= 2)
    numCTAsList.push_back(n);

  std::vector<std::vector<uint32_t>> validRangeCdnaParams = {
      mPerBlock,              // M/block
      nPerBlock,              // N/block
      {1},                    // kPackList
      numWavesRange,          // numWaves
      matrixInstrNonkdimList, // matrixInstrNonkdim
      {1, 2, 3},              // numStages
      wavesPerEUList,         // wavesPerEU
      gridGroupSizeList,      // gridGroupSize
      numCTAsList             // numCTAs
  };

  // kPack is always one for WMMA
  std::vector<std::vector<uint32_t>> validRangeWmmaParams = {
      mPerBlock,         // M/block
      nPerBlock,         // N/block
      {1},               // kPackList
      {4, 8},            // numWaves
      {0},               // matrixInstrNonkdim
      {1, 2, 3},         // numStages
      wavesPerEUList,    // wavesPerEU
      gridGroupSizeList, // gridGroupSize
      numCTAsList        // numCTAs
  };

  // Non-accel (FMA) parameters. M/N tiles reuse computeDPerBlock (the same
  // {16, 32, 64, 128, 256} space as the accel paths, capped by the actual M/N
  // dimension); createGemmTuningRangeBF then drops the overwide M/N pairs.
  std::vector<uint32_t> numWavesNonAccel;
  for (uint32_t blockSize : {64u, 128u, 256u}) {
    if (blockSize % waveSize == 0)
      numWavesNonAccel.push_back(blockSize / waveSize);
  }
  assert(!numWavesNonAccel.empty() && "numWavesNonAccel must be non-empty");
  std::vector<std::vector<uint32_t>> validRangeNonAccelParams = {
      mPerBlock,         // M/block
      nPerBlock,         // N/block
      {1},               // kPackList (no kpack on non-accel FMA path)
      numWavesNonAccel,  // numWaves (= blockSize / waveSize)
      {0},               // matrixInstrNonkdim (no matrix-accel instr)
      {1, 2, 3},         // numStages
      wavesPerEUList,    // wavesPerEU
      gridGroupSizeList, // gridGroupSize
      numCTAsList        // numCTAs
  };

  if (isWideAccel)
    return validRangeCdnaParams;
  if (isAccel)
    return validRangeWmmaParams;
  return validRangeNonAccelParams;
}

std::vector<std::vector<uint32_t>>
getRangeGemmGemm(RockGemmGemmWrapperInterface gemmGemmOp, int64_t waveSize,
                 TuningParamSetKind kind) {
  std::vector<uint32_t> numWavesRange = computeNumWaves(kind, waveSize);
  auto mPerBlock = computeDPerBlock(gemmGemmOp, kind, GemmMNDim::M);
  auto nPerBlock = computeDPerBlock(gemmGemmOp, kind, GemmMNDim::N);
  std::vector<uint32_t> wavesPerEUList = {0};
  std::vector<uint32_t> gridGroupSizeList = {0};

  int64_t gemm0K = gemmGemmOp.getGemmGemmSize().k;
  // use the actual K dimension, typically it's 128 for attention
  uint32_t gemm0KPerBlock =
      std::min<uint64_t>(llvm::PowerOf2Ceil(gemm0K), kMaxKPerBlock);
  std::vector<uint32_t> kPerBlock = {gemm0KPerBlock};
  if (kind == TuningParamSetKind::Exhaustive) {
    for (uint32_t k : {16, 32, 64, 128, 512})
      if (!llvm::is_contained(kPerBlock, k))
        kPerBlock.push_back(k);
  }
  // Drop K tiles larger than PowerOf2Ceil(K).
  capKPerBlockByK(kPerBlock, gemm0K);

  auto arch = rock::getArchValue(gemmGemmOp);
  bool isAccel = rock::hasAccel(arch, gemmGemmOp);
  // Checking the first gemm is sufficient: the two gemms in attention always
  // share the same accel kind on every currently supported arch.
  MatrixAccelKind firstGemmKind =
      rock::getMatrixAccelKind(arch, gemmGemmOp).first;
  bool isMfma = firstGemmKind == MatrixAccelKind::MFMA ||
                firstGemmKind == MatrixAccelKind::ScaledMFMA;
  bool isWideAccel = isAccel && prefersWideTuningSpace(arch);

  // An MFMA-only knob (see kMatrixInstrNonkdims), so every WMMA arch -- gfx1250
  // included -- keeps the 0 that spells "no instruction tile to choose".
  std::vector<uint32_t> matrixInstrNonkdimList =
      isMfma ? std::vector<uint32_t>(std::begin(kMatrixInstrNonkdims),
                                     std::end(kMatrixInstrNonkdims))
             : std::vector<uint32_t>{0};

  std::vector<uint32_t> numCTAsList;
  for (uint32_t n = 1; n <= rock::getMaxNumCTAs(arch); n *= 2)
    numCTAsList.push_back(n);

  // kpack > 1 is not supported on certain architectures.
  std::vector<uint32_t> kPackList;
  for (uint32_t kp = 1; kp <= rock::getMaxKpack(arch); kp *= 2)
    kPackList.push_back(kp);

  // Second-GEMM N (attention head-dim) tile candidates. `0` keeps the second
  // GEMM untiled; the power-of-two candidates tile the head dim, shrinking the
  // V LDS staging tile (gemm1KPerBlock * nPerBlockG1). Only worth exploring
  // when the head dim is large, so otherwise keep just `0`.
  std::vector<uint32_t> nPerBlockG1List = {0};
  if (gemmGemmOp.getGemmGemmSize().o > 256)
    for (uint32_t t = 64; t <= 256; t *= 2)
      nPerBlockG1List.push_back(t);

  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParamsCDNA = {
      /*gemm0MPerBlock=*/mPerBlock,
      /*gemm0NPerBlock=*/nPerBlock,
      /*nPerBlockG1=*/nPerBlockG1List,
      kPerBlock,
      kPackList,
      numWavesRange,
      matrixInstrNonkdimList,
      {1, 2, 3},
      wavesPerEUList,
      gridGroupSizeList,
      numCTAsList};
  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParamsWMMA = {
      /*gemm0MPerBlock=*/mPerBlock,
      /*gemm0NPerBlock=*/nPerBlock,
      /*nPerBlockG1=*/nPerBlockG1List,
      kPerBlock,
      /*kPackList=*/{1},
      numWavesRange,
      /*matrixInstrNonkdim=*/{0},
      {1, 2},
      wavesPerEUList,
      gridGroupSizeList,
      numCTAsList};

  // Non-accel path. Attention keeps the reduced {32, 64, 128} M/N space:
  // exhaustive tuning on Navi showed neither 16 nor 256 is worth widening the
  // attention tuning space (16 helped only one shape, 256 never won and is
  // LDS-bound). See AIROCMLIR-938.
  std::vector<uint32_t> dPerBlockNonAccel = {32, 64, 128};
  std::vector<uint32_t> numWavesNonAccel;
  for (uint32_t blockSize : {64, 128, 256}) {
    if (blockSize % waveSize == 0)
      numWavesNonAccel.push_back(blockSize / waveSize);
  }
  assert(!numWavesNonAccel.empty() && "numWavesNonAccel must be non-empty");
  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParamsNonAccel = {
      /*gemm0MPerBlock=*/dPerBlockNonAccel,
      /*gemm0NPerBlock=*/dPerBlockNonAccel,
      /*nPerBlockG1=*/nPerBlockG1List,
      kPerBlock,
      /*kPackList=*/{1},
      numWavesNonAccel,
      /*matrixInstrNonkdim=*/{0},
      {1, 2},
      wavesPerEUList,
      gridGroupSizeList,
      numCTAsList};

  if (isWideAccel)
    return validRangeGemmGemmParamsCDNA;
  if (isAccel)
    return validRangeGemmGemmParamsWMMA;
  return validRangeGemmGemmParamsNonAccel;
}

} // namespace rock
} // namespace mlir
