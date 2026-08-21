//===- RockTuningImpl.cpp - tuning API implementation ----*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//
//
// This file implements the tuning interfaces
//
//===----------------------------------------------------------------------===//

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/LdsBlacklist.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>

namespace mlir {
namespace rock {

// Which GEMM output dimension a per-block tile list is being built for. Unlike
// `GemmDimension` (GridwiseGemmParams.h), which collapses M and N into a single
// `MorN`, this distinguishes the two so a tile list can be tailored per axis.
enum class GemmMNDim { M, N };

// Largest per-block M/N tile size we tune for. Also acts as the threshold below
// which a small dimension is covered by a single tightly-fitting tile.
#define MAX_MN_PER_BLOCK 256

// Largest per-block K tile size we tune for.
#define MAX_K_PER_BLOCK 512

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
// exceed MAX_K_PER_BLOCK (bigger tiles waste LDS and are ~never optimal).
static void capKPerBlockByK(std::vector<uint32_t> &kPerBlockList, int64_t k) {
  assert(k > 0 && !kPerBlockList.empty() &&
         "capKPerBlockByK expects a positive K and a non-empty candidate list");
  uint32_t cap = static_cast<uint32_t>(llvm::PowerOf2Ceil(k));
  llvm::erase_if(kPerBlockList, [&](uint32_t v) { return v > cap; });
  if (!llvm::is_contained(kPerBlockList, cap) && k <= MAX_K_PER_BLOCK)
    kPerBlockList.push_back(cap);
  assert(llvm::all_of(kPerBlockList,
                      [](uint32_t v) { return v <= MAX_K_PER_BLOCK; }) &&
         "kPerBlock candidates must not exceed MAX_K_PER_BLOCK");
}

// Whether any kPerBlock candidate tiles K cleanly, i.e. divides it evenly and
// is a multiple of `alignTo` (see kPerBlockAlignmentFactor). We skip kPerBlock
// of 1, since it always divides K, but degenerates the K loop into one iteration
// per K element, so it does not count as a usable tiling.
static bool pow2KPerBlockTilesK(ArrayRef<uint32_t> kPerBlockList, int64_t k,
                                int64_t alignTo) {
  return llvm::any_of(kPerBlockList, [&](uint32_t v) {
    return v > 1 && k % v == 0 && v % alignTo == 0;
  });
}

// The factor a kPerBlock should be a multiple of for the GEMM's K index
// computation to be cheap to advance. Returns 1 when K carries no such
// structure.
//
// We only look for forward convolutions that are NCHW, since NHWC does not
// need kPerBlock to be divisible in order to generate efficient im2col code, since
// in that layout X and Y are constants inside the loop. So we only extend
// search space for NCHW convolutions.
static int64_t kPerBlockAlignmentFactor(RockGemmWrapperInterface gemmOp) {
  // Only forward convs supported for now.
  if (gemmOp.getKernelType() != KernelType::Conv)
    return 1;

  Operation *op = gemmOp.getOperation();
  auto inputLayout = op->getAttrOfType<ArrayAttr>("input_layout");
  if (!inputLayout)
    return 1;

  ConvolutionDims convDims = ConvolutionDims::fromOp(op);
  // Walking the input layout in order collects the gemmK-merged dims in the
  // same order the merge lists them in, so the first entry is the merge's
  // outermost dim.
  SmallVector<int64_t> mergedExtents;
  for (Attribute nameAttr : inputLayout) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    if (name == "ci") {
      mergedExtents.push_back(convDims.c);
      continue;
    }
    for (auto [i, filLen] : llvm::enumerate(convDims.fil)) {
      SmallString<4> spatial;
      (Twine(i) + "i").toVector(spatial);
      // "hi"/"wi" are the legacy spellings of "0i"/"1i".
      bool isLegacy = (i == 0 && name == "hi") || (i == 1 && name == "wi");
      if (name == spatial || isLegacy)
        mergedExtents.push_back(filLen);
    }
  }
  if (mergedExtents.size() != convDims.fil.size() + 1)
    return 1;

  // Only the outermost dim advances without a carry, so the tile has to be a
  // multiple of everything below it.
  int64_t trailing = 1;
  for (int64_t len : llvm::drop_begin(mergedExtents))
    trailing *= len;
  return trailing > 0 ? trailing : 1;
}

// Triton caps every tensor at 2^20 elements and enforces it via
// OpTrait::impl::verifyTensorSize. Hand-mirrored from `maxTensorNumElements` in
// external/triton/include/triton/Dialect/Triton/IR/Traits.h; re-audited on
// every Triton bump (see docs/bump_triton_version.md section 5.4.1).
static constexpr int64_t kTritonMaxTensorNumElements = 1048576;

// A GEMM tile lowered through the Rock->Triton bridge materialises kPerBlock x
// (M|N)PerBlock index/mask tensors (a tt.broadcast from tensor<kPerBlock x 1>).
// If kPerBlock * max(mPerBlock, nPerBlock) exceeds Triton's per-tensor element
// cap, that broadcast fails verifyTensorSize and the kernel cannot compile. In
// exhaustive mode a single such combo aborts the whole config, so drop these
// combos from the tuning space up front.
static bool exceedsTritonTensorCap(uint32_t mPerBlock, uint32_t nPerBlock,
                                   uint32_t kPerBlock) {
  int64_t maxMN = std::max(mPerBlock, nPerBlock);
  return static_cast<int64_t>(kPerBlock) * maxMN > kTritonMaxTensorNumElements;
}

// The non-accel (FMA) blockwise GEMM is a fully unrolled scalar loop, so M/N
// pairs with both tiles >= 128 cost ~70% of the space's compile time while
// never winning a shape in exhaustive Navi tuning.
static bool isOverwideNonAccelMNPair(uint32_t mPerBlock, uint32_t nPerBlock) {
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

static SmallVector<int64_t>
computeOptimalSplitKFactors(RockGemmGemmWrapperInterface gemmGemmOp,
                            int64_t gemm0NPerBlock) {
  SmallVector<int64_t> splitKValues = {1};

  auto func = cast<func::FuncOp>(gemmGemmOp->getParentOp());
  if (!func->hasAttr(rock::EnableSplitKForTuningAttr::getMnemonic())) {
    return splitKValues;
  }

  uint32_t numCUs = rock::getMinNumCU(rock::getArchValue(gemmGemmOp));
  auto opNumCUs = rock::getNumCU(gemmGemmOp);
  if (succeeded(opNumCUs))
    numCUs = opNumCUs.value();

  SmallVector<int64_t, 3> aShape =
      llvm::to_vector<3>(cast<ShapedType>(gemmGemmOp.getAType()).getShape());
  SmallVector<int64_t, 3> bShape =
      llvm::to_vector<3>(cast<ShapedType>(gemmGemmOp.getBType()).getShape());
  SmallVector<int64_t, 3> cShape =
      llvm::to_vector<3>(cast<ShapedType>(gemmGemmOp.getCType()).getShape());

  GemmSize gemm0Size(/*g=*/aShape[0], /*m=*/bShape[2],
                     /*k=*/aShape[1],
                     /*n=*/aShape[2]);
  int64_t gridSize = ((gemm0Size.n) / gemm0NPerBlock) * gemm0Size.g;

  // Simple heuristic, if gridSize >= numCUs, don't use splitK
  // TODO: improve this heuristic
  if (gridSize >= numCUs)
    return splitKValues;

  // Try splitK factors 3, 4 tend to help even when M is small
  // TODO: improve this heuristic
  return {1, 3, 4};
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

// Bounds of the flat range relaxed mode adds on top of rule (3)'s window.
//
// The low bound is the largest non-accel pow2 kPerBlock, and it is exclusive:
// at or below it the pow2 list {1,4,8,16} already samples K densely, and the
// window itself reaches the small two-segment divisors (9, 12) on a 16- or
// 32-wide tile. It only ever bites on the non-accel path, since the caller
// clamps the range from below by the path's own smallest pow2 tile, which is
// already above it on an accel path: 17 for MFMA and 32 for WMMA.
static constexpr uint32_t kRelaxedMinExclusiveKPerBlock = 16;

// The high bound depends on accel / non-accel, because the resource that
// caps kPerBlock is different for each.
//
// - Non-accel: The dot operands live in registers, so kPerBlock affects
// VGPR pressure directly. Higher values become inefficient very quickly.
//
// - Accel path: It holds its operands through LDS, so it is not bound by
// that ceiling, and it needs a higher one to be useful at all.
static constexpr uint32_t kNonAccelRelaxedMaxKPerBlock = 64;
static constexpr uint32_t kAccelRelaxedMaxKPerBlock = 128;

// The matrixInstrNonkdim values the tuner tries, i.e. the MFMA instruction tile
// sizes. Ignored on WMMA, whose instructions are all 16x16.
static constexpr uint32_t kMatrixInstrNonkdims[] = {16, 32};

// The narrowest K any matrix instruction reachable from this tuning space can
// consume, or 0 if the op has no accel path. A kPerBlock that is not a multiple
// of it wastes the accelerator.
//
// decomposePow2 peels a non-pow2 kPerBlock into pow2 segments, and a segment
// narrower than an instruction's K cannot fill one. Segments and instruction
// widths are both powers of two, and Triton selects the widest instruction no
// wider than the segment, so "every segment fills an instruction" is exactly
// "kPerBlock is a multiple of the narrowest one". Falling short is not a
// miscompile, just waste: MFMA drops off its layout (the inputKSize % kDim check
// in AccelerateAMDMatmul.cpp) while WMMA zero-pads up to kDim (computeKPadding
// in DotOpToLLVM/WMMA.cpp). kPerBlock=126 peels into 64+32+16+8+4+2, so on WMMA
// it issues ten instructions to do less work than kPerBlock=128 does in eight.
//
// MFMA's kDim halves as matrixInstrNonkdim goes from 16 to 32, and that knob is
// swept outside the K axis, so take the narrowest over the whole sweep rather
// than pruning a tile that one of its settings could still use.
static int64_t minAccelInstrKDim(StringRef arch,
                                 RockGemmWrapperInterface gemmOp) {
  int64_t minKDim = 0;
  for (uint32_t instrNonKDim : kMatrixInstrNonkdims) {
    int64_t kDim = rock::getAccelInstrMinKDim(arch, gemmOp, instrNonKDim);
    if (kDim != 0)
      minKDim = minKDim ? std::min(minKDim, kDim) : kDim;
  }
  return minKDim;
}

static SmallVector<uint32_t, 8>
windowDividingKPerBlock(int64_t gemmK, uint32_t mPerBlock, uint32_t nPerBlock,
                        uint32_t minBaseK, uint32_t maxK, uint32_t relaxedMaxK,
                        int64_t relaxedMultipleOf) {
  SmallVector<uint32_t, 8> candidates;
  assert(gemmK > 0 && "gemmK must be greater than 0");

  uint32_t minMN = std::min(mPerBlock, nPerBlock);
  uint32_t lo = std::max(minBaseK, minMN / 2);
  uint32_t hi = std::min<uint32_t>(maxK, minMN);
  for (uint32_t d = lo; d < hi && static_cast<int64_t>(d) <= gemmK; ++d) {
    if (gemmK % d == 0 && llvm::popcount(d) == 2)
      candidates.push_back(d);
  }
  if (relaxedMaxK == 0)
    return candidates;

  // Relaxed mode only ever adds to the window's candidates, so that enabling it
  // cannot take a kPerBlock away from a shape that the default rules do serve.
  uint32_t relaxedLo = std::max(minBaseK, kRelaxedMinExclusiveKPerBlock + 1);
  uint32_t relaxedHi = std::min<uint32_t>(maxK, relaxedMaxK) + 1;
  for (uint32_t d = relaxedLo;
       d < relaxedHi && static_cast<int64_t>(d) <= gemmK; ++d) {
    if (d % relaxedMultipleOf != 0)
      continue;
    if (gemmK % d == 0 && !llvm::is_contained(candidates, d))
      candidates.push_back(d);
  }
  return candidates;
}

static std::vector<uint32_t> computeKPerBlock(RockGemmWrapperInterface gemmOp,
                                              TuningParamSetKind kind,
                                              uint32_t mPerBlock,
                                              uint32_t nPerBlock) {
  auto arch = rock::getArchValue(gemmOp);
  auto accelKind = rock::getMatrixAccelKind(arch, gemmOp);
  bool isMfma = accelKind == MatrixAccelKind::MFMA ||
                accelKind == MatrixAccelKind::ScaledMFMA;
  bool isWmma = accelKind == MatrixAccelKind::WMMA ||
                accelKind == MatrixAccelKind::ScaledWMMA;

  int64_t k = gemmOp.getGemmSize().k;

  std::vector<uint32_t> kList;
  if (isMfma)
    kList = kind == TuningParamSetKind::Exhaustive
                ? std::vector<uint32_t>{16, 32, 64, 128, 256, 512}
                : std::vector<uint32_t>{16, 32, 64, 128};
  else if (isWmma)
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
    // A forward conv advances its K index without a coordinate carry only on
    // tiles that are a multiple of its merged trailing extents, which is what
    // keeps the address arithmetic affine and the padding mask hoistable. This
    // is a no-op wherever K carries no such structure.
    int64_t alignTo = kPerBlockAlignmentFactor(gemmOp);
    // On an accel path every peeled segment must also fill a matrix
    // instruction, so the two requirements combine: a 3x3 conv on WMMA wants a
    // multiple of lcm(9, 16) = 144.
    if (int64_t instrKDim = minAccelInstrKDim(arch, gemmOp))
      alignTo = std::lcm(alignTo, instrKDim);
    // A K that the default space cannot tile cleanly has no good kPerBlock
    // there, so widen the search for it. Both halves of "cleanly" bite: a
    // channels-first 3x3 conv over C=256 has K = 2304, which 32 divides evenly,
    // but 32 spans two and a half filter windows, so every iteration still
    // straddles one. Since no power of two is a multiple of an odd trailing
    // product, this opens for every such conv. What counts as clean is per-path,
    // since kList is: the accel lists bottom out at 16 (MFMA) or 32 (WMMA), so
    // the gate opens for them on more shapes than for {1,4,8,16}.
    constexpr uint32_t maxK = 256;
    uint32_t relaxedMaxK = 0;
    if (!pow2KPerBlockTilesK(kList, k, alignTo))
      // How far the widened range runs is per-path; see
      // kNonAccelRelaxedMaxKPerBlock. It always runs at least to the smallest
      // tile that can satisfy alignTo at all, since a lower ceiling would open
      // the range and then admit nothing from it. maxK still caps that, so a
      // filter too large to align inside the space at all -- 5x5 on WMMA wants
      // a multiple of lcm(25, 16) = 400 -- costs no tuning time.
      relaxedMaxK = std::max<uint32_t>((isMfma || isWmma)
                                           ? kAccelRelaxedMaxKPerBlock
                                           : kNonAccelRelaxedMaxKPerBlock,
                                       alignTo);
    for (uint32_t d :
         windowDividingKPerBlock(k, mPerBlock, nPerBlock, baseMinK, maxK,
                                 relaxedMaxK, alignTo)) {
      if (isIntegerGemm && d % 4 != 0)
        continue;
      if (!llvm::is_contained(kList, d))
        kList.push_back(d);
    }
    llvm::sort(kList);
  }
  return kList;
}

static std::vector<std::vector<uint32_t>>
getRangeGemm(RockGemmWrapperInterface gemmOp, int64_t waveSize,
             int64_t maxWavesPerEU, TuningParamSetKind kind) {
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
  bool isWmma = accelKind == MatrixAccelKind::WMMA ||
                accelKind == MatrixAccelKind::ScaledWMMA;

  // The K/block axis is not returned here: it depends on the (m,n) tile and is
  // built per tile by computeKPerBlock in createGemmTuningRangeBF.
  std::vector<uint32_t> numCTAsList;
  for (uint32_t n = 1; n <= rock::getMaxNumCTAs(arch); n *= 2)
    numCTAsList.push_back(n);

  std::vector<std::vector<uint32_t>> validRangeMfmaParams = {
      mPerBlock,         // M/block
      nPerBlock,         // N/block
      {1},               // kPackList
      numWavesRange,     // numWaves
      {std::begin(kMatrixInstrNonkdims),
       std::end(kMatrixInstrNonkdims)}, // matrixInstrNonkdim
      {1, 2, 3},         // numStages
      wavesPerEUList,    // wavesPerEU
      gridGroupSizeList, // gridGroupSize
      numCTAsList        // numCTAs
  };

  // WMMA (RDNA) parameters
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

  if (isMfma)
    return validRangeMfmaParams;
  if (isWmma)
    return validRangeWmmaParams;
  return validRangeNonAccelParams;
}

static std::vector<std::vector<uint32_t>>
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
      std::min<uint64_t>(llvm::PowerOf2Ceil(gemm0K), MAX_K_PER_BLOCK);
  std::vector<uint32_t> kPerBlock = {gemm0KPerBlock};
  if (kind == TuningParamSetKind::Exhaustive) {
    for (uint32_t k : {16, 32, 64, 128, 512})
      if (!llvm::is_contained(kPerBlock, k))
        kPerBlock.push_back(k);
  }
  // Drop K tiles larger than PowerOf2Ceil(K).
  capKPerBlockByK(kPerBlock, gemm0K);

  auto arch = rock::getArchValue(gemmGemmOp);
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

  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParamsMFMA = {
      /*gemm0MPerBlock=*/mPerBlock,
      /*gemm0NPerBlock=*/nPerBlock,
      /*nPerBlockG1=*/nPerBlockG1List,
      kPerBlock,
      kPackList,
      numWavesRange,
      /*matrixInstrNonkdim=*/{16, 32},
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

  auto [firstGemmKind, secondGemmKind] =
      rock::getMatrixAccelKind(rock::getArchValue(gemmGemmOp), gemmGemmOp);

  // Checking the first gemm is sufficient: the two gemms in attention always
  // share the same accel kind on every currently supported arch.
  bool isMfma = firstGemmKind == MatrixAccelKind::MFMA ||
                firstGemmKind == MatrixAccelKind::ScaledMFMA;
  bool isWmma = firstGemmKind == MatrixAccelKind::WMMA ||
                firstGemmKind == MatrixAccelKind::ScaledWMMA;
  if (isMfma)
    return validRangeGemmGemmParamsMFMA;
  if (isWmma)
    return validRangeGemmGemmParamsWMMA;
  return validRangeGemmGemmParamsNonAccel;
}

// Keep in sync with attentionSweeps.py
// The full space is a brute-force search for attention kernels
//
// NOTE: We intentionally don't tune the following parameters:
// - `wavesPerEU`                       (set to 0)
// - `gridGroupSize`                    (set to 0)
// - `useAsyncCopy`                     (set to kKnobDefault)
// - `useBlockPingpong`                 (set to kKnobDefault)
// - `useInThreadTranspose`             (set to kKnobDefault)
// - `useBufferOps`                     (set to kKnobDefault)
// - `useBufferAtomics`                 (set to kKnobDefault)
// - `useReductionLayout`               (set to kKnobDefault)
// - `useOptimizeEpilogue`              (set to kKnobDefault)
// - `useBf16x3ForF32`                        (set to kKnobDefault)
static void createGemmGemmTuningRangeBF(TuningParamSet *newSpace,
                                        RockGemmGemmWrapperInterface gemmGemmOp,
                                        TuningParamSetKind kind) {
  auto waveSize = rock::getWaveSize(rock::getArchValue(gemmGemmOp));
  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParams =
      getRangeGemmGemm(gemmGemmOp, waveSize, kind);
  // gemm1's untiled N tile: PowerOf2Ceil(head dim of V) (see
  // PopulateParamsGemmGemm::getGemm1Params). GemmGemmSize::o is that
  // head-dim-of-V, already extracted transpose-aware. A tuned nPerBlockG1 of 0
  // keeps the second GEMM untiled and processes this full width; it is used
  // below to guard gemm1's lowered index/mask tensor against the same Triton
  // per-tensor cap as gemm0.
  uint32_t derivedGemm1NPerBlock =
      static_cast<uint32_t>(llvm::PowerOf2Ceil(gemmGemmOp.getGemmGemmSize().o));
  OpBuilder b(gemmGemmOp.getContext());
  for (uint32_t gemm0MPerBlock : validRangeGemmGemmParams[0]) {
    for (uint32_t gemm0NPerBlock : validRangeGemmGemmParams[1]) {
      auto optimalSplitKFactors =
          computeOptimalSplitKFactors(gemmGemmOp, gemm0MPerBlock);

      for (uint32_t gemm1NPerBlock : validRangeGemmGemmParams[2]) {
        // gemm1 lowers a gemm0NPerBlock x max(gemm0MPerBlock, gemm1NPerBlock)
        // index/mask tensor (its contraction tile is gemm0NPerBlock). A tuned
        // nPerBlockG1 of 0 keeps the second GEMM untiled, i.e. the full derived
        // head-dim width; a non-zero tile shrinks that dim. Guard it against
        // the same cap as gemm0; this depends only on the gemm0 M/N tiles and
        // the gemm1 N tile, so check it once outside the gemmKPerBlock loop.
        uint32_t effectiveGemm1NPerBlock =
            gemm1NPerBlock == 0 ? derivedGemm1NPerBlock : gemm1NPerBlock;
        if (exceedsTritonTensorCap(gemm0MPerBlock, effectiveGemm1NPerBlock,
                                   gemm0NPerBlock))
          continue;
        for (uint32_t gemmKPerBlock : validRangeGemmGemmParams[3]) {
          // Skip tiles whose lowered index/mask tensors would exceed Triton's
          // per-tensor element cap (see exceedsTritonTensorCap).
          if (exceedsTritonTensorCap(gemm0MPerBlock, gemm0NPerBlock,
                                     gemmKPerBlock))
            continue;
          for (uint32_t gemmKPack : validRangeGemmGemmParams[4]) {
            for (uint32_t numWaves : validRangeGemmGemmParams[5]) {
              for (uint32_t matrixInstrNonkdim : validRangeGemmGemmParams[6]) {
                for (int64_t splitKFactor : optimalSplitKFactors) {
                  for (uint32_t numStages : validRangeGemmGemmParams[7]) {
                    for (uint32_t wavesPerEU : validRangeGemmGemmParams[8]) {
                      for (uint32_t gridGroupSize :
                           validRangeGemmGemmParams[9]) {
                        for (uint32_t numCTAs : validRangeGemmGemmParams[10]) {
                          auto gemmGemmParams = GemmGemmParamsAttr::get(
                              gemmGemmOp.getContext(), gemm0MPerBlock,
                              gemm0NPerBlock, gemm1NPerBlock, gemmKPerBlock,
                              gemmKPack, numCTAs, numWaves, matrixInstrNonkdim,
                              splitKFactor, numStages, wavesPerEU,
                              gridGroupSize,
                              /*useAsyncCopy=*/kKnobDefault,
                              /*useBlockPingpong=*/kKnobDefault,
                              /*useInThreadTranspose=*/kKnobDefault,
                              /*useBufferOps=*/kKnobDefault,
                              /*useBufferAtomics=*/kKnobDefault,
                              /*useReductionLayout=*/kKnobDefault,
                              /*useOptimizeEpilogue=*/kKnobDefault,
                              /*useBf16x3ForF32=*/kKnobDefault);
                          newSpace->tuningRange.insert(
                              cast<RockTuningParamAttrInterface>(
                                  gemmGemmParams));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

static double computeWorkImbalance(GemmSize origGemmSize, int32_t gemmMPerBlock,
                                   int32_t gemmNPerBlock, int32_t gemmKPerBlock,
                                   uint32_t numCUs, int32_t splitKFactor = 1) {
  // Use calculatePaddedGemmSize with individual parameters
  const GemmSize gemmSize = calculatePaddedGemmSize(
      gemmKPerBlock, gemmMPerBlock, gemmNPerBlock, origGemmSize);
  const auto numMTiles = (gemmSize.m + gemmMPerBlock - 1) / gemmMPerBlock;
  const auto numNTiles = (gemmSize.n + gemmNPerBlock - 1) / gemmNPerBlock;

  const double totalNumWorkGroups =
      gemmSize.g * numMTiles * numNTiles * splitKFactor;
  const double maxWorkGroupsPerCU = std::ceil(totalNumWorkGroups / numCUs);
  // imbalances = max. CU work / average work per CU
  return (maxWorkGroupsPerCU * numCUs) / totalNumWorkGroups;
}

static SmallVector<int64_t> computeOptimalSplitKFactors(GemmSize origGemmSize,
                                                        int32_t gemmMPerBlock,
                                                        int32_t gemmNPerBlock,
                                                        int32_t gemmKPerBlock,
                                                        uint32_t numCUs) {
  SmallVector<int64_t> splitKValues = {1};

  const auto dataParallelGemmImbalance = computeWorkImbalance(
      origGemmSize, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock, numCUs);

  constexpr double imbalaceThreshold = 1.20;
  if (dataParallelGemmImbalance < imbalaceThreshold) {
    return splitKValues;
  }

  struct LocalData {
    int64_t splitKValue = 0;
    double workImbalance = 0.0;
  };
  SmallVector<LocalData> factors;
  constexpr double minGain = 1.30;
  // A large set of splitK values significantly increases tuning time,
  // after analysis, we've determined that using only splitK factors 3 and 4 is
  // sufficient.
  for (int64_t splitKFactor : {3, 4}) {
    const double imbalance =
        computeWorkImbalance(origGemmSize, gemmMPerBlock, gemmNPerBlock,
                             gemmKPerBlock, numCUs, splitKFactor);
    const auto gain = dataParallelGemmImbalance / imbalance;
    if (gain > minGain) {
      factors.emplace_back(LocalData{splitKFactor, imbalance});
    }
  }

  if (factors.empty()) {
    return splitKValues;
  }

  llvm::sort(factors.rbegin(), factors.rend(), [](LocalData &a, LocalData &b) {
    return a.workImbalance < b.workImbalance;
  });

  llvm::ArrayRef<LocalData> view(factors.data(), factors.size());
  llvm::for_each(view, [&](const LocalData &item) {
    splitKValues.push_back(item.splitKValue);
  });

  return splitKValues;
}

static SmallVector<int64_t>
computeOptimalSplitKFactors(RockGemmWrapperInterface gemmOp,
                            int32_t gemmMPerBlock, int32_t gemmNPerBlock,
                            int32_t gemmKPerBlock) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  SmallVector<int64_t> splitKValues = {1};

  auto func = cast<func::FuncOp>(gemmOp->getParentOp());
  if (!func->hasAttr(rock::EnableSplitKForTuningAttr::getMnemonic())) {
    return splitKValues;
  }

  uint32_t numCUs = rock::getMinNumCU(rock::getArchValue(gemmOp));
  if (succeeded(rock::getNumCU(gemmOp))) {
    numCUs = rock::getNumCU(gemmOp).value();
  }

  return computeOptimalSplitKFactors(info.gemmSize, gemmMPerBlock,
                                     gemmNPerBlock, gemmKPerBlock, numCUs);
}

// The full space is a brute-force search starting with the configs that have
// the smallest parameters. This filters out perf configs that are
// known to be impossible during tthe AffixTuningParams check.
// If `kind` is Full, also filters out unlikely-to-be-good configurations.
//
// NOTE: We intentionally don't tune the following parameters:
// - `wavesPerEU`                       (set to 0)
// - `gridGroupSize`                    (set to 0)
// - `useAsyncCopy`                     (set to kKnobDefault)
// - `useBlockPingpong`                 (set to kKnobDefault)
// - `useInThreadTranspose`             (set to kKnobDefault)
// - `useBufferOps`                     (set to kKnobDefault)
// - `useBufferAtomics`                 (set to kKnobDefault)
// - `useReductionLayout`               (set to kKnobDefault)
// - `useOptimizeEpilogue`              (set to kKnobDefault)
// - `useBf16x3ForF32`                        (set to kKnobDefault)
static void createGemmTuningRangeBF(TuningParamSet *newSpace,
                                    RockGemmWrapperInterface gemmOp,
                                    TuningParamSetKind kind) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);

  int64_t maxWavesPerEU = rock::getMaxWavesPerEU(rock::getArchValue(gemmOp));
  int64_t waveSize = rock::getWaveSize(rock::getArchValue(gemmOp));
  const std::vector<std::vector<uint32_t>> params =
      getRangeGemm(gemmOp, waveSize, maxWavesPerEU, kind);

  auto tuningInfo = std::make_unique<PopulateParams>();
  bool isNonAccel = !rock::hasAccel(rock::getArchValue(gemmOp), gemmOp);

  // Tile shapes known to overflow LDS for this (arch, input dtype) are dropped
  // up front: LDS usage of a plain GEMM is independent of M/N/K, so this cache
  // (populated offline, see LdsBlacklist.h) never depends on the problem shape.
  // It is keyed on only the LDS-relevant fields (see GemmLdsKey), so it also
  // covers configs that differ only in an LDS-irrelevant field (kpack, splitK,
  // ...). A miss returns an empty set, so an un-populated blacklist is a no-op.
  GemmLdsKeySet ldsBlacklistSet =
      LdsBlacklist::lookupGemm(rock::getArchValue(gemmOp), info.gemmAType);

  OpBuilder b(gemmOp.getContext());
  for (uint32_t gemmMPerBlock : params[0]) {
    for (uint32_t gemmNPerBlock : params[1]) {
      // Skip M/N pairs that are too costly to compile to be worth tuning on
      // the FMA path (see isOverwideNonAccelMNPair).
      if (isNonAccel && isOverwideNonAccelMNPair(gemmMPerBlock, gemmNPerBlock))
        continue;
      for (uint32_t gemmKPerBlock :
           computeKPerBlock(gemmOp, kind, gemmMPerBlock, gemmNPerBlock)) {
        // Skip tiles whose lowered index/mask tensors would exceed Triton's
        // per-tensor element cap; they cannot compile (see
        // exceedsTritonTensorCap).
        if (exceedsTritonTensorCap(gemmMPerBlock, gemmNPerBlock, gemmKPerBlock))
          continue;
        for (uint32_t gemmKPack : params[2]) {
          for (uint32_t numWaves : params[3]) {
            for (uint32_t matrixInstrNonkdim : params[4]) {
              auto optimalSplitKFactors = computeOptimalSplitKFactors(
                  gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock);
              for (int64_t splitKFactor : optimalSplitKFactors) {
                for (int64_t numStages : params[5]) {
                  // Drop LDS-overflowing tiles before building any attribute or
                  // running the (expensive) performance filter below: the
                  // projection depends only on fields fixed by this point, so a
                  // blacklisted tile is dead for every inner splitK/wave/CTA
                  // combination. Field order must match GemmLdsKey /
                  // PROJECTION_NAMES.
                  if (!ldsBlacklistSet.empty() &&
                      ldsBlacklistSet.count({gemmMPerBlock, gemmNPerBlock,
                                             gemmKPerBlock, numWaves,
                                             matrixInstrNonkdim, numStages}))
                    continue;
                  for (int64_t wavesPerEU : params[6]) {
                    for (int64_t gridGroupSize : params[7]) {
                      for (uint32_t numCTAs : params[8]) {
                        auto gemmParams = GemmParamsAttr::get(
                            b.getContext(), gemmMPerBlock, gemmNPerBlock,
                            gemmKPerBlock, gemmKPack, numCTAs, numWaves,
                            matrixInstrNonkdim, splitKFactor, numStages,
                            wavesPerEU, gridGroupSize,
                            /*useAsyncCopy=*/kKnobDefault,
                            /*useBlockPingpong=*/kKnobDefault,
                            /*useInThreadTranspose=*/kKnobDefault,
                            /*useBufferOps=*/kKnobDefault,
                            /*useBufferAtomics=*/kKnobDefault,
                            /*useReductionLayout=*/kKnobDefault,
                            /*useOptimizeEpilogue=*/kKnobDefault,
                            /*useBf16x3ForF32=*/kKnobDefault);
                        if (kind == TuningParamSetKind::Full &&
                            failed(tuningInfo->couldBePerformant(info,
                                                                 gemmParams)))
                          continue;
                        newSpace->tuningRange.insert(
                            cast<RockTuningParamAttrInterface>(gemmParams));
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

static void createGemmTuningRangeQuick(TuningParamSet *newSpace,
                                       RockGemmWrapperInterface gemmOp) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);
  OpBuilder b(gemmOp.getContext());
  PopulateParams tuningInfo;

  // `getTuningParameters` already bumps the first conservatively-applicable
  // config to the front of the list.
  for (GemmParamsAttr param : tuningInfo.getTuningParameters(
           b, info.kernelType, info.gemmAType, info.gemmBType, info.arch,
           info.quantBlockSize, info.aScaleType, info.bScaleType)) {
    newSpace->tuningRange.insert(cast<RockTuningParamAttrInterface>(param));
  }
}

static void
createGemmGemmTuningRangeQuick(TuningParamSet *newSpace,
                               RockGemmGemmWrapperInterface gemmGemmOp) {
  OpBuilder b(gemmGemmOp.getContext());
  // `getTuningParameters` already bumps the first conservatively-applicable
  // config to the front of the list.
  for (GemmGemmParamsAttr params :
       PopulateParamsGemmGemm::getTuningParameters(b, gemmGemmOp)) {
    newSpace->tuningRange.insert(cast<RockTuningParamAttrInterface>(params));
  }
}

TuningParamSet *createTunableParamSpace(ModuleOp mod, TuningParamSetKind kind) {
  struct TuningParamSet *newSpace;
  newSpace = new TuningParamSet();

  // create range and heuristic
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createGemmTuningRangeBF(newSpace, op, kind);
          // Full and exhaustive tuning must include every quick-tuning config
          // so they cannot produce a worse search space than quick tuning.
          [[fallthrough]];
        case TuningParamSetKind::Quick:
          createGemmTuningRangeQuick(newSpace, op);
          break;
        }
        newSpace->primaryOpType = op.getKernelType();
        return WalkResult::interrupt();
      });
  WalkResult findGemmGemm =
      mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createGemmGemmTuningRangeBF(newSpace, op, kind);
          // Full and exhaustive tuning must include every quick-tuning config
          // so they cannot produce a worse search space than quick tuning.
          [[fallthrough]];
        case TuningParamSetKind::Quick:
          createGemmGemmTuningRangeQuick(newSpace, op);
          break;
        }
        return WalkResult::interrupt();
      });
  if (!findPrimary.wasInterrupted() && !findGemmGemm.wasInterrupted()) {
    llvm::report_fatal_error("Expected to find GEMM, convolution, attention, "
                             "gemm+gemm or conv+gemm op, and didn't.");
  }
  return newSpace;
}

bool tuningGetParam(TuningParamSet *tuningSpace, unsigned pos,
                    ParamEntry *paramEntry) {
  // out of bound check.
  if (pos > tuningSpace->tuningRange.size() - 1)
    return false;
  paramEntry->param = tuningSpace->tuningRange[pos];
  return true;
}

bool tuningSetParam(ModuleOp &mod, ParamEntry *paramEntry) {
  WalkResult setPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        SmallString<ROCMLIR_TUNING_PARAM_STRING_BUFSZ> perfConfig;
        paramEntry->param.getPerfConfigStr(perfConfig);
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  WalkResult setGemmGemm =
      mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        SmallString<ROCMLIR_TUNING_PARAM_STRING_BUFSZ> perfConfig;
        paramEntry->param.getPerfConfigStr(perfConfig);
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  return setPrimary.wasInterrupted() || setGemmGemm.wasInterrupted();
}

bool tuningSetStr(ModuleOp &mod, StringRef perfConfig) {
  // This stamps a single perf config onto a single tunable op, so we expect at
  // most one gemm or gemm+gemm op in the module. Walk fully (rather than
  // interrupting on the first match) so the asserts below can catch a module
  // that violates that assumption instead of silently stamping just one op.
  unsigned numGemm = 0;
  mod->walk([&](rock::RockGemmWrapperInterface op) {
    auto *ctx = op.getContext();
    op->setAttr("perf_config", StringAttr::get(ctx, perfConfig));
    ++numGemm;
  });
  unsigned numGemmGemm = 0;
  mod->walk([&](rock::RockGemmGemmWrapperInterface op) {
    auto *ctx = op.getContext();
    op->setAttr("perf_config", StringAttr::get(ctx, perfConfig));
    ++numGemmGemm;
  });
  assert(numGemm <= 1 && "expected at most one gemm op to stamp");
  assert(numGemmGemm <= 1 && "expected at most one gemm+gemm op to stamp");
  assert(!(numGemm && numGemmGemm) &&
         "expected either a gemm or a gemm+gemm op, not both");
  return numGemm || numGemmGemm;
}

TuningTable *tuningTableCreate() {
  struct TuningTable *newTable = new TuningTable();
  return newTable;
}

static LogicalResult
extractLayouts(Operation *op, llvm::StringMap<unsigned> &fLayoutMap,
               llvm::StringMap<unsigned> &iLayoutMap,
               llvm::StringMap<unsigned> &oLayoutMap, SmallString<6> &fLayout,
               SmallString<6> &iLayout, SmallString<6> &oLayout,
               bool computeOutput = true) {
  // Extract layout information
  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  ArrayAttr outputLayoutAttr;
  if (computeOutput)
    outputLayoutAttr = op->getAttrOfType<ArrayAttr>("output_layout");

  unsigned size = filterLayoutAttr.size();

  for (unsigned i = 0; i < size; ++i) {
    auto filterAttr = cast<StringAttr>(filterLayoutAttr.getValue()[i]);
    StringRef fKey = filterAttr.getValue();
    if (fKey == "y")
      fKey = "0";
    if (fKey == "x")
      fKey = "1";
    fLayoutMap[fKey] = i;
    auto inputAttr = cast<StringAttr>(inputLayoutAttr.getValue()[i]);
    StringRef iKey = inputAttr.getValue();
    if (iKey == "hi")
      iKey = "0i";
    if (iKey == "wi")
      iKey = "1i";
    iLayoutMap[iKey] = i;
    if (computeOutput) {
      auto outputAttr = cast<StringAttr>(outputLayoutAttr.getValue()[i]);
      StringRef oKey = outputAttr.getValue();
      if (oKey == "ho")
        oKey = "0o";
      if (oKey == "wo")
        oKey = "1o";
      oLayoutMap[oKey] = i;
    }
  }

  fLayout.assign(size, '#');
  iLayout.assign(size, '#');
  oLayout.assign(size, '#');

  // dimensions need to be mapped 1 to 1.
  fLayout[fLayoutMap["k"]] = 'N';
  fLayout[fLayoutMap["c"]] = 'C';
  fLayout[fLayoutMap["g"]] = 'G';
  iLayout[iLayoutMap["ni"]] = 'N';
  iLayout[iLayoutMap["ci"]] = 'C';
  iLayout[iLayoutMap["gi"]] = 'G';
  if (computeOutput) {
    oLayout[oLayoutMap["no"]] = 'N';
    oLayout[oLayoutMap["ko"]] = 'C';
    oLayout[oLayoutMap["go"]] = 'G';
  }

  for (unsigned i = 0; i < size - 3; i++) {
    std::string key = std::to_string(i);
    char val = '0' + i;
    fLayout[fLayoutMap[key]] = val;
    iLayout[iLayoutMap[key + "i"]] = val;
    if (computeOutput)
      oLayout[oLayoutMap[key + "o"]] = val;
  }

  if (computeOutput) {
    if (llvm::any_of(llvm::concat<const char>(fLayout, iLayout, oLayout),
                     [](const char c) { return c == '#'; }))
      return failure();
  } else {
    if (llvm::any_of(llvm::concat<const char>(fLayout, iLayout),
                     [](const char c) { return c == '#'; }))
      return failure();
  }
  return success();
}

// Walk backward through a transform chain and report whether any individual
// transform preserves rank and element extents while applying a non-identity
// permutation. This catches layout-only changes, such as transposed attention
// bias, even when they are surrounded by flatten/unflatten transforms.
static bool hasRankPreservingNonIdentityPermutation(Value value) {
  while (auto transformOp = value.getDefiningOp<TransformOp>()) {
    auto valueType = dyn_cast<ShapedType>(transformOp.getResult().getType());
    auto inputType = dyn_cast<ShapedType>(transformOp.getInput().getType());
    if (valueType && inputType && valueType.getRank() == inputType.getRank()) {
      SmallVector<int64_t> valueShape(valueType.getShape());
      SmallVector<int64_t> inputShape(inputType.getShape());
      llvm::sort(valueShape);
      llvm::sort(inputShape);

      AffineMap map = transformOp.getTransform().getMap().getAffineMap();
      if (valueShape == inputShape && map && map.isPermutation() &&
          !isIdentityOnShape(map, valueType.getShape()))
        return true;
    }
    value = transformOp.getInput();
  }
  return false;
}

// Determine whether an attention op fuses a pre-softmax scale and/or bias, as
// created by rocmlir-gen's `--with-attn-scale` / `--with-attn-bias`. Scale
// folds an extra elementwise multiply of the QK^T scores by an external input;
// bias folds an elementwise add. Both change the generated kernel (and thus its
// optimal perf config), so they are part of the tuning-problem identity and
// must appear in the tuning key. A rank-preserving non-identity permutation on
// the bias input records that the bias is loaded transposed.
//
// The fusion is encoded as ops inside the `preSoftmaxBody` region that consume
// the region's block arguments. Block argument 0 is the QK^T product; the
// remaining block arguments map 1:1 to `preSoftmaxElemWiseInputs`. Constant
// scales/biases
// and causal masks are not external inputs (they are folded into the body or
// captured by the `causal` attribute), so they are intentionally not treated as
// attn scale/bias here. For quantized (i8) attention the first two elementwise
// inputs are dequantization operands, not the attention scale/bias, so they are
// skipped.
static void getAttentionScaleBias(AttentionOp attnOp, bool isQuantized,
                                  bool &hasAttnScale, bool &hasAttnBias,
                                  bool &hasTransposedAttnBias) {
  hasAttnScale = false;
  hasAttnBias = false;
  hasTransposedAttnBias = false;
  Region &body = attnOp.getPreSoftmaxBody();
  if (body.empty())
    return;
  Block &entry = body.front();
  unsigned numInputs = attnOp.getPreSoftmaxElemWiseInputs().size();
  unsigned numQuantInputs = isQuantized ? 2u : 0u;
  if (numInputs <= numQuantInputs)
    return;

  // Entry block argument 0 is the QK^T product; arguments 1.. correspond 1:1 to
  // the pre-softmax elementwise inputs. AttentionOp::verify pins this arity
  // (see verifyGemmPlusGemmLikeOp).
  assert(entry.getNumArguments() == 1 + numInputs &&
         "pre-softmax body arguments must match elementwise inputs");

  for (unsigned i = numQuantInputs; i < numInputs; ++i) {
    // Walk forward from the input through its use chain, stepping over any
    // intermediate op (e.g. a `rock.transform` reshaping the input to match the
    // scores) until we reach the multiply (scale) or add (bias) that consumes
    // it.
    SmallVector<Value> worklist{entry.getArgument(i + 1)};
    llvm::SmallPtrSet<Value, 8> seen;
    bool isTransposedInput = hasRankPreservingNonIdentityPermutation(
        attnOp.getPreSoftmaxElemWiseInputs()[i]);
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      for (Operation *user : v.getUsers()) {
        if (isa<arith::MulFOp>(user))
          hasAttnScale = true;
        else if (isa<arith::AddFOp>(user)) {
          hasAttnBias = true;
          hasTransposedAttnBias |= isTransposedInput;
        } else {
          llvm::append_range(worklist, user->getResults());
        }
      }
    }
  }
}

// Keep this problem-key serialization in sync with the corresponding
// configuration parsing and serialization in perfRunner.py.
static LogicalResult
getTuningProblemStr(RockGemmGemmWrapperInterface gemmGemmOp,
                    SmallVectorImpl<char> &out) {
  int64_t numCU = rock::getNumCUValue(gemmGemmOp);
  int64_t numChiplets = rock::getNumChipletsValue(gemmGemmOp);
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  int64_t headDimQK;
  int64_t headDimV;
  int64_t seqLenQ;
  int64_t seqLenK;
  llvm::raw_svector_ostream problemOS(out);
  // ARCH string
  problemOS << StringRef(rock::getArchValue(gemmGemmOp)) << tab;
  // Number of Compute Units
  problemOS << numCU << tab;
  // Number of chiplets
  problemOS << numChiplets << tab;

  ArrayRef<int64_t> qShape = cast<ShapedType>(gemmGemmOp.getAType()).getShape();
  ArrayRef<int64_t> kShape = cast<ShapedType>(gemmGemmOp.getBType()).getShape();
  ArrayRef<int64_t> vShape = cast<ShapedType>(gemmGemmOp.getCType()).getShape();

  bool isAttention = isa<AttentionOp>(gemmGemmOp);
  bool isConvGemm = isa<ConvElementwiseGemmOp>(gemmGemmOp);

  Type elemTypeQ = cast<ShapedType>(gemmGemmOp.getAType()).getElementType();
  problemOS << "-t ";
  if (elemTypeQ.isF32()) {
    problemOS << "f32" << sep;
  } else if (elemTypeQ.isF16()) {
    problemOS << "f16" << sep;
  } else if (elemTypeQ.isBF16()) {
    problemOS << "bf16" << sep;
  } else if (elemTypeQ.isInteger(8) && isAttention) {
    problemOS << "i8" << sep;
  } else {
    return gemmGemmOp.emitError("invalid type:") << elemTypeQ << "\n";
  }

  // Extract layout information
  llvm::StringMap<unsigned> fLayoutMap, iLayoutMap, oLayoutMap;
  SmallString<6> fLayout, iLayout, oLayout;

  if (isConvGemm) {
    if (failed(extractLayouts(gemmGemmOp, fLayoutMap, iLayoutMap, oLayoutMap,
                              fLayout, iLayout, oLayout, false)))
      return gemmGemmOp.emitError("layout can't be extracted");

    // filter layout
    problemOS << "-f " << fLayout << sep;
    // input layout
    problemOS << "-I " << iLayout << sep;
  } else {
    // TransQ
    if (isAttention)
      problemOS << "-transQ ";
    else
      problemOS << "-transA ";
    if (gemmGemmOp.getTransposedA()) {
      seqLenQ = qShape[2];
      headDimQK = qShape[1];
      problemOS << "true" << sep;
    } else {
      seqLenQ = qShape[1];
      headDimQK = qShape[2];
      problemOS << "false" << sep;
    }

    // TransK
    if (isAttention)
      problemOS << "-transK ";
    else
      problemOS << "-transB ";
    if (gemmGemmOp.getTransposedB()) {
      seqLenK = kShape[1];
      problemOS << "true" << sep;
    } else {
      seqLenK = kShape[2];
      problemOS << "false" << sep;
    }
  }

  // TransV
  if (isAttention)
    problemOS << "-transV ";
  else
    problemOS << "-transC ";
  if (gemmGemmOp.getTransposedC()) {
    headDimV = vShape[1];
    problemOS << "true" << sep;
  } else {
    headDimV = vShape[2];
    problemOS << "false" << sep;
  }

  // TransO
  problemOS << "-transO ";
  if (gemmGemmOp.getTransposedOut())
    problemOS << "true" << sep;
  else
    problemOS << "false" << sep;

  bool hasAttnScale = false, hasAttnBias = false;
  bool hasTransposedAttnBias = false;
  if (isAttention) {
    auto attentionOp = cast<AttentionOp>(gemmGemmOp);
    getAttentionScaleBias(attentionOp, elemTypeQ.isInteger(8), hasAttnScale,
                          hasAttnBias, hasTransposedAttnBias);
    problemOS << "-causal ";
    if (attentionOp.getCausal())
      problemOS << "true" << sep;
    else
      problemOS << "false" << sep;

    problemOS << "-return_lse ";
    if (attentionOp.getLse())
      problemOS << "true" << sep;
    else
      problemOS << "false" << sep;

    problemOS << "-split_kv " << attentionOp.getSplitKV() << sep;
    // The look-back is optional; only emit it when set so non-sliding problems
    // omit the field from their tuning identity.
    if (auto slidingWindowLookBack = attentionOp.getSlidingWindowLookBack();
        slidingWindowLookBack && *slidingWindowLookBack > 0)
      problemOS << "-sliding_window_look_back " << *slidingWindowLookBack
                << sep;
    problemOS << "-num_heads_q " << attentionOp.getNumHeadsQ() << sep;
    problemOS << "-num_heads_kv " << attentionOp.getNumHeadsKV() << sep;
    problemOS << "-g " << qShape[0] / attentionOp.getNumHeadsQ() << sep;
  }

  if (!isConvGemm && !isAttention)
    problemOS << "-g " << qShape[0] << sep;

  if (isAttention) {
    problemOS << "-seq_len_q " << seqLenQ << sep;
    problemOS << "-seq_len_k " << seqLenK << sep;
    problemOS << "-head_dim_qk " << headDimQK << sep;
    problemOS << "-head_dim_v " << headDimV;
    // Keep these last and in this order to match the layout parsed by
    // AttentionConfiguration.from_command_line() in perfRunner.py.
    problemOS << sep << "-with-attn-scale "
              << (hasAttnScale ? "true" : "false");
    problemOS << sep << "-with-attn-bias " << (hasAttnBias ? "true" : "false");
    problemOS << sep << "-transBias "
              << (hasTransposedAttnBias ? "true" : "false");
  } else if (isConvGemm) {
    auto convGemmOp = cast<ConvElementwiseGemmOp>(gemmGemmOp);
    ArrayRef<int64_t> inShape = convGemmOp.getInput().getType().getShape();
    ArrayRef<int64_t> filShape = convGemmOp.getFilter().getType().getShape();

    // N
    problemOS << "-n " << inShape[iLayoutMap["ni"]] << sep;
    // C
    problemOS << "-c " << inShape[iLayoutMap["ci"]] * inShape[iLayoutMap["gi"]]
              << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["0i"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["1i"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] * filShape[fLayoutMap["g"]]
              << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["0"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["1"]] << sep;

    auto paddingVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getPadding());
    auto strideVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getStrides());
    auto dilationVal =
        extractFromIntegerArrayAttr<int64_t>(convGemmOp.getDilations());

    // padding
    problemOS << "-p " << paddingVal[0] << " -q " << paddingVal[2] << sep;
    // stride
    problemOS << "-u " << strideVal[0] << " -v " << strideVal[1] << sep;
    // dilation
    problemOS << "-l " << dilationVal[0] << " -j " << dilationVal[1] << sep;
    // group
    problemOS << "-g " << inShape[iLayoutMap["gi"]] << sep;
    problemOS << "-gemmO " << headDimV;
  } else {
    problemOS << "-m " << seqLenQ << sep;
    problemOS << "-n " << seqLenK << sep;
    problemOS << "-k " << headDimQK << sep;
    problemOS << "-gemmO " << headDimV;
  }
  return success();
}

static LogicalResult getTuningProblemStr(rock::RockGemmWrapperInterface gemmIF,
                                         SmallVectorImpl<char> &out) {
  int64_t numCU = rock::getNumCUValue(gemmIF);
  int64_t numChiplets = rock::getNumChipletsValue(gemmIF);
  constexpr char sep = ' ';
  constexpr char tab = '\t';
  llvm::raw_svector_ostream problemOS(out);

  KernelType opType = gemmIF.getKernelType();
  Operation *gemmOp = gemmIF.getOperation();

  auto f8TypeStr = [](const Type &type) -> std::optional<StringLiteral> {
    if (isa<Float8E4M3FNUZType, Float8E4M3FNType>(type))
      return StringLiteral("fp8");
    if (isa<Float8E5M2FNUZType, Float8E5M2Type>(type))
      return StringLiteral("bf8");
    return std::nullopt;
  };

  // ARCH string
  problemOS << StringRef(rock::getArchValue(gemmIF)).trim("\"") << tab;
  // Number of Compute Units
  problemOS << numCU << tab;
  // Number of chiplets
  problemOS << numChiplets << tab;

  if (opType == KernelType::Conv || opType == KernelType::ConvBwdData ||
      opType == KernelType::ConvBwdWeight) { // conv cases
    RockConvInterface convIF = dyn_cast<RockConvInterface>(gemmOp);

    ShapedType inType = convIF.getConvInput().getType();
    ArrayRef<int64_t> inShape = inType.getShape();
    ShapedType filType = convIF.getConvFilter().getType();
    ArrayRef<int64_t> filShape = filType.getShape();

    // Extract layout information
    llvm::StringMap<unsigned> fLayoutMap, iLayoutMap, oLayoutMap;
    SmallString<6> fLayout, iLayout, oLayout;
    if (failed(extractLayouts(gemmOp, fLayoutMap, iLayoutMap, oLayoutMap,
                              fLayout, iLayout, oLayout)))
      return convIF.emitError("layout can't be extracted");

    // Please keep these in sync with mlir/utils/performance/perfRunner.py

    // OP datatype
    Type inElemType = inType.getElementType();
    Type filElemType = filType.getElementType();
    if (inElemType.isF32()) {
      problemOS << "conv ";
    } else if (inElemType.isF16()) {
      problemOS << "convfp16 ";
    } else if (inElemType.isBF16()) {
      problemOS << "convbfp16 ";
    } else if (inElemType.isInteger(8)) {
      problemOS << "convint8 ";
    } else {
      auto inString = f8TypeStr(inElemType);
      auto filString = f8TypeStr(filElemType);
      if (inString && filString)
        problemOS << llvm::formatv("conv{0}_{1} ", *inString, *filString);
      else
        return failure();
    }

    // OP direction
    switch (opType) {
    case KernelType::Conv:
      problemOS << "-F 1" << sep;
      break;
    case KernelType::ConvBwdData:
      problemOS << "-F 2" << sep;
      break;
    case KernelType::ConvBwdWeight:
      problemOS << "-F 4" << sep;
      break;
    default:
      return failure();
    }

    // filter layout
    problemOS << "-f " << fLayout << sep;
    // input layout
    problemOS << "-I " << iLayout << sep;
    // output layout
    problemOS << "-O " << oLayout << sep;
    // N
    problemOS << "-n " << inShape[iLayoutMap["ni"]] << sep;
    // C
    problemOS << "-c " << inShape[iLayoutMap["ci"]] * inShape[iLayoutMap["gi"]]
              << sep;
    // H
    problemOS << "-H " << inShape[iLayoutMap["0i"]] << sep;
    // W
    problemOS << "-W " << inShape[iLayoutMap["1i"]] << sep;
    // K
    problemOS << "-k " << filShape[fLayoutMap["k"]] * filShape[fLayoutMap["g"]]
              << sep;
    // Y
    problemOS << "-y " << filShape[fLayoutMap["0"]] << sep;
    // X
    problemOS << "-x " << filShape[fLayoutMap["1"]] << sep;

    auto paddingVal = extractFromIntegerArrayAttr<int64_t>(convIF.getPadding());
    auto strideVal = extractFromIntegerArrayAttr<int64_t>(convIF.getStrides());
    auto dilationVal =
        extractFromIntegerArrayAttr<int64_t>(convIF.getDilations());
    // padding
    problemOS << "-p " << paddingVal[0] << " -q " << paddingVal[2] << sep;
    // stride
    problemOS << "-u " << strideVal[0] << " -v " << strideVal[1] << sep;
    // dilation
    problemOS << "-l " << dilationVal[0] << " -j " << dilationVal[1] << sep;
    // group
    problemOS << "-g " << inShape[iLayoutMap["gi"]] << sep;

  } else if (opType == KernelType::Gemm) { // gemm case
    rock::GemmOp rGemmOp = dyn_cast<rock::GemmOp>(gemmOp);
    bool isScaledGemm =
        rGemmOp.getScaleA() != nullptr && rGemmOp.getScaleB() != nullptr;
    // Please keep these in sync with mlir/utils/performance/perfRunner.py
    // Data type
    problemOS << "-t ";
    Type elemTypeA = gemmIF.getAType(), elemTypeB = gemmIF.getBType();
    if (elemTypeA.isF32() && elemTypeB.isF32()) {
      problemOS << "f32";
    } else if (elemTypeA.isF16() && elemTypeB.isF16()) {
      problemOS << "f16";
    } else if (elemTypeA.isBF16() && elemTypeB.isBF16()) {
      problemOS << "bf16";
    } else if (elemTypeA.isInteger(8) && elemTypeB.isInteger(8)) {
      problemOS << "i8";
    } else if (isa<Float4E2M1FNType>(elemTypeA) &&
               isa<Float4E2M1FNType>(elemTypeB)) {
      problemOS << "f4E2M1FN";
    } else {
      auto aString = f8TypeStr(elemTypeA);
      auto bString = f8TypeStr(elemTypeB);
      if (aString && bString)
        problemOS << llvm::formatv("{0}_{1}", *aString, *bString);
      else
        return failure();
    }

    // Output datatype
    Type outType = gemmIF->getResult(0).getType();
    Type elemTypeC;
    if (auto shapedType = dyn_cast<ShapedType>(outType))
      elemTypeC = shapedType.getElementType();
    else
      elemTypeC = outType;
    problemOS << " -out_datatype ";
    auto outStr = f8TypeStr(elemTypeC);
    if (outStr)
      problemOS << *outStr << sep;
    else
      problemOS << elemTypeC << sep;

    // TransA
    problemOS << "-transA ";
    if (rGemmOp.getATransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // TransB
    problemOS << "-transB ";
    if (rGemmOp.getBTransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    // TransO
    problemOS << "-transO ";
    if (rGemmOp.getOTransposed())
      problemOS << "true ";
    else
      problemOS << "false ";

    if (isScaledGemm) {
      problemOS << "-scaledGemm" << sep;
      auto scaleA = rGemmOp.getScaleA();
      auto scaleB = rGemmOp.getScaleB();
      problemOS << "-scale_a_dtype ";
      auto scaleAElemType = scaleA.getType().getElementType();
      auto scaleBElemType = scaleB.getType().getElementType();
      if (scaleAElemType.isF32()) {
        problemOS << "f32";
      } else if (isa<Float8E8M0FNUType>(scaleAElemType)) {
        problemOS << "f8E8M0FNU";
      } else {
        llvm_unreachable("Unsupported scale A element type");
      }
      problemOS << sep;
      problemOS << "-scale_b_dtype ";
      if (scaleBElemType.isF32()) {
        problemOS << "f32";
      } else if (isa<Float8E8M0FNUType>(scaleBElemType)) {
        problemOS << "f8E8M0FNU";
      } else {
        llvm_unreachable("Unsupported scale B element type");
      }
      problemOS << sep;
      problemOS << "-transScaleA" << sep;
      if (rGemmOp.getAScaleTransposed()) {
        problemOS << "true" << sep;
      } else {
        problemOS << "false" << sep;
      }
      problemOS << "-transScaleB" << sep;
      if (rGemmOp.getBScaleTransposed()) {
        problemOS << "true" << sep;
      } else {
        problemOS << "false" << sep;
      }
    }

    // Gemmsize G/M/N/K
    problemOS << "-g " << gemmIF.getGemmSize().g << sep;
    problemOS << "-m " << gemmIF.getGemmSize().m << sep;
    problemOS << "-n " << gemmIF.getGemmSize().n << sep;
    problemOS << "-k " << gemmIF.getGemmSize().k << sep;
  } else {
    // Unknown op type, unreachable.
    return failure();
  }

  while (out.back() == sep) {
    // remove trailing whitespace
    out.pop_back();
  }

  return success();
}

// Suppose to return the structure of the given problem to tune, currently
// combines the string representation of the selected field of the primary
// operation. String format of the problem will not be required by the DB,
// since it can store each field separately.
// Currently serialize the problem in MIOpenDriver command friendly format
LogicalResult getTuningProblemStr(ModuleOp mod, SmallVectorImpl<char> &out) {
  {
    rock::RockGemmWrapperInterface gemmIF;
    WalkResult findPrimary =
        mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
          gemmIF = op;
          return WalkResult::interrupt();
        });
    if (findPrimary.wasInterrupted())
      return getTuningProblemStr(gemmIF, out);
  }
  {
    rock::RockGemmGemmWrapperInterface gemmGemmOp;
    WalkResult findGemmGemm =
        mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
          gemmGemmOp = op;
          return WalkResult::interrupt();
        });
    if (findGemmGemm.wasInterrupted())
      return getTuningProblemStr(gemmGemmOp, out);
  }
  return failure();
}

bool tuningTableUpdate(TuningTable *perfTable, StringRef problem,
                       StringRef perfConfig, float time) {
  if (problem.empty())
    return false;
  llvm::sys::SmartScopedWriter<true> guard(perfTable->lock);
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    if (entry.second <= time) {
      return false;
    }
  }
  perfTable->tuningMap[problem] = std::make_pair(perfConfig, time);
  return true;
}

LogicalResult tuningTableLookup(TuningTable *perfTable, ModuleOp &mod,
                                SmallVectorImpl<char> &out) {
  SmallString<ROCMLIR_TUNING_KEY_BUFSZ> problem;
  if (failed(getTuningProblemStr(mod, problem)))
    return failure();
  llvm::sys::SmartScopedReader<true> guard(perfTable->lock);
  auto search = perfTable->tuningMap.find(problem);
  if (search != perfTable->tuningMap.end()) {
    auto entry = perfTable->tuningMap[problem];
    out.assign(entry.first);
    return success();
  }
  return failure();
}

int64_t retrieveSplitKValue(StringAttr perfConfig) {
  auto gemmGemmPerfConfig = GemmGemmParamsAttr::get(perfConfig);
  if (gemmGemmPerfConfig)
    return gemmGemmPerfConfig.getSplitKFactor();

  auto params = GemmParamsAttr::get(perfConfig);
  return params ? params.getSplitKFactor() : 1;
}

bool isSplitKRequested(StringAttr perfConfig) {
  return retrieveSplitKValue(perfConfig) > 1;
}

bool isSplitKRequested(ModuleOp mod, StringRef perfConfig) {
  auto perfConfigAttr = StringAttr::get(mod->getContext(), perfConfig);
  WalkResult walkResult = mod.walk([&](Operation *op) -> WalkResult {
    if (isa<RockGemmWrapperInterface, RockGemmGemmWrapperInterface>(op) &&
        isSplitKRequested(perfConfigAttr))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return walkResult.wasInterrupted();
}

RocmlirSplitKSelectionLikelihood isSplitKFaster(int64_t gDim, int64_t mDim,
                                                int64_t nDim, int64_t kDim,
                                                int64_t numCUs) {
  return RocmlirSplitKSelectionLikelihood::never;
}

bool isModuleFusible(ModuleOp module, StringRef perfConfig) {
  bool fusible = succeeded(rock::testFusionLegalityBwdDataConv(module));
  if (!rock::isSplitKRequested(module, perfConfig))
    return fusible;
  return fusible && succeeded(rock::testFusionLegalitySplitK(module));
}

} // namespace rock
} // namespace mlir
