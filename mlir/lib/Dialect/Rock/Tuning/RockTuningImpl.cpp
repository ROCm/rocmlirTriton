//===- RockTuningImpl.cpp - tuning API implementation ----*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file implements the tuning interfaces
//
//===----------------------------------------------------------------------===//

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cstdint>
#include <random>

namespace mlir {
namespace rock {

// Which GEMM output dimension a per-block tile list is being built for. Unlike
// `GemmDimension` (GridwiseGemmParams.h), which collapses M and N into a single
// `MorN`, this distinguishes the two so a tile list can be tailored per axis.
enum class GemmMNDim { M, N };

// Largest per-block M/N tile size we tune for. Also acts as the threshold below
// which a small dimension is covered by a single tightly-fitting tile.
#define MAX_MN_PER_BLOCK 256

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
  auto arch = rock::getArchValue(op);
  bool hasAcceleration = false;
  if (auto gemmOp = dyn_cast<RockGemmWrapperInterface>(op))
    hasAcceleration = rock::hasAccel(arch, gemmOp);
  else if (auto gemmGemmOp = dyn_cast<RockGemmGemmWrapperInterface>(op))
    hasAcceleration = rock::hasAccel(arch, gemmGemmOp);
  else
    llvm_unreachable("Unexpected op");

  std::vector<uint32_t> dPerBlockList;
  if (hasAcceleration) {
    for (uint32_t dPerBlock = 16; dPerBlock <= MAX_MN_PER_BLOCK; dPerBlock *= 2)
      dPerBlockList.push_back(dPerBlock);
  } else {
    // non-accel
    dPerBlockList = {32, 64, 128, 256};
  }

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
// work.
static void capKPerBlockByK(std::vector<uint32_t> &kPerBlockList, int64_t k) {
  assert(k > 0 && !kPerBlockList.empty() &&
         "capKPerBlockByK expects a positive K and a non-empty candidate list");
  uint32_t cap = static_cast<uint32_t>(llvm::PowerOf2Ceil(k));
  uint32_t maxCandidateK = *llvm::max_element(kPerBlockList);
  llvm::erase_if(kPerBlockList, [&](uint32_t v) { return v > cap; });
  if (!llvm::is_contained(kPerBlockList, cap) && cap < maxCandidateK)
    kPerBlockList.push_back(cap);
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

  std::vector<uint32_t> kPerBlockMFMA =
      kind == TuningParamSetKind::Exhaustive
          ? std::vector<uint32_t>{16, 32, 64, 128, 256, 512}
          : std::vector<uint32_t>{16, 32, 64, 128};

  std::vector<uint32_t> kPerBlockWMMA =
      kind == TuningParamSetKind::Exhaustive
          ? std::vector<uint32_t>{32, 64, 128, 256}
          : std::vector<uint32_t>{32, 64};

  // Cap the K tiles by the actual K dimension so we don't tune (and pad) K
  // tiles that are far larger than the problem's K.
  int64_t gemmK = gemmOp.getGemmSize().k;
  capKPerBlockByK(kPerBlockMFMA, gemmK);
  capKPerBlockByK(kPerBlockWMMA, gemmK);

  std::vector<uint32_t> numCTAsList;
  for (uint32_t n = 1; n <= rock::getMaxNumCTAs(arch); n *= 2)
    numCTAsList.push_back(n);

  std::vector<std::vector<uint32_t>> validRangeMfmaParams = {
      mPerBlock,         // M/block
      nPerBlock,         // N/block
      kPerBlockMFMA,     // K/block
      {1},               // kPackList
      numWavesRange,     // numWaves
      {16, 32},          // matrixInstrNonkdim
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
      kPerBlockWMMA,     // K/block
      {1},               // kPackList
      {4, 8},            // numWaves
      {0},               // matrixInstrNonkdim
      {1, 2, 3},         // numStages
      wavesPerEUList,    // wavesPerEU
      gridGroupSizeList, // gridGroupSize
      numCTAsList        // numCTAs
  };

  // Non-accel (FMA) parameters. M/N tiles reuse computeDPerBlock (which has a
  // dedicated non-accel branch and caps by the actual M/N dimension).
  std::vector<uint32_t> kPerBlockNonAccel = {1, 4, 8, 16};
  capKPerBlockByK(kPerBlockNonAccel, gemmK);
  std::vector<uint32_t> numWavesNonAccel;
  for (uint32_t blockSize : {64u, 128u, 256u}) {
    if (blockSize % waveSize == 0)
      numWavesNonAccel.push_back(blockSize / waveSize);
  }
  assert(!numWavesNonAccel.empty() && "numWavesNonAccel must be non-empty");
  std::vector<std::vector<uint32_t>> validRangeNonAccelParams = {
      mPerBlock,         // M/block
      nPerBlock,         // N/block
      kPerBlockNonAccel, // K/block
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

  std::vector<uint32_t> kPerBlock = {16, 32, 64, 128, 512, 1024, 2048};
  // use the actual K dimension, typically it's 128 for attention
  if (kind != TuningParamSetKind::Exhaustive) {
    kPerBlock = {static_cast<uint32_t>(llvm::PowerOf2Ceil(gemm0K))};
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

  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParamsMFMA = {
      /*gemm0MPerBlock=*/mPerBlock,
      /*gemm0NPerBlock=*/nPerBlock,
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
      kPerBlock,
      /*kPackList=*/{1},
      numWavesRange,
      /*matrixInstrNonkdim=*/{0},
      {1, 2},
      wavesPerEUList,
      gridGroupSizeList,
      numCTAsList};

  // Non-accel path.
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
static void createGemmGemmTuningRangeBF(TuningParamSet *newSpace,
                                        RockGemmGemmWrapperInterface gemmGemmOp,
                                        TuningParamSetKind kind) {
  auto waveSize = rock::getWaveSize(rock::getArchValue(gemmGemmOp));
  const std::vector<std::vector<uint32_t>> validRangeGemmGemmParams =
      getRangeGemmGemm(gemmGemmOp, waveSize, kind);
  // gemm1's N tile is derived, not tuned: gemm1NPerBlock = PowerOf2Ceil(head
  // dim of V) (see PopulateParamsGemmGemm::getGemm1Params). GemmGemmSize::o is
  // that head-dim-of-V, already extracted transpose-aware, so guard gemm1's
  // lowered tensor against the same Triton per-tensor cap as gemm0.
  uint32_t gemm1NPerBlock =
      static_cast<uint32_t>(llvm::PowerOf2Ceil(gemmGemmOp.getGemmGemmSize().o));
  OpBuilder b(gemmGemmOp.getContext());
  for (uint32_t gemm0MPerBlock : validRangeGemmGemmParams[0]) {
    for (uint32_t gemm0NPerBlock : validRangeGemmGemmParams[1]) {
      auto optimalSplitKFactors =
          computeOptimalSplitKFactors(gemmGemmOp, gemm0MPerBlock);

      // gemm1 lowers a gemm0NPerBlock x max(gemm0MPerBlock, gemm1NPerBlock)
      // index/mask tensor (its contraction tile is gemm0NPerBlock). Guard it
      // against the same cap as gemm0; this depends only on the gemm0 M/N
      // tiles, so check it once outside the gemmKPerBlock loop.
      if (exceedsTritonTensorCap(gemm0MPerBlock, gemm1NPerBlock,
                                 gemm0NPerBlock))
        continue;

      for (uint32_t gemmKPerBlock : validRangeGemmGemmParams[2]) {
        // Skip tiles whose lowered index/mask tensors would exceed Triton's
        // per-tensor element cap (see exceedsTritonTensorCap).
        if (exceedsTritonTensorCap(gemm0MPerBlock, gemm0NPerBlock,
                                   gemmKPerBlock))
          continue;
        for (uint32_t gemmKPack : validRangeGemmGemmParams[3]) {
          for (uint32_t numWaves : validRangeGemmGemmParams[4]) {
            for (uint32_t matrixInstrNonkdim : validRangeGemmGemmParams[5]) {
              for (int64_t splitKFactor : optimalSplitKFactors) {
                for (uint32_t numStages : validRangeGemmGemmParams[6]) {
                  for (uint32_t wavesPerEU : validRangeGemmGemmParams[7]) {
                    for (uint32_t gridGroupSize : validRangeGemmGemmParams[8]) {
                      for (uint32_t numCTAs : validRangeGemmGemmParams[9]) {
                        auto gemmGemmParams = GemmGemmParamsAttr::get(
                            gemmGemmOp.getContext(), gemm0MPerBlock,
                            gemm0NPerBlock, gemmKPerBlock, gemmKPack, numCTAs,
                            numWaves, matrixInstrNonkdim, splitKFactor,
                            numStages, wavesPerEU, gridGroupSize,
                            /*useAsyncCopy=*/kKnobDefault,
                            /*useBlockPingpong=*/kKnobDefault,
                            /*useInThreadTranspose=*/kKnobDefault,
                            /*useBufferOps=*/kKnobDefault,
                            /*useBufferAtomics=*/kKnobDefault,
                            /*useReductionLayout=*/kKnobDefault);
                        newSpace->tuningRange.push_back(
                            cast<RockTuningParamAttrInterface>(gemmGemmParams));
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
static void createGemmTuningRangeBF(TuningParamSet *newSpace,
                                    RockGemmWrapperInterface gemmOp,
                                    TuningParamSetKind kind) {
  auto info = PopulateParamsInfo::fromOp(gemmOp);

  int64_t maxWavesPerEU = rock::getMaxWavesPerEU(rock::getArchValue(gemmOp));
  int64_t waveSize = rock::getWaveSize(rock::getArchValue(gemmOp));
  const std::vector<std::vector<uint32_t>> params =
      getRangeGemm(gemmOp, waveSize, maxWavesPerEU, kind);

  auto tuningInfo = std::make_unique<PopulateParams>();

  OpBuilder b(gemmOp.getContext());
  for (uint32_t gemmMPerBlock : params[0]) {
    for (uint32_t gemmNPerBlock : params[1]) {
      for (uint32_t gemmKPerBlock : params[2]) {
        // Skip tiles whose lowered index/mask tensors would exceed Triton's
        // per-tensor element cap; they cannot compile (see
        // exceedsTritonTensorCap).
        if (exceedsTritonTensorCap(gemmMPerBlock, gemmNPerBlock, gemmKPerBlock))
          continue;
        for (uint32_t gemmKPack : params[3]) {
          for (uint32_t numWaves : params[4]) {
            for (uint32_t matrixInstrNonkdim : params[5]) {
              auto optimalSplitKFactors = computeOptimalSplitKFactors(
                  gemmOp, gemmMPerBlock, gemmNPerBlock, gemmKPerBlock);
              for (int64_t splitKFactor : optimalSplitKFactors) {
                for (int64_t numStages : params[6]) {
                  for (int64_t wavesPerEU : params[7]) {
                    for (int64_t gridGroupSize : params[8]) {
                      for (uint32_t numCTAs : params[9]) {
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
                            /*useReductionLayout=*/kKnobDefault);
                        if (kind != TuningParamSetKind::Full ||
                            succeeded(tuningInfo->couldBePerformant(
                                info, gemmParams)))
                          newSpace->tuningRange.push_back(
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
    newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(param));
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
    newSpace->tuningRange.push_back(cast<RockTuningParamAttrInterface>(params));
  }
}

bool needToUpdateBest(rock::TuningParamSetKind kind) {
  switch (kind) {
  case TuningParamSetKind::Quick:
  case TuningParamSetKind::Full:
  case TuningParamSetKind::Exhaustive:
    return false;
  }
  llvm_unreachable("invalid tuning kind");
}

unsigned getNumberOfIterations(TuningParamSetKind kind) {
  switch (kind) {
  case TuningParamSetKind::Quick:
  case TuningParamSetKind::Full:
  case TuningParamSetKind::Exhaustive:
    return 1;
  }
  llvm_unreachable("invalid tuning kind");
}

TuningParamSet *
createTunableParamSpace(ModuleOp mod, TuningParamSetKind kind,
                        rock::TuningParamSpaceSettings &settings) {
  struct TuningParamSet *newSpace;
  newSpace = new TuningParamSet();

  // create range and heuristic
  WalkResult findPrimary =
      mod->walk([&](rock::RockGemmWrapperInterface op) -> WalkResult {
        switch (kind) {
        case TuningParamSetKind::Full:
        case TuningParamSetKind::Exhaustive:
          createGemmTuningRangeBF(newSpace, op, kind);
          break;
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
          break;
        case TuningParamSetKind::Quick:
          createGemmGemmTuningRangeQuick(newSpace, op);
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
        SmallString<64> perfConfig;
        paramEntry->param.getPerfConfigStr(perfConfig);
        StringAttr attr = StringAttr::get(ctx, perfConfig);
        op->setAttr("perf_config", attr);
        return WalkResult::interrupt();
      });
  WalkResult setGemmGemm =
      mod->walk([&](rock::RockGemmGemmWrapperInterface op) -> WalkResult {
        auto *ctx = op.getContext();
        SmallString<64> perfConfig;
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
    // sliding_window_size is optional (KV-cache only); only emit it when set so
    // non-sliding-window problems keep their existing tuning identity.
    if (auto slidingWindowSize = attentionOp.getSlidingWindowSize();
        slidingWindowSize && *slidingWindowSize > 0)
      problemOS << "-sliding_window_size " << *slidingWindowSize << sep;
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
  SmallString<2048> problem;
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
  bool fusible = succeeded(rock::testFusionLegalityReduce(module)) &&
                 succeeded(rock::testFusionLegalityBwdDataConv(module));
  if (!rock::isSplitKRequested(module, perfConfig))
    return fusible;
  return fusible && succeeded(rock::testFusionLegalitySplitK(module));
}

} // namespace rock
} // namespace mlir
