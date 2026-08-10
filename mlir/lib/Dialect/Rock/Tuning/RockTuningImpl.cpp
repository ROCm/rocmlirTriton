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

#include "TuningRanges.h"
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
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <set>

namespace mlir {
namespace rock {

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
                          gemmGemmParams.getPerfConfigStr(
                              newSpace->tuningRange.emplace_back());
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
  // covers configs that differ only in an LDS-irrelevant field (kpack,
  // wavesPerEU, ...). A miss returns an empty set, so an un-populated blacklist
  // is a no-op.
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
                  if (isBlacklisted(ldsBlacklistSet,
                                    {gemmMPerBlock, gemmNPerBlock,
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
                        if (kind != TuningParamSetKind::Full ||
                            succeeded(tuningInfo->couldBePerformant(
                                info, gemmParams)))
                          gemmParams.getPerfConfigStr(
                              newSpace->tuningRange.emplace_back());
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
    param.getPerfConfigStr(newSpace->tuningRange.emplace_back());
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
    params.getPerfConfigStr(newSpace->tuningRange.emplace_back());
  }
}

// Drops the configs the space lists more than once, keeping the first of each.
// The full and exhaustive spaces run the quick list on top of their
// enumeration, which already covers part of it, and a repeat would cost a
// second benchmark of a kernel the tuner has just timed.
static void dropRepeatedConfigs(TuningParamSet *space) {
  llvm::StringSet<> seen;
  llvm::erase_if(space->tuningRange, [&seen](const PerfConfigString &config) {
    return !seen.insert(config).second;
  });
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
  dropRepeatedConfigs(newSpace);
  return newSpace;
}

bool tuningGetParam(TuningParamSet *tuningSpace, unsigned pos,
                    ParamEntry *paramEntry) {
  // out of bound check.
  if (pos >= tuningSpace->tuningRange.size())
    return false;
  paramEntry->param = tuningSpace->tuningRange[pos];
  paramEntry->primaryOpType = tuningSpace->primaryOpType;
  return true;
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

  if (opType == KernelType::Conv ||
      opType == KernelType::ConvBwdData) { // conv cases
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
