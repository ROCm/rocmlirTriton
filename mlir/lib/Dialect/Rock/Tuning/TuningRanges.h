//===- TuningRanges.h - Values a perf config parameter may take -----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What each tunable parameter of a perf config may be set to, and which
// combinations of those values a kernel can actually be built from. Defined in
// TuningRanges.cpp and shared by the two consumers of that space:
// RockTuningImpl.cpp, which enumerates it into a list of configs, and
// TuningSearch.cpp, which offers it to a search as axes it can move along (see
// `TuningParamAxes`). The two must agree about what the space is, so neither
// gets its own copy of these.
//
// Internal to the Rock tuning library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TUNING_TUNINGRANGES_H
#define MLIR_LIB_DIALECT_ROCK_TUNING_TUNINGRANGES_H

#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <vector>

namespace mlir {
namespace rock {

/// Largest per-block K tile size we tune for. Tiles above it waste LDS and are
/// ~never optimal, so no ladder here reaches past it whatever the problem's K.
constexpr uint32_t kMaxKPerBlock = 512;

/// The values of each independently tunable parameter, in the order
/// `createGemmTuningRangeBF` loops over them: mPerBlock, nPerBlock, kpack,
/// numWaves, matrixInstrNonkdim, numStages, wavesPerEU, gridGroupSize, numCTAs.
/// `kPerBlock` and `splitKFactor` are absent because they follow from the tile;
/// see `computeKPerBlock` and `computeOptimalSplitKFactors`
/// (RockTuningImpl.cpp).
std::vector<std::vector<uint32_t>> getRangeGemm(RockGemmWrapperInterface gemmOp,
                                                int64_t waveSize,
                                                int64_t maxWavesPerEU,
                                                TuningParamSetKind kind);

/// As `getRangeGemm`, in the order `createGemmGemmTuningRangeBF` loops over
/// them: mPerBlockG0, nPerBlockG0, nPerBlockG1, kPerBlock, kpack, numWaves,
/// matrixInstrNonkdim, numStages, wavesPerEU, gridGroupSize, numCTAs. Only
/// `splitKFactor` is derived here.
std::vector<std::vector<uint32_t>>
getRangeGemmGemm(RockGemmGemmWrapperInterface gemmGemmOp, int64_t waveSize,
                 TuningParamSetKind kind);

/// The K tiles worth trying for one (m, n) tile of `gemmOp`.
std::vector<uint32_t> computeKPerBlock(RockGemmWrapperInterface gemmOp,
                                       TuningParamSetKind kind,
                                       uint32_t mPerBlock, uint32_t nPerBlock);

/// Whether a tile's lowered index/mask tensors would exceed Triton's per-tensor
/// element cap, which no kernel using that tile can get past.
bool exceedsTritonTensorCap(uint32_t mPerBlock, uint32_t nPerBlock,
                            uint32_t kPerBlock);

/// Whether an M/N pair is one the non-accel (FMA) path is too slow to compile
/// for. `getRangeGemm` keeps both tiles on their own axis, so the pair can only
/// be rejected once the two are combined.
bool isOverwideNonAccelMNPair(uint32_t mPerBlock, uint32_t nPerBlock);

} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TUNING_TUNINGRANGES_H
