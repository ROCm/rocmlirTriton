//===- GridwiseGemmGemmParams.h - MLIR tuning parameter generation --------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation for gemm+gemm (attn) ops
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
#define MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {

class PopulateParamsGemmGemm {
public:
  static std::vector<GemmGemmParamsAttr>
  getTuningParameters(OpBuilder &b, RockGemmGemmWrapperInterface op);

  // Same as above, but for callers without a concrete op (e.g. rocmlir-gen
  // computing default block sizes before the kernel exists).
  static std::vector<GemmGemmParamsAttr>
  getTuningParameters(OpBuilder &b, StringRef arch, KernelType kernelType,
                      Type elementType);

  static FailureOr<std::pair<GemmParamsAttr, GemmParamsAttr>>
  getGemmParams(OpBuilder &b, RockGemmGemmWrapperInterface op,
                     GemmGemmParamsAttr params);

  static FailureOr<GemmGemmParamsAttr>
  obtainTuningParameters(OpBuilder &b, RockGemmGemmWrapperInterface op);

  static GemmParamsAttr getGemm0Params(OpBuilder &b, GemmGemmParamsAttr params);

  static GemmParamsAttr getGemm1Params(OpBuilder &b,
                                       RockGemmGemmWrapperInterface op,
                                       GemmGemmParamsAttr params);

private:
#define GemmGemm_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_DECLARATIONS_GEN

  friend class ParamLookupTable<GemmGemmParamsAttr>;
};

/// Pure-arithmetic core of the gemm+gemm/attention peak-LDS footprint.
/// Covers the three concurrently-live tiles: gemm0's A (Q) and B (K) tiles
/// plus gemm1's B (V) tile, the latter sized by `gemm1NPerBlock`; the gemm0
/// result P stays in registers. `aBits`/`bBits`/`cBits` are the A (Q), B (K),
/// and C (V) element bit widths. The footprint is accumulated in bits so
/// sub-byte element types are not rounded per element, then converted to bytes
/// once and scaled by the pipeline stage count. This is the single source of
/// truth shared by `isGemmGemmParamsConservativelyApplicable` and
/// `estimateGemmGemmLdsBytes`.
static inline int64_t gemmGemmLdsBytes(GemmGemmParamsAttr p,
                                       int64_t gemm1NPerBlock, int64_t aBits,
                                       int64_t bBits, int64_t cBits) {
  int64_t totalBits = (p.getMPerBlockG0() * p.getKPerBlock() * aBits) +
                      (p.getNPerBlockG0() * p.getKPerBlock() * bBits) +
                      (p.getNPerBlockG0() * gemm1NPerBlock * cBits);
  return llvm::divideCeil(totalBits, static_cast<int64_t>(8)) *
         p.getNumStages();
}

/// Gemm+gemm/attention counterpart of isGemmParamsConservativelyApplicable.
/// LDS bound covers gemm0's A+B tiles (Q, K) plus gemm1's B tile (V); P stays
/// in registers. The gemm1 tile uses `getGemm1Params(...).getNPerBlock()` so
/// there's a single source of truth.
inline bool isGemmGemmParamsConservativelyApplicable(
    OpBuilder &b, GemmGemmParamsAttr p, Type aElemType, Type bElemType,
    Type cElemType, StringRef arch, RockGemmGemmWrapperInterface op) {
  if (p.getKpack() != 1 || p.getSplitKFactor() != 1 || p.getNumCTAs() != 1)
    return false;
  int64_t gemm1NPerBlock =
      PopulateParamsGemmGemm::getGemm1Params(b, op, p).getNPerBlock();
  int64_t bytes = gemmGemmLdsBytes(
      p, gemm1NPerBlock, aElemType.getIntOrFloatBitWidth(),
      bElemType.getIntOrFloatBitWidth(), cElemType.getIntOrFloatBitWidth());
  return bytes <= getLDSSize(arch);
}

/// Default config used as a guaranteed-applicable fallback when no entry in
/// the quick-tuning table satisfies isGemmGemmParamsConservativelyApplicable.
inline GemmGemmParamsAttr
getConservativeDefaultGemmGemmParams(MLIRContext *ctx) {
  return GemmGemmParamsAttr::get(ctx,
                                 /*mPerBlockG0=*/32, /*nPerBlockG0=*/32,
                                 /*kPerBlock=*/32, /*kpack=*/1, /*numCTAs=*/1,
                                 /*numWaves=*/4, /*matrixInstrNonkdim=*/0,
                                 /*splitKFactor=*/1, /*numStages=*/1,
                                 /*wavesPerEU=*/0, /*gridGroupSize=*/0,
                                 /*useAsyncCopy=*/kKnobDefault,
                                 /*useBlockPingpong=*/kKnobDefault,
                                 /*useInThreadTranspose=*/kKnobDefault,
                                 /*useBufferOps=*/kKnobDefault,
                                 /*useBufferAtomics=*/kKnobDefault,
                                 /*useReductionLayout=*/kKnobDefault);
}

/// Estimate the peak LDS (shared memory) bytes a fused gemm+gemm/attention
/// kernel would use for a problem whose second-gemm output (gemmO) dimension
/// is `gemmO`, evaluated against the conservative default perf config. This is
/// the problem-sizes-only sibling of
/// `isGemmGemmParamsConservativelyApplicable`: it needs neither an op nor a
/// module, so callers such as the MIGraphX CAPI can gate on problem sizes
/// alone. Returns failure for a non-positive `gemmO` or a type that has no
/// integer or floating-point bit width.
inline FailureOr<int64_t> estimateGemmGemmLdsBytes(Type elemType,
                                                   int64_t gemmO) {
  if (gemmO <= 0)
    return failure();
  // LDS usage depends only on element width. Operation-specific type support
  // is validated by the operation or compilation pipeline, not this estimator.
  if (!isa<IntegerType, FloatType>(elemType))
    return failure();
  unsigned bits = elemType.getIntOrFloatBitWidth();

  GemmGemmParamsAttr params =
      getConservativeDefaultGemmGemmParams(elemType.getContext());
  // The second gemm's N-per-block tile covers the power-of-two-padded gemmO,
  // matching getGemm1Params().
  int64_t gemm1NPerBlock = llvm::PowerOf2Ceil(gemmO);
  return gemmGemmLdsBytes(params, gemm1NPerBlock, static_cast<int64_t>(bits),
                          static_cast<int64_t>(bits),
                          static_cast<int64_t>(bits));
}

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
