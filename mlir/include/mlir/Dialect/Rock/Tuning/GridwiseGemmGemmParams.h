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
#include "mlir/IR/Attributes.h"
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

/// Per-workgroup LDS footprint for the gemm+gemm/attention kernel: gemm0's A
/// and B tiles (Q, K) plus gemm1's B tile (V) for the given gemm1NPerBlock,
/// scaled by numStages. P stays in registers. V's bitwidth is approximated as
/// `bElemType` (true for plain attention; conservative when V is wider).
inline int64_t computeGemmGemmLdsBytes(GemmGemmParamsAttr p, Type aElemType,
                                       Type bElemType,
                                       int64_t gemm1NPerBlock) {
  int64_t aBits = aElemType.getIntOrFloatBitWidth();
  int64_t bBits = bElemType.getIntOrFloatBitWidth();
  int64_t totalBits = (p.getMPerBlockG0() * p.getKPerBlock() * aBits) +
                      (p.getNPerBlockG0() * p.getKPerBlock() * bBits) +
                      (p.getNPerBlockG0() * gemm1NPerBlock * bBits);
  return llvm::divideCeil(totalBits, static_cast<int64_t>(8)) *
         p.getNumStages();
}

/// Gemm+gemm/attention counterpart of isGemmParamsConservativelyApplicable.
/// The gemm1 tile size comes from `getGemm1Params(...).getNPerBlock()` so
/// there's a single source of truth, and the LDS formula is shared via
/// `computeGemmGemmLdsBytes()`.
inline bool isGemmGemmParamsConservativelyApplicable(
    OpBuilder &b, GemmGemmParamsAttr p, Type aElemType, Type bElemType,
    StringRef arch, RockGemmGemmWrapperInterface op) {
  if (p.getKpack() != 1 || p.getSplitKFactor() != 1 || p.getNumCTAs() != 1)
    return false;
  int64_t gemm1NPerBlock =
      PopulateParamsGemmGemm::getGemm1Params(b, op, p).getNPerBlock();
  int64_t bytes =
      computeGemmGemmLdsBytes(p, aElemType, bElemType, gemm1NPerBlock);
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
                                 /*wavesPerEU=*/0, /*gridGroupSize=*/0);
}

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
