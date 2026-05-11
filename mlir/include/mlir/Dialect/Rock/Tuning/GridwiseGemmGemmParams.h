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

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/IR/Attributes.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {

class PopulateParamsGemmGemm {
public:
  static std::vector<GemmGemmParamsAttr>
  getTuningParameters(OpBuilder &b, RockGemmGemmWrapperInterface op);

  static FailureOr<std::pair<GemmParamsAttr, GemmParamsAttr>>
  getGemmParams(OpBuilder &b, RockGemmGemmWrapperInterface op,
                     GemmGemmParamsAttr params);

  static FailureOr<GemmGemmParamsAttr>
  obtainTuningParameters(OpBuilder &b, RockGemmGemmWrapperInterface op);

  /// Trailing N (gemmO) dimension of `op`'s C operand, accounting for the
  /// optional G dim and the transposed-C flag. This is the N dimension of
  /// the second GEMM (the one whose tile size feeds into gemm1NPerBlock).
  static int64_t getGemm1N(RockGemmGemmWrapperInterface op);

  /// Per-block K tile size for the second GEMM, derived from `params`. Pinned
  /// to `nPerBlockG0` because the second GEMM's K dim is the first GEMM's N
  /// dim (gemm0's per-block N tile becomes gemm1's per-block K tile). Kept
  /// here so the lowering (`getGemm1Params`) and any pre-lowering analyses
  /// (e.g. LDS budgeting in `fusionUtils`) share a single source of truth.
  static int64_t getGemm1KPerBlock(GemmGemmParamsAttr params);

protected:
  static GemmGemmParamsAttr
  deserializePerfConfig(OpBuilder &b, RockGemmGemmWrapperInterface op,
                        StringRef config);

  static std::vector<GemmGemmParamsAttr>
  deserializePerfConfigs(OpBuilder &b, RockGemmGemmWrapperInterface op,
                         ArrayRef<StringRef> configs);

  static GemmParamsAttr getGemm0Params(OpBuilder &b,
                                            GemmGemmParamsAttr params);

  static GemmParamsAttr getGemm1Params(OpBuilder &b,
                                       RockGemmGemmWrapperInterface op,
                                       GemmGemmParamsAttr params);

private:
#define GemmGemm_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_DECLARATIONS_GEN

  friend class ParamLookupTable<GemmGemmParamsAttr>;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_GEMM_PARAMS_H
