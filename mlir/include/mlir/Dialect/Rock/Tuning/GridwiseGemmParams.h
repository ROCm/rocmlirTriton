//===- GridwiseGemmParams.h - MLIR tuning parameter generation ------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter generation for single-gemm ops
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
#define MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <optional>

namespace llvm {
class raw_ostream;
} // end namespace llvm

namespace mlir {
class Type;
namespace rock {
enum class GemmDimension : uint32_t { G = 0, K = 1, MorN = 2 };
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, GemmDimension dim);

// This core function to calculate the required padding amount
// given a gemm size.
std::optional<GemmSize> calculatePadding(int64_t kPerBlock, int64_t mPerBlock,
                                         int64_t nPerBlock,
                                         const GemmSize &gemmSize);

GemmSize calculatePaddedGemmSize(int64_t kPerBlock, int64_t mPerBlock,
                                 int64_t nPerBlock, GemmSize gemmSize);

/// Given a tuning parameter struct, determine how much padding the gemm with
/// a given gemm size requires. Returns None if no padding is needed. The
/// values in the returned gemm context represent the number of 0s that need to
/// be added to the given dimension. The mulBy* arguments multiply the
/// corresponding dimension of the attributes.
std::optional<GemmSize> requiredPadding(Attribute paramsAttr, GemmSize gemmSize,
                                        int64_t mulByKPerBlock = 1,
                                        int64_t mulByMPerBlock = 1,
                                        int64_t mulByNPerBlock = 1);

int64_t obtainBlockSize(int64_t waveSize, GemmParamsAttr params);

/// Build a `ParamsAttr` from `perfConfig`, falling back to `defaults.front()`
/// if `perfConfig` is empty. Fails if both are empty or the perfConfig fails
/// to parse.
template <typename ParamsAttr>
FailureOr<ParamsAttr> materializeTuningParams(OpBuilder &b,
                                              StringRef perfConfig,
                                              ArrayRef<ParamsAttr> defaults);

/// Store information useful for populating perf configurations
struct PopulateParamsInfo {
  GemmSize gemmSize;
  SmallString<32> arch;
  Type gemmAType;
  Type gemmBType;
  KernelType kernelType;
  int64_t batchSize;
  uint32_t numCu;
  bool hasFusedReduction;
  // Block-scaled (MXFP-style) GEMM metadata. `quantBlockSize` is unset for
  // non-scaled ops; the scale element types come from
  // `RockGemmWrapperInterface::getScale{A,B}Type()` and are null for ops that
  // don't carry that operand.
  std::optional<int64_t> quantBlockSize;
  Type aScaleType;
  Type bScaleType;

  PopulateParamsInfo(GemmSize gemmSize, StringRef arch, Type gemmAType,
                     Type gemmBType, KernelType kernelType)
      : gemmSize(gemmSize), arch(arch), gemmAType(gemmAType),
        gemmBType(gemmBType), kernelType(kernelType), hasFusedReduction(false) {
  }

  PopulateParamsInfo(GemmSize gemmSize, StringRef arch, Type gemmAType,
                     Type gemmBType, KernelType kernelType, int64_t batchSize,
                     uint32_t numCu)
      : gemmSize(gemmSize), arch(arch), gemmAType(gemmAType),
        gemmBType(gemmBType), kernelType(kernelType), batchSize(batchSize),
        numCu(numCu), hasFusedReduction(false) {}

  /// Extract the relevant information from a RockGemmWrapperInterface operation
  static PopulateParamsInfo fromOp(RockGemmWrapperInterface op);
};

/// FP4 (4-bit float) operands are upcast in the AMD Triton backend's
/// `Fp4ToFpOp` lowering, which consumes packed values 8 at a time. A
/// `kPerBlock` that is not a multiple of `kFp4KPerBlockMultiple` leaves a
/// partial group and crashes that lowering. The curated FP4 quick-tuning
/// entries always used `kPerBlock >= 64`; this makes the constraint explicit
/// now that FP4 borrows the i8 tuning space (see `ParamLookupTable` fallback).
static constexpr int64_t kFp4KPerBlockMultiple = 64;

/// True when `type` (or its element type) is a 4-bit float, i.e. FP4/MXFP4.
inline bool isFp4ElementType(Type type) {
  Type elem = getElementTypeOrSelf(type);
  return isa<FloatType>(elem) && elem.getIntOrFloatBitWidth() == 4;
}

/// Conservative GEMM applicability: covers the perfconfig-driven
/// `markAsNotApplicable` sites (kpack/splitK/numCTAs constraints + LDS
/// budget over A+B tiles, `numStages`-buffered).
///
/// For block-scaled (MXFP-style) GEMMs, pass `quantBlockSize` and the scale
/// element types. Element types are extracted from whatever is passed (so
/// shaped tensor types coming from
/// `RockGemmWrapperInterface::getScale{A,B}Type()` are handled too). The
/// check then also:
///   - requires `kPerBlock % quantBlockSize == 0` (matches
///     `GridwiseGemmToBlockwise`'s `markAsNotApplicable` site), and
///   - charges per-tile scale storage to the LDS budget.
/// Non-scaled callers leave these arguments at their defaults and behave
/// exactly as before.
inline bool isGemmParamsConservativelyApplicable(
    GemmParamsAttr p, Type aElemType, Type bElemType, StringRef arch,
    std::optional<int64_t> quantBlockSize = std::nullopt,
    Type aScaleType = nullptr, Type bScaleType = nullptr) {
  if (p.getKpack() != 1 || p.getSplitKFactor() != 1 || p.getNumCTAs() != 1)
    return false;
  if (quantBlockSize.has_value() && p.getKPerBlock() % *quantBlockSize != 0)
    return false;
  // Scaled GEMMs require power-of-two tiles
  if (aScaleType || bScaleType) {
    if (!llvm::isPowerOf2_64(p.getMPerBlock()) ||
        !llvm::isPowerOf2_64(p.getNPerBlock()) ||
        !llvm::isPowerOf2_64(p.getKPerBlock()))
      return false;
  }
  // The element-type args may have been threaded from interface methods that
  // hand back shaped types; normalize so `getIntOrFloatBitWidth()` is safe.
  Type aElem = getElementTypeOrSelf(aElemType);
  Type bElem = getElementTypeOrSelf(bElemType);
  // FP4 operands are upcast 8-at-a-time by the AMD Triton `Fp4ToFpOp`
  // lowering; a `kPerBlock` that is not a multiple of `kFp4KPerBlockMultiple`
  // leaves a partial group and crashes that pass.
  if ((isFp4ElementType(aElem) || isFp4ElementType(bElem)) &&
      p.getKPerBlock() % kFp4KPerBlockMultiple != 0)
    return false;
  int64_t totalBits =
      (p.getMPerBlock() * p.getKPerBlock() * aElem.getIntOrFloatBitWidth()) +
      (p.getNPerBlock() * p.getKPerBlock() * bElem.getIntOrFloatBitWidth());
  if (quantBlockSize.has_value() && (aScaleType || bScaleType)) {
    int64_t scaleK = llvm::divideCeil(p.getKPerBlock(), *quantBlockSize);
    if (aScaleType) {
      Type aScaleElem = getElementTypeOrSelf(aScaleType);
      totalBits +=
          p.getMPerBlock() * scaleK * aScaleElem.getIntOrFloatBitWidth();
    }
    if (bScaleType) {
      Type bScaleElem = getElementTypeOrSelf(bScaleType);
      totalBits +=
          p.getNPerBlock() * scaleK * bScaleElem.getIntOrFloatBitWidth();
    }
  }
  int64_t bytes =
      llvm::divideCeil(totalBits, static_cast<int64_t>(8)) * p.getNumStages();
  return bytes <= getLDSSize(arch);
}

/// Default config used as a guaranteed-applicable fallback when no entry in
/// the quick-tuning table satisfies isGemmParamsConservativelyApplicable.
///
/// When `quantBlockSize` is provided, `kPerBlock` is rounded up to a multiple
/// of it so the default also satisfies the divisibility constraint enforced
/// in `GridwiseGemmToBlockwise`.
inline GemmParamsAttr getConservativeDefaultGemmParams(
    MLIRContext *ctx, std::optional<int64_t> quantBlockSize = std::nullopt,
    Type aElemType = nullptr, Type bElemType = nullptr) {
  int64_t kPerBlock = 32;
  if (quantBlockSize.has_value() && *quantBlockSize > 0)
    kPerBlock = llvm::alignTo(kPerBlock, *quantBlockSize);
  // Keep the guaranteed-applicable fallback valid for FP4 (see
  // `isGemmParamsConservativelyApplicable` and `kFp4KPerBlockMultiple`).
  if ((aElemType && isFp4ElementType(aElemType)) ||
      (bElemType && isFp4ElementType(bElemType)))
    kPerBlock = llvm::alignTo(kPerBlock, kFp4KPerBlockMultiple);
  return GemmParamsAttr::get(ctx,
                             /*mPerBlock=*/32, /*nPerBlock=*/32,
                             /*kPerBlock=*/kPerBlock, /*kpack=*/1,
                             /*numCTAs=*/1, /*numWaves=*/4,
                             /*matrixInstrNonkdim=*/0,
                             /*splitKFactor=*/1, /*numStages=*/1,
                             /*wavesPerEU=*/0, /*gridGroupSize=*/0,
                             /*useAsyncCopy=*/kKnobDefault,
                             /*useBlockPingpong=*/kKnobDefault,
                             /*useInThreadTranspose=*/kKnobDefault,
                             /*useBufferOps=*/kKnobDefault,
                             /*useBufferAtomics=*/kKnobDefault,
                             /*useReductionLayout=*/kKnobDefault,
                             /*useOptimizeEpilogue=*/kKnobDefault,
                             /*useBf16x3ForF32=*/kKnobDefault);
}

/// Bump the first param matching `isApplicable` to the front, preserving the
/// relative order of the rest. Used by skip-benchmarking consumers that pick
/// `front()` (e.g. MIGraphX with `MIGRAPHX_SKIP_BENCHMARKING`).
template <typename ParamAttrType>
std::vector<ParamAttrType>
orderParams(ArrayRef<ParamAttrType> params,
            llvm::function_ref<bool(ParamAttrType)> isApplicable) {
  std::vector<ParamAttrType> ordered(params.begin(), params.end());
  auto it = std::find_if(ordered.begin(), ordered.end(), isApplicable);
  if (it != ordered.end() && it != ordered.begin())
    std::rotate(ordered.begin(), it, std::next(it));
  return ordered;
}

template <typename ParamAttrType>
class BasePopulateParams {
public:
  // Succeed if `params` should be included in a "full" tuning space that
  // excludes those known to not yield good performance on the problem described
  // in `info`. This function uses hardcoded heuristics.
  virtual LogicalResult couldBePerformant(const PopulateParamsInfo &info,
                                          ParamAttrType params) = 0;

  virtual ~BasePopulateParams() {}
};

//
// Data holder for static tuning parameter arrays from generated .inc file.
// Used by ParamLookupTable.
//
struct PopulateParamsGemm {
#define Gemm_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Gemm_DECLARATIONS_GEN

  friend class ParamLookupTable<GemmParamsAttr>;
};

//
// Tuning-parameter interface for single-gemm ops.
//
class PopulateParams : public BasePopulateParams<GemmParamsAttr> {
public:
  FailureOr<GemmParamsAttr> obtainTuningParameters(OpBuilder &b,
                                                   RockGemmWrapperInterface op);

  FailureOr<GemmParamsAttr>
  obtainTuningParameters(OpBuilder &b, const PopulateParamsInfo &info,
                         const StringRef perfConfig);

  int64_t calculatePaddingAmount(GemmParamsAttr params,
                                 const GemmSize &gemmSize) const;

  // Return the set of heuristic tuning parameters for the given opType, data
  // types, and architecture. Pass `quantBlockSize` / `aScaleType` /
  // `bScaleType` for block-scaled (MXFP-style) GEMMs so the applicability
  // check accounts for scale-tile LDS use and the `kPerBlock %
  // quantBlockSize == 0` constraint.
  std::vector<GemmParamsAttr> getTuningParameters(
      OpBuilder &b, KernelType opType, Type dataTypeA, Type dataTypeB,
      StringRef arch, std::optional<int64_t> quantBlockSize = std::nullopt,
      Type aScaleType = nullptr, Type bScaleType = nullptr) const;

  LogicalResult couldBePerformant(const PopulateParamsInfo &info,
                                  GemmParamsAttr params) override;

private:
  LogicalResult specificCouldBePerformant(GemmParamsAttr params, Type dataTypeA,
                                          Type dataTypeB, StringRef arch);
};

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_GRIDWISE_GEMM_PARAMS_H
