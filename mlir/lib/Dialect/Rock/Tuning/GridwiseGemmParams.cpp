#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir-c/Dialect/Rock.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <memory>

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

llvm::raw_ostream &mlir::rock::operator<<(llvm::raw_ostream &os,
                                          GemmDimension dim) {
  switch (dim) {
  case GemmDimension::G:
    return os << "GemmDimmension::G";
  case GemmDimension::K:
    return os << "GemmDimension::K";
  case GemmDimension::MorN:
    return os << "GemmDimension::MorN";
  }
  return os;
}

/// Static data for tuning parameters (used by ParamLookupTable).
// clang-format off
#define Gemm_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Gemm_DEFINITIONS_GEN
// clang-format on

PopulateParamsInfo PopulateParamsInfo::fromOp(RockGemmWrapperInterface op) {
  PopulateParamsInfo info{op.getGemmSize(), rock::getArchValue(op),
                          op.getAType(), op.getBType(), op.getKernelType()};

  if (auto convOp = dyn_cast<ConvBwdWeightOp>(*op)) {
    auto convDims = ConvolutionDims::fromOp(op);
    info.numCu = rock::getNumCUValue(convOp);
    info.batchSize = convDims.n;
  }
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  WalkResult wRes = func.walk(
      [&](ReduceOp rOp) -> WalkResult { return WalkResult::interrupt(); });
  info.hasFusedReduction = wRes.wasInterrupted();

  // Block-scaled GEMM metadata: `quantBlockSize` lives on `GemmOp`, scale
  // element types come from the interface. `getScale{A,B}Type` returns the
  // shaped tensor type (or null if no scale operand), so peel off the
  // element type before storing.
  if (auto gemmOp = dyn_cast<GemmOp>(*op))
    info.quantBlockSize = gemmOp.getQuantBlockSize();
  if (Type aScaleTy = op.getScaleAType())
    info.aScaleType = getElementTypeOrSelf(aScaleTy);
  if (Type bScaleTy = op.getScaleBType())
    info.bScaleType = getElementTypeOrSelf(bScaleTy);
  return info;
}

std::optional<GemmSize> mlir::rock::calculatePadding(int64_t kPerBlock,
                                                     int64_t mPerBlock,
                                                     int64_t nPerBlock,
                                                     const GemmSize &gemmSize) {
  int64_t kExtra = llvm::alignTo(gemmSize.k, kPerBlock) - gemmSize.k;
  int64_t mExtra = llvm::alignTo(gemmSize.m, mPerBlock) - gemmSize.m;
  int64_t nExtra = llvm::alignTo(gemmSize.n, nPerBlock) - gemmSize.n;
  if (mExtra == 0 && kExtra == 0 && nExtra == 0)
    return std::nullopt;
  return GemmSize(0, mExtra, kExtra, nExtra);
}

GemmSize mlir::rock::calculatePaddedGemmSize(int64_t kPerBlock,
                                             int64_t mPerBlock,
                                             int64_t nPerBlock,
                                             GemmSize gemmSize) {
  auto gemmExtraPad =
      calculatePadding(kPerBlock, mPerBlock, nPerBlock, gemmSize);

  if (gemmExtraPad.has_value()) {
    gemmSize.m += gemmExtraPad->m;
    gemmSize.k += gemmExtraPad->k;
    gemmSize.n += gemmExtraPad->n;
  }
  return gemmSize;
}

std::optional<GemmSize> mlir::rock::requiredPadding(Attribute paramsAttr,
                                                    GemmSize gemmSize,
                                                    int64_t mulByKPerBlock,
                                                    int64_t mulByMPerBlock,
                                                    int64_t mulByNPerBlock) {
  int64_t kPerBlock, mPerBlock, nPerBlock;
  if (auto params = dyn_cast<GemmParamsAttr>(paramsAttr)) {
    kPerBlock = params.getKPerBlock();
    mPerBlock = params.getMPerBlock();
    nPerBlock = params.getNPerBlock();
  } else {
    llvm_unreachable("The tuning parameters are general or xdlops");
  }
  return calculatePadding(kPerBlock * mulByKPerBlock,
                          mPerBlock * mulByMPerBlock,
                          nPerBlock * mulByNPerBlock, gemmSize);
}

int64_t mlir::rock::obtainBlockSize(int64_t waveSize, GemmParamsAttr params) {
  return waveSize * params.getNumWaves();
}

template <typename ParamsAttr>
FailureOr<ParamsAttr>
mlir::rock::materializeTuningParams(OpBuilder &b, StringRef perfConfig,
                                    ArrayRef<ParamsAttr> defaults) {
  StringAttr perfConfigAttr;
  if (!perfConfig.empty()) {
    perfConfigAttr = b.getStringAttr(perfConfig);
  } else {
    if (defaults.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Quick tuning list is empty\n");
      return failure();
    }
    ParamsAttr validParams = defaults.front();
    LLVM_DEBUG(llvm::dbgs() << validParams << "\n");
    SmallVector<char, ROCMLIR_TUNING_PARAM_STRING_BUFSZ> buf;
    validParams.getPerfConfigStr(buf);
    perfConfigAttr = b.getStringAttr(buf);
  }
  ParamsAttr params = ParamsAttr::get(perfConfigAttr);
  if (!params) {
    LLVM_DEBUG(llvm::dbgs()
               << "Invalid perfConfig: " << perfConfigAttr << "\n");
    return failure();
  }
  return params;
}

// Explicit instantiations. Add new instantiations here if you need a new
// ParamsAttr type to use this helper.
template FailureOr<GemmParamsAttr>
mlir::rock::materializeTuningParams<GemmParamsAttr>(OpBuilder &, StringRef,
                                                    ArrayRef<GemmParamsAttr>);
template FailureOr<GemmGemmParamsAttr>
mlir::rock::materializeTuningParams<GemmGemmParamsAttr>(
    OpBuilder &, StringRef, ArrayRef<GemmGemmParamsAttr>);

static LogicalResult couldFusedReductionBePerformant(const GemmSize &gemmSize,
                                                     int64_t mPerBlock,
                                                     int64_t nPerBlock) {
  // 16 is practically lowest m in MFMAs/WMMAs
  // that could be performant. If the gemm sizes
  // are not divisible by that, then we definitely
  // need padding. Therefore, it can't use blockwise
  // reductions.

  // Thus, it becomes a competition among
  // atomic_store based reduction kernels.
  // So basically, all configs could be performant relative to each other.
  if (gemmSize.m % 16 != 0) {
    return success();
  }
  if (gemmSize.n % 16 != 0) {
    return success();
  }
  // We can skip knowing that dPerBlock=16
  // is there on the tuning space that should
  // be faster than anyone that use m or n
  // padding.
  if (gemmSize.m % mPerBlock != 0) {
    return failure();
  }
  if (gemmSize.n % nPerBlock != 0) {
    return failure();
  }
  return success();
}

static int64_t calculatePaddingComplexity(const GemmSize &paddingAmount,
                                          const GemmSize &gemmSize) {
  int64_t nonPaddedComplexity = gemmSize.m * gemmSize.k * gemmSize.n;
  int64_t paddedComplexity = (gemmSize.m + paddingAmount.m) *
                             (gemmSize.k + paddingAmount.k) *
                             (gemmSize.n + paddingAmount.n);
  return paddedComplexity - nonPaddedComplexity;
}

int64_t PopulateParams::calculatePaddingAmount(GemmParamsAttr params,
                                               const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.getKPerBlock(), params.getMPerBlock(),
                       params.getNPerBlock(), gemmSize);
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

LogicalResult PopulateParams::couldBePerformant(const PopulateParamsInfo &info,
                                                GemmParamsAttr params) {
  if (info.hasFusedReduction) {
    return couldFusedReductionBePerformant(info.gemmSize, params.getMPerBlock(),
                                           params.getNPerBlock());
  }

  return specificCouldBePerformant(params, info.gemmAType, info.gemmBType,
                                   info.arch);
}

FailureOr<GemmParamsAttr> PopulateParams::obtainTuningParameters(
    OpBuilder &b, const PopulateParamsInfo &info, const StringRef perfConfig) {
  // Under two scenarios can we receive a non-empty perfConfig:
  // 1. This is tuning mode
  // 2. This is running mode and we have succeeded with a perfdb load.
  // Otherwise we fall back to the first entry of the quick-tuning list, which
  // `getTuningParameters` already reorders so that the first conservatively-
  // applicable config (LDS budget, kpack/splitK/numCTAs constraints, plus
  // block-scaling divisibility/LDS for scaled ops) is up front.
  return materializeTuningParams<GemmParamsAttr>(
      b, perfConfig,
      getTuningParameters(b, info.kernelType, info.gemmAType, info.gemmBType,
                          info.arch, info.quantBlockSize, info.aScaleType,
                          info.bScaleType));
}

FailureOr<GemmParamsAttr>
PopulateParams::obtainTuningParameters(OpBuilder &b,
                                       RockGemmWrapperInterface op) {
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);

  StringRef perfConfig;
  if (auto perfConfigAttr =
          op->template getAttrOfType<StringAttr>("perf_config")) {
    perfConfig = perfConfigAttr.getValue();
  }
  return obtainTuningParameters(b, info, perfConfig);
}

std::vector<GemmParamsAttr> PopulateParams::getTuningParameters(
    OpBuilder &b, KernelType opType, Type dataTypeA, Type dataTypeB,
    StringRef arch, std::optional<int64_t> quantBlockSize, Type aScaleType,
    Type bScaleType) const {
  auto perfConfigs =
      ParamLookupTable<GemmParamsAttr>::lookup(arch, opType, dataTypeA);

  LLVM_DEBUG(
      llvm::dbgs() << "PopulateParams::getTuningParameters: perfConfigs: "
                   << perfConfigs.size() << "\n");
  std::vector<GemmParamsAttr> res;
  for (StringRef perfConfig : perfConfigs) {
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto params = GemmParamsAttr::get(perfConfigAttr);

    LLVM_DEBUG(llvm::dbgs()
               << "PopulateParams::getTuningParameters: perfConfigAttr: "
               << perfConfigAttr << "\n");
    LLVM_DEBUG(llvm::dbgs() << "PopulateParams::getTuningParameters: params: "
                            << params << "\n");
    if (!params)
      continue;

    res.push_back(params);
  }
  auto ordered = orderParams<GemmParamsAttr>(res, [&](GemmParamsAttr p) {
    return isGemmParamsConservativelyApplicable(
        p, dataTypeA, dataTypeB, arch, quantBlockSize, aScaleType, bScaleType);
  });
  // Guarantee MIGRAPHX_SKIP_BENCHMARKING consumers see an applicable
  // `front()`: if no table entry passed the check, prepend the conservative
  // default (which is rounded up to a multiple of `quantBlockSize` for
  // scaled GEMMs so it also satisfies the divisibility constraint).
  if (ordered.empty() || !isGemmParamsConservativelyApplicable(
                             ordered.front(), dataTypeA, dataTypeB, arch,
                             quantBlockSize, aScaleType, bScaleType))
    ordered.insert(ordered.begin(), getConservativeDefaultGemmParams(
                                        b.getContext(), quantBlockSize));
  return ordered;
}

// TODO(rocmlirTriton): Check and re-design the heuristics after performance work.
LogicalResult PopulateParams::specificCouldBePerformant(GemmParamsAttr params,
                                                        Type dataTypeA,
                                                        Type dataTypeB,
                                                        StringRef arch) {
  (void)dataTypeA;
  (void)dataTypeB;

  /// MFMA/XDL-only heuristic (rocMLIR `PopulateParamsXDL::specificCouldBePerformant`):
  /// factor total wave count into an M×N wave grid; `nPerWave` is
  /// `nPerBlock / nWaves`; `mnPerXdl` is `matrixInstrNonkdim`.

  /// WMMA uses `matrixInstrNonkdim == 0`; do not apply XDL pruning here so the
  /// full tuning space stays aligned with `computeNumWaves` (e.g. 2/4/8 on RDNA).
  MatrixAccelKind accelKind = getMatrixAccelKind(arch, dataTypeA, dataTypeB);
  bool isMFMA = accelKind == MatrixAccelKind::MFMA ||
                accelKind == MatrixAccelKind::ScaledMFMA;
  if (!isMFMA)
    return success();

  int64_t numWaves = params.getNumWaves();
  // XDL: limit to wave counts this heuristic was derived for (see rocMLIR).
  if (numWaves != 1 && numWaves != 2 && numWaves != 4)
    return failure();

  // Filter the `mPerBlock = nPerBlock = 256, numWaves = 1` corner: it blows
  // the per-thread accumulator register budget, no winning config on gfx90a /
  // gfx942 / gfx950 uses it, and it has been observed to derail tuning.
  // Threshold 512 catches that case (256·256 / (1·64) = 1024) but not the
  // same tile with `numWaves >= 2` (e.g. 512 at two waves, not greater).
  static constexpr int64_t kGemmMaxAccPerThread = 512;
  int64_t waveSize = rock::getWaveSize(arch);
  int64_t accPerThread =
      (params.getMPerBlock() * params.getNPerBlock()) / (numWaves * waveSize);
  if (accPerThread > kGemmMaxAccPerThread)
    return failure();

  return success();
}
