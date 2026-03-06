#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/IR/BuiltinTypes.h"
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

/// Non-xdlops
// clang-format off
#define NonAccel_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef NonAccel_DEFINITIONS_GEN
// clang-format on

/// Static data for XDL tuning parameters (used by ParamLookupTable)
// clang-format off
#define XDL_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef XDL_DEFINITIONS_GEN
// clang-format on

/// Static data for WMMA tuning parameters (used by ParamLookupTable)
// clang-format off
#define Wmma_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef Wmma_DEFINITIONS_GEN
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
  return info;
}

std::optional<GemmSize> mlir::rock::calculatePadding(int64_t kPerBlock,
                                                     int64_t mPerBlock,
                                                     int64_t nPerBlock,
                                                     const GemmSize &gemmSize) {
  int64_t kExtra = kPerBlock - math_util::mod_1_to_n(gemmSize.k, kPerBlock);
  int64_t mExtra = mPerBlock - math_util::mod_1_to_n(gemmSize.m, mPerBlock);
  int64_t nExtra = nPerBlock - math_util::mod_1_to_n(gemmSize.n, nPerBlock);
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

std::optional<GemmSize> mlir::rock::requiredPadding(Attribute params,
                                                    GemmSize gemmSize,
                                                    int64_t mulByKPerBlock,
                                                    int64_t mulByMPerBlock,
                                                    int64_t mulByNPerBlock) {
  int64_t kPerBlock, mPerBlock, nPerBlock;
  if (auto accelParams = dyn_cast<GemmParamsAttr>(params)) {
    kPerBlock = accelParams.getKPerBlock();
    mPerBlock = accelParams.getMPerBlock();
    nPerBlock = accelParams.getNPerBlock();
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

int64_t
PopulateParamsAccel::calculatePaddingAmount(GemmParamsAttr params,
                                            const GemmSize &gemmSize) const {
  std::optional<GemmSize> maybeGemmExtraPad =
      calculatePadding(params.getKPerBlock(), params.getMPerBlock(),
                       params.getNPerBlock(), gemmSize);
  if (maybeGemmExtraPad.has_value()) {
    return calculatePaddingComplexity(maybeGemmExtraPad.value(), gemmSize);
  }
  return 0;
}

LogicalResult
PopulateParamsAccel::couldBePerformant(const PopulateParamsInfo &info,
                                       GemmParamsAttr params) {
  if (info.hasFusedReduction) {
    return couldFusedReductionBePerformant(info.gemmSize, params.getMPerBlock(),
                                           params.getNPerBlock());
  }

  return specificCouldBePerformant(params, info.gemmAType, info.gemmBType);
}

FailureOr<GemmParamsAttr> PopulateParamsAccel::obtainTuningParameters(
    OpBuilder &b, const PopulateParamsInfo &info, const StringRef perfConfig) {

  StringAttr perfConfigAttr;
  if (!perfConfig.empty()) {
    // Under two scenarios can we receive a perfConfig:
    // 1. This is tuning mode
    // 2. This is running mode and we have succeeded with a perfdb load
    perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
  } else {
    auto paramSets = getTuningParameters(b, info.kernelType, info.gemmAType,
                                         info.gemmBType, info.arch);

    auto orderedParams = orderParams(paramSets, info.gemmSize);
    if (orderedParams.empty())
      return failure();

    GemmParamsAttr validParams = orderedParams.front();
    LLVM_DEBUG(llvm::dbgs() << validParams << "\n");
    SmallVector<char, 64> perfConfigBuf;
    validParams.getPerfConfigStr(perfConfigBuf);
    perfConfigAttr = StringAttr::get(b.getContext(), perfConfigBuf);
  }

  GemmParamsAttr params = GemmParamsAttr::get(perfConfigAttr);
  if (!params)
    return failure();

  return params;
}

FailureOr<GemmParamsAttr>
PopulateParamsAccel::obtainTuningParameters(OpBuilder &b,
                                            RockGemmWrapperInterface op) {
  PopulateParamsInfo info = PopulateParamsInfo::fromOp(op);

  StringRef perfConfig;
  if (auto perfConfigAttr =
          op->template getAttrOfType<StringAttr>("perf_config")) {
    perfConfig = perfConfigAttr.getValue();
  }
  return obtainTuningParameters(b, info, perfConfig);
}

std::vector<GemmParamsAttr>
PopulateParams::getTuningParameters(OpBuilder &b, KernelType opType,
                                    Type dataTypeA, Type dataTypeB,
                                    StringRef arch) const {
  auto perfConfigs =
      ParamLookupTable<GemmParamsAttr>::lookup(arch, opType, dataTypeA);

  LLVM_DEBUG(llvm::dbgs() << "PopulateParams::getTuningParameters: perfConfigs: "
                          << perfConfigs.size() << "\n");
  std::vector<GemmParamsAttr> res;
  for (StringRef perfConfig : perfConfigs) {
    auto perfConfigAttr = StringAttr::get(b.getContext(), perfConfig);
    auto params = GemmParamsAttr::get(perfConfigAttr);

    LLVM_DEBUG(llvm::dbgs()
               << "PopulateParams::getTuningParameters: perfConfigAttr: "
               << perfConfigAttr << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "PopulateParams::getTuningParameters: params: " << params
               << "\n");
    if (!params)
      continue;

    res.push_back(params);
  }
  return res;
}

LogicalResult PopulateParams::specificCouldBePerformant(GemmParamsAttr params,
                                                        Type dataTypeA,
                                                        Type dataTypeB) {
  // TODO(roctriton): We should probably implement this.
  (void)params;
  (void)dataTypeA;
  (void)dataTypeB;
  return success();
}
