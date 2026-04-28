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
    if (orderedParams.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Quick tuning list is empty\n");
      return failure();
    }

    GemmParamsAttr validParams = orderedParams.front();
    LLVM_DEBUG(llvm::dbgs() << validParams << "\n");
    SmallVector<char, 64> perfConfigBuf;
    validParams.getPerfConfigStr(perfConfigBuf);
    perfConfigAttr = StringAttr::get(b.getContext(), perfConfigBuf);
  }

  GemmParamsAttr params = GemmParamsAttr::get(perfConfigAttr);
  if (!params) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid perfConfig: " << perfConfigAttr << "\n");
    return failure();
  }

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

/// Hard legality check that must apply in every tuning mode (Quick / Full /
/// Exhaustive), unlike `specificCouldBePerformant` which is a Full-only
/// performance heuristic.
///
/// CDNA2/3/4 (gfx90a / gfx940-942 / gfx950) per the AMD ISA references
/// (MI200 ISA §3.6.4, MI300 ISA §3.6.4, CDNA4 ISA §3.6.4): "A wave may
/// have up to 512 total VGPRs, 256 of each type [arch + accumulation]";
/// LLVM mirrors this in `getAddressableNumVGPRs` returning 512 when
/// `FeatureGFX90AInsts` is set. With f32 accumulation each MFMA output
/// element occupies one slot in that unified 512-entry register file, so
/// once
///   accPerWave = mPerBlock * nPerBlock / numWaves    (lanes * slots)
///   accPerLane = accPerWave / waveSize(64)           // gfx9 wave is 64
/// crosses 512 the codegen must spill the accumulator to scratch. The
/// non-pipelined BlockwiseGemm lowering used at `numStages=1` then takes
/// a runtime memory access fault on splitK K-padded shapes (notably
/// `numWaves=1`, 256x256, `mnPerXdl=16`, splitK=3 on K=1024). Even for
/// shapes that do not crash, these configs are 3-20x slower than the
/// multi-wave / smaller-tile alternatives, so dropping them does not
/// regress achievable performance.
///
/// CDNA1 (gfx908 / MI100) is stricter: the MI100 ISA §3.6.4 says
/// "two sets of VGPRs: normal and accumulation. Waves are allocated an
/// equal number of each type", so the AGPR cap is 256 (not 256-of-512
/// flexible). The 512 threshold here is therefore lenient on gfx908; a
/// future arch-aware tightening could lower it. We keep 512 as the common
/// ceiling because the original crash repro is on gfx942, and using a
/// tighter threshold on gfx908 would silently shrink its tuning space.
///
/// WMMA / RDNA paths take the `matrixInstrNonkdim == 0` early-out below;
/// they have a different (256-VGPR per wave) register file and reuse
/// VGPRs (no AGPR), so the gfx9 64-lane wave assumption in the threshold
/// does not apply.
bool mlir::rock::gemmParamsExceedRegisterBudget(GemmParamsAttr params) {
  int64_t mnPerXdl = params.getMatrixInstrNonkdim();
  if (mnPerXdl == 0)
    return false;

  int64_t numWaves = params.getNumWaves();
  if (numWaves <= 0)
    return false;

  constexpr int64_t kCdnaWaveSize = 64;
  constexpr int64_t kMaxAccPerThread = 512;
  int64_t accPerThread = (params.getMPerBlock() * params.getNPerBlock()) /
                         (numWaves * kCdnaWaveSize);
  return accPerThread > kMaxAccPerThread;
}

LogicalResult PopulateParams::specificCouldBePerformant(GemmParamsAttr params,
                                                        Type dataTypeA,
                                                        Type dataTypeB) {
  (void)dataTypeA;
  (void)dataTypeB;

  /// MFMA/XDL-only heuristic (rocMLIR `PopulateParamsXDL::specificCouldBePerformant`):
  /// factor total wave count into an M×N wave grid; `nPerWave` is
  /// `nPerBlock / nWaves`; `mnPerXdl` is `matrixInstrNonkdim`.


  /// WMMA uses `matrixInstrNonkdim == 0`; do not apply XDL pruning here so the
  /// full tuning space stays aligned with `computeNumWaves` (e.g. 2/4/8 on RDNA).
  int64_t mnPerXdl = params.getMatrixInstrNonkdim();
  if (mnPerXdl == 0)
    return success();

  int64_t numWaves = params.getNumWaves();
  // XDL: limit to wave counts this heuristic was derived for (see rocMLIR).
  if (numWaves != 1 && numWaves != 2 && numWaves != 4)
    return failure();

  // Defense-in-depth: also reject oversized per-thread accumulator tiles in
  // the Full performance heuristic. The same check runs unconditionally via
  // `gemmParamsExceedRegisterBudget` for the brute-force enumeration loop;
  // mirroring it here keeps direct `couldBePerformant` callers consistent.
  if (gemmParamsExceedRegisterBudget(params))
    return failure();

  return success();
}
