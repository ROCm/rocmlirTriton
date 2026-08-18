#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

#define GemmGemm_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_DEFINITIONS_GEN

FailureOr<GemmGemmParamsAttr> PopulateParamsGemmGemm::obtainTuningParameters(
    OpBuilder &b, RockGemmGemmWrapperInterface op) {
  // Prefer the op's `perf_config`; otherwise fall back to the
  // (already-ordered) quick-tuning list and pick its front.
  StringRef perfConfig;
  if (auto mayBePerfConfig =
          dyn_cast_or_null<StringAttr>(op->getAttr("perf_config")))
    perfConfig = mayBePerfConfig.getValue();
  return materializeTuningParams<GemmGemmParamsAttr>(
      b, perfConfig, getTuningParameters(b, op));
}

std::vector<GemmGemmParamsAttr>
PopulateParamsGemmGemm::getTuningParameters(OpBuilder &b,
                                            RockGemmGemmWrapperInterface op) {
  // Bump the first applicable config (Q + K + V LDS fit, kpack/splitK/numCTAs
  // == 1) to the front for skip-benchmarking consumers.
  auto aElemType = cast<ShapedType>(op.getAType()).getElementType();
  auto bElemType = cast<ShapedType>(op.getBType()).getElementType();
  auto cElemType = cast<ShapedType>(op.getCType()).getElementType();
  auto arch = rock::getArchValue(op);
  auto list = getTuningParameters(b, arch, op.getKernelType(), aElemType);
  auto ordered =
      orderParams<GemmGemmParamsAttr>(list, [&](GemmGemmParamsAttr p) {
        return isGemmGemmParamsConservativelyApplicable(
            b, p, aElemType, bElemType, cElemType, arch, op);
      });
  // Guarantee MIGRAPHX_SKIP_BENCHMARKING consumers see an applicable
  // `front()`: if no table entry passed the check, prepend the conservative
  // default.
  if (ordered.empty() ||
      !isGemmGemmParamsConservativelyApplicable(b, ordered.front(), aElemType,
                                                bElemType, cElemType, arch, op))
    ordered.insert(ordered.begin(),
                   getConservativeDefaultGemmGemmParams(b.getContext()));
  return ordered;
}

std::vector<GemmGemmParamsAttr> PopulateParamsGemmGemm::getTuningParameters(
    OpBuilder &b, StringRef arch, KernelType kernelType, Type elementType) {
  auto perfConfigs = ParamLookupTable<GemmGemmParamsAttr>::lookup(
      arch, kernelType, elementType);
  std::vector<GemmGemmParamsAttr> ret;
  ret.reserve(perfConfigs.size());
  for (StringRef config : perfConfigs) {
    if (auto params = GemmGemmParamsAttr::get(b.getStringAttr(config)))
      ret.push_back(params);
  }
  return ret;
}

FailureOr<std::pair<GemmParamsAttr, GemmParamsAttr>>
PopulateParamsGemmGemm::getGemmParams(OpBuilder &b,
                                      RockGemmGemmWrapperInterface op,
                                      GemmGemmParamsAttr params) {
  GemmParamsAttr params0 = getGemm0Params(b, params);
  GemmParamsAttr params1 = getGemm1Params(b, op, params);

  return std::make_pair(params0, params1);
}

GemmParamsAttr
PopulateParamsGemmGemm::getGemm0Params(OpBuilder &b,
                                       GemmGemmParamsAttr params) {
  constexpr auto splitKFactor = 1;

  return GemmParamsAttr::get(
      b.getContext(), params.getMPerBlockG0(), params.getNPerBlockG0(),
      params.getKPerBlock(), params.getKpack(), params.getNumCTAs(),
      params.getNumWaves(), params.getMatrixInstrNonkdim(), splitKFactor,
      params.getNumStages(), params.getWavesPerEU(), params.getGridGroupSize(),
      params.getUseAsyncCopy(), params.getUseBlockPingpong(),
      params.getUseInThreadTranspose(), params.getUseBufferOps(),
      params.getUseBufferAtomics(), params.getUseReductionLayout(),
      params.getUseOptimizeEpilogue());
}

GemmParamsAttr PopulateParamsGemmGemm::getGemm1Params(
    OpBuilder &b, RockGemmGemmWrapperInterface op, GemmGemmParamsAttr params) {
  // `nPerBlockG1 == 0` keeps the second GEMM untiled (process the full gemm1N);
  // otherwise it tiles the head dim. The untiled head dim is rounded up to a
  // tight `tileReducingPartitions` size (a few pow2 segments) rather than the
  // next power of two, so rock-decompose-nonpow2-tiles can split it into pow2
  // sub-attentions instead of padding e.g. 80 up to 128.
  int64_t gemm1N = rock::tileReducingPartitions(
      static_cast<uint32_t>(op.getGemmGemmSize().o));
  int64_t gemm1NPerBlock =
      params.getNPerBlockG1() > 0 ? params.getNPerBlockG1() : gemm1N;
  return GemmParamsAttr::get(
      b.getContext(), params.getMPerBlockG0(), gemm1NPerBlock,
      params.getNPerBlockG0(), params.getKpack(), params.getNumCTAs(),
      params.getNumWaves(), params.getMatrixInstrNonkdim(),
      params.getSplitKFactor(), params.getNumStages(), params.getWavesPerEU(),
      params.getGridGroupSize(), params.getUseAsyncCopy(),
      params.getUseBlockPingpong(), params.getUseInThreadTranspose(),
      params.getUseBufferOps(), params.getUseBufferAtomics(),
      params.getUseReductionLayout(), params.getUseOptimizeEpilogue());
}
