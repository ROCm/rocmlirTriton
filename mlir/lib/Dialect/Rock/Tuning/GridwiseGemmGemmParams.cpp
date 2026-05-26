#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
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
  auto elemType = cast<ShapedType>(op.getAType()).getElementType();
  auto arch = rock::getArchValue(op);
  auto list = getTuningParameters(b, arch, op.getKernelType(), elemType);
  auto ordered =
      orderParams<GemmGemmParamsAttr>(list, [&](GemmGemmParamsAttr p) {
        return isGemmGemmParamsConservativelyApplicable(b, p, elemType,
                                                        elemType, arch, op);
      });
  // Guarantee MIGRAPHX_SKIP_BENCHMARKING consumers see an applicable
  // `front()`: if no table entry passed the check, prepend the conservative
  // default.
  if (ordered.empty() || !isGemmGemmParamsConservativelyApplicable(
                             b, ordered.front(), elemType, elemType, arch, op))
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
      b.getContext(), params.getMPerBlockG0(),
      params.getNPerBlockG0(), params.getKPerBlock(), params.getKpack(), params.getNumCTAs(),
      params.getNumWaves(), params.getMatrixInstrNonkdim(), splitKFactor,
      params.getNumStages(),
      params.getWavesPerEU(), params.getGridGroupSize());
}

GemmParamsAttr PopulateParamsGemmGemm::getGemm1Params(
    OpBuilder &b, RockGemmGemmWrapperInterface op, GemmGemmParamsAttr params) {
  // Due to limitations, gemm1NPerBlock must be equal to gemm1N
  // and gemm1NPerBlock must be a power of two.
  auto cShape = cast<ShapedType>(op.getCType()).getShape();
  int idx = op.getTransposedC() ? 0 : 1;
  assert(cShape.size() == 3 || cShape.size() == 2);
  if (cShape.size() == 3)
    idx++;
  int64_t gemm1NPerBlock = llvm::PowerOf2Ceil(cShape[idx]);
  return GemmParamsAttr::get(
      b.getContext(), params.getMPerBlockG0(), gemm1NPerBlock,
      params.getNPerBlockG0(), params.getKpack(), params.getNumCTAs(),
      params.getNumWaves(), params.getMatrixInstrNonkdim(),
      params.getSplitKFactor(), params.getNumStages(), params.getWavesPerEU(),
      params.getGridGroupSize());
}
