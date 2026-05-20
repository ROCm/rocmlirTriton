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
  // Prefer the user-supplied `perf_config` attribute on the op; if it's
  // absent, fall back to the first entry of the per-(arch, kernel, dtype)
  // quick tuning list as the default perfConfig.
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
  return getTuningParameters(b, rock::getArchValue(op), op.getKernelType(),
                             cast<ShapedType>(op.getAType()).getElementType());
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
  // and gemm1NPerBlock must be a power of two
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
