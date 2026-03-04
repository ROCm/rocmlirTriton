#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "rock-tuning-parameter"

using namespace mlir;
using namespace mlir::rock;

#define GemmGemm_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/QuickTuningPerfconfigs.inc"
#undef GemmGemm_DEFINITIONS_GEN

StringAttr PopulateParamsGemmGemm::obtainTuningParameters(
    OpBuilder &b, RockGemmGemmWrapperInterface op) {
  // default perfConfig
  StringAttr perfConfig = b.getStringAttr("attn:v1:32,32,32,1,1,4,0,1,1,0,0");
  if (StringAttr mayBePerfConfig =
          dyn_cast_or_null<StringAttr>(op->getAttr("perf_config"))) {
    perfConfig = mayBePerfConfig;
  }
  return perfConfig;
}

std::vector<GemmGemmParamsAttr>
PopulateParamsGemmGemm::getTuningParameters(OpBuilder &b,
                                            RockGemmGemmWrapperInterface op) {
  auto perfConfigs = ParamLookupTable<GemmGemmParamsAttr>::lookup(
      rock::getArchValue(op), op.getKernelType(),
      cast<ShapedType>(op.getAType()).getElementType());
  return deserializePerfConfigs(b, op, perfConfigs);
}

GemmGemmParamsAttr PopulateParamsGemmGemm::deserializePerfConfig(
    OpBuilder &b, RockGemmGemmWrapperInterface op, StringRef config) {
  auto stringAttr = b.getStringAttr(config);
  return GemmGemmParamsAttr::get(stringAttr);
}

std::vector<GemmGemmParamsAttr>
PopulateParamsGemmGemm::deserializePerfConfigs(OpBuilder &b,
                                               RockGemmGemmWrapperInterface op,
                                               ArrayRef<StringRef> configs) {
  std::vector<GemmGemmParamsAttr> ret;
  ret.reserve(configs.size());
  std::transform(
      configs.begin(), configs.end(), std::back_inserter(ret),
      [&](StringRef config) { return deserializePerfConfig(b, op, config); });
  return ret;
}

FailureOr<std::pair<GemmParamsAttr, GemmParamsAttr>>
PopulateParamsGemmGemm::getGemmParams(OpBuilder &b,
                                      RockGemmGemmWrapperInterface op,
                                      GemmGemmParamsAttr params) {
  GemmParamsAttr accelParams0 = getGemm0Params(b, params);
  GemmParamsAttr accelParams1 = getGemm1Params(b, op, params);

  return std::make_pair(accelParams0, accelParams1);
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
