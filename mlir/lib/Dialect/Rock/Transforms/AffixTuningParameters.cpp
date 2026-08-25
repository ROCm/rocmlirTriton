// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/Tuning/UtilityParams.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/MathExtras.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKAFFIXTUNINGPARAMETERSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-affix-params"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct AffixTuningParameters
    : public rock::impl::RockAffixTuningParametersPassBase<
          AffixTuningParameters> {
public:
  using rock::impl::RockAffixTuningParametersPassBase<
      AffixTuningParameters>::RockAffixTuningParametersPassBase;
  void runOnOperation() override;

private:
  // Actual implementation.
  void affixTuningParametersImpl(RockGemmWrapperInterface op);
  void affixTuningParametersImpl(RockGemmGemmWrapperInterface op);

  LogicalResult validateRockAttributes(func::FuncOp func);
};
} // anonymous namespace

LogicalResult AffixTuningParameters::validateRockAttributes(func::FuncOp func) {
  static const llvm::StringSet<> knownFuncRockAttrs = {
      EnableSplitKForTuningAttr::getMnemonic(),
      ArchAttr::getMnemonic(),
      KernelAttr::getMnemonic(),
      ConvKernelAttr::getMnemonic(),
      NumCUAttr::getMnemonic(),
      NumChipletsAttr::getMnemonic(),
      BlockSizeAttr::getMnemonic(),
      UseOptimizeEpilogueAttr::getMnemonic(),
      GridSizeAttr::getMnemonic(),
      CpuVerifierAttr::getMnemonic(),
  };
  static const llvm::StringSet<> knownArgRockAttrs = {
      PrefillAttr::getMnemonic(),
  };

  for (auto &namedAttr : func->getDiscardableAttrs()) {
    StringRef name = namedAttr.getName().getValue();
    if (!knownFuncRockAttrs.contains(name)) {
      return func.emitError() << "unknown attribute '" << name
                              << "' on function '" << func.getSymName() << "'";
    }
  }
  for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
    if (auto argAttrs = func.getArgAttrDict(i)) {
      for (auto &namedAttr : argAttrs) {
        StringRef name = namedAttr.getName().getValue();
        if (!knownArgRockAttrs.contains(name)) {
          return func.emitError()
                 << "unknown attribute '" << name << "' on argument " << i
                 << " of function '" << func.getSymName() << "'";
        }
      }
    }
  }
  return success();
}

void AffixTuningParameters::runOnOperation() {
  func::FuncOp func = getOperation();

  if (failed(validateRockAttributes(func)))
    return signalPassFailure();

  // currently, in rocMLIR we only support one Fusion Root per function.
  // Therefore we check for that here. Note that rocMLIR does generate multiple
  // kernels for the conv_bwd_data but that decomposition happens later in the
  // pipeline.
  uint32_t fusionRootCnt = 0;
  func.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::rock::FusionRoot>()) {
      fusionRootCnt++;
    }
  });
  if (fusionRootCnt > 1) {
    func.emitError() << "Multiple Fusion Roots detected in a single "
                        "function. This is not supported.";
    signalPassFailure();
    return;
  }
  func.walk([&](RockGemmWrapperInterface op) {
    affixTuningParametersImpl(op);
    // Make sure the op has a params attribute
    if (!op.getGemmParams().has_value()) {
      return signalPassFailure();
    }
  });
  func.walk([&](RockGemmGemmWrapperInterface op) {
    affixTuningParametersImpl(op);
    // Make sure the op has a params attribute
    if (!op.getGemm0Params().has_value() || !op.getGemm1Params().has_value()) {
      return signalPassFailure();
    }
  });
}

void AffixTuningParameters::affixTuningParametersImpl(
    RockGemmWrapperInterface op) {
  OpBuilder b(op.getContext());
  auto funcParent = op->getParentOfType<func::FuncOp>();

  auto populateParamsPtr = std::make_unique<PopulateParams>();
  auto maybeValidParams = populateParamsPtr->obtainTuningParameters(b, op);

  if (failed(maybeValidParams)) {
    LLVM_DEBUG(llvm::dbgs() << "obtainTuningParameters call fails.\n");
    return signalPassFailure();
  }
  GemmParamsAttr gemmParams = maybeValidParams.value();
  StringAttr perfConfigAttr = gemmParams.getPerfConfigAttr();

  LLVM_DEBUG(llvm::dbgs() << "affixTuningParametersImpl: perfConfig: "
                          << perfConfigAttr << "\n");

  // Scaled GEMMs are not yet handled by rock-decompose-nonpow2-tiles (M/N) nor
  // by the K decomposition in rock-gridwise-gemm-to-blockwise, so keep the
  // power-of-two M/N and K requirements for them; plain GEMMs allow both to be
  // non-pow2. Arches where the peeled K loop miscompiles keep the power-of-two
  // K requirement as well (see rock::supportsNonPow2KPerBlock).
  bool isScaledGemm = op.getScaleA() || op.getScaleB();
  bool requirePow2K =
      isScaledGemm || !rock::supportsNonPow2KPerBlock(rock::getArchValue(op));
  if (failed(validatePerfConfig(op, gemmParams, /*requirePow2MN=*/isScaledGemm,
                                /*requirePow2K=*/requirePow2K)))
    return signalPassFailure();

  // Set kblocks attribute only for backward weight convolutions.
  if (auto bwdOp = dyn_cast<ConvBwdWeightOp>(op.getOperation())) {
    FailureOr<int64_t> maybeKBlocks =
        computeBwdWeightKBlocks(op, gemmParams);
    if (failed(maybeKBlocks)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Invalid tuning parameters for computing KBlocks.\n");
      return signalPassFailure();
    }
    bwdOp->setAttr(bwdOp.getKBlocksAttrName(),
                   b.getIndexAttr(*maybeKBlocks));
  }

  int64_t waveSize = rock::getWaveSize(rock::getArchValue(op));
  int64_t blockSize = obtainBlockSize(waveSize, gemmParams);
  assert(blockSize > 0);
  op.setGemmParamsAttr(gemmParams);

  // Set attributes on the function.
  getOperation()->setAttr(rock::BlockSizeAttr::getMnemonic(),
                          b.getI32IntegerAttr(blockSize));
  getOperation()->setAttr(
      rock::UseOptimizeEpilogueAttr::getMnemonic(),
      b.getI64IntegerAttr(gemmParams.getUseOptimizeEpilogue()));
  getOperation()->setAttr(rock::UseBf16x3ForF32Attr::getMnemonic(),
                          b.getI64IntegerAttr(gemmParams.getUseBf16x3ForF32()));

  // Check fusion legality. These checks should happen after perfConfig is
  // picked either through heuristics or user provided.
  auto fusionInfo = rock::collectFusionInfo(op->getResult(0));
  if (!fusionInfo.fusionOps.empty()) {
    if (failed(testFusionLegalityBwdDataConv(funcParent))) {
      op->emitError() << "Fusion with backward data convolution is not legal";
      return signalPassFailure();
    }
  }
  if (rock::isSplitKRequested(perfConfigAttr)) {
    if (failed(testFusionLegalitySplitK(funcParent))) {
      rock::markAsNotApplicable(op);
      op->emitError() << "Fusion with SplitK perfConfig is not legal";
      return signalPassFailure();
    }
  }
}

void AffixTuningParameters::affixTuningParametersImpl(
    RockGemmGemmWrapperInterface op) {
  OpBuilder builder(op.getContext());
  auto funcParent = op->getParentOfType<func::FuncOp>();

  // set a default one if params is not provided
  auto maybeAttnPerfConfig =
      PopulateParamsGemmGemm::obtainTuningParameters(builder, op);
  if (failed(maybeAttnPerfConfig)) {
    op.emitError() << "perf config string has an incorrect format.";
    return signalPassFailure();
  }
  auto attnPerfConfig = maybeAttnPerfConfig.value();
  StringAttr perfConfigAttr = attnPerfConfig.getPerfConfigAttr();

  if (failed(validatePerfConfig(op, attnPerfConfig, /*requirePow2MN=*/true,
                                /*requirePow2K=*/true)))
    return signalPassFailure();

  auto params =
      PopulateParamsGemmGemm::getGemmParams(builder, op, attnPerfConfig);
  if (failed(params)) {
    op.emitError() << "The provided perf config is not valid";
    return signalPassFailure();
  }
  // Check fusion legality.
  auto fusionInfo = rock::collectFusionInfo(op->getResult(0));
  if (!fusionInfo.fusionOps.empty()) {
    if (failed(testFusionLegalityBwdDataConv(funcParent))) {
      op->emitError() << "Fusion with backward data convolution is not legal";
      return signalPassFailure();
    }
  }
  if (rock::isSplitKRequested(perfConfigAttr)) {
    if (failed(testFusionLegalitySplitK(funcParent))) {
      rock::markAsNotApplicable(op);
      op->emitError() << "Fusion with SplitK perfConfig is not legal";
      return signalPassFailure();
    }
  }
  GemmParamsAttr params0, params1;
  params0 = params->first;
  params1 = params->second;
  LLVM_DEBUG(llvm::dbgs() << "params0=" << params0 << "\n");
  LLVM_DEBUG(llvm::dbgs() << "params1=" << params1 << "\n");
  op.setGemm0ParamsAttr(params0);
  op.setGemm1ParamsAttr(params1);

  // Set block size on the function (use gemm0 params since both gemms
  // share the same wave/block configuration).
  int64_t waveSize = rock::getWaveSize(rock::getArchValue(op));
  int64_t blockSize = obtainBlockSize(waveSize, params0);
  assert(blockSize > 0);
  getOperation()->setAttr(rock::BlockSizeAttr::getMnemonic(),
                          builder.getI32IntegerAttr(blockSize));
  getOperation()->setAttr(
      rock::UseOptimizeEpilogueAttr::getMnemonic(),
      builder.getI64IntegerAttr(attnPerfConfig.getUseOptimizeEpilogue()));
  getOperation()->setAttr(
      rock::UseBf16x3ForF32Attr::getMnemonic(),
      builder.getI64IntegerAttr(attnPerfConfig.getUseBf16x3ForF32()));
}
