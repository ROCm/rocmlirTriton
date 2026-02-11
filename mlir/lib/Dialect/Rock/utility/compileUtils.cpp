//===- compileUtils.cpp - Rock compile utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>
using namespace mlir;
using namespace mlir::rock;

#define DEBUG_TYPE "rock-compile-utils"

namespace mlir {
namespace rock {

LLVM::LLVMFuncOp findKernelFunc(ModuleOp moduleOp, StringRef kernelName) {
  LLVM::LLVMFuncOp result = nullptr;
  moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
      return WalkResult::advance();
    if (funcOp.getName() == kernelName) {
      result = funcOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

void populateKernelArgTypes(ModuleOp moduleOp,
                            SmallVectorImpl<KernelInfo> &kernels) {
  for (auto &kernel : kernels) {
    if (auto funcOp = findKernelFunc(moduleOp, kernel.name)) {
      auto llvmFuncType = funcOp.getFunctionType();
      kernel.argTypes.clear();
      for (unsigned i = 0; i < llvmFuncType.getNumParams(); ++i) {
        kernel.argTypes.push_back(llvmFuncType.getParamType(i));
      }
    }
  }
}

std::optional<unsigned> getKernelArgCount(ModuleOp moduleOp,
                                          StringRef kernelName) {
  if (auto funcOp = findKernelFunc(moduleOp, kernelName)) {
    return funcOp.getFunctionType().getNumParams();
  }
  return std::nullopt;
}

LogicalResult fillCompilationConfigs(StringAttr perfConfig,
                                     rock::TritonOptions &tritonOpts,
                                     rock::BackendOptions &backendOpts) {
  // TODO(roctriton): add common params to RockTuningParamAttrInterface
  if (auto gemmParams = rock::GemmParamsAttr::get(perfConfig)) {
    tritonOpts.numWarps = gemmParams.getNumWaves();
    tritonOpts.numCTAs = gemmParams.getNumCTAs();
    tritonOpts.numStages = gemmParams.getNumStages();
    tritonOpts.matrixInstrNonkdim = gemmParams.getMatrixInstrNonkdim();
    tritonOpts.kpack = gemmParams.getKpack();

    backendOpts.numStages = gemmParams.getNumStages();
    backendOpts.numWarps = gemmParams.getNumWaves();
    backendOpts.numCTAs = gemmParams.getNumCTAs();
    backendOpts.wavesPerEU = gemmParams.getWavesPerEU();
    return success();
  }
  if (auto gemmGemmParams = rock::GemmGemmParamsAttr::get(perfConfig)) {
    tritonOpts.numWarps = gemmGemmParams.getNumWaves();
    tritonOpts.numCTAs = gemmGemmParams.getNumCTAs();
    tritonOpts.numStages = gemmGemmParams.getNumStages();
    tritonOpts.matrixInstrNonkdim = gemmGemmParams.getMatrixInstrNonkdim();
    tritonOpts.kpack = gemmGemmParams.getKpack();

    backendOpts.numStages = gemmGemmParams.getNumStages();
    backendOpts.numWarps = gemmGemmParams.getNumWaves();
    backendOpts.wavesPerEU = gemmGemmParams.getWavesPerEU();
    return success();
  }
  return failure();
}

} // namespace rock
} // namespace mlir
