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

LogicalResult collectKernelInfo(ModuleOp moduleOp, int64_t maxSharedMemPerWG,
                                SmallVectorImpl<KernelInfo> &kernels) {
  // Get Triton metadata from module attributes
  int64_t numWarps = -1;
  int64_t warpSize = -1;
  int64_t sharedMemory = 0;

  // Try ttg.total-num-warps first (set by warp-specialization pass),
  // fall back to ttg.num-warps
  if (auto totalNumWarpsAttr =
          moduleOp->getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
    numWarps = totalNumWarpsAttr.getInt();
  else if (auto numWarpsAttr =
               moduleOp->getAttrOfType<IntegerAttr>("ttg.num-warps"))
    numWarps = numWarpsAttr.getInt();
  if (auto warpSizeAttr =
          moduleOp->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
    warpSize = warpSizeAttr.getInt();
  if (auto sharedAttr = moduleOp->getAttrOfType<IntegerAttr>("ttg.shared"))
    sharedMemory = sharedAttr.getInt();

  // Validate LDS usage
  if (sharedMemory > maxSharedMemPerWG) {
    LLVM_DEBUG(llvm::dbgs()
               << "ttg.shared: too much LDS usage (" << sharedMemory << " > "
               << maxSharedMemPerWG << ")\n");
    return failure();
  }

  if (numWarps == -1) {
    LLVM_DEBUG(llvm::dbgs() << "ttg.num-warps not found\n");
    return failure();
  }
  if (warpSize == -1) {
    LLVM_DEBUG(llvm::dbgs() << "ttg.threads-per-warp not found\n");
    return failure();
  }

  int64_t tritonBlockSize = numWarps * warpSize;

  // Walk LLVM functions with KernelAttr
  moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
    if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
      return;

    KernelInfo info;
    info.name = funcOp.getName().str();
    info.llvmFunc = funcOp;
    info.blockSize = tritonBlockSize; // Use Triton's block size (matches HSACO)
    info.sharedMemorySize = sharedMemory;

    // Get grid_size from module attribute (set by FuncToTritonFunc)
    std::string gridAttrName = "rock.grid_size." + info.name;
    if (auto gridAttr = moduleOp->getAttrOfType<IntegerAttr>(gridAttrName))
      info.gridSize = gridAttr.getInt();

    // Get argument types from LLVM function signature
    auto llvmFuncType = funcOp.getFunctionType();
    unsigned numParams = llvmFuncType.getNumParams();
    info.argTypes.clear();
    for (unsigned i = 0; i < numParams; ++i) {
      info.argTypes.push_back(llvmFuncType.getParamType(i));
    }

    kernels.push_back(info);
  });

  return success();
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

    backendOpts.numWarps = gemmGemmParams.getNumWaves();
    backendOpts.wavesPerEU = gemmGemmParams.getWavesPerEU();
    return success();
  }
  return failure();
}

} // namespace rock
} // namespace mlir
