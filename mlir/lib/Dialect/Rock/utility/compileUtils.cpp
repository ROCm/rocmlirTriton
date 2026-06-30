//===- compileUtils.cpp - Rock compile utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

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

LogicalResult collectKernelInfo(ModuleOp moduleOp,
                                SmallVectorImpl<KernelInfo> &kernels) {
  // Get Triton metadata from module attributes
  int64_t numWarps = -1;
  int64_t warpSize = -1;

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

  if (numWarps == -1) {
    LLVM_DEBUG(llvm::dbgs() << "ttg.num-warps not found\n");
    return failure();
  }
  if (warpSize == -1) {
    LLVM_DEBUG(llvm::dbgs() << "ttg.threads-per-warp not found\n");
    return failure();
  }

  int64_t tritonBlockSize = numWarps * warpSize;

  auto numCTAsAttr = moduleOp->getAttrOfType<IntegerAttr>("ttg.num-ctas");
  if (!numCTAsAttr) {
    LLVM_DEBUG(llvm::dbgs() << "ttg.num-ctas not found\n");
    return failure();
  }
  int64_t numCTAs = numCTAsAttr.getInt();

  // Walk LLVM functions with KernelAttr
  auto walkResult = moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) -> WalkResult {
    if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
      return WalkResult::advance();

    KernelInfo info;
    info.name = funcOp.getName().str();
    info.llvmFunc = funcOp;
    info.blockSize = tritonBlockSize;
    info.clusterSize = numCTAs;

    // Get grid_size from module attribute (set by RockTensorToTritonPtr)
    std::string gridAttrName = "rock.grid_size." + info.name;
    if (auto gridAttr = moduleOp->getAttrOfType<IntegerAttr>(gridAttrName))
      info.gridSize = gridAttr.getInt();

    // Get prefill arg info from module attribute (set by RockTensorToTritonPtr)
    std::string prefillAttrName = "rock.prefill_args." + info.name;
    if (auto prefillArr = moduleOp->getAttrOfType<ArrayAttr>(prefillAttrName)) {
      for (Attribute entry : prefillArr) {
        auto dict = dyn_cast<DictionaryAttr>(entry);
        if (!dict) {
          funcOp.emitOpError("malformed ")
              << prefillAttrName << ": entry is not a DictionaryAttr";
          return WalkResult::interrupt();
        }
        auto indexAttr = dict.getAs<IntegerAttr>("index");
        if (!indexAttr) {
          funcOp.emitOpError("malformed ")
              << prefillAttrName << ": entry missing 'index' IntegerAttr";
          return WalkResult::interrupt();
        }
        Attribute valueAttr = dict.get("value");
        if (!valueAttr) {
          funcOp.emitOpError("malformed ")
              << prefillAttrName << ": entry missing 'value' attribute";
          return WalkResult::interrupt();
        }
        PrefillInfo pi;
        pi.argIndex = indexAttr.getValue().getZExtValue();
        pi.initValue = valueAttr;
        info.prefillArgs.push_back(pi);
      }
    }

    // Get argument types from LLVM function signature
    auto llvmFuncType = funcOp.getFunctionType();
    unsigned numParams = llvmFuncType.getNumParams();
    info.argTypes.clear();
    for (unsigned i = 0; i < numParams; ++i) {
      info.argTypes.push_back(llvmFuncType.getParamType(i));
    }

    for (const PrefillInfo &pi : info.prefillArgs) {
      assert(pi.argIndex < info.argTypes.size() &&
             "prefill arg index out of range");
      assert(isa<LLVM::LLVMPointerType>(info.argTypes[pi.argIndex]) &&
             "prefill arg must be a pointer (buffer) argument");
    }

    kernels.push_back(info);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  for (KernelInfo &k : kernels) {
    if (k.gridSize <= 0) {
      return k.llvmFunc.emitOpError("missing rock.grid_size." + k.name +
                                    " module attribute");
    }
  }

  return success();
}

FailureOr<ArrayAttr> getPrefillArrayFromBinary(ModuleOp moduleOp) {
  ArrayAttr result;
  LogicalResult status = success();
  moduleOp.walk([&](gpu::BinaryOp binary) {
    auto kernelTable =
        cast<gpu::ObjectAttr>(binary.getObjects()[0]).getKernels();
    size_t numKernels = 0;
    for (auto kernel : kernelTable) {
      ++numKernels;
      if (auto arr =
              kernel.getAttr<ArrayAttr>(rock::PrefillAttr::getMnemonic())) {
        result = arr;
      }
    }
    if (numKernels != 1) {
      binary.emitOpError("expected exactly one kernel in binary, got ")
          << numKernels;
      status = failure();
    }
  });
  if (failed(status))
    return failure();
  return result;
}

LogicalResult fillCompilationConfigs(Attribute perfConfig,
                                     rock::TritonOptions &tritonOpts,
                                     rock::BackendOptions &backendOpts) {
  auto params = dyn_cast<RockTuningParamAttrInterface>(perfConfig);
  if (!params)
    return failure();

  tritonOpts.numWarps = params.getNumWaves();
  tritonOpts.numCTAs = params.getNumCTAs();
  tritonOpts.numStages = params.getNumStages();
  tritonOpts.matrixInstrNonkdim = params.getMatrixInstrNonkdim();
  tritonOpts.kpack = params.getKpack();
  tritonOpts.useAsyncCopy = params.getUseAsyncCopy();
  tritonOpts.useBlockPingpong = params.getUseBlockPingpong();
  tritonOpts.useInThreadTranspose = params.getUseInThreadTranspose();
  tritonOpts.useBufferOps = params.getUseBufferOps();
  tritonOpts.useBufferAtomics = params.getUseBufferAtomics();
  tritonOpts.scheduleHint = params.getScheduleHint();

  backendOpts.numWarps = params.getNumWaves();
  backendOpts.numCTAs = params.getNumCTAs();
  backendOpts.wavesPerEU = params.getWavesPerEU();
  backendOpts.scheduleHint = params.getScheduleHint();
  return success();
}

LogicalResult fillCompilationConfigs(MLIRContext *ctx, StringRef perfConfig,
                                     rock::TritonOptions &tritonOpts,
                                     rock::BackendOptions &backendOpts) {
  if (perfConfig.empty())
    return failure();
  auto strAttr = StringAttr::get(ctx, perfConfig);
  Attribute params = GemmParamsAttr::get(strAttr);
  if (!params)
    params = GemmGemmParamsAttr::get(strAttr);
  if (!params)
    return failure();
  return fillCompilationConfigs(params, tritonOpts, backendOpts);
}

} // namespace rock
} // namespace mlir
