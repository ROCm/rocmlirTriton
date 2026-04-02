//===- MIGraphX.cpp - C Interface for MIGraphX dialect
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Pipeline/Pipeline.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/utility/RocmDeviceName.h"
#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TargetSelect.h"
#include <mutex>
#include <vector>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(MIGraphX, migraphx,
                                      mlir::migraphx::MIGraphXDialect)

MlirTypeID rocmlirMIXRShapedTypeGetTypeId() {
  return wrap(mlir::migraphx::MIXRShapedType::getTypeID());
}

bool rocmlirIsAMIXRShapedType(MlirType type) {
  return llvm::isa<mlir::migraphx::MIXRShapedType>(unwrap(type));
}

MlirType rocmlirMIXRShapedTypeGet(intptr_t rank, const int64_t *shape,
                                  const int64_t *strides,
                                  MlirType elementType) {
  return wrap(mlir::migraphx::MIXRShapedType::get(
      llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      llvm::ArrayRef(strides, static_cast<size_t>(rank)), unwrap(elementType)));
}

MlirType rocmlirMIXRShapedTypeAsTensor(MlirType type) {
  return wrap(
      llvm::cast<mlir::migraphx::MIXRShapedType>(unwrap(type)).asTensor());
}

// Returns the required buffer size if called with null buffer
// and fill information in the passed ptr when provided.
MLIR_CAPI_EXPORTED
void mlirGetKernelInfo(MlirModule module, int *size, void *data) {
  auto mod = unwrap(module);
  int argNum = 0;
  int argIdx = 0;
  llvm::StringRef kernelName;

  // Either of pointers should be provided.
  assert((size != nullptr || data != nullptr) &&
         "Either size or data pointer should be provided");
  std::vector<int> info;
  mod.walk([&](mlir::func::FuncOp f) {
    auto args = f.getArguments();
    for (auto arg : args) {
      argNum++;
      auto sType = mlir::cast<mlir::ShapedType>(arg.getType());
      auto rank = sType.getRank();
      info.push_back(rank);
      for (int i = 0; i < rank; i++)
        info.push_back(sType.getDimSize(i));
      argIdx += rank;
    }
    kernelName = f.getName();
  });
  if (data == nullptr && size != nullptr) {
    *size = (1 + argNum + argIdx) * sizeof(int) + kernelName.size();
  } else if (data != nullptr) {
    int argSize = argNum + argIdx;
    int *argData = (int *)data;
    argData[0] = argNum;
    for (int i = 0; i < argSize; i++)
      argData[i + 1] = info[i];
    char *nameData = (char *)(argData + argSize + 1);
    for (size_t i = 0, e = kernelName.size(); i < e; ++i) {
      nameData[i] = kernelName[i];
    }
  }
}

// Returns block_size and grid_size as uint32_t[2]
MLIR_CAPI_EXPORTED void mlirGetKernelAttrs(MlirModule module, uint32_t *attrs) {
  auto mod = unwrap(module);
  size_t count = 0;
  mod.walk([&](mlir::gpu::BinaryOp binary) {
    mlir::gpu::KernelTableAttr metadata =
        mlir::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]).getKernels();
    for (auto kernel : metadata) {
      auto block = kernel.getAttr<mlir::IntegerAttr>(
          mlir::rock::BlockSizeAttr::getMnemonic());
      auto grid = kernel.getAttr<mlir::IntegerAttr>(
          mlir::rock::GridSizeAttr::getMnemonic());
      if (!block || !grid)
        continue;
      attrs[0] = block.getInt();
      attrs[1] = grid.getInt();
      ++count;
    }
  });
  assert(count == 1 && "invalid number of kernels");
}

// Returns the size of compiled binary if called with null ptr
// and return the compiled binary when buffer is provided
MLIR_CAPI_EXPORTED bool mlirGetBinary(MlirModule module, size_t *size,
                                      char *bin) {
  bool success = false;
  auto mod = unwrap(module);
  if (bin == nullptr && size == nullptr)
    return success;
  mod.walk([&](mlir::gpu::BinaryOp binary) {
    auto object = llvm::cast<mlir::gpu::ObjectAttr>(binary.getObjects()[0]);
    if (bin != nullptr) { // return binary regardless the presence of *size
      llvm::StringRef hsaco = object.getObject().getValue();
      std::copy(hsaco.begin(), hsaco.end(), bin);
      success = true;
    } else {
      *size = object.getObject().getValue().size();
      success = true;
    }
  });
  return success;
}

// pipelines

MLIR_CAPI_EXPORTED
void mlirMIGraphXAddHighLevelPipeline(MlirPassManager pm) {
  auto passMan = unwrap(pm);
  if (failed(applyPassManagerCLOptions(*passMan)))
    llvm::errs() << "Failed to apply command-line options.\n";
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::migraphx::addMIGraphXPipeline(*passMan);
  mlir::rock::buildHighlevelPipeline(*passMan);
}

static bool parseArchAndPerfConfig(MlirPassManager pm, const char *arch,
                                   const char *perfConfig,
                                   mlir::rock::TritonOptions &tritonOpts,
                                   mlir::rock::BackendOptions &backendOpts) {
  if (!arch || !perfConfig) {
    llvm::errs() << "arch and perfConfig must not be null\n";
    return false;
  }
  llvm::StringRef archStr(arch);
  mlir::RocmDeviceName devName;
  if (archStr.empty() || mlir::failed(devName.parse(archStr))) {
    llvm::errs() << "Invalid architecture: " << archStr << "\n";
    return false;
  }

  tritonOpts.arch = devName.getChip().str();
  backendOpts.triple = devName.getTriple().str();
  backendOpts.chip = devName.getChip().str();
  backendOpts.features = devName.getFeaturesForBackend();
  backendOpts.optLevel = 3;

  mlir::MLIRContext *ctx = unwrap(pm)->getContext();
  llvm::StringRef configStr(perfConfig);
  if (configStr.empty()) {
    llvm::errs() << "perfConfig must not be empty\n";
    return false;
  }
  auto strAttr = mlir::StringAttr::get(ctx, configStr);
  mlir::Attribute configAttr;
  if (auto gemm = mlir::rock::GemmParamsAttr::get(strAttr)) {
    configAttr = gemm;
  } else if (auto attn = mlir::rock::GemmGemmParamsAttr::get(strAttr)) {
    configAttr = attn;
  } else {
    llvm::errs() << "Invalid perfConfig: " << configStr << "\n";
    return false;
  }
  if (mlir::failed(mlir::rock::fillCompilationConfigs(configAttr, tritonOpts,
                                                      backendOpts))) {
    llvm::errs() << "Failed to apply perfConfig: " << configStr << "\n";
    return false;
  }
  return true;
}

MLIR_CAPI_EXPORTED bool
mlirMIGraphXAddApplicabilityPipeline(MlirPassManager pm, const char *arch,
                                     const char *perfConfig) {
  auto *passMan = unwrap(pm);
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::rock::KernelOptions kOpts;
  mlir::rock::buildKernelPipeline(*passMan, kOpts);

  mlir::rock::TritonOptions tritonOpts;
  mlir::rock::BackendOptions backendOpts;
  if (!parseArchAndPerfConfig(pm, arch, perfConfig, tritonOpts, backendOpts))
    return false;

  mlir::rock::buildTritonPipeline(*passMan, tritonOpts);
  return true;
}

MLIR_CAPI_EXPORTED bool mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                                                       const char *arch,
                                                       const char *perfConfig) {
  auto *passMan = unwrap(pm);
  if (failed(applyPassManagerCLOptions(*passMan)))
    return false;
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::rock::KernelOptions kOpts;
  mlir::rock::buildKernelPipeline(*passMan, kOpts);

  mlir::rock::TritonOptions tritonOpts;
  mlir::rock::BackendOptions backendOpts;
  if (!parseArchAndPerfConfig(pm, arch, perfConfig, tritonOpts, backendOpts))
    return false;

  mlir::rock::buildTritonPipeline(*passMan, tritonOpts);
  mlir::rock::buildBackendPipeline(*passMan, backendOpts);
  return true;
}
