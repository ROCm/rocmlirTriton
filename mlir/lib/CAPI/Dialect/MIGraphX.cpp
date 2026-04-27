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

// Returns block_size, grid_size and cluster_size as uint32_t[3]
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
      auto cluster = kernel.getAttr<mlir::IntegerAttr>(
          mlir::rock::ClusterSizeAttr::getMnemonic());
      if (!block || !grid || !cluster)
        continue;
      attrs[0] = block.getInt();
      attrs[1] = grid.getInt();
      attrs[2] = cluster.getInt();
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

static bool parseBackendOptions(MlirPassManager pm,
                                const MlirMIGraphXBackendOptions &opts,
                                mlir::rock::TritonOptions &tritonOpts,
                                mlir::rock::BackendOptions &backendOpts) {
  if (!opts.arch || !opts.perfConfig) {
    llvm::errs() << "opts.arch and opts.perfConfig must not be null\n";
    return false;
  }
  if (opts.optLevel < 0 || opts.optLevel > 3) {
    llvm::errs() << "opts.optLevel must be 0, 1, 2, or 3; got " << opts.optLevel
                 << "\n";
    return false;
  }
  llvm::StringRef archStr(opts.arch);
  mlir::RocmDeviceName devName;
  if (archStr.empty() || mlir::failed(devName.parse(archStr))) {
    llvm::errs() << "Invalid architecture: " << archStr << "\n";
    return false;
  }

  tritonOpts.arch = devName.getChip().str();
  backendOpts.triple = devName.getTriple().str();
  backendOpts.chip = devName.getChip().str();
  backendOpts.features = devName.getFeaturesForBackend();
  backendOpts.optLevel = opts.optLevel;

  mlir::MLIRContext *ctx = unwrap(pm)->getContext();
  llvm::StringRef configStr(opts.perfConfig);
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
mlirMIGraphXAddBackendPipeline(MlirPassManager pm,
                               const MlirMIGraphXBackendOptions *opts) {
  if (!opts) {
    llvm::errs() << "opts is null\n";
    return false;
  }
  auto *passMan = unwrap(pm);
  if (failed(applyPassManagerCLOptions(*passMan)))
    return false;
  passMan->setNesting(mlir::PassManager::Nesting::Implicit);
  mlir::rock::KernelOptions kOpts;
  mlir::rock::buildKernelPipeline(*passMan, kOpts);

  mlir::rock::TritonOptions tritonOpts;
  mlir::rock::BackendOptions backendOpts;
  if (!parseBackendOptions(pm, *opts, tritonOpts, backendOpts))
    return false;

  mlir::rock::buildTritonPipeline(*passMan, tritonOpts);
  mlir::rock::buildBackendPipeline(*passMan, backendOpts);
  return true;
}
