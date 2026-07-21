//===- MIGraphX.cpp - C Interface for MIGraphX dialect
//------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/MIGraphX.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/MIGraphX/Pipeline/Pipeline.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmGemmParams.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/RocmDeviceName.h"
#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
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

// Map a parsed ROCm device name onto the Triton/backend option structs. Shared
// by the module LDS gate and `parseBackendOptions` so both derive arch-specific
// options the same way.
static void applyDeviceNameToOptions(const mlir::RocmDeviceName &devName,
                                     int optLevel,
                                     mlir::rock::TritonOptions &tritonOpts,
                                     mlir::rock::BackendOptions &backendOpts) {
  tritonOpts.arch = devName.getChip().str();
  backendOpts.triple = devName.getTriple().str();
  backendOpts.chip = devName.getChip().str();
  backendOpts.features = devName.getFeaturesForBackend();
  backendOpts.optLevel = optLevel;
}

// Authoritative LDS check for a concrete module: lower a clone through the
// kernel + Triton + backend pipeline and let the hardware LDS gate
// (ResolveKernelLaunchParams) decide. Returns true iff the pipeline succeeds
// and no rock pass marked the module `rock.not_applicable`.
static bool ldsUsageFitsForModule(MlirModule module) {
  // Operate on a clone: lowering is destructive and the caller's module must
  // be left untouched.
  mlir::ModuleOp clonedMod = unwrap(module).clone();
  mlir::OwningOpRef<mlir::ModuleOp> cloneGuard(clonedMod);
  mlir::MLIRContext *ctx = clonedMod.getContext();

  // This is a gate, so an inapplicable problem is an expected, non-fatal
  // outcome; swallow the diagnostics the pipelines emit on failure.
  mlir::ScopedDiagnosticHandler diagHandler(
      ctx, [](mlir::Diagnostic &) { return mlir::success(); });

  // Phase 1: MIGraphX -> Rock (high level). Must complete before we can stamp
  // perf configs on the resulting Rock ops.
  {
    mlir::PassManager pm(clonedMod->getName(),
                         mlir::PassManager::Nesting::Implicit);
    mlir::migraphx::addMIGraphXPipeline(pm);
    mlir::rock::buildHighlevelPipeline(pm);
    if (mlir::failed(pm.run(clonedMod)))
      return false;
  }

  // Honor an existing `perf_config`; otherwise stamp the smallest conservative
  // config. Keep the chosen config to derive the Triton/backend options below.
  mlir::rock::RockTuningParamAttrInterface params;
  mlir::StringAttr archAttr;
  auto resolveTunableOp = [&](auto op, auto parse, auto makeDefault) {
    mlir::rock::RockTuningParamAttrInterface cfg;
    if (auto existing =
            op->template getAttrOfType<mlir::StringAttr>("perf_config"))
      cfg = parse(existing);
    if (!cfg) {
      auto def = makeDefault(ctx);
      op->setAttr("perf_config", def.getPerfConfigAttr());
      cfg = def;
    }
    params = cfg;
    if (auto a = mlir::rock::getArch(op.getOperation()); mlir::succeeded(a))
      archAttr = *a;
  };

  clonedMod->walk([&](mlir::rock::RockGemmGemmWrapperInterface op) {
    resolveTunableOp(
        op,
        [](mlir::StringAttr s) {
          return mlir::rock::GemmGemmParamsAttr::get(s);
        },
        [](mlir::MLIRContext *c) {
          return mlir::rock::getConservativeDefaultGemmGemmParams(c);
        });
  });
  if (!params)
    clonedMod->walk([&](mlir::rock::RockGemmWrapperInterface op) {
      resolveTunableOp(
          op,
          [](mlir::StringAttr s) { return mlir::rock::GemmParamsAttr::get(s); },
          [](mlir::MLIRContext *c) {
            return mlir::rock::getConservativeDefaultGemmParams(c);
          });
    });

  // No tunable kernel means nothing allocates LDS through this path; trivially
  // fits.
  if (!params)
    return true;

  if (!archAttr)
    return false;

  mlir::RocmDeviceName devName;
  if (mlir::failed(devName.parse(archAttr.strref())))
    return false;

  mlir::rock::TritonOptions tritonOpts;
  mlir::rock::BackendOptions backendOpts;
  applyDeviceNameToOptions(devName, /*optLevel=*/3, tritonOpts, backendOpts);
  backendOpts.compile = true;
  if (mlir::failed(
          mlir::rock::fillCompilationConfigs(params, tritonOpts, backendOpts)))
    return false;

  // Phase 2: Rock -> Triton -> backend (runs the LDS gate; full codegen).
  {
    mlir::PassManager pm(clonedMod->getName(),
                         mlir::PassManager::Nesting::Implicit);
    mlir::rock::KernelOptions kOpts;
    mlir::rock::buildKernelPipeline(pm, kOpts);
    mlir::rock::buildTritonPipeline(pm, tritonOpts);
    mlir::rock::buildBackendPipeline(pm, backendOpts);
    if (mlir::failed(pm.run(clonedMod)))
      return false;
  }

  // A `rock.not_applicable` marker means a rock pass (e.g. the LDS gate)
  // refused the config; treat that as "does not fit".
  return !clonedMod->hasAttr(mlir::rock::NotApplicableAttr::getMnemonic());
}

MLIR_CAPI_EXPORTED bool mlirMIGraphXLDSUsageFitsArch(int64_t gemmO,
                                                     const char *arch,
                                                     MlirType elementType,
                                                     MlirModule module) {
  // When a module is provided, its lowering is the authoritative answer and
  // supersedes the problem-size estimate below.
  if (!mlirModuleIsNull(module))
    return ldsUsageFitsForModule(module);

  if (!arch) {
    llvm::errs() << "arch must not be null when checking LDS usage\n";
    return false;
  }

  mlir::Type elemType = unwrap(elementType);
  if (!elemType) {
    llvm::errs() << "elementType must not be null when checking LDS usage\n";
    return false;
  }

  llvm::StringRef archStr(arch);
  mlir::RocmDeviceName devName;
  if (archStr.empty() || mlir::failed(devName.parse(archStr))) {
    llvm::errs() << "could not estimate LDS usage: invalid architecture '"
                 << archStr << "'\n";
    return false;
  }

  mlir::FailureOr<int64_t> ldsBytes =
      mlir::rock::estimateGemmGemmLdsBytes(elemType, gemmO);
  if (mlir::failed(ldsBytes)) {
    llvm::errs() << "could not estimate LDS usage for the given problem on "
                 << archStr << "\n";
    return false;
  }

  // The estimate fits when it is within the arch's shared-memory capacity.
  return ldsBytes.value() <= mlir::rock::getLDSSize(archStr);
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

  applyDeviceNameToOptions(devName, opts.optLevel, tritonOpts, backendOpts);

  llvm::StringRef configStr(opts.perfConfig);
  if (configStr.empty()) {
    llvm::errs() << "perfConfig must not be empty\n";
    return false;
  }
  if (mlir::failed(mlir::rock::fillCompilationConfigs(
          unwrap(pm)->getContext(), configStr, tritonOpts, backendOpts))) {
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
