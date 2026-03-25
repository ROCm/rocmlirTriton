//===- Schedules.cpp - CPU lowering transform schedules -------------------===//
//
// Copyright 2026 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file contains the transform schedules used for CPU lowering.
// The schedules are MLIR transform dialect modules that define the
// optimization and lowering pipeline.
//
//===----------------------------------------------------------------------===//

#include "Schedules.h"
#include "Schedules/LowerToLLVMSchedule.h"
#include "Schedules/UnrollSchedule.h"
#include "Schedules/PrePostSchedules.h"
#include "Schedules/TilingSchedule.h"
#include "Schedules/VectorizationSchedule.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/TransformOps/FuncTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cpu-schedules"

using namespace mlir;
using namespace mlir::cpu;

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

void cpu::registerScheduleDialectExtensions(DialectRegistry &registry) {
  // Register transform dialect extensions needed for parsing the transform
  // sequences. This must be done in getDependentDialects() to avoid
  // thread-safety issues. The extensions provide ops like
  // transform.structured.match, transform.bufferization.one_shot_bufferize.
  linalg::registerTransformDialectExtension(registry);
  bufferization::registerTransformDialectExtension(registry);
  tensor::registerTransformDialectExtension(registry);
  vector::registerTransformDialectExtension(registry);
  func::registerTransformDialectExtension(registry);
  memref::registerTransformDialectExtension(registry);
  scf::registerTransformDialectExtension(registry);

  // Register ValueBoundsOpInterface for dialects, required by
  // transform.structured.tile_using_for and other tiling transforms
  affine::registerValueBoundsOpInterfaceExternalModels(registry);
  arith::registerValueBoundsOpInterfaceExternalModels(registry);
  scf::registerValueBoundsOpInterfaceExternalModels(registry);
  tensor::registerValueBoundsOpInterfaceExternalModels(registry);

  // Register BufferizableOpInterface and SubsetOpInterface for vector dialect,
  // required by one_shot_bufferize when vector ops are present
  vector::registerBufferizableOpInterfaceExternalModels(registry);
  vector::registerSubsetOpInterfaceExternalModels(registry);

  // Register AllocationOpInterface for memref dialect,
  // required by buffer-hoisting and buffer-loop-hoisting passes
  memref::registerAllocationOpInterfaceExternalModels(registry);
}

FailureOr<TransformSchedules> cpu::createTransformSchedules(MLIRContext *ctx) {
  TransformSchedules schedules;

  schedules.preModule = buildPreSchedule(ctx);
  if (!schedules.preModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to build pre transform sequence for CPU verifier";
    return failure();
  }

  schedules.postModule = buildPostSchedule(ctx);
  if (!schedules.postModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to build post transform sequence for CPU verifier";
    return failure();
  }

  schedules.tilingModule = buildTilingSchedule(ctx);
  if (!schedules.tilingModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to build tiling transform sequence for CPU verifier";
    return failure();
  }

  schedules.vectorizationModule = buildVectorizationSchedule(ctx);
  if (!schedules.vectorizationModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to build vectorization transform sequence for CPU verifier";
    return failure();
  }

  schedules.unrollModule = buildUnrollSchedule(ctx);
  if (!schedules.unrollModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to build unroll transform sequence for CPU verifier";
    return failure();
  }

  schedules.lowerToLLVMModule = buildLowerToLLVMSchedule(ctx);
  if (!schedules.lowerToLLVMModule) {
    emitError(UnknownLoc::get(ctx))
        << "Failed to build lowerToLLVM transform sequence for CPU verifier";
    return failure();
  }

  return schedules;
}

/// Unwrap nested module structure if present.
/// Some transforms may wrap their result in an additional module, which would
/// cause subsequent transforms to not process the inner function properly.
static void unwrapNestedModule(OwningOpRef<ModuleOp> &module,
                               StringRef funcName) {
  // Check if there's a nested module inside the outer module
  ModuleOp nestedModule = nullptr;
  for (Operation &op : module->getBody()->getOperations()) {
    if (auto nested = dyn_cast<ModuleOp>(&op)) {
      nestedModule = nested;
      break;
    }
  }

  if (!nestedModule) {
    return; // No nested module, nothing to do
  }

  LLVM_DEBUG(llvm::dbgs() << "  Unwrapping nested module structure\n");

  // Find the function inside the nested module
  func::FuncOp funcOp = nullptr;
  nestedModule.walk([&](func::FuncOp f) {
    if (f.getName() == funcName)
      funcOp = f;
  });

  if (!funcOp) {
    // Function not found in nested module, leave as-is
    return;
  }

  // Create a new clean module with just the function
  MLIRContext *ctx = module->getContext();
  Location loc = module->getLoc();
  OwningOpRef<ModuleOp> newModule = ModuleOp::create(loc);

  // Clone the function into the new module
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(newModule->getBody());
  builder.clone(*funcOp.getOperation());

  // Replace the old module with the new one
  module = std::move(newModule);
}

LogicalResult cpu::applyTransformSequence(OwningOpRef<ModuleOp> &targetModule,
                                          ModuleOp transformModule,
                                          StringRef sequenceName,
                                          StringRef funcName) {
  // Find the transform entry point
  transform::TransformOpInterface entryPoint =
      transform::detail::findTransformEntryPoint(targetModule.get(),
                                                 transformModule);
  if (!entryPoint) {
    return targetModule->emitError("Could not find transform entry point for ")
           << sequenceName;
  }

  // Apply the transform sequence to the target module
  transform::TransformOptions options;
  if (failed(transform::applyTransformNamedSequence(
          targetModule.get(), entryPoint, transformModule, options))) {
    return targetModule->emitError("Transform interpreter failed for ")
           << sequenceName;
  }

  // Unwrap any nested module structure created by the transform
  unwrapNestedModule(targetModule, funcName);

  return success();
}
