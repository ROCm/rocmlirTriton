//===- RestoreHostCode.cpp - Restore host functions after Triton compilation
//-------------------------===//
//
// Copyright 2026 The MLIR Authors.
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
// This pass restores host functions that were stored during
// RockFuncToTritonFuncPass and converts them to use gpu.launch_func with a
// gpu.binary containing the HSACO.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/compileUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKRESTOREHOSTCODEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-restore-host-code"

using namespace mlir;
using namespace mlir::rock;

static FailureOr<std::pair<gpu::ObjectAttr, DenseMap<StringRef, size_t>>>
createGpuBinary(OpBuilder builder, ModuleOp moduleOp,
                RockRestoreHostCodePassOptions &options,
                SmallVectorImpl<KernelInfo> &kernels) {
  // Get the HSACO binary from the triton.hsaco attribute
  auto hsacoAttr = moduleOp->getAttrOfType<StringAttr>("triton.hsaco");
  if (!hsacoAttr) {
    return failure();
  }

  // Build a map from kernel names to their info
  DenseMap<StringRef, size_t> kernelMap;
  for (size_t i = 0; i < kernels.size(); ++i) {
    kernelMap[kernels[i].name] = i;
  }

  // Create kernel metadata for the gpu.binary
  MLIRContext *ctx = builder.getContext();
  SmallVector<gpu::KernelMetadataAttr> kernelMetadata;
  auto ptrType = LLVM::LLVMPointerType::get(ctx);

  for (const KernelInfo &kernel : kernels) {
    // Create a function type with the correct number of pointer arguments
    // (matching the LLVM function signature from the HSACO).
    // The number of arguments varies by operation type and fusions.
    // Triton adds 2 workspace pointers (global scratch, profile scratch).
    SmallVector<Type> argTypes(kernel.argTypes.size(), ptrType);
    auto kernelFuncType = FunctionType::get(ctx, argTypes, {});

    // Build metadata dictionary with block/grid sizes and prefill info so
    // that downstream consumers can retrieve them.
    SmallVector<NamedAttribute> metadataEntries;
    metadataEntries.push_back(
        builder.getNamedAttr(rock::BlockSizeAttr::getMnemonic(),
                             builder.getI64IntegerAttr(kernel.blockSize)));
    metadataEntries.push_back(
        builder.getNamedAttr(rock::GridSizeAttr::getMnemonic(),
                             builder.getI64IntegerAttr(kernel.gridSize)));
    metadataEntries.push_back(
        builder.getNamedAttr(rock::ClusterSizeAttr::getMnemonic(),
                             builder.getI64IntegerAttr(kernel.clusterSize)));

    if (!kernel.prefillArgs.empty()) {
      SmallVector<Attribute> prefillEntries;
      for (const PrefillInfo &pi : kernel.prefillArgs) {
        SmallVector<NamedAttribute> entry;
        entry.push_back(builder.getNamedAttr(
            "index", builder.getI64IntegerAttr(pi.argIndex)));
        entry.push_back(builder.getNamedAttr("value", pi.initValue));
        prefillEntries.push_back(builder.getDictionaryAttr(entry));
      }
      metadataEntries.push_back(
          builder.getNamedAttr(rock::PrefillAttr::getMnemonic(),
                               builder.getArrayAttr(prefillEntries)));
    }

    auto metadataDict = builder.getDictionaryAttr(metadataEntries);

    auto metadata =
        gpu::KernelMetadataAttr::get(builder.getStringAttr(kernel.name),
                                     /*functionType=*/kernelFuncType,
                                     /*argAttrs=*/nullptr,
                                     /*metadata=*/metadataDict);
    kernelMetadata.push_back(metadata);
  }

  // Create the kernel table
  auto kernelTable = gpu::KernelTableAttr::get(ctx, kernelMetadata);

  // Create the ROCDL target attribute
  // ROCDLTargetAttr::get(ctx, optLevel, triple, chip, features, abiVersion,
  // ...)
  auto rocdlTarget = ROCDL::ROCDLTargetAttr::get(ctx,
                                                 /*optLevel=*/options.optLevel,
                                                 /*triple=*/options.triple,
                                                 /*chip=*/options.arch,
                                                 /*features=*/options.features,
                                                 /*abiVersion=*/"500");

  // Create the object attribute with the HSACO
  // ObjectAttr::get(Attribute target, CompilationTarget format, StringAttr
  // object, ...)
  auto objectAttr = gpu::ObjectAttr::get(
      rocdlTarget,
      gpu::CompilationTarget::Binary, // format enum directly
      hsacoAttr,
      /*properties=*/nullptr, kernelTable);
  return std::make_pair(objectAttr, kernelMap);
}

namespace {

struct RockRestoreHostCodePass
    : public rock::impl::RockRestoreHostCodePassBase<RockRestoreHostCodePass> {
  using RockRestoreHostCodePassBase::RockRestoreHostCodePassBase;

  void runOnOperation() override;

private:
  /// Parse and restore host functions from the serialized attribute
  bool restoreHostFunctions(ModuleOp moduleOp);

  /// Create gpu.binary from HSACO and convert calls to gpu.launch_func
  LogicalResult
  createGpuBinaryAndLaunchFuncs(ModuleOp moduleOp,
                                RockRestoreHostCodePassOptions &options,
                                SmallVector<KernelInfo> &kernels);

  /// Remove kernel LLVM functions (they're now in the binary)
  void removeKernelFunctions(SmallVector<KernelInfo> &kernels);
};

} // end anonymous namespace

/// Parse and restore host functions from the serialized attribute
bool RockRestoreHostCodePass::restoreHostFunctions(ModuleOp moduleOp) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  auto hostFuncsAttr =
      moduleOp->getAttrOfType<ArrayAttr>("rock.host_functions");
  if (!hostFuncsAttr || hostFuncsAttr.empty())
    return false;

  builder.setInsertionPointToEnd(moduleOp.getBody());

  for (Attribute attr : hostFuncsAttr) {
    auto strAttr = dyn_cast<StringAttr>(attr);
    if (!strAttr)
      continue;

    // Parse the function from the stored string
    // Wrap it in a module for parsing
    std::string moduleStr = "module {\n" + strAttr.getValue().str() + "\n}";

    // Use parseSourceString with verification disabled for symbols
    ParserConfig config(ctx, /*verifyAfterParse=*/false);
    auto parsedModule = parseSourceString<ModuleOp>(moduleStr, config);
    if (!parsedModule) {
      emitWarning(moduleOp.getLoc())
          << "Failed to parse stored host function, skipping";
      continue;
    }

    // Move each operation from the parsed module to our module
    for (Operation &op :
         llvm::make_early_inc_range(parsedModule->getBody()->getOperations())) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        continue;
      op.moveBefore(&moduleOp.getBody()->back());
    }
  }

  // Remove the attribute
  moduleOp->removeAttr("rock.host_functions");
  return true;
}

LogicalResult RockRestoreHostCodePass::createGpuBinaryAndLaunchFuncs(
    ModuleOp moduleOp, RockRestoreHostCodePassOptions &options,
    SmallVector<KernelInfo> &kernels) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  Location loc = moduleOp.getLoc();

  FailureOr<std::pair<gpu::ObjectAttr, DenseMap<StringRef, size_t>>>
      maybeBinary = createGpuBinary(builder, moduleOp, options, kernels);
  if (failed(maybeBinary)) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find binary\n");
    return failure();
  }
  gpu::ObjectAttr objectAttr = maybeBinary.value().first;
  DenseMap<StringRef, size_t> kernelMap = maybeBinary.value().second;

  // Create the gpu.binary operation at module level
  // BinaryOp::create(builder, loc, name, offloadingHandler, objects)
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto binaryOp = gpu::BinaryOp::create(builder, loc, "rock_kernels",
                                        /*offloadingHandler=*/nullptr,
                                        builder.getArrayAttr({objectAttr}));

  // Collect all func.call ops that call kernels (use callee name for lookup)
  SmallVector<func::CallOp> callsToConvert;
  moduleOp.walk([&](func::CallOp callOp) {
    StringRef calleeName = callOp.getCalleeAttr().getValue();
    if (kernelMap.count(calleeName)) {
      callsToConvert.push_back(callOp);
    }
  });

  // Convert each call to gpu.launch_func
  for (func::CallOp callOp : callsToConvert) {
    StringRef calleeName = callOp.getCalleeAttr().getValue();
    auto it = kernelMap.find(calleeName);
    if (it == kernelMap.end())
      continue;
    KernelInfo &kernel = kernels[it->second];

    builder.setInsertionPoint(callOp);
    Location callLoc = callOp.getLoc();

    // Create grid and block dimensions
    Value one = arith::ConstantIndexOp::create(builder, callLoc, 1);
    Value gridX = arith::ConstantIndexOp::create(
        builder, callLoc, kernel.gridSize * kernel.clusterSize);
    Value blockX =
        arith::ConstantIndexOp::create(builder, callLoc, kernel.blockSize);

    // Convert memref arguments to LLVM pointers for the kernel
    SmallVector<Value> launchArgs;
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    for (Value operand : callOp.getOperands()) {
      Value memrefVal = operand;

      // If it's a tensor, first convert to memref
      if (auto tensorType = dyn_cast<TensorType>(operand.getType())) {
        auto memrefType =
            MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        memrefVal =
            bufferization::ToBufferOp::create(builder, callLoc, memrefType, operand);
      }

      if (isa<MemRefType>(memrefVal.getType())) {
        // Extract aligned pointer from memref and convert to LLVM pointer
        Value indexPtr = memref::ExtractAlignedPointerAsIndexOp::create(
            builder, callLoc, memrefVal);
        // Convert index to i64 then to pointer
        Value i64Val = arith::IndexCastOp::create(
            builder, callLoc, builder.getI64Type(), indexPtr);
        Value llvmPtr =
            LLVM::IntToPtrOp::create(builder, callLoc, ptrType, i64Val);
        launchArgs.push_back(llvmPtr);
      } else {
        launchArgs.push_back(operand);
      }
    }

    // ResolveKernelLaunchParams has already stripped unused workspace args and
    // baked LDS into the binary, so no padding or dynamic shared memory needed.
    if (launchArgs.size() != kernel.argTypes.size()) {
      callOp.emitError("launch arg count (")
          << launchArgs.size() << ") does not match kernel signature ("
          << kernel.argTypes.size()
          << ") — ResolveKernelLaunchParams may not have run or workspace args "
             "are unexpectedly used";
      return failure();
    }

    // Set cluster size for multi-CTA launch
    std::optional<gpu::KernelDim3> clusterSize;
    if (kernel.clusterSize > 1) {
      Value numCTAsVal =
          arith::ConstantIndexOp::create(builder, callLoc, kernel.clusterSize);
      clusterSize = gpu::KernelDim3{numCTAsVal, one, one};
    }

    // Create gpu.launch_func
    // Note: gpu.launch_func expects kernel operands to have proper types
    gpu::LaunchFuncOp::create(
        builder, callLoc,
        SymbolRefAttr::get(ctx, binaryOp.getName(),
                           {SymbolRefAttr::get(ctx, kernel.name)}),
        gpu::KernelDim3{gridX, one, one},  // grid dimensions
        gpu::KernelDim3{blockX, one, one}, // block dimensions
        /*dynamicSharedMemorySize=*/nullptr, launchArgs,
        /*asyncTokenType=*/nullptr,
        /*asyncDependencies=*/ValueRange{}, clusterSize);

    // gpu.launch_func doesn't return values - it modifies buffers in-place.
    // Replace uses of the func.call results with the corresponding output
    // operands. Support a variable number of results: the last N tensor/memref
    // operands (in reverse order) correspond to the N results (e.g. GEMM: 1
    // result = last operand; attention with return_lse: 2 results = last two).
    const unsigned numResults = callOp.getNumResults();
    if (numResults > 0) {
      // Collect redundant to_buffer + copy ops before replacing (they use the
      // call results). The kernel writes in-place, so these are no-ops;
      // removing them avoids issues with multi-result kernels (e.g.
      // -return_lse).
      SmallVector<Operation *> redundantOpsToErase;
      for (Value result : callOp.getResults()) {
        for (Operation *user : result.getUsers()) {
          if (isa<bufferization::ToBufferOp>(user)) {
            redundantOpsToErase.push_back(user);
            for (Operation *copyUser : user->getResult(0).getUsers())
              if (isa<memref::CopyOp>(copyUser))
                redundantOpsToErase.push_back(copyUser);
          }
        }
      }
      SmallVector<Value> outputOperandsInReverseOrder;
      outputOperandsInReverseOrder.reserve(numResults);
      for (Value operand : llvm::reverse(callOp.getOperands())) {
        if (isa<TensorType, MemRefType>(operand.getType())) {
          outputOperandsInReverseOrder.push_back(operand);
          if (outputOperandsInReverseOrder.size() >= numResults)
            break;
        }
      }
      for (auto [resultIdx, result] : llvm::enumerate(callOp.getResults())) {
        if (resultIdx < outputOperandsInReverseOrder.size())
          result.replaceAllUsesWith(outputOperandsInReverseOrder[resultIdx]);
      }
      for (Operation *op : llvm::reverse(redundantOpsToErase))
        op->erase();
    }

    // Erase the old func.call
    callOp.erase();
  }
  return success();
}

void RockRestoreHostCodePass::removeKernelFunctions(
    SmallVector<KernelInfo> &kernels) {
  // Remove the LLVM kernel functions since they're now in the binary
  for (KernelInfo &kernel : kernels) {
    if (kernel.llvmFunc)
      kernel.llvmFunc.erase();
  }
}

void RockRestoreHostCodePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Build options from pass parameters
  RockRestoreHostCodePassOptions options;
  options.triple = triple.getValue();
  options.arch = arch.getValue();
  options.features = features.getValue();
  options.optLevel = optLevel.getValue();

  // Mark the module as containing GPU code
  moduleOp->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                    builder.getUnitAttr());

  // Collect kernel information from LLVM functions
  SmallVector<KernelInfo> kernels;
  if (failed(rock::collectKernelInfo(moduleOp, kernels))) {
    signalPassFailure();
    return;
  }

  // Restore host functions from the serialized attribute (if any).
  bool hasHostFuncs = restoreHostFunctions(moduleOp);

  // Create gpu.binary and (if host functions were restored) convert calls
  // to gpu.launch_func. The gpu.binary is needed even without host functions
  // so that downstream consumers (e.g. mlirGetKernelAttrs) can find kernel
  // metadata.
  if (!kernels.empty()) {
    if (failed(createGpuBinaryAndLaunchFuncs(moduleOp, options, kernels)))
      return signalPassFailure();
    // Only remove LLVM kernel functions when host functions were restored
    // (gpu.launch_func now references the kernel via gpu.binary).
    if (hasHostFuncs)
      removeKernelFunctions(kernels);
  }
}
