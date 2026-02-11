#ifndef MLIR_DIALECT_ROCK_UTILITY_COMPILEUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_COMPILEUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/Pipelines/Pipelines.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
namespace rock {

/// Information about a compiled kernel
struct KernelInfo {
  std::string name;
  LLVM::LLVMFuncOp llvmFunc;
  int64_t gridSize;
  int64_t blockSize;
  int64_t sharedMemorySize;
  SmallVector<Type> argTypes; // Original func argument types
};

/// Find an LLVM kernel function by name in the module.
/// Returns nullptr if not found.
LLVM::LLVMFuncOp findKernelFunc(ModuleOp moduleOp, StringRef kernelName);

/// Populate argTypes for kernels by walking LLVM functions in the module.
/// Each kernel in the vector should have its name set before calling this.
void populateKernelArgTypes(ModuleOp moduleOp,
                            SmallVectorImpl<KernelInfo> &kernels);

/// Returns the argument count for a kernel by name.
std::optional<unsigned> getKernelArgCount(ModuleOp moduleOp,
                                          StringRef kernelName);

/// Create a gpu.ObjectAttr from the HSACO binary in moduleOp and kernel info.
/// Returns the ObjectAttr and a mapping from kernel names to their indices.
FailureOr<std::pair<gpu::ObjectAttr, DenseMap<StringRef, size_t>>>
createGpuBinary(OpBuilder builder, ModuleOp moduleOp,
                SmallVectorImpl<KernelInfo> &kernels);

LogicalResult fillCompilationConfigs(StringAttr perfConfig,
                                     rock::TritonOptions &tritonOpts,
                                     rock::BackendOptions &backendOpts);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_COMPILEUTILS_H
