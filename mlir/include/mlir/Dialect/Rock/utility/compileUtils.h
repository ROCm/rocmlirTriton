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
  unsigned argCount = 0;      // Total argument count
};

/// Collect kernel information from a compiled module.
/// Walks LLVM functions with KernelAttr and extracts launch parameters:
/// - Block size from ttg.num-warps (or ttg.total-num-warps) * ttg.threads-per-warp
/// - Shared memory from ttg.shared
/// - Grid size from rock.grid_size.{kernelName} module attribute
/// - Argument types and count from LLVM function signature
/// Returns failure if LDS usage exceeds maxSharedMemPerWG or required
/// attributes are missing.
LogicalResult collectKernelInfo(ModuleOp moduleOp, int64_t maxSharedMemPerWG,
                                SmallVectorImpl<KernelInfo> &kernels);

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
