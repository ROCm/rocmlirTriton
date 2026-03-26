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

/// A kernel argument that must be pre-initialized before launch.
struct PrefillInfo {
  unsigned argIndex;
  Attribute initValue;
};

/// Information about a compiled kernel
struct KernelInfo {
  std::string name;
  LLVM::LLVMFuncOp llvmFunc;
  int64_t gridSize = -1;
  int64_t blockSize = -1;
  int64_t sharedMemorySize = 0;
  SmallVector<Type> argTypes;
  SmallVector<PrefillInfo> prefillArgs;
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

/// Check that the module's LDS (shared memory) usage does not exceed the
/// hardware limit. Reads the "ttg.shared" attribute from `moduleOp` and
/// compares it against `maxSharedMemPerWG`.
/// Returns the shared memory size on success, or failure if it exceeds the
/// limit.
FailureOr<int64_t> checkLDSUsage(ModuleOp moduleOp, int64_t maxSharedMemPerWG);

/// Create a gpu.ObjectAttr from the HSACO binary in moduleOp and kernel info.
/// Returns the ObjectAttr and a mapping from kernel names to their indices.
FailureOr<std::pair<gpu::ObjectAttr, DenseMap<StringRef, size_t>>>
createGpuBinary(OpBuilder builder, ModuleOp moduleOp,
                SmallVectorImpl<KernelInfo> &kernels);

/// Retrieve the prefill argument array from the module's gpu.binary, or
/// a default-constructed ArrayAttr if none exists. Returns failure if the
/// binary does not contain exactly one kernel.
FailureOr<ArrayAttr> getPrefillArrayFromBinary(ModuleOp moduleOp);

/// Parse a performance-config string into Triton and backend compilation
/// options. Attempts to interpret `perfConfig` as a GemmParamsAttr or
/// GemmGemmParamsAttr and populates `tritonOpts` (numWarps, numCTAs,
/// numStages, matrixInstrNonkdim, kpack) and `backendOpts` (numWarps,
/// numCTAs, wavesPerEU) accordingly.
/// Returns failure if `perfConfig` does not match any known parameter format.
LogicalResult fillCompilationConfigs(Attribute perfConfig,
                                     rock::TritonOptions &tritonOpts,
                                     rock::BackendOptions &backendOpts);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_COMPILEUTILS_H
