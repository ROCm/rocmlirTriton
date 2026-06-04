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
  int64_t clusterSize = -1;
  SmallVector<Type> argTypes;
  SmallVector<PrefillInfo> prefillArgs;
};

/// Collect kernel information from a compiled module.
/// Walks LLVM functions with KernelAttr and extracts launch parameters:
/// - Block size from ttg.num-warps (or ttg.total-num-warps) *
/// ttg.threads-per-warp
/// - Grid size from rock.grid_size.{kernelName} module attribute
/// - Argument types and count from LLVM function signature
/// Returns failure if required attributes are missing.
LogicalResult collectKernelInfo(ModuleOp moduleOp,
                                SmallVectorImpl<KernelInfo> &kernels);

/// Create a gpu.ObjectAttr from the HSACO binary in moduleOp and kernel info.
/// Returns the ObjectAttr and a mapping from kernel names to their indices.
FailureOr<std::pair<gpu::ObjectAttr, DenseMap<StringRef, size_t>>>
createGpuBinary(OpBuilder builder, ModuleOp moduleOp,
                SmallVectorImpl<KernelInfo> &kernels);

/// Retrieve the prefill argument array from the module's gpu.binary, or
/// a default-constructed ArrayAttr if none exists. Returns failure if the
/// binary does not contain exactly one kernel.
FailureOr<ArrayAttr> getPrefillArrayFromBinary(ModuleOp moduleOp);

/// Populate `tritonOpts` (numWarps, numCTAs, numStages, matrixInstrNonkdim,
/// kpack) and `backendOpts` (numWarps, numCTAs, wavesPerEU) from a perf-config
/// attribute. `perfConfig` must implement `RockTuningParamAttrInterface` (i.e.
/// be a `GemmParamsAttr` or `GemmGemmParamsAttr`); otherwise this fails.
LogicalResult fillCompilationConfigs(Attribute perfConfig,
                                     rock::TritonOptions &tritonOpts,
                                     rock::BackendOptions &backendOpts);

/// Same as above but parses a perf-config string. Tries `GemmParamsAttr` and
/// then `GemmGemmParamsAttr`; returns failure if neither parses or if the
/// resulting attribute can't be applied.
LogicalResult fillCompilationConfigs(MLIRContext *ctx, StringRef perfConfig,
                                     rock::TritonOptions &tritonOpts,
                                     rock::BackendOptions &backendOpts);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_COMPILEUTILS_H
