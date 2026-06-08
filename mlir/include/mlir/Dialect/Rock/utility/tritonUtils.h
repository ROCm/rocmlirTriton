#ifndef MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H

//===- tritonUtils.h - Triton-dependent utilities for Rock ----------------===//
//
// Centralizes C++ replicas of Triton-internal functions that must be kept in
// sync on every Triton version bump.  Having them in one place makes the
// bump_triton_version.md checklist easier to follow.
//
// Upstream sources:
//   getMfmaVersion / getWmmaVersion
//     ->
//     triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp
//   mlirTypeToScaleDotElemType
//     ->
//     triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp
//        (mlirTypeToScaledElemType, extended with BF16/FP16)
//===----------------------------------------------------------------------===//

#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include "hip/hip_runtime_api.h"

namespace mlir {

namespace triton {
enum class ScaleDotElemType : uint32_t;
namespace amdgpu {
enum class ISAFamily;
} // namespace amdgpu
} // namespace triton

namespace rock {

/// Get MFMA version from ISA family.
/// Mirrors the internal getMfmaVersion() in AccelerateAMDMatmul.cpp.
int getMfmaVersion(triton::amdgpu::ISAFamily isaFamily);

/// Get WMMA version from ISA family.
/// Mirrors the internal getWmmaVersion() in AccelerateAMDMatmul.cpp.
int getWmmaVersion(triton::amdgpu::ISAFamily isaFamily);

/// Return true if `t` is one of the types in Triton's TT_Float set.
/// Mirrors the TT_Float type constraint from TritonTypes.td:
///   {F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64}
bool isTTFloat(Type t);

/// Map an MLIR element type to the corresponding triton::ScaleDotElemType.
/// Covers F8 (E4M3, E5M2), F6 (E2M3, E3M2), F4 (E2M1), BF16, and FP16.
/// Returns failure() for unsupported types.
///
/// Adapted from mlirTypeToScaledElemType in AccelerateAMDMatmul.cpp with
/// additional BF16/FP16 coverage.
FailureOr<triton::ScaleDotElemType> mlirTypeToScaleDotElemType(Type type);

// Mirrors _launch() from external/triton/third_party/amd/backend/driver.c
LogicalResult launchKernel(hipFunction_t function, uint32_t gridX,
                           uint32_t blockSize, uint32_t shared_memory,
                           uint32_t num_ctas, hipStream_t stream,
                           void **params);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H
