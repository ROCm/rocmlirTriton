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

namespace mlir {

namespace triton {
enum class ScaleDotElemType : uint32_t;
namespace AMD {
enum class ISAFamily;
} // namespace AMD
} // namespace triton

namespace rock {

/// Get MFMA version from ISA family.
/// Mirrors the internal getMfmaVersion() in AccelerateAMDMatmul.cpp.
int getMfmaVersion(triton::AMD::ISAFamily isaFamily);

/// Get WMMA version from architecture chip string (e.g. "gfx1100").
/// Mirrors the internal getWmmaVersion() in AccelerateAMDMatmul.cpp.
int getWmmaVersion(StringRef arch);

/// Map an MLIR element type to the corresponding triton::ScaleDotElemType.
/// Covers F8 (E4M3, E5M2), F6 (E2M3, E3M2), F4 (E2M1), BF16, and FP16.
/// Returns failure() for unsupported types.
///
/// Adapted from mlirTypeToScaledElemType in AccelerateAMDMatmul.cpp with
/// additional BF16/FP16 coverage.
FailureOr<triton::ScaleDotElemType> mlirTypeToScaleDotElemType(Type type);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H
