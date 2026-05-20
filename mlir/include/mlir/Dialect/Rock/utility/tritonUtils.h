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
//   emitFloatAtomicMax
//     -> triton/python/triton/language/semantic.py::atomic_max
//===----------------------------------------------------------------------===//

#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include "hip/hip_runtime_api.h"

namespace mlir {

class Operation;
class PatternRewriter;
class Value;

namespace triton {
enum class ScaleDotElemType : uint32_t;
enum class MemSemantic : uint32_t;
enum class MemSyncScope : uint32_t;
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

/// Emit a float atomic_max as two integer atomic_rmw ops on disjoint masks,
/// mirroring upstream Triton's frontend trick in
/// python/triton/language/semantic.py::atomic_max.
///
/// The float operand is bitcast to a signless integer of matching width and
/// the operation is split by the sign bit:
///   * positive lanes (signbit == 0)  -> RMWOp::MAX  (signed int)
///   * negative lanes (signbit == 1)  -> RMWOp::UMIN (unsigned int)
///
/// For non-negative IEEE floats the int reinterpretation preserves order, so
/// a signed integer MAX is equivalent to fmax. For negative IEEE floats, a
/// larger magnitude corresponds to a larger unsigned bit pattern, so unsigned
/// MIN picks the one closest to zero, i.e. the maximum among negatives.
///
/// Only f32 is supported today; other widths emit a diagnostic on `op`.
LogicalResult emitFloatAtomicMax(PatternRewriter &rewriter, Operation *op,
                                 Value value, Value ptrTensor, Value mask,
                                 triton::MemSemantic sem,
                                 triton::MemSyncScope scope);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H
