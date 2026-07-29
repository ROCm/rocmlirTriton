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
//   classifyDotLowering / getRequiredDotKMultiple
//     ->
//     triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp
//        (chooseMfmaInstruction, BlockedToWMMA,
//         AccelerateBlocked::isLegalFMAForm / ::tryLegalizeFMA)
//===----------------------------------------------------------------------===//

#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

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

/// How Triton's AMD backend will end up serving a `tt.dot`.
enum class DotLowering {
  /// An MFMA (CDNA) or WMMA (RDNA) matrix-core instruction.
  MatrixCore,
  /// A packed `v_dot4` / `v_dot2` instruction. Exact.
  PackedVDot,
  /// Scalar `llvm.fmuladd` over operands that already share a float element
  /// type. Exact.
  ScalarFMA,
  /// None of the above applies, so `AccelerateBlocked::tryLegalizeFMA` recasts
  /// A, B and the accumulator to a common float type and dots in that type.
  /// With an integer accumulator the common type is always f32, so sums beyond
  /// 2^24 are rounded.
  UpcastedFMA,
};

/// Classify how `isaFamily` will serve a `tt.dot` whose operands have element
/// types `aElemTy`/`bElemTy`, whose accumulator element type is `cElemTy`, and
/// whose contraction length is `kDim`.
///
/// Mirrors the selection in AccelerateAMDMatmul.cpp: `chooseMfmaInstruction` /
/// `BlockedToWMMA` for the matrix-core cases, then
/// `AccelerateBlocked::isLegalFMAForm` and `::tryLegalizeFMA` for the rest.
///
/// Matrix-core availability is deliberately over-approximated: every MFMA tile
/// shape is tried rather than only the one `chooseMfmaInstruction` derives from
/// M and N, because a perf config can force `matrixInstructionSize`. Callers
/// therefore get false `MatrixCore` answers before they get false
/// `UpcastedFMA` ones, which keeps this predicate from rejecting a dot the
/// backend can in fact accelerate.
DotLowering classifyDotLowering(triton::amdgpu::ISAFamily isaFamily,
                                Type aElemTy, Type bElemTy, Type cElemTy,
                                int64_t kDim);

/// The value K must be a multiple of for a dot that no matrix-core instruction
/// can serve to stay on an exact path: 4 for `i8 x i8 -> i32` (`v_dot4`), 2 for
/// the 16-bit `-> f32` cases (`v_dot2`), and 1 otherwise.
///
/// This is the AMD counterpart of `get_min_dot_size()` in
/// third_party/amd/backend/compiler.py, which returns (1, 1, 1) unconditionally
/// and so constrains nothing. Only the K component is mirrored here: the M and
/// N bounds really are 1 on AMD, because an M/N that no matrix-core tile can
/// serve falls back to FMA, which is exact.
int64_t getRequiredDotKMultiple(triton::amdgpu::ISAFamily isaFamily,
                                Type aElemTy, Type bElemTy, Type cElemTy);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H
