//===- AmdArchDb.h - Database of AMD GPU features ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_AMDARCHDB_H
#define MLIR_DIALECT_ROCK_IR_AMDARCHDB_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <tuple>

namespace mlir {

namespace triton {
namespace amdgpu {
enum class ISAFamily;
} // namespace amdgpu
} // namespace triton

namespace rock {

/// Result of checking matrix acceleration support
enum class MatrixAccelKind {
  None,       /// No hardware acceleration available
  MFMA,       /// AMD CDNA Matrix Fused Multiply-Add
  WMMA,       /// AMD RDNA Wave Matrix Multiply-Accumulate
  ScaledMFMA, /// AMD CDNA4 Scaled Matrix Fused Multiply-Add (F8/F6/F4 with scales)
  ScaledWMMA  /// AMD gfx1250 Scaled Wave Matrix Multiply-Accumulate (F8/F4 with scales)
};

/// Check if hardware matrix acceleration (MFMA or WMMA) is available
/// for the given architecture and data types.
///
/// \param arch The target architecture string (e.g., "gfx942", "gfx1100")
/// \param inputTypeA Element type of the first input matrix
/// \param inputTypeB Element type of the second input matrix
/// \param scaleAType Optional scale tensor type for input A. If provided
///                   (non-null), indicates this is a scaled GEMM operation.
/// \param scaleBType Optional scale tensor type for input B. If provided
///                   (non-null), indicates this is a scaled GEMM operation.
/// \return The kind of matrix acceleration available, or None if not supported
///
/// Scaled GEMM behavior (when scaleAType or scaleBType is provided):
/// - If input types don't match requirements, falls back to regular
///   (non-scaled) MFMA/WMMA if available for the given types
MatrixAccelKind getMatrixAccelKind(StringRef arch, Type inputTypeA,
                                   Type inputTypeB, Type scaleAType = Type(),
                                   Type scaleBType = Type());

/// Get the matrix acceleration kind for a GEMM operation.
/// Note: This does not check for scale types - use the full getMatrixAccelKind()
/// with explicit scale types for scaled operations.
MatrixAccelKind getMatrixAccelKind(StringRef arch,
                                   RockGemmWrapperInterface gemmOp);

/// Extract the bare gfx token from an architecture string, stripping any
/// target-triple prefix and `:feature` suffixes.
std::tuple<StringRef, unsigned> parseArchString(StringRef arch);

/// Extract the ISAFamily and chip name from an architecture string. The chip
/// name is the bare gfx token (e.g. "gfx1100"), with any target-triple prefix
/// and `:feature` suffixes stripped.
std::tuple<triton::amdgpu::ISAFamily, StringRef> getArch(StringRef arch);

/// Whether this architecture belongs to the CDNA (data-center, "big GPU")
/// line. Mirrors Triton's `triton::amdgpu::isCDNA`, which counts gfx1250 as
/// CDNA even though its matrix instructions are WMMA.
bool isCDNA(StringRef arch);

/// Whether this architecture belongs to the RDNA (consumer) line. Mirrors
/// Triton's `triton::amdgpu::isRDNA`. Note that gfx906 (GCN5_1) is in neither
/// line, so `isCDNA` and `isRDNA` are not complements.
bool isRDNA(StringRef arch);

/// Check if hardware matrix acceleration is available for the given GEMM op.
/// Returns true if any acceleration (MFMA, WMMA, ScaledMFMA, ScaledWMMA)
/// is available for the operation's types on the specified architecture.
bool hasAccel(StringRef arch, RockGemmWrapperInterface gemmOp);

/// Same as above but for gemm+gemm
std::pair<MatrixAccelKind, MatrixAccelKind>
getMatrixAccelKind(StringRef arch, RockGemmGemmWrapperInterface gemmOp);

/// Same as above but for gemm+gemm
bool hasAccel(StringRef arch, RockGemmGemmWrapperInterface gemmOp);

/// The K extent of the narrowest matrix instruction this arch offers for the
/// given operand types, at an instruction tile of `instrNonKDim` x
/// `instrNonKDim`. Fails when the arch has no matrix instruction for them,
/// which callers should read as "no instruction constrains K here" rather than
/// as an error.
///
/// `instrNonKDim` is the `matrixInstrNonkdim` perf-config field, and is ignored
/// on WMMA, whose instructions are all 16x16.
FailureOr<int64_t> getAccelInstrMinKDim(StringRef arch, Type inputTypeA,
                                        Type inputTypeB, uint32_t instrNonKDim,
                                        Type scaleAType = Type(),
                                        Type scaleBType = Type());

/// Same as above, for the operand types of a GEMM operation.
FailureOr<int64_t> getAccelInstrMinKDim(StringRef arch,
                                        RockGemmWrapperInterface gemmOp,
                                        uint32_t instrNonKDim);

/// Get minimum number of CUs per arch
int64_t getMinNumCU(StringRef arch);

/// Get maximum number of chiplets per arch
int64_t getMaxNumChiplets(StringRef arch);

/// Infer the active chiplet count from the architecture and live CU count.
int64_t inferNumChiplets(StringRef arch, int64_t numCUs);

/// Get maximum number of waves per EU per arch
int64_t getMaxWavesPerEU(StringRef arch);

/// Get the SIMD VGPR file size per execution unit for this arch.
int64_t getVGPRsPerEU(StringRef arch);

/// Whether this architecture has any FP8 matrix-acceleration intrinsics
/// (MFMA on CDNA3+, WMMA on RDNA4+ / GFX1250). Independent of any specific
/// operation; useful for gating test suites that require an FP8 hardware
/// reference (e.g. hipBLASLt validation).
bool archSupportsAccelFp8(StringRef arch);

/// Whether this architecture has scaled-GEMM matrix acceleration (scaled
/// MFMA on CDNA4 / gfx950, scaled WMMA on GFX1250).
bool archSupportsScaledGemm(StringRef arch);

/// Whether this architecture can lower a scaled GEMM whose sub-byte (fp4)
/// operand is packed along the non-K (M/N) dimension rather than K.
bool archSupportsNonKPackedScaledInput(StringRef arch);

/// Get wave size
int64_t getWaveSize(StringRef arch);

/// Get LDS size
int64_t getLDSSize(StringRef arch);

/// Get the size in bytes of the last-level cache for this chip (the AMD
/// Infinity Cache / MALL where present, otherwise the L2), taking the maximum
/// across the cache configurations that chip ships in. Chips whose caches AMD
/// has not published are estimated from their closest published sibling, or
/// fall back to the maximum across their ISA family.
int64_t getLastLevelCacheSize(StringRef arch);

/// Whether the architecture supports multi-CTA
bool supportsMultiCTALaunch(StringRef arch);

/// Get maximum number of CTAs for cluster launch.
/// Returns 1 if multi-CTA is not supported.
int64_t getMaxNumCTAs(StringRef arch);

/// Get the maximum supported `kpack` perf-config value for this arch.
int64_t getMaxKpack(StringRef arch);

/// Whether an f32 `tt.dot` should be emulated with the 3xBF16 trick
/// (`InputPrecision::BF16x3`) instead of being issued as an IEEE f32 dot.
/// This is the heuristic behind the `useBf16x3ForF32` perfConfig knob's `-1`
/// setting; an explicit `0`/`1` in the perfConfig overrides it.
bool preferBf16x3ForF32Dot(StringRef arch);

/// Check if architecture supports TDM (Tensor Descriptor Memory)
bool supportsTDM(StringRef arch);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_IR_AMDARCHDB_H
