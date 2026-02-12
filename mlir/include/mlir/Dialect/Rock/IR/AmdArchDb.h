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
#include "mlir/Support/LLVM.h"

namespace mlir {
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

/// Check if hardware matrix acceleration is available for the given GEMM op.
/// Returns true if any acceleration (MFMA, WMMA, ScaledMFMA, ScaledWMMA)
/// is available for the operation's types on the specified architecture.
bool hasAccel(StringRef arch, RockGemmWrapperInterface gemmOp);

/// Same as above but for gemm+gemm
std::pair<MatrixAccelKind, MatrixAccelKind>
getMatrixAccelKind(StringRef arch, RockGemmGemmWrapperInterface gemmOp);

/// Same as above but for gemm+gemm
bool hasAccel(StringRef arch, RockGemmGemmWrapperInterface gemmOp);

/// Get minimum number of CUs per arch
int64_t getMinNumCU(StringRef arch);

/// Get maximum number of chiplets per arch
int64_t getMaxNumChiplets(StringRef arch);

/// Get maximum number of waves per EU per arch
int64_t getMaxWavesPerEU(StringRef arch);

/// Whether there's fast atomic add support
bool isFastAtomicAddSupported(StringRef arch, Type type);

/// Whether there's fast atomic max support
bool isFastAtomicMaxSupported(StringRef arch, Type type);

/// Get wave size
int64_t getWaveSize(StringRef arch);

/// Get LDS size
int64_t getLDSSize(StringRef arch);

/// Check if architecture supports TDM (Tensor Descriptor Memory)
bool supportsTDM(StringRef arch);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_IR_AMDARCHDB_H
