//===- ParamLookupTable.h - MLIR tuning parameter lookup ------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR tuning parameter lookup
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_PARAM_LOOKUP_TABLE_H
#define MLIR_DIALECT_ROCK_PARAM_LOOKUP_TABLE_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace rock {

// Canonicalize an arch string to its leading `gfxNNN...` token, stripping any
// feature suffix (e.g. "gfx942:sramecc+" -> "gfx942"). Asserts if `arch`
// contains no gfx token. Shared by every tuning-table lookup so their keys
// agree (see also LdsBlacklist.cpp).
StringRef normalizeArch(StringRef arch);

// Canonicalize a data type to its tuning-key spelling: all 4-bit floats -> f4,
// all 8-bit floats -> fp8, all 16-bit floats (incl. bf16) -> f16; other types
// print as-is with a leading integer sign char ('s'/'u') dropped. Shared so the
// keys emitted here match those baked into the generated .inc tables.
std::string getDataTypeString(Type dataType);

template <typename ParamsType>
class ParamLookupTable {
public:
  static ArrayRef<StringRef> lookup(StringRef arch, KernelType op,
                                    Type dataType);

  // Finds the lexicographically closest architecture variant when the exact
  // target key is not found in the lookup table.
  //
  // A "relative" entry must have:
  // - Same suffix (operation + data type, e.g., "_gemm_f16")
  // - Same architecture prefix (e.g., "gfx9" for gfx908, gfx90a, gfx942)
  //
  // Example: If target "gfx1151_gemm_f16" is missing but "gfx1101_gemm_f16"
  // and "gfx1201_gemm_f16" exist, this picks the lexicographically closest one
  // (gfx1101_gemm_f16). This enables graceful fallback between similar GPU
  // architectures.
  static StringRef findFallback(StringRef target);

private:
  static constexpr char separator = '_';

  static std::string makeKey(StringRef arch, KernelType op, Type dataType) {
    return (Twine(arch) + Twine(separator) + getKernelTypeString(op) +
            Twine(separator) + getDataTypeString(dataType))
        .str();
  }

  // Returns the closest datatype to borrow tuning configs from when `dataType`
  // has no entries of its own (e.g. fp8 -> i8, f4 -> i8). Returns an empty
  // StringRef when there is no fallback datatype.
  static StringRef getFallbackDataType(StringRef dataType);

  static const std::map<StringRef, ArrayRef<StringRef>> &getTable() {
    static const std::map<StringRef, ArrayRef<StringRef>> table = buildTable();
    return table;
  }

  static std::map<StringRef, ArrayRef<StringRef>> buildTable();

  static std::string getKernelTypeString(KernelType kernelType);

  // Get all related entries sorted lexicographically
  static SmallVector<StringRef, 12> getRelatives(StringRef target);
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PARAM_LOOKUP_TABLE_H
