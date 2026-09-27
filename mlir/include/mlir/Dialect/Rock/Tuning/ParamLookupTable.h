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
#include "mlir/Dialect/Rock/Tuning/QuickTuningProblemMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
namespace rock {

// Canonicalize an arch string to its leading `gfxNNN...` token, stripping any
// feature suffix (e.g. "gfx942:sramecc+" -> "gfx942"). Asserts if `arch`
// contains no gfx token. Shared by every tuning-table lookup so their keys
// agree (see also LdsBlacklist.cpp).
StringRef normalizeArch(StringRef arch);

// Canonicalize a data type to its tuning-key spelling: all 4-bit floats -> f4,
// all 8-bit floats -> fp8, 16-bit floats except bf16 -> f16; other
// types print as-is with a leading integer sign char ('s'/'u') dropped. Shared
// so the keys emitted here match those baked into the generated .inc tables.
std::string getDataTypeString(Type dataType);

template <typename ParamsType>
class ParamLookupTable {
public:
  /// Perfconfigs to try, narrowed to `problemKey` when this key has a
  /// compatible generated map and an entry for its problem hash.
  ///
  /// `supportsSplitK` selects between the set-cover tables: the regular one
  /// when true and the no-split-K one otherwise, with fallback between the
  /// pair always enabled. It does not gate the per-problem rankings, whose
  /// split-K-illegal members are dropped by the caller.
  static SmallVector<StringRef>
  lookup(StringRef arch, KernelType op, Type dataType, bool supportsSplitK,
         std::optional<QuickTuningProblemKey> problemKey = std::nullopt);

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
  //
  // When the target key itself is absent, the kernel type and data type may
  // also be substituted (see getFallbackKernelType and getFallbackDataType).
  // Candidates are tried cheapest-first along three axes: kernel type is
  // cheapest, then architecture, then data type. So a same-architecture
  // attention list is preferred over the same fusion tuned for a relative
  // architecture, and both are preferred over any change of precision.
  // Returns an empty StringRef when nothing applies.
  //
  // Split-K pairing is unconditional: if the regular entry is missing, its
  // no-split-K pair is tried before kernel type, architecture, or data type,
  // and candidates on those later axes may also come from either table.
  static StringRef findFallback(StringRef target);

private:
  static constexpr char separator = '_';

  static std::string makeKey(StringRef arch, KernelType op, Type dataType) {
    return (Twine(arch) + Twine(separator) + getKernelTypeString(op) +
            Twine(separator) + getDataTypeString(dataType))
        .str();
  }

  // Returns the closest datatype to borrow tuning configs from when `dataType`
  // has no entries of its own (e.g. bf16 -> f16, fp8 -> i8, f4 -> i8). Returns
  // an empty StringRef when there is no fallback datatype.
  static StringRef getFallbackDataType(StringRef dataType);

  // Returns the closest kernel type to borrow tuning configs from when
  // `kernelType` has no entries of its own. The gemm+elementwise+gemm and
  // conv+elementwise+gemm fusions lower through the same gridwise code and
  // share attention's perf-config format, so they can use the attention lists
  // until they are tuned in their own right. Returns an empty StringRef when
  // there is no fallback kernel type.
  static StringRef getFallbackKernelType(StringRef kernelType);

  // Splits `key` into its `<arch>_<kernelType>_<dataType>` components. Returns
  // false, leaving the outputs untouched, when `key` has no such structure.
  static bool splitKey(StringRef key, StringRef &arch, StringRef &kernelType,
                       StringRef &dataType);

  // Of `relatives`, all of which share `target`'s suffix and architecture
  // family, returns the one whose key diverges from `target` latest, preferring
  // the newer architecture when two are equidistant.
  //
  // `relatives` must be non-empty and sorted ascending: the search for
  // `target`'s neighbours is a binary search, and the two endpoint cases
  // dereference the extremes directly. Both preconditions are asserted.
  static StringRef pickClosestRelative(StringRef target,
                                       ArrayRef<StringRef> relatives);

  static const std::map<StringRef, ArrayRef<StringRef>> &getTable() {
    static const std::map<StringRef, ArrayRef<StringRef>> table = buildTable();
    return table;
  }

  static const std::map<StringRef, ArrayRef<StringRef>> &getNoSplitKTable() {
    static const std::map<StringRef, ArrayRef<StringRef>> table =
        buildNoSplitKTable();
    return table;
  }

  static std::map<StringRef, ArrayRef<StringRef>> buildTable();
  static std::map<StringRef, ArrayRef<StringRef>> buildNoSplitKTable();

  static const llvm::StringMap<QuickTuningProblemMap> &getProblemMap() {
    static const llvm::StringMap<QuickTuningProblemMap> map = buildProblemMap();
    return map;
  }

  static llvm::StringMap<QuickTuningProblemMap> buildProblemMap();

  static std::string getKernelTypeString(KernelType kernelType);

  // Get all related entries across both tables, sorted lexicographically.
  static SmallVector<StringRef, 12> getRelatives(StringRef target);
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PARAM_LOOKUP_TABLE_H
