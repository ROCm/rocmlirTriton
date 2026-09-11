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
#include "mlir/Dialect/Rock/Tuning/QuickTuningShardDb.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <map>

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

/// Environment variable overriding how many configs a known problem's
/// quick-tuning list may hold, the recorded bests included. Only the list of a
/// problem the database has measurements for is capped; an unknown problem
/// still sweeps the whole set cover.
inline constexpr StringLiteral kQuickTuningListMaxEnvVar =
    "ROCMLIR_QUICK_TUNING_LIST_MAX";

/// Cap used when `kQuickTuningListMaxEnvVar` is unset or unparseable.
inline constexpr size_t kQuickTuningListMaxDefault = 30;

template <typename ParamsType>
class ParamLookupTable {
public:
  /// The quick-tuning list for a problem: the configs to sweep, best first.
  ///
  /// `problemHash` identifies the problem within the resolved key (see
  /// QuickTuningProblemKey.h). When the database holds measurements for it,
  /// the list leads with the best non-split-K and best split-K config recorded
  /// for that exact problem and is then backfilled from the set cover, without
  /// repeats, to a total of `kQuickTuningListMaxEnvVar`. Otherwise -- an
  /// unknown problem, an untuned key, or `kQuickTuningNoProblem` from a caller
  /// that has no problem to name -- the list is the whole set cover, in its
  /// recorded order, exactly as before per-problem data existed.
  static SmallVector<StringRef>
  lookup(StringRef arch, KernelType op, Type dataType,
         uint64_t problemHash = kQuickTuningNoProblem);

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
  static StringRef pickClosestRelative(StringRef target,
                                       ArrayRef<StringRef> relatives);

  // Every key the compiled-in quick-tuning database holds, mapped to the shard
  // holding its data. Ordered rather than hashed because `getRelatives`
  // depends on iterating it in key order.
  static const std::map<StringRef, const QuickTuningShard *> &getTable();

  static std::string getKernelTypeString(KernelType kernelType);

  // Get all related entries sorted lexicographically
  static SmallVector<StringRef, 12> getRelatives(StringRef target);
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_PARAM_LOOKUP_TABLE_H
