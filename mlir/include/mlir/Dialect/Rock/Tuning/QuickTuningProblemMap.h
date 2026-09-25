//===- QuickTuningProblemMap.h - Per-problem quick tuning map -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_QUICK_TUNING_PROBLEM_MAP_H
#define MLIR_DIALECT_ROCK_QUICK_TUNING_PROBLEM_MAP_H

#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/xxhash.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace rock {

using QuickTuningProblemKeyHash = uint64_t;
using QuickTuningTableLookUpKeyVersionHash = uint64_t;

/// The hash identifying one problem and a hash of the ordered field names used
/// to build it. The latter changes automatically when the lookup-key schema
/// changes, without depending on the problem's field values.
struct QuickTuningProblemKey {
  QuickTuningProblemKey(QuickTuningProblemKeyHash hash,
                        QuickTuningTableLookUpKeyVersionHash versionHash,
                        std::string unsupportedFields = {},
                        std::string untunableFields = {})
      : hash(hash), versionHash(versionHash),
        unsupportedFields(std::move(unsupportedFields)),
        untunableFields(std::move(untunableFields)) {}

  bool hasUnsupportedFields() const {
    return !unsupportedFields.empty() || !untunableFields.empty();
  }

  QuickTuningProblemKeyHash hash;
  QuickTuningTableLookUpKeyVersionHash versionHash;
  /// Fields or modes understood by the compiler but not represented by the
  /// shipped per-problem maps. Such a key must fall back to the set cover
  /// rather than reusing a ranking measured for a different problem schema.
  /// Exhaustively tuning the mode and regenerating the maps would add it, so
  /// the lookup diagnoses these.
  std::string unsupportedFields;
  /// Like `unsupportedFields`, but for modes the tuning pipeline cannot
  /// express (e.g. asymmetric padding, which the tuning problem string does
  /// not record). Retuning cannot help, so these fall back silently.
  std::string untunableFields;
};

inline QuickTuningProblemKeyHash hashQuickTuningProblemKey(StringRef key) {
  return llvm::xxh3_64bits(key);
}

inline QuickTuningTableLookUpKeyVersionHash
hashQuickTuningTableLookUpKeyVersion(StringRef fields) {
  return llvm::xxh3_64bits(fields);
}

/// Key identifying `op`'s problem, or nullopt when it has none. Only the
/// operation's own fields take part, so the table lookup key must still carry
/// the architecture and data type.
std::optional<QuickTuningProblemKey>
getQuickTuningProblemKey(RockGemmWrapperInterface op);
std::optional<QuickTuningProblemKey>
getQuickTuningProblemKey(RockGemmGemmWrapperInterface op);
std::optional<QuickTuningProblemKey> getQuickTuningProblemKey(ModuleOp mod);

/// A problem's key hash and the slice of `perfConfigIndices` holding its
/// perfconfigs.
struct QuickTuningProblemRef {
  QuickTuningProblemKeyHash hash;
  uint32_t offset;
  uint32_t count;
};

/// Generated per-problem perfconfigs for one architecture/kernel/data-type
/// key. `problems` must be sorted by hash, and `keyVersionHash` is the hash of
/// the ordered field names used to compute their problem hashes.
class QuickTuningProblemMap {
public:
  QuickTuningProblemMap(QuickTuningTableLookUpKeyVersionHash keyVersionHash,
                        ArrayRef<QuickTuningProblemRef> problems,
                        ArrayRef<uint16_t> perfConfigIndices,
                        ArrayRef<StringRef> perfConfigs)
      : keyVersionHash(keyVersionHash), problems(problems),
        perfConfigIndices(perfConfigIndices), perfConfigs(perfConfigs) {
    assert(llvm::is_sorted(problems,
                           [](const QuickTuningProblemRef &lhs,
                              const QuickTuningProblemRef &rhs) {
                             return lhs.hash < rhs.hash;
                           }) &&
           "problems must be sorted by hash for lookup's binary search");
  }

  SmallVector<StringRef> lookup(QuickTuningProblemKeyHash hash) const {
    const QuickTuningProblemRef *it = llvm::lower_bound(
        problems, hash,
        [](const QuickTuningProblemRef &ref, QuickTuningProblemKeyHash h) {
          return ref.hash < h;
        });
    if (it == problems.end() || it->hash != hash)
      return {};

    SmallVector<StringRef> result;
    result.reserve(it->count);
    for (uint16_t index : perfConfigIndices.slice(it->offset, it->count))
      result.push_back(perfConfigs[index]);
    return result;
  }

  QuickTuningTableLookUpKeyVersionHash getKeyVersionHash() const {
    return keyVersionHash;
  }

private:
  QuickTuningTableLookUpKeyVersionHash keyVersionHash;
  ArrayRef<QuickTuningProblemRef> problems;
  ArrayRef<uint16_t> perfConfigIndices;
  ArrayRef<StringRef> perfConfigs;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_QUICK_TUNING_PROBLEM_MAP_H
