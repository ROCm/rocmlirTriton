//===- QuickTuningProblemMap.h - Per-problem quick tuning map ---*- C++ -*-===//
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

#include <cstdint>
#include <optional>

namespace mlir {
namespace rock {

using QuickTuningProblemKeyHash = uint64_t;

inline QuickTuningProblemKeyHash hashQuickTuningProblemKey(StringRef key) {
  return llvm::xxh3_64bits(key);
}

/// Hash identifying `op`'s problem, or nullopt when it has none. Only the
/// operation's own fields take part, so the lookup key must still carry the
/// architecture and data type.
std::optional<QuickTuningProblemKeyHash>
getQuickTuningProblemKeyHash(RockGemmWrapperInterface op);
std::optional<QuickTuningProblemKeyHash>
getQuickTuningProblemKeyHash(RockGemmGemmWrapperInterface op);
std::optional<QuickTuningProblemKeyHash>
getQuickTuningProblemKeyHash(ModuleOp mod);

/// A problem's key hash and the slice of `perfConfigIndices` holding its
/// perfconfigs.
struct QuickTuningProblemRef {
  QuickTuningProblemKeyHash hash;
  uint32_t offset;
  uint32_t count;
};

/// Generated per-problem perfconfigs for one architecture/kernel/data-type
/// key. `problems` must be sorted by hash.
class QuickTuningProblemMap {
public:
  QuickTuningProblemMap(ArrayRef<QuickTuningProblemRef> problems,
                        ArrayRef<uint16_t> perfConfigIndices,
                        ArrayRef<StringRef> perfConfigs)
      : problems(problems), perfConfigIndices(perfConfigIndices),
        perfConfigs(perfConfigs) {}

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

private:
  ArrayRef<QuickTuningProblemRef> problems;
  ArrayRef<uint16_t> perfConfigIndices;
  ArrayRef<StringRef> perfConfigs;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_QUICK_TUNING_PROBLEM_MAP_H
