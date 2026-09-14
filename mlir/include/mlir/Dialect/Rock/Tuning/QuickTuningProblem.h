//===- QuickTuningProblem.h - Per-problem quick tuning ---------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_QUICK_TUNING_PROBLEM_H
#define MLIR_DIALECT_ROCK_QUICK_TUNING_PROBLEM_H

#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>

namespace mlir {
namespace rock {

/// Compact generated data for one <architecture, kernel, data type> key.
///
/// Names are sorted for binary search. Config strings are interned within the
/// key; offsets delimit each problem's indices into the config dictionary.
class ProblemTuningData {
public:
  ProblemTuningData(ArrayRef<StringRef> names, ArrayRef<uint32_t> offsets,
                    ArrayRef<uint16_t> configIndices,
                    ArrayRef<StringRef> configs)
      : names(names), offsets(offsets), configIndices(configIndices),
        configs(configs) {}

  SmallVector<StringRef, 8> lookup(StringRef name) const {
    auto it = std::lower_bound(names.begin(), names.end(), name);
    if (it == names.end() || *it != name)
      return {};

    size_t problemIndex = std::distance(names.begin(), it);
    uint32_t begin = offsets[problemIndex];
    uint32_t end = offsets[problemIndex + 1];
    SmallVector<StringRef, 8> result;
    result.reserve(end - begin);
    for (uint16_t configIndex : configIndices.slice(begin, end - begin))
      result.push_back(configs[configIndex]);
    return result;
  }

private:
  ArrayRef<StringRef> names;
  ArrayRef<uint32_t> offsets;
  ArrayRef<uint16_t> configIndices;
  ArrayRef<StringRef> configs;
};

/// Build the stable identity used by the generated per-problem quick-tuning
/// data. Hardware topology and every field substituted by ParamLookupTable's
/// fallback (architecture, kernel type, and data type) are deliberately absent.
LogicalResult getQuickTuningProblemName(RockGemmWrapperInterface op,
                                        SmallVectorImpl<char> &out);
LogicalResult getQuickTuningProblemName(RockGemmGemmWrapperInterface op,
                                        SmallVectorImpl<char> &out);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_QUICK_TUNING_PROBLEM_H
