//===- LdsBlacklist.cpp - GEMM perf configs that overflow LDS -------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Tuning/LdsBlacklist.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/Tuning/ParamLookupTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

#define DEBUG_TYPE "rock-lds-blacklist"

using namespace mlir;
using namespace mlir::rock;

namespace {

constexpr char kSeparator = '_';

// Out-of-line storage for the per-(arch,dtype) config arrays.
struct LdsBlacklistGemm {
#define GemmLdsBlacklist_DECLARATIONS_GEN
#include "mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc"
#undef GemmLdsBlacklist_DECLARATIONS_GEN
};

#define GemmLdsBlacklist_DEFINITIONS_GEN
#include "mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc"
#undef GemmLdsBlacklist_DEFINITIONS_GEN

static const llvm::StringMap<ArrayRef<GemmLdsKey>> &getTable() {
  static const llvm::StringMap<ArrayRef<GemmLdsKey>> table = {
#define GemmLdsBlacklist_LOOKUP_TABLE_GEN
#include "mlir/Dialect/Rock/Tuning/LdsBlacklistPerfconfigs.inc"
#undef GemmLdsBlacklist_LOOKUP_TABLE_GEN
  };
  return table;
}

// The architecture token of a "<arch>_gemm_<dtype>" key.
static StringRef archOfKey(StringRef key) {
  return key.substr(0, key.find(kSeparator));
}

// The "_gemm_<dtype>" suffix of a "<arch>_gemm_<dtype>" key.
static StringRef suffixOfKey(StringRef key) {
  return key.substr(key.find(kSeparator));
}

// Numeric id of a "gfxNNN[a]" arch token, parsed as hex like getMaxKpack in
// AmdArchDb.cpp (e.g. "gfx90a" -> 0x90a). nullopt for a non-gfx/malformed
// token.
static std::optional<unsigned> gfxId(StringRef arch) {
  if (!arch.consume_front("gfx"))
    return std::nullopt;
  unsigned n = 0;
  if (arch.getAsInteger(/*radix=*/16, n))
    return std::nullopt;
  return n;
}

// Returns the raw config array for (arch, dtype), applying the LDS-size arch
// fallback. Empty on a miss. lookupGemm wraps this into a queryable set.
static ArrayRef<GemmLdsKey> findGemmKeys(StringRef arch, Type dataType) {
  std::string key = (Twine(normalizeArch(arch)) + Twine(kSeparator) + "gemm" +
                     Twine(kSeparator) + getDataTypeString(dataType))
                        .str();

  const auto &table = getTable();
  auto it = table.find(key);
  if (it != table.end())
    return it->getValue();

  // Fallback is only sound when the substitute arch shares both the target's
  // LDS budget *and* its ISA family: an LDS-overflow verdict is a function of
  // shared-memory usage, and that usage depends on the matmul lowering (MFMA
  // vs WMMA vs non-accel), which differs between families that happen to share
  // a 64 KB budget. So one family's overflow set is not a safe substitute for
  // another's. Among same-family, same-LDS candidates, pick the arch whose
  // gfx id is numerically closest to the target (ids parse as hex, e.g.
  // gfx1151 -> gfx1150); ties prefer the lower id for determinism.
  StringRef keyRef(key);
  StringRef targetArch = archOfKey(keyRef);
  int64_t targetLds = getLDSSize(targetArch);
  triton::amdgpu::ISAFamily targetFamily = std::get<0>(getArch(targetArch));
  StringRef targetSuffix = suffixOfKey(keyRef);
  std::optional<unsigned> targetId = gfxId(targetArch);
  if (!targetId)
    return {};

  StringRef bestKey;
  unsigned bestDist = 0;
  unsigned bestId = 0;
  for (const auto &entry : table) {
    StringRef candidate = entry.getKey();
    if (suffixOfKey(candidate) != targetSuffix)
      continue;
    StringRef candidateArch = archOfKey(candidate);
    if (getLDSSize(candidateArch) != targetLds)
      continue;
    if (std::get<0>(getArch(candidateArch)) != targetFamily)
      continue;
    std::optional<unsigned> candId = gfxId(candidateArch);
    if (!candId)
      continue;
    unsigned dist =
        *candId > *targetId ? *candId - *targetId : *targetId - *candId;
    if (bestKey.empty() || dist < bestDist ||
        (dist == bestDist && *candId < bestId)) {
      bestKey = candidate;
      bestDist = dist;
      bestId = *candId;
    }
  }

  if (!bestKey.empty())
    return table.at(bestKey);
  return {};
}

} // namespace

GemmLdsKeySet LdsBlacklist::lookupGemm(StringRef arch, Type dataType) {
  // Escape hatch for the offline generator (generateLDSBlacklist.py): it drives
  // `--emit-tuning-space=exhaustive`, which flows through this filter, so it
  // must be able to observe the *unfiltered* space to (re)discover overflowing
  // configs. Production tuning leaves this unset and gets the filtered space.
  if (std::getenv("ROCMLIR_DISABLE_LDS_BLACKLIST"))
    return {};

  GemmLdsKeySet keys;
  for (const GemmLdsKey &k : findGemmKeys(arch, dataType))
    keys.insert(k.asArray());
  return keys;
}
