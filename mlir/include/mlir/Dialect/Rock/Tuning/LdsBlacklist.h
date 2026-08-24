//===- LdsBlacklist.h - GEMM perf configs that overflow LDS ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A compiled-in blacklist of GEMM perf configs that overflow LDS (shared
// memory) for a given (arch, input dtype), letting the tuning-space builder
// drop known-dead configs before they are ever compiled.
//
// Per-block LDS is *not* strictly a function of (arch, dtype, perf config): it
// also grows with the K-loop trip count (K / kPerBlock), because `numStages`
// software pipelining multi-buffers the LDS operands, and that buffering only
// materialises once there are enough K iterations to fill the pipeline. It is
// monotonic non-decreasing in the trip count and *saturates* for large K (e.g.
// on gfx942/f16, kPerBlock=512/numStages=3 uses 32768 B at K=512 but 98304 B at
// K>=1024). The table is therefore generated at a large K (DEFAULT_DIMS in
// generateLDSBlacklist.py), i.e. at the saturated, worst-case footprint. This
// keeps the blacklist safe and effectively problem-shape independent:
//
//   * It never blacklists a config that fits at large K, so it never prunes a
//     config that could overflow at runtime for *any* K (LDS only shrinks as K
//     drops).
//   * The only shape-dependent gap is over-pruning at trip count 1 (K <=
//     kPerBlock), where a blacklisted config would actually fit. That is
//     harmless: with a single K iteration the pipeliner caps the effective
//     stage count to 1, so such a config produces the *identical* kernel (same
//     LDS) as its numStages=1 variant -- which fits even at large K, is never
//     blacklisted, and remains in the tuning space. So no non-degenerate config
//     is ever lost.
//
// The table is generated offline by
// mlir/utils/performance/generateLDSBlacklist.py into
// LdsBlacklistPerfconfigs.inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_LDS_BLACKLIST_H
#define MLIR_DIALECT_ROCK_LDS_BLACKLIST_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/BuiltinTypes.h"

#include <array>
#include <cstdint>
#include <set>

namespace mlir {
namespace rock {

// The only perf-config fields that change a GEMM's per-block LDS footprint.
// Determined empirically (see generateLDSBlacklist.py); kpack, numCTAs,
// splitKFactor, wavesPerEU, gridGroupSize and the knob fields have no effect.
// Field order must match PROJECTION_NAMES in generateLDSBlacklist.py.
struct GemmLdsKey {
  int64_t mPerBlock;
  int64_t nPerBlock;
  int64_t kPerBlock;
  int64_t numWaves;
  int64_t matrixInstrNonkdim;
  int64_t numStages;

  std::array<int64_t, 6> asArray() const {
    return {mPerBlock, nPerBlock,          kPerBlock,
            numWaves,  matrixInstrNonkdim, numStages};
  }
};

// Set of GemmLdsKey projections, ready for O(log n) membership tests by the
// tuning-space builder. Keyed on the 6-tuple (GemmLdsKey::asArray) rather than
// the struct so callers can query with a brace-initialized tuple.
using GemmLdsKeySet = std::set<std::array<int64_t, 6>>;

class LdsBlacklist {
public:
  // Returns the set of tile-shape keys known to overflow LDS for a GEMM on
  // `arch` with input element type `dataType`. On an exact miss, falls back
  // only to an architecture in the *same ISA family* with an *identical* LDS
  // size (e.g. gfx1151 -> gfx1150); if no such entry exists, returns an empty
  // set. Never fails: a miss simply means "nothing to skip".
  static GemmLdsKeySet lookupGemm(StringRef arch, Type dataType);
};

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_LDS_BLACKLIST_H
