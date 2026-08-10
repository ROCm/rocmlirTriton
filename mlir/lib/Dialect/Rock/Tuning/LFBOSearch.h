//===- LFBOSearch.h - Surrogate-guided perf config search -------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Likelihood-Free Bayesian Optimization (LFBO) tree search: a pattern search
// over the tuning space in which a random forest decides which candidates are
// worth benchmarking, so that only a fraction of the space is ever compiled.
//
// Ported from Helion's LFBOTreeSearch
// (helion/autotuner/surrogate_pattern_search.py), which is in turn based on
// Song et al., "A General Recipe for Likelihood-free Bayesian Optimization"
// (2022) and Misic, "Optimization of tree ensembles" (2020).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TUNING_LFBOSEARCH_H
#define MLIR_LIB_DIALECT_ROCK_TUNING_LFBOSEARCH_H

#include "mlir/Dialect/Rock/Tuning/TuningSearch.h"

namespace mlir {
namespace rock {

/// Tuning knobs of the LFBO search.
///
/// Every default below is Helion's, so that a tuning run here searches the way
/// an equivalent Helion run would. They come from the signature of
/// `LFBOTreeSearch.__init__`, and, for the three population sizes, from
/// `PATTERN_SEARCH_DEFAULTS` in effort_profile.py, which is what Helion's
/// "full" autotuning effort uses. `setEffort` below switches those three to
/// Helion's cheaper "quick" effort.
///
/// Helion runs 200 neighbours, selects 0.10 of them and moves by a radius of 2,
/// which is what the values below are. Its docstrings say otherwise and are
/// stale.
///
/// Only `candidateSpace` and `seed` have no Helion counterpart at all.
struct LFBOOptions {
  /// The space the search moves through, so that it covers what the equivalent
  /// brute-force run would without benchmarking all of it. It is taken as its
  /// axes (see `createTunableParamAxes`) rather than enumerated, so widening it
  /// costs the search nothing up front, and the axes are wider than the
  /// enumerated space in the parameters that space declines to vary.
  ///
  /// Every config the search builds belongs to this space. The quick list,
  /// which seeds the first batch, need not: it comes from the heuristic and is
  /// benchmarked for what it knows, not because the space contains it. rocMLIR
  /// specific: Helion generates configs rather than drawing them from a space.
  TuningParamSetKind candidateSpace = TuningParamSetKind::Exhaustive;
  /// Number of configs to benchmark before the first model is fitted. The quick
  /// tuning list is always used; `padInitialPopulation` tops it up from the
  /// candidate space, which the model needs to see any variety at all. This
  /// mirrors Helion's FROM_BEST_AVAILABLE strategy padding with random configs.
  unsigned initialPopulation = 100;
  bool padInitialPopulation = true;
  /// Number of independent pattern searches run from the best initial configs.
  unsigned copies = 5;
  /// Cap on the number of generations each copy runs for.
  unsigned maxGenerations = 20;
  /// Neighbours proposed per copy per generation, before model filtering.
  unsigned numNeighbors = 200;
  /// Fraction of those neighbours that actually gets benchmarked.
  double fracSelected = 0.10;
  /// How far a coarse move may travel, counted in doublings: a value may change
  /// by at most a factor of two to the power of this. Helion's radius is that
  /// same distance in log2 space; see `radiusMoves` for why counting places
  /// along a parameter's values would not be, here.
  unsigned radius = 2;
  /// Configs in this quantile of measured times are labelled "good" and the
  /// rest "bad"; the model learns to tell the two apart.
  double quantile = 0.1;
  /// Generations without a real improvement a copy may spend before it stops.
  unsigned patience = 1;
  /// Relative improvement below which a generation counts as no improvement.
  double minImprovementDelta = 0.001;
  /// How strongly a candidate is penalised for resembling one already picked
  /// for the same batch.
  double similarityPenalty = 1.0;
  /// Fixed so that repeated tuning runs of the same problem agree. rocMLIR
  /// specific: Helion seeds from its distributed `sync_seed` instead.
  uint64_t seed = 42;
  /// File to append a JSON record of every iteration to, one object per line:
  /// what the search knew, what it proposed and what that cost, so that a run
  /// can be plotted afterwards (see analysis/plotLFBOTrace.py). A search only
  /// ever reports on itself here; nothing it does depends on this being set.
  std::string tracePath;

  /// Sets the three population sizes - how many configs the first batch holds,
  /// how many searches run in parallel and how long each may run - to what
  /// `effort` allows, and leaves every other knob alone.
  void setEffort(LFBOEffort effort);
};

/// Creates an LFBO search over the tuning space of the primary op in `mod`.
std::unique_ptr<TuningSearchStrategy>
createLFBOSearchStrategy(ModuleOp mod, const LFBOOptions &options);

/// The values a coarse move can reach from `value`, out of `ladder`, which
/// holds one parameter's values in increasing order. `value` itself is never
/// among them and need not be on the ladder: a copy can start from a config the
/// search would not have proposed, such as a quick-list config spelling a knob
/// as its default, and then the value at the insertion point is a move like any
/// other rather than the place being moved from.
///
/// A coarse move travels up to `radius` doublings, which is Helion's radius. On
/// a power-of-two axis that is `radius` places either way, so the two readings
/// agree; on an axis carrying every multiple of 16, which is what a tile is
/// given where it need not be a power of two (`tileValues`, TuningSearch.cpp),
/// counting places instead would mean a move of 16 where Helion moves by a
/// factor of four, and it would do so on the very parameters a trial sets out
/// to move. A ladder carrying a value that is not positive, such as a knob's
/// default or the zero standing for "untiled", has no ratio to take, so it
/// moves by places.
///
/// Whether the config a move leaves behind is one the search space contains is
/// a separate question, which `feasibleNeighbors` puts to the space.
void radiusMoves(ArrayRef<int64_t> ladder, int64_t value, unsigned radius,
                 SmallVectorImpl<int64_t> &out);

/// The values a fine move can reach from `value`: the next one either way,
/// which is the smallest move an axis allows and the reason for listing tiles
/// that are not powers of two at all. As with `radiusMoves`, `value` is
/// excluded and need not be on the ladder.
void stepMoves(ArrayRef<int64_t> ladder, int64_t value,
               SmallVectorImpl<int64_t> &out);

} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TUNING_LFBOSEARCH_H
