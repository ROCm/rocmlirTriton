//===- LLMSearch.h - Tuning search guided by a language model ---*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A `TuningSearchStrategy` that asks a language model which perf configs to
// benchmark, a round at a time, showing it what the previous rounds measured.
//
// Ported from Helion's `LLMGuidedSearch` (helion/autotuner/llm_search.py),
// following the same discipline as the LFBO port next door: mirror the
// upstream algorithm, and say so wherever this deliberately does not.
//
// The idea it rests on is that the search space is not arbitrary. A tile shape
// is a tile shape, `splitKFactor` is there for skinny GEMMs, and a model that
// has read about GPU kernels knows both -- so it can start from a good guess
// instead of from the uniform prior every other search here begins with. What
// it cannot do is measure anything, which is why it proposes and this file
// benchmarks.
//
// Concept mapping (Helion -> rocmlirTriton):
//
//   `Config` (a dict of fields)         `PerfConfigString` + `ConfigValues`
//   `config_spec._flat_fields()`        `TuningParamAxes::getParamNames`
//   `describe_config_space`             `TuningParamAxes::getAxes` ladders
//   `config_spec.normalize`             `TuningParamAxes::isFeasible`
//   `autotune_reference_config()`       the first quick-list config
//   kernel source + input tensors       `PopulateParamsInfo`
//   `workload.py::_gpu_hardware_lines`  `AmdArchDb` + `GetRockInfo`
//   compiler-derived seed configs       the quick tuning list
//   `BenchmarkResult.perf` (ms)         `BenchmarkResult::timeNs`
//   `llm/transport.py` (provider HTTP)  `LLMProposer` -> llm/proposer.py
//
// Deliberate deviations from Helion:
//
// - No `rebenchmark_population` or `final_rebenchmark_best`. Helion
//   rebenchmarks its top configs between rounds so that later prompts see
//   stabilized timings rather than one-shot noise; the tuning driver has no
//   rebenchmark path, so a round ends by updating best-so-far and the model
//   sees the measurement it got.
// - Seeds are the quick tuning list rather than Helion's default config plus
//   random draws, which is what `LFBOSearch` already does, and what makes the
//   two comparable.
// - Round 0's overlap between the first request and the seed benchmarks is
//   kept (see `LLMSearchOptions::waitForSeeds`); the later rounds are
//   synchronous, exactly as upstream.
// - `isFeasible` runs before anything is compiled, so a proposal can be turned
//   down without ever being measured. Helion has no such filter and so nothing
//   to report; here the rejections are fed back to the model, since a
//   constraint it cannot see is one it will otherwise violate every round.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TUNING_LLMSEARCH_H
#define MLIR_LIB_DIALECT_ROCK_TUNING_LLMSEARCH_H

#include "SearchTrace.h"
#include "mlir/Dialect/Rock/Tuning/TuningSearch.h"

#include <memory>
#include <string>

namespace mlir {
namespace rock {

/// How `LLMSearch` searches. `search` holds what a command line sets; the rest
/// are the constants Helion hard-codes, exposed here because a test has to be
/// able to reach them.
struct LLMOptions {
  LLMSearchOptions search;

  /// Which space to propose into. `Exhaustive` because the model is being
  /// asked to reason about what will be fast, and `Full`'s heuristic filter
  /// would silently overrule it; it is also the space `LFBOSearch` refines in,
  /// which the hybrid needs them to agree on.
  TuningParamSetKind candidateSpace = TuningParamSetKind::Exhaustive;

  /// Seeds the draws that pad the first batch. Fixed so that repeated runs of
  /// the same problem agree, insofar as a model lets them.
  uint64_t seed = 42;

  /// How much faster a round has to make the best config to count as progress,
  /// as a fraction, and how many rounds may fail to before the search stops.
  /// Helion's `min_improvement_delta` and `_MAX_STAGNANT_ROUNDS`.
  double minImprovementDelta = 0.005;
  unsigned maxStagnantRounds = 2;

  /// Where the helper keeps the conversation between rounds, so that a later
  /// prompt can refer to an earlier one. Empty means every round starts cold,
  /// which costs the model its memory of what it already tried. Helion's
  /// rolling window over `self._messages` (`_MAX_CONTEXT_ROUNDS`) lives on the
  /// Python side of this file, since it is the prompt's business.
  std::string sessionPath;

  SharedTrace trace;

  void setEffort(SearchEffort effort);
};

std::unique_ptr<TuningSearchStrategy>
createLLMSearchStrategy(ModuleOp mod, const LLMOptions &options);

} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TUNING_LLMSEARCH_H
