//===- TuningSearch.h - Perf config search strategies -----------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface tuning clients use to obtain the perf configs
// they should compile and benchmark.
//
// A strategy is driven as a loop: the client asks for a batch of perf configs,
// benchmarks them, and hands the timings back on the next call. An empty batch
// ends the search:
//
//   auto strategy = createTuningSearchStrategy(mod, kind);
//   std::vector<BenchmarkResult> results;
//   while (true) {
//     std::vector<PerfConfigString> batch =
//     strategy->getPerfConfigBatch(results); if (batch.empty())
//       break;
//     results = benchmarkEachOf(batch);
//   }
//
// The brute-force spaces (quick/full/exhaustive) are expressed in these terms
// as strategies that return the whole space in one batch and then stop, so a
// client only has to implement the loop above. A strategy that consumes the
// timings (such as LFBO) can instead steer the search towards the fastest
// configs and only ever benchmark a fraction of the space.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TUNING_TUNINGSEARCH_H
#define MLIR_DIALECT_ROCK_TUNING_TUNINGSEARCH_H

#include "mlir-c/Dialect/Rock.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace mlir {
namespace rock {

/// What happened when a client tried to benchmark one perf config. Strategies
/// that learn from their results are told about the failures too: a config that
/// cannot be compiled or run is a data point about the shape of the search
/// space, not just a gap in it.
struct BenchmarkResult {
  enum class Status {
    /// The kernel compiled and ran; `timeNs` is a real measurement.
    Success,
    /// The config was rejected before it could be measured, e.g. because it
    /// overflows LDS or makes the surrounding fusion illegal.
    NotApplicable,
    /// The config should have been measurable but wasn't: a compilation
    /// failure, a timeout, or a crash.
    Failed,
  };

  PerfConfigString perfConfig;
  /// Measured time in nanoseconds, infinity unless `status` is `Success`.
  double timeNs = std::numeric_limits<double>::infinity();
  Status status = Status::Failed;

  bool measured() const { return effectiveStatus() == Status::Success; }

  /// `status`, except that a `Success` carrying no finite time is reported as
  /// `Failed`. A client that sets one without the other has produced a result
  /// nothing can be learned from, and calling it a success would put an
  /// unusable timing into a surrogate model or a prompt.
  Status effectiveStatus() const {
    if (status == Status::Success && !std::isfinite(timeNs))
      return Status::Failed;
    return status;
  }
};

/// The enumerator's own name, lowerCamel so that it can be a value in the
/// JSON a search traces. Prefer `BenchmarkResult::effectiveStatus` over the
/// raw field when naming one, so that "success" always means a usable timing.
StringRef getNameForBenchmarkStatus(BenchmarkResult::Status status);

/// Streams that name. Note the direction: `<<` is what writes a value out to a
/// stream, which is what "convert to string" means here, and it composes with
/// `Twine`, `formatv` and the diagnostic streams. Reach for
/// `getNameForBenchmarkStatus` directly where a `StringRef` is wanted rather
/// than an insertion, as when building a JSON object.
llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              BenchmarkResult::Status status);

/// The available search strategies. The first three enumerate a fixed space
/// (see `TuningParamSetKind`); the rest search adaptively.
enum class SearchStrategyKind : uint32_t {
  Quick = 0,
  Full = 1,
  Exhaustive = 2,
  /// Surrogate-guided pattern search; see LFBOSearch.h.
  LFBO = 3,
  /// A language model proposes the configs; see LLMSearch.h.
  LLM = 4,
  /// One round of `LLM` to find a good config, then `LFBO` to refine around
  /// it, learning from everything the first stage measured.
  LLMSeededLFBO = 5,
};

/// How much benchmarking an adaptive search is allowed to spend, mirroring the
/// two autotuning efforts Helion offers: `Quick` looks at a fraction of the
/// configs `Full` does, and so is likelier to settle for a slower one. The
/// strategies that enumerate a fixed space have nothing to spend it on.
enum class SearchEffort : uint32_t {
  Quick = 0,
  Full = 1,
};

/// Maps the names used on command lines ("quick", "full", "exhaustive",
/// "lfbo", "llm", "llm-lfbo") onto their enumerator. Returns std::nullopt for
/// anything else.
std::optional<SearchStrategyKind> parseSearchStrategyKind(StringRef name);

/// The inverse of `parseSearchStrategyKind`.
StringRef getSearchStrategyKindName(SearchStrategyKind kind);

/// Produces the perf configs a tuning client should benchmark. See the file
/// comment for the protocol. Strategies are stateful and are not thread-safe.
class TuningSearchStrategy {
public:
  virtual ~TuningSearchStrategy() = default;

  /// Returns the next batch of perf configs to compile and benchmark, or an
  /// empty batch once the search is over. `prevResults` holds the outcome of
  /// the batch returned by the previous call, in any order, and must be empty
  /// on the first call. Results for configs this strategy never proposed are
  /// ignored.
  virtual std::vector<PerfConfigString>
  getPerfConfigBatch(ArrayRef<BenchmarkResult> prevResults) = 0;

  /// Whether this strategy can return more than one batch.
  virtual bool isIterative() const = 0;
};

/// What a language model is allowed to spend proposing configs. Every default
/// is Helion's, from `LLM_SEARCH_DEFAULTS` in effort_profile.py, except
/// `model`, which names a Cursor model rather than Helion's `gpt-5-2`.
struct LLMSearchOptions {
  /// A model id, optionally followed by `:name=value` parameters of that
  /// model, comma-separated. The default uses Nano without reasoning: measured
  /// convolution rounds average under ten seconds while retaining useful
  /// config quality. A parameter the account does not offer fails the run
  /// rather than being dropped; `Cursor.models.list()` says what is available.
  std::string model = "gpt-5.4-nano:reasoning=none";
  /// Configs to ask for per round.
  unsigned configsPerRound = 15;
  /// Rounds of proposal, including the first. One means a single call.
  unsigned maxRounds = 4;
  /// Configs drawn from the space to pad the first batch with, so that the
  /// model's first proposal is judged against something other than the quick
  /// list alone.
  unsigned initialRandomConfigs = 10;
  /// Seconds a single proposal may take before the helper is killed.
  unsigned requestTimeoutSec = 120;
  /// Per-config compile timeout to impose while the model's proposals are
  /// being benchmarked, or 0 to leave the client's own budget alone. Helion
  /// caps this because "LLM proposals timed out more often per config, so fail
  /// them fast" (`_llm_search_settings_context`).
  unsigned compileTimeoutSec = 15;
  /// Whether round 0 waits for the seed batch to be measured before asking for
  /// the first proposal. Helion overlaps the two instead, since its first
  /// prompt does not depend on any measurement; waiting buys a better-informed
  /// first proposal at the cost of a serial round trip.
  bool waitForSeeds = false;
  /// The proposer script to run. Empty means the one installed next to the
  /// running executable, unless $ROCMLIR_LLM_PROPOSER overrides it.
  std::string proposerPath;
  /// Where the helper keeps the conversation between this search's rounds, so
  /// that round 3 is the same conversation as round 0 and the model can be
  /// asked to improve on what it already said rather than told about it.
  ///
  /// Empty means a temporary file of the search's own, removed when the search
  /// ends, which is what almost every client wants: the file is a mechanism
  /// and llm/transcript.py is what a person reads. Naming one keeps it.
  ///
  /// One file per search. Two searches sharing a session would resume the same
  /// conversation from two problems, and problems tune concurrently, so they
  /// would also be two rounds contending for one agent.
  std::string sessionPath;
  /// Where to write down every prompt and every reply, readably, or empty for
  /// nowhere. A debugging aid that nothing the search does depends on, and the
  /// only record of the conversation there is: `tracePath` counts rounds and
  /// says nothing of what was said in them. One problem per file -- a search
  /// appends wherever it is pointed -- since problems tune concurrently.
  std::string transcriptPath;
};

/// Everything the strategies can be steered by. A single struct because the
/// adaptive searches each read a different part of it and the fixed spaces
/// read none of it, so threading them as parameters means every caller
/// spelling out defaults it has no opinion on.
struct SearchOptions {
  /// The budget of an adaptive search (`LFBO`, `LLM`, `LLMSeededLFBO`); a
  /// fixed space is as large as it is and ignores this.
  SearchEffort effort = SearchEffort::Full;
  /// A debugging aid that nothing a search does depends on: when it is not
  /// empty, a strategy that searches in iterations writes a record of each of
  /// them there so that a run can be examined afterwards (see
  /// mlir/utils/performance/analysis/plotSearchTrace.py). The strategies that
  /// enumerate a fixed space have no iterations to report and ignore it.
  std::string tracePath;
  LLMSearchOptions llm;
};

/// Creates the strategy for `kind`, searching the space of the primary GEMM,
/// convolution or attention op in `mod`.
std::unique_ptr<TuningSearchStrategy>
createTuningSearchStrategy(ModuleOp mod, SearchStrategyKind kind,
                           const SearchOptions &options = {});

/// Creates a strategy that proposes `configs` as its one and only batch. Lets a
/// client that already knows what it wants to benchmark - one pinned config,
/// say - go through the same loop as a real search.
std::unique_ptr<TuningSearchStrategy>
createFixedBatchSearchStrategy(std::vector<PerfConfigString> configs);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TUNING_TUNINGSEARCH_H
