//===- LFBOSearch.cpp - Surrogate-guided perf config search ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LFBOSearch.h"
#include "RandomForest.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>

#define DEBUG_TYPE "rock-lfbo-search"

using namespace mlir;
using namespace mlir::rock;

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

//===----------------------------------------------------------------------===//
// Perf config encoding
//===----------------------------------------------------------------------===//

// A perf config reaches the search as `ConfigValues`, the values of its
// parameters in the order the attribute reports them (see RockTuning.h).
// The search never learns what any of them mean, nor how many there are; it
// only moves a value to another value the same parameter is known to take.
// That keeps it working unchanged when a parameter is added to, or removed
// from, a perf config.

/// Whether every value a parameter takes is a power of two, in which case it is
/// encoded in log2, as Helion encodes its own power-of-two parameters. Derived
/// from those values rather than declared, since growing geometrically is a
/// property of a parameter's values and not of what it is called.
///
/// Only the encoding depends on this: how far a move may travel is
/// `radiusMoves`' own question, and since the forest compares a feature against
/// thresholds it placed between values it observed, this monotone transform
/// cannot move a decision boundary either.
bool isLogScaled(ArrayRef<int64_t> ladder) {
  return llvm::all_of(ladder, [](int64_t value) {
    return value > 0 && llvm::isPowerOf2_64(value);
  });
}

float encodeValue(bool logScaled, int64_t value) {
  if (logScaled && value > 0)
    return std::log2(static_cast<float>(value));
  return static_cast<float>(value);
}

/// The perf-config keys of the tile sizes and the wave count. Helion gives a
/// block size and the warp count a chance to move in every neighbour it
/// proposes, since they dominate performance; these are their counterparts
/// here. Every spelling is listed because gemm+gemm suffixes the first gemm's
/// m/n tiles with `G0` and tiles the second gemm's head dim as well; a config
/// carries one spelling or the other. Only this preference lives here, while
/// which parameters a config actually has stays the attribute's answer to give.
constexpr StringRef kTileKeys[] = {"mPerBlock",   "nPerBlock",   "kPerBlock",
                                   "mPerBlockG0", "nPerBlockG0", "nPerBlockG1"};
constexpr StringRef kWaveCountKey = "numWaves";

/// numpy's default (linearly interpolated) quantile. Sorts `values` in place.
double quantileOf(MutableArrayRef<double> values, double quantile) {
  assert(!values.empty() && "quantile of an empty sample");
  llvm::sort(values);
  double position = quantile * (values.size() - 1);
  size_t lower = static_cast<size_t>(std::floor(position));
  size_t upper = static_cast<size_t>(std::ceil(position));
  return values[lower] +
         (values[upper] - values[lower]) * (position - double(lower));
}

//===----------------------------------------------------------------------===//
// LFBOSearch
//===----------------------------------------------------------------------===//

class LFBOSearch : public TuningSearchStrategy {
public:
  LFBOSearch(ModuleOp mod, const LFBOOptions &options)
      : mod(mod), options(options), rng(options.seed), trace(options.trace) {}

  std::vector<PerfConfigString>
  getPerfConfigBatch(ArrayRef<BenchmarkResult> prevResults) override;

  bool isIterative() const override { return true; }

private:
  /// Where a config the search handed out came from, so that a trace can say
  /// whether the best config so far is one the heuristic already knew or one
  /// the search went and found.
  enum class Origin { QuickSeed, RandomPad, Proposal };
  struct Provenance {
    Origin origin = Origin::QuickSeed;
    /// The copy that proposed the config, and the generation it did so in.
    /// Meaningless unless `origin` is `Proposal`.
    unsigned copy = 0;
    unsigned generation = 0;
  };

  /// One pattern search, walking downhill from its own starting config. Several
  /// run at once so that the search does not commit to a single basin.
  struct SearchCopy {
    ConfigValues current;
    double currentPerf = kInfinity;
    /// Generations without improvement this copy may still spend.
    unsigned patience = 0;
    bool active = true;
    /// The configs this copy contributed to the batch currently in flight.
    std::vector<PerfConfigString> pending;
  };

  /// A benchmarked config: its time, and the values the search manipulates, so
  /// that a config never has to be parsed twice.
  struct Measured {
    double perf = kInfinity;
    ConfigValues values;
  };

  /// Picks up the space's axes and the seeds to start from. Returns false if
  /// the module has no tuning space, leaving the search with nothing to do.
  bool buildSearchSpace();
  /// Draws configs uniformly from the axes and offers those that prove to be
  /// configs of the space to `emit`, which returns whether it kept one, until
  /// `count` are kept or the attempts run out. Sampling the axes and rejecting
  /// is what lets a space too large to enumerate still be seeded.
  void
  sampleFeasibleConfigs(unsigned count,
                        llvm::function_ref<bool(const ConfigValues &)> emit);
  /// Folds a benchmarked batch into the model's training set.
  void recordResults(ArrayRef<BenchmarkResult> prevResults);
  std::vector<PerfConfigString> buildInitialPopulation();
  /// Starts one copy from each of the fastest configs measured so far. Returns
  /// false if nothing was measurable, in which case the search cannot continue.
  bool seedSearchCopies();
  /// Moves each copy onto the best config of its last batch and retires the
  /// ones that have stopped improving.
  void advanceSearchCopies();
  std::vector<PerfConfigString> runGeneration();
  /// One iteration of the search, which is what `getPerfConfigBatch` is once
  /// the tracing around it is taken away.
  std::vector<PerfConfigString> proposeBatch(ArrayRef<BenchmarkResult> results);

  /// Notes that `perfConfig` was handed out, and by whom.
  void recordProvenance(const PerfConfigString &perfConfig, Origin origin,
                        unsigned copy = 0);
  /// Describes the search space and the knobs it is being searched with, once,
  /// so that the iterations that follow can be read without the command line
  /// that produced them.
  void traceHeader();
  /// Reports on the iteration that just ran: see the fields for what of it is
  /// worth keeping.
  void traceIteration(double elapsedMs, ArrayRef<PerfConfigString> batch);

  void fitSurrogate();
  /// Greedily picks `numSelected` of `features`, trading the model's score
  /// against how much a candidate resembles the ones already picked. Returns
  /// indices into `features`, best first.
  std::vector<unsigned> surrogateSelect(ArrayRef<std::vector<float>> features,
                                        unsigned numSelected);

  void generateNeighbors(const ConfigValues &base,
                         std::vector<ConfigValues> &out);
  void generateRandomNeighbors(const ConfigValues &base,
                               std::vector<ConfigValues> &out);
  void generateTreeGuidedNeighbors(const ConfigValues &base,
                                   std::vector<ConfigValues> &out);
  /// How far a move along a parameter's values travels: to the next value
  /// either way, or up to `options.radius` doublings of the one it holds. See
  /// `stepMoves` and `radiusMoves`.
  enum class MoveSize { Step, Radius };
  /// The values parameter `param` can move to under `move`, of which only those
  /// that leave a config the space contains are kept.
  void feasibleNeighbors(const ConfigValues &base, unsigned param,
                         MoveSize move, SmallVectorImpl<int64_t> &out);

  /// Turns a config back into the string the driver benchmarks. The space does
  /// the spelling, so a config the search proposes is indistinguishable from
  /// one the tuning space would have handed out.
  void serialize(const ConfigValues &values, PerfConfigString &out) {
    axes->serialize(values, out);
  }

  void encode(const ConfigValues &values, std::vector<float> &out) {
    out.resize(values.size());
    for (auto [value, logScale, encoded] :
         llvm::zip_equal(values, logScaled, out))
      encoded = encodeValue(logScale, value);
  }

  size_t pickIndex(size_t count) {
    assert(count > 0 && "picking from an empty range");
    return std::uniform_int_distribution<size_t>(0, count - 1)(rng);
  }

  ModuleOp mod;
  LFBOOptions options;
  std::mt19937_64 rng;

  /// The space being searched, as the values of each parameter plus the test of
  /// whether a combination of them is a config the space contains. Asking it
  /// rather than holding the space is what keeps a brute-force space
  /// searchable: the product of the axes is far larger than memory.
  std::unique_ptr<TuningParamAxes> axes;
  /// The values each parameter takes, in increasing order, and whether those
  /// values call for a log2 encoding.
  std::vector<std::vector<int64_t>> ladders;
  std::vector<bool> logScaled;
  /// The tile-size parameters, and the wave count if the config has one; see
  /// `kTileKeys`. Empty when a config names none of them.
  SmallVector<unsigned> tileParams;
  std::optional<unsigned> waveCountParam;
  std::vector<ConfigValues> quickSeeds;

  /// Every config considered so far, so none is ever benchmarked twice.
  std::set<ConfigValues> visited;
  llvm::StringMap<Measured> perfByConfig;
  std::vector<std::vector<float>> trainX;
  std::vector<double> trainY;

  RandomForestClassifier surrogate;
  std::vector<SearchCopy> searchCopies;
  unsigned generation = 0;
  bool started = false;
  bool done = false;

  /// The fastest config measured so far, which is what the search is for.
  double bestSoFar = kInfinity;
  PerfConfigString bestConfig;

  //===--------------------------------------------------------------------===//
  // Trace state. Nothing here is read by the search itself: it exists so that
  // a run can be explained afterwards, and is only maintained when a trace is
  // being written.
  //===--------------------------------------------------------------------===//

  SharedTrace trace;
  llvm::StringMap<Provenance> provenance;
  /// When the last batch was handed out, so that the next call can report how
  /// long the client spent compiling and benchmarking it. Iterations differ
  /// wildly in size, so a plot against iteration number alone flatters the
  /// search; this is the cost it actually spent.
  std::chrono::steady_clock::time_point handedOut;
  double totalMs = 0.0;
  /// How the batch folded in by the current iteration turned out.
  unsigned numSucceeded = 0;
  unsigned numNotApplicable = 0;
  unsigned numFailed = 0;
  /// Why copies stopped during the current iteration. A copy left without
  /// unvisited neighbours has run out of space to walk through rather than out
  /// of improvements, which is what the feasibility filters look like from
  /// here; the other two are the search deciding it is finished.
  unsigned stoppedOutOfPatience = 0;
  unsigned stoppedWithoutNeighbors = 0;
  unsigned stoppedWithoutSelection = 0;
  /// Neighbour bookkeeping of the current iteration: how many were generated,
  /// how many had already been visited, and how many single-parameter moves
  /// `feasibleNeighbors` refused because the space does not contain them.
  unsigned neighborsGenerated = 0;
  unsigned neighborsSeenBefore = 0;
  unsigned movesRejected = 0;
  /// What the last fit of the surrogate saw: the size of its training set, how
  /// much of it counted as good, and the time that made a config good.
  size_t trainSize = 0;
  unsigned numPositives = 0;
  double goodThreshold = kInfinity;
  /// The batch in flight and the threshold it was picked against, so that the
  /// next iteration can say what fraction of the model's picks turned out to be
  /// good. Negative precision means there is nothing to judge yet.
  std::vector<PerfConfigString> inFlight;
  double inFlightThreshold = kInfinity;
  double pickPrecision = -1.0;
};

bool LFBOSearch::buildSearchSpace() {
  axes = createTunableParamAxes(mod, options.candidateSpace);
  if (!axes)
    return false;

  ArrayRef<std::vector<int64_t>> spaceAxes = axes->getAxes();
  ladders.assign(spaceAxes.begin(), spaceAxes.end());
  for (std::vector<int64_t> &ladder : ladders) {
    // The search walks a ladder by index, so it has to be ordered and free of
    // repeats even if the space listed a parameter's values in another order.
    llvm::sort(ladder);
    ladder.erase(std::unique(ladder.begin(), ladder.end()), ladder.end());
    logScaled.push_back(isLogScaled(ladder));
  }

  SmallVector<StringRef> names;
  axes->getParamNames(names);
  for (unsigned param = 0; param < names.size(); ++param) {
    StringRef name = names[param];
    if (llvm::is_contained(kTileKeys, name))
      tileParams.push_back(param);
    else if (name == kWaveCountKey)
      waveCountParam = param;
  }

  // The quick list is the best guess available before anything is measured, so
  // it seeds the search.
  MLIRContext *ctx = mod.getContext();
  std::unique_ptr<TuningParamSet> quickSpace(
      createTunableParamSpace(mod, TuningParamSetKind::Quick));
  for (const PerfConfigString &perfConfig : quickSpace->tuningRange) {
    RockTuningParamAttrInterface params = parsePerfConfig(ctx, perfConfig);
    assert(params && "a tuning space spelled a config its attribute rejects");
    quickSeeds.emplace_back();
    params.getParamValues(quickSeeds.back());
    assert(quickSeeds.back().size() == ladders.size() &&
           "the quick and searched spaces disagree on what a config is");
  }

  LLVM_DEBUG({
    llvm::dbgs() << "LFBO: searching " << ladders.size() << " parameters over ";
    for (auto [name, ladder] : llvm::zip_equal(names, ladders))
      llvm::dbgs() << name << "=" << ladder.size() << " ";
    llvm::dbgs() << "values, " << quickSeeds.size() << " quick seeds\n";
  });
  return true;
}

void LFBOSearch::sampleFeasibleConfigs(
    unsigned count, llvm::function_ref<bool(const ConfigValues &)> emit) {
  // A draw from the axes need not be a config of the space, and how often it is
  // depends on the space, so cap the attempts rather than the failures: seeding
  // is worth some work but must not become the search.
  const unsigned maxAttempts = 100 * count;
  ConfigValues candidate(ladders.size());
  for (unsigned attempt = 0, kept = 0; attempt < maxAttempts && kept < count;
       ++attempt) {
    for (auto [value, ladder] : llvm::zip_equal(candidate, ladders))
      value = ladder[pickIndex(ladder.size())];
    if (axes->isFeasible(candidate) && emit(candidate))
      ++kept;
  }
}

void LFBOSearch::recordResults(ArrayRef<BenchmarkResult> prevResults) {
  MLIRContext *ctx = mod.getContext();
  for (const BenchmarkResult &result : prevResults) {
    switch (result.status) {
    case BenchmarkResult::Status::Success:
      numSucceeded += result.measured();
      numFailed += !result.measured();
      break;
    case BenchmarkResult::Status::NotApplicable:
      ++numNotApplicable;
      break;
    case BenchmarkResult::Status::Failed:
      ++numFailed;
      break;
    }

    RockTuningParamAttrInterface params =
        parsePerfConfig(ctx, result.perfConfig);
    if (!params)
      continue;
    auto [entry, isNew] = perfByConfig.try_emplace(result.perfConfig);
    if (!isNew)
      continue;
    // A config that could not be compiled or run is kept with an infinite time:
    // the model classifies rather than regresses, so it learns from the
    // failures as much as from the fast configs.
    Measured &measured = entry->second;
    measured.perf = result.measured() ? result.timeNs : kInfinity;
    params.getParamValues(measured.values);
    // A config this search proposed is already here, but a seeded one is not,
    // and nothing measured should ever be handed out again.
    visited.insert(measured.values);
    trainX.emplace_back();
    encode(measured.values, trainX.back());
    trainY.push_back(measured.perf);

    if (measured.perf < bestSoFar) {
      bestSoFar = measured.perf;
      bestConfig = result.perfConfig;
    }
  }

  // Of the configs the model picked last time, how many landed in the quantile
  // it was aiming at. This is what tells a surrogate that has learnt something
  // apart from one that is sampling the neighbourhood at random.
  pickPrecision = -1.0;
  if (!traceEnabled(trace) || !std::isfinite(inFlightThreshold))
    return;
  unsigned judged = 0, good = 0;
  for (const PerfConfigString &perfConfig : inFlight) {
    auto it = perfByConfig.find(perfConfig);
    if (it == perfByConfig.end() || !std::isfinite(it->second.perf))
      continue;
    ++judged;
    good += it->second.perf <= inFlightThreshold;
  }
  if (judged > 0)
    pickPrecision = static_cast<double>(good) / judged;
}

std::vector<PerfConfigString> LFBOSearch::buildInitialPopulation() {
  std::vector<PerfConfigString> batch;
  Origin origin = Origin::QuickSeed;
  auto emit = [&](const ConfigValues &values) {
    if (!visited.insert(values).second)
      return false;
    batch.emplace_back();
    serialize(values, batch.back());
    recordProvenance(batch.back(), origin);
    return true;
  };

  for (const ConfigValues &values : quickSeeds)
    emit(values);
  origin = Origin::RandomPad;

  // The quick list is far too small and too uniform to fit a model on, so top
  // it up with a random sample of the space.
  if (options.padInitialPopulation && batch.size() < options.initialPopulation)
    sampleFeasibleConfigs(options.initialPopulation - batch.size(), emit);
  return batch;
}

bool LFBOSearch::seedSearchCopies() {
  SmallVector<std::pair<StringRef, const Measured *>> ranked;
  for (const auto &entry : perfByConfig) {
    if (!std::isfinite(entry.second.perf))
      continue;
    // A copy walks from wherever it starts by moving one parameter at a time
    // onto the axes, so a config the space does not admit is a starting point
    // it could never move away from. Such a config is still worth measuring,
    // since a measurement is training data either way.
    if (!axes->isFeasible(entry.second.values))
      continue;
    ranked.emplace_back(entry.first(), &entry.second);
  }
  if (ranked.empty())
    return false;

  // Ties are broken by config string so that the starting points, and with
  // them the whole search, do not depend on hash table iteration order.
  llvm::sort(ranked, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.second->perf, lhs.first) <
           std::tie(rhs.second->perf, rhs.first);
  });

  for (const auto &entry : llvm::ArrayRef(ranked).take_front(options.copies)) {
    SearchCopy copy;
    copy.current = entry.second->values;
    copy.currentPerf = entry.second->perf;
    copy.patience = options.patience;
    searchCopies.push_back(std::move(copy));
  }
  return true;
}

void LFBOSearch::advanceSearchCopies() {
  for (SearchCopy &copy : searchCopies) {
    if (!copy.active)
      continue;

    const ConfigValues *best = &copy.current;
    double bestPerf = copy.currentPerf;
    for (const PerfConfigString &perfConfig : copy.pending) {
      auto it = perfByConfig.find(perfConfig);
      if (it == perfByConfig.end() || it->second.perf >= bestPerf)
        continue;
      best = &it->second.values;
      bestPerf = it->second.perf;
    }
    copy.pending.clear();

    bool improved = std::isfinite(bestPerf) &&
                    std::isfinite(copy.currentPerf) &&
                    std::abs(bestPerf / copy.currentPerf - 1.0) >=
                        options.minImprovementDelta;
    if (!improved) {
      if (copy.patience == 0) {
        copy.active = false;
        ++stoppedOutOfPatience;
        continue;
      }
      --copy.patience;
    }
    copy.current = *best;
    copy.currentPerf = bestPerf;
  }
}

std::vector<PerfConfigString> LFBOSearch::runGeneration() {
  std::vector<PerfConfigString> batch;
  std::vector<ConfigValues> neighbors;
  std::vector<ConfigValues> candidates;
  std::vector<std::vector<float>> features;

  for (auto [copyIdx, copy] : llvm::enumerate(searchCopies)) {
    if (!copy.active)
      continue;

    generateNeighbors(copy.current, neighbors);
    neighborsGenerated += neighbors.size();

    // The current config is scored alongside its neighbours (it is already
    // benchmarked, so it is never handed out again) so that the model can rate
    // the neighbourhood against the point it came from.
    candidates.assign(1, copy.current);
    for (const ConfigValues &neighbor : neighbors)
      if (visited.insert(neighbor).second)
        candidates.push_back(neighbor);
      else
        ++neighborsSeenBefore;
    if (candidates.size() <= 1) {
      copy.active = false;
      ++stoppedWithoutNeighbors;
      continue;
    }

    features.resize(candidates.size());
    for (auto [candidate, encoded] : llvm::zip_equal(candidates, features))
      encode(candidate, encoded);

    // At least two slots, of which the current config can take only one, so a
    // copy that still has unvisited neighbours is never retired below because
    // the model rated the config it came from above all of them. Helion floors
    // its own quota at two for the same reason.
    unsigned numSelected = std::max<unsigned>(
        2, static_cast<unsigned>(candidates.size() * options.fracSelected));
    for (unsigned idx : surrogateSelect(features, numSelected)) {
      if (idx == 0)
        continue;
      PerfConfigString perfConfig;
      serialize(candidates[idx], perfConfig);
      recordProvenance(perfConfig, Origin::Proposal, copyIdx);
      copy.pending.push_back(perfConfig);
      batch.push_back(perfConfig);
    }
    if (copy.pending.empty()) {
      copy.active = false;
      ++stoppedWithoutSelection;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "LFBO: generation " << generation << " proposes "
                          << batch.size() << " configs\n");
  return batch;
}

void LFBOSearch::fitSurrogate() {
  assert(trainX.size() == trainY.size() &&
         "a training point is a config and the time it was measured at");
  RandomForestOptions forestOptions;
  forestOptions.seed = options.seed;
  surrogate = RandomForestClassifier(forestOptions);
  trainSize = trainX.size();
  numPositives = 0;
  goodThreshold = kInfinity;
  if (trainX.empty())
    return;

  SmallVector<double> measured;
  for (double perf : trainY)
    if (std::isfinite(perf))
      measured.push_back(perf);
  if (measured.empty())
    return;

  // Label the fastest `quantile` of the configs "good" and everything else,
  // failures included, "bad".
  double threshold = quantileOf(measured, options.quantile);
  std::vector<float> labels(trainY.size(), 0.0f);
  std::vector<float> weights(trainY.size(), 1.0f);
  double positiveWeight = 0.0;
  unsigned positiveCount = 0;
  for (auto [idx, perf] : llvm::enumerate(trainY)) {
    if (perf > threshold)
      continue;
    labels[idx] = 1.0f;
    // Weight a good config by how far below the threshold it lands, so the
    // model is pulled towards the configs that beat it by the most rather than
    // treating everything in the quantile alike. The floor keeps the weights
    // usable when every measurement is identical.
    weights[idx] = std::max(1e-5, threshold - perf);
    positiveWeight += weights[idx];
    ++positiveCount;
  }
  numPositives = positiveCount;
  goodThreshold = threshold;
  if (positiveCount == 0 || positiveCount == trainY.size())
    return;

  // Normalize so that a good config weighs 1 on average, matching the weight of
  // a bad one.
  float meanPositiveWeight = positiveWeight / positiveCount;
  for (auto [label, weight] : llvm::zip_equal(labels, weights))
    if (label > 0.5f)
      weight /= meanPositiveWeight;

  LLVM_DEBUG(llvm::dbgs() << "LFBO: fitting surrogate on " << trainX.size()
                          << " configs, " << positiveCount << " good\n");
  surrogate.fit(trainX, labels, weights);
}

std::vector<unsigned>
LFBOSearch::surrogateSelect(ArrayRef<std::vector<float>> features,
                            unsigned numSelected) {
  std::vector<unsigned> selected;
  unsigned numCandidates = features.size();
  numSelected = std::min(numSelected, numCandidates);
  if (numSelected == 0)
    return selected;

  if (!surrogate.isFitted()) {
    // No model to rank with, which here means the measurements were all alike
    // and so told it nothing: fall back to a random sample of the
    // neighbourhood.
    selected.resize(numCandidates);
    std::iota(selected.begin(), selected.end(), 0u);
    std::shuffle(selected.begin(), selected.end(), rng);
    selected.resize(numSelected);
    return selected;
  }

  std::vector<float> proba(numCandidates);
  std::vector<SmallVector<int, 128>> leaves(numCandidates);
  for (unsigned idx = 0; idx < numCandidates; ++idx) {
    proba[idx] = surrogate.predictProba(features[idx]);
    surrogate.applyLeaves(features[idx], leaves[idx]);
  }

  unsigned numTrees = surrogate.getNumTrees();
  std::vector<double> similaritySums(numCandidates, 0.0);
  std::vector<bool> taken(numCandidates, false);
  selected.reserve(numSelected);
  while (selected.size() < numSelected) {
    // Two candidates are similar to the degree that the forest sends them to
    // the same leaves. Subtracting the mean similarity to what is already
    // picked keeps a batch from being one config in a dozen disguises.
    int best = -1;
    double bestScore = -kInfinity;
    for (unsigned idx = 0; idx < numCandidates; ++idx) {
      if (taken[idx])
        continue;
      double score = proba[idx];
      if (!selected.empty())
        score -=
            options.similarityPenalty * (similaritySums[idx] / selected.size());
      if (score > bestScore) {
        bestScore = score;
        best = idx;
      }
    }
    if (best < 0)
      break;

    taken[best] = true;
    selected.push_back(best);
    if (selected.size() == numSelected)
      break;
    for (unsigned idx = 0; idx < numCandidates; ++idx) {
      if (taken[idx])
        continue;
      unsigned shared = 0;
      for (auto [lhs, rhs] : llvm::zip_equal(leaves[idx], leaves[best]))
        shared += lhs == rhs;
      similaritySums[idx] += double(shared) / numTrees;
    }
  }
  return selected;
}

void LFBOSearch::feasibleNeighbors(const ConfigValues &base, unsigned param,
                                   MoveSize move,
                                   SmallVectorImpl<int64_t> &out) {
  if (move == MoveSize::Step)
    stepMoves(ladders[param], base[param], out);
  else
    radiusMoves(ladders[param], base[param], options.radius, out);

  // Whether a move lands on a config the space contains is a question no one
  // parameter's values can answer, so put each of them to the space.
  ConfigValues candidate = base;
  llvm::erase_if(out, [&](int64_t value) {
    candidate[param] = value;
    if (axes->isFeasible(candidate))
      return false;
    ++movesRejected;
    return true;
  });
}

void LFBOSearch::generateNeighbors(const ConfigValues &base,
                                   std::vector<ConfigValues> &out) {
  out.clear();
  // The first generation has only the initial population to go on, which says
  // little about which parameters matter, so it perturbs at random.
  if (surrogate.isFitted() && generation > 1)
    generateTreeGuidedNeighbors(base, out);
  else
    generateRandomNeighbors(base, out);
}

void LFBOSearch::generateRandomNeighbors(const ConfigValues &base,
                                         std::vector<ConfigValues> &out) {
  SmallVector<int64_t, 8> values;
  SmallVector<unsigned> movable;
  SmallVector<unsigned, 2> emphasized;
  unsigned numParams = ladders.size();
  for (unsigned trial = 0; trial < options.numNeighbors; ++trial) {
    ConfigValues candidate = base;

    // One tile size and the wave count move by up to `radius` doublings, ...
    emphasized.clear();
    if (!tileParams.empty())
      emphasized.push_back(tileParams[pickIndex(tileParams.size())]);
    if (waveCountParam)
      emphasized.push_back(*waveCountParam);
    // ... unless the config names neither, in which case any one parameter
    // moves, so that a trial still travels somewhere.
    if (emphasized.empty())
      emphasized.push_back(pickIndex(numParams));
    for (unsigned param : emphasized) {
      feasibleNeighbors(candidate, param, MoveSize::Radius, values);
      if (!values.empty())
        candidate[param] = values[pickIndex(values.size())];
    }

    // ... then up to `radius` of the others are moved to the next value along.
    movable.clear();
    for (unsigned param = 0; param < numParams; ++param) {
      if (llvm::is_contained(emphasized, param))
        continue;
      feasibleNeighbors(candidate, param, MoveSize::Step, values);
      if (!values.empty())
        movable.push_back(param);
    }
    unsigned count =
        pickIndex(std::min<size_t>(options.radius, movable.size()) + 1);
    for (unsigned i = 0; i < count; ++i) {
      std::swap(movable[i], movable[pickIndex(movable.size() - i) + i]);
      feasibleNeighbors(candidate, movable[i], MoveSize::Step, values);
      if (!values.empty())
        candidate[movable[i]] = values[pickIndex(values.size())];
    }

    if (candidate != base)
      out.push_back(candidate);
  }
}

void LFBOSearch::generateTreeGuidedNeighbors(const ConfigValues &base,
                                             std::vector<ConfigValues> &out) {
  std::vector<float> baseFeatures;
  encode(base, baseFeatures);

  SmallVector<int> path;
  SmallVector<unsigned> pathParams;
  SmallVector<int64_t, 8> values;
  SmallVector<int64_t, 8> bestValues;
  std::vector<float> features;

  for (unsigned trial = 0; trial < options.numNeighbors; ++trial) {
    // Using a single tree rather than the whole forest is what makes the trials
    // differ: each tree has its own opinion of which parameters matter.
    const DecisionTree &tree =
        surrogate.getTree(pickIndex(surrogate.getNumTrees()));
    if (tree.empty())
      continue;
    tree.decisionPath(baseFeatures, path);

    // The parameters this tree tests on the way to `base` are the ones it
    // considers decisive, so those are the ones worth tuning.
    std::vector<bool> seen(ladders.size(), false);
    pathParams.clear();
    for (int node : path) {
      int feature = tree.getNodes()[node].feature;
      if (feature < 0 || seen[feature])
        continue;
      seen[feature] = true;
      pathParams.push_back(feature);
    }
    // The tile sizes and the wave count are tuned whether or not this tree
    // tests them, as in the random generator.
    auto emphasize = [&](unsigned param) {
      if (!seen[param]) {
        seen[param] = true;
        pathParams.push_back(param);
      }
    };
    if (!tileParams.empty())
      emphasize(tileParams[pickIndex(tileParams.size())]);
    if (waveCountParam)
      emphasize(*waveCountParam);
    // A tree that tests nothing on this path has no opinion to follow, and a
    // trial that moves nothing is wasted, so pick a parameter at random.
    if (pathParams.empty())
      pathParams.push_back(pickIndex(ladders.size()));

    ConfigValues candidate = base;
    features = baseFeatures;
    for (unsigned param : pathParams) {
      feasibleNeighbors(candidate, param, MoveSize::Radius, values);
      if (values.empty())
        continue;

      // Greedily keep whichever value this tree scores highest, breaking ties
      // at random so that repeated trials explore different plateaus. A leaf is
      // grown until it is pure, so it scores 1 or 0 unless the tree ran out of
      // splits, and equal scores are a plateau rather than a rounding artefact.
      float bestProba = tree.predictProba(features);
      bestValues.assign(1, candidate[param]);
      for (int64_t value : values) {
        features[param] = encodeValue(logScaled[param], value);
        float proba = tree.predictProba(features);
        if (proba > bestProba) {
          bestProba = proba;
          bestValues.assign(1, value);
        } else if (proba == bestProba) {
          bestValues.push_back(value);
        }
      }
      candidate[param] = bestValues[pickIndex(bestValues.size())];
      features[param] = encodeValue(logScaled[param], candidate[param]);
    }

    if (candidate != base)
      out.push_back(candidate);
  }
}

void LFBOSearch::recordProvenance(const PerfConfigString &perfConfig,
                                  Origin origin, unsigned copy) {
  if (!traceEnabled(trace))
    return;
  provenance[perfConfig] = Provenance{origin, copy, generation};
}

void LFBOSearch::traceHeader() {
  if (!traceEnabled(trace))
    return;

  SmallVector<StringRef> names;
  axes->getParamNames(names);
  llvm::json::Array params;
  for (auto [name, ladder] : llvm::zip_equal(names, ladders))
    params.push_back(
        llvm::json::Object{{"name", name}, {"values", ladder.size()}});

  StringRef arch;
  if (auto archAttr = mod->getAttrOfType<StringAttr>(ArchAttr::getMnemonic()))
    arch = archAttr.getValue();

  traceWrite(trace, llvm::json::Object{
                        {"kind", "header"},
                        {"arch", arch},
                        {"seed", options.seed},
                        {"initialPopulation", options.initialPopulation},
                        {"copies", options.copies},
                        {"maxGenerations", options.maxGenerations},
                        {"numNeighbors", options.numNeighbors},
                        {"fracSelected", options.fracSelected},
                        {"radius", options.radius},
                        {"quantile", options.quantile},
                        {"patience", options.patience},
                        {"minImprovementDelta", options.minImprovementDelta},
                        {"similarityPenalty", options.similarityPenalty},
                        {"quickSeeds", quickSeeds.size()},
                        {"params", std::move(params)}});
}

void LFBOSearch::traceIteration(double elapsedMs,
                                ArrayRef<PerfConfigString> batch) {
  if (!traceEnabled(trace))
    return;
  totalMs += elapsedMs;

  // Where the best config came from is what says whether the search has earned
  // its keep yet, or whether the heuristic's own guess is still winning.
  std::string bestOrigin;
  auto entry = provenance.find(bestConfig);
  if (entry != provenance.end()) {
    switch (entry->second.origin) {
    case Origin::QuickSeed:
      bestOrigin = "quick";
      break;
    case Origin::RandomPad:
      bestOrigin = "random";
      break;
    case Origin::Proposal:
      bestOrigin = ("copy" + Twine(entry->second.copy) + "/gen" +
                    Twine(entry->second.generation))
                       .str();
      break;
    }
  }

  auto isActive = [](const SearchCopy &copy) { return copy.active; };
  traceWrite(trace,
             llvm::json::Object{
                 {"kind", "iteration"},
                 {"generation", generation},
                 {"proposed", batch.size()},
                 {"done", done},
                 {"copies", searchCopies.size()},
                 {"alive", llvm::count_if(searchCopies, isActive)},
                 {"elapsedMs", elapsedMs},
                 {"totalMs", totalMs},
                 {"measured", perfByConfig.size()},
                 {"visited", visited.size()},
                 {"succeeded", numSucceeded},
                 {"notApplicable", numNotApplicable},
                 {"failed", numFailed},
                 {"bestNs", finiteOrNull(bestSoFar)},
                 {"bestConfig", bestConfig.str()},
                 {"bestOrigin", bestOrigin},
                 {"stoppedOutOfPatience", stoppedOutOfPatience},
                 {"stoppedWithoutNeighbors", stoppedWithoutNeighbors},
                 {"stoppedWithoutSelection", stoppedWithoutSelection},
                 {"neighborsGenerated", neighborsGenerated},
                 {"neighborsSeenBefore", neighborsSeenBefore},
                 {"movesRejected", movesRejected},
                 {"trainSize", trainSize},
                 {"positives", numPositives},
                 {"goodThresholdNs", finiteOrNull(goodThreshold)},
                 {"pickPrecision", pickPrecision < 0.0
                                       ? llvm::json::Value(nullptr)
                                       : llvm::json::Value(pickPrecision)},
             });
}

std::vector<PerfConfigString>
LFBOSearch::getPerfConfigBatch(ArrayRef<BenchmarkResult> prevResults) {
  auto now = std::chrono::steady_clock::now();
  double elapsedMs =
      started
          ? std::chrono::duration<double, std::milli>(now - handedOut).count()
          : 0.0;
  numSucceeded = numNotApplicable = numFailed = 0;
  stoppedOutOfPatience = stoppedWithoutNeighbors = stoppedWithoutSelection = 0;
  neighborsGenerated = neighborsSeenBefore = movesRejected = 0;

  std::vector<PerfConfigString> batch = proposeBatch(prevResults);

  traceIteration(elapsedMs, batch);
  // The threshold the batch was picked against is the one the model has now,
  // and it is what the next iteration judges these picks by.
  inFlight = batch;
  inFlightThreshold = goodThreshold;
  handedOut = std::chrono::steady_clock::now();
  return batch;
}

std::vector<PerfConfigString>
LFBOSearch::proposeBatch(ArrayRef<BenchmarkResult> prevResults) {
  if (done)
    return {};
  recordResults(prevResults);

  if (!started) {
    started = true;
    if (!buildSearchSpace()) {
      done = true;
      return {};
    }
    traceHeader();
    // Whatever an earlier stage measured is training data and a starting point
    // alike; see `LFBOOptions::seedResults`. `prevResults` is empty on this
    // call by the protocol, so anything in `perfByConfig` afterwards came from
    // the seeds.
    recordResults(options.seedResults);
    if (perfByConfig.empty()) {
      // Nothing to measure means nothing to fit a model on, so the search ends
      // before it starts rather than proposing configs blindly.
      std::vector<PerfConfigString> batch = buildInitialPopulation();
      done = batch.empty();
      return batch;
    }
    // Seeded: the initial population is already in hand, so fall through and
    // spend this batch on the first generation instead of re-measuring it.
  }

  if (searchCopies.empty()) {
    if (!seedSearchCopies()) {
      done = true;
      return {};
    }
  } else {
    advanceSearchCopies();
  }
  if (llvm::none_of(searchCopies,
                    [](const SearchCopy &copy) { return copy.active; })) {
    done = true;
    return {};
  }

  if (++generation > options.maxGenerations) {
    done = true;
    return {};
  }

  fitSurrogate();
  std::vector<PerfConfigString> batch = runGeneration();
  done = batch.empty();
  return batch;
}

} // namespace

//===----------------------------------------------------------------------===//
// Moves along a parameter's values
//===----------------------------------------------------------------------===//

/// Where `value` sits on `ladder`, or where it would be inserted if the ladder
/// does not carry it.
static int64_t placeOf(ArrayRef<int64_t> ladder, int64_t value) {
  return llvm::lower_bound(ladder, value) - ladder.begin();
}

/// The values of `ladder` from place `first` to place `last`, both clamped to
/// the ladder, other than `value` itself.
static void movesBetween(ArrayRef<int64_t> ladder, int64_t value, int64_t first,
                         int64_t last, SmallVectorImpl<int64_t> &out) {
  out.clear();
  first = std::max<int64_t>(first, 0);
  last = std::min<int64_t>(last, static_cast<int64_t>(ladder.size()) - 1);
  for (int64_t idx = first; idx <= last; ++idx)
    if (ladder[idx] != value)
      out.push_back(ladder[idx]);
}

void mlir::rock::radiusMoves(ArrayRef<int64_t> ladder, int64_t value,
                             unsigned radius, SmallVectorImpl<int64_t> &out) {
  // Nothing to take the ratio of; see the header.
  if (ladder.empty() || ladder.front() <= 0 || value <= 0) {
    int64_t place = placeOf(ladder, value);
    movesBetween(ladder, value, place - radius, place + radius, out);
    return;
  }

  // Everything within a factor of two to the `radius`. A reach past the top of
  // the ladder is taken as the top itself, which is also what keeps the product
  // from overflowing.
  int64_t factor = int64_t(1) << std::min(radius, 32u);
  int64_t lowest = std::max<int64_t>(1, value / factor);
  int64_t highest =
      value > ladder.back() / factor ? ladder.back() : value * factor;
  movesBetween(ladder, value, placeOf(ladder, lowest),
               (llvm::upper_bound(ladder, highest) - ladder.begin()) - 1, out);
}

void mlir::rock::stepMoves(ArrayRef<int64_t> ladder, int64_t value,
                           SmallVectorImpl<int64_t> &out) {
  int64_t place = placeOf(ladder, value);
  movesBetween(ladder, value, place - 1, place + 1, out);
}

void mlir::rock::LFBOOptions::setEffort(SearchEffort effort) {
  switch (effort) {
  case SearchEffort::Full:
    // What the defaults already are.
    return;
  case SearchEffort::Quick:
    // Helion's "quick" effort profile. That profile also stops padding the
    // first batch with random configs, which we keep doing: how long the quick
    // list is depends on the problem, and a model fitted on that list alone
    // may have seen no variety at all.
    initialPopulation = 30;
    copies = 2;
    maxGenerations = 5;
    return;
  }
  llvm_unreachable("unhandled search effort");
}

std::unique_ptr<TuningSearchStrategy>
mlir::rock::createLFBOSearchStrategy(ModuleOp mod, const LFBOOptions &options) {
  return std::make_unique<LFBOSearch>(mod, options);
}
