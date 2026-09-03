//===- LLMSeededLFBOSearch.cpp - LLM proposals, then LFBO -----------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLMSeededLFBOSearch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rock-llm-lfbo-search"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Two searches driven as one. The client sees a single strategy running the
/// usual batch loop and never learns that the batches stop coming from one
/// place and start coming from another.
class LLMSeededLFBOSearch : public TuningSearchStrategy {
public:
  LLMSeededLFBOSearch(ModuleOp mod, const LLMOptions &llmOptions,
                      const LFBOOptions &lfboOptions)
      : mod(mod), lfboOptions(lfboOptions),
        llm(createLLMSearchStrategy(mod, llmOptions)) {}

  std::vector<PerfConfigString>
  getPerfConfigBatch(ArrayRef<BenchmarkResult> prevResults) override;

  bool isIterative() const override { return true; }

private:
  ModuleOp mod;
  LFBOOptions lfboOptions;
  /// The stage in charge, in order. `llm` is released as the handoff happens,
  /// which is also what says which stage a call belongs to.
  std::unique_ptr<TuningSearchStrategy> llm;
  std::unique_ptr<TuningSearchStrategy> lfbo;
  /// Everything the first stage measured, which is what the second starts
  /// from; see `LFBOOptions::seedResults`.
  std::vector<BenchmarkResult> seedResults;
};

std::vector<PerfConfigString>
LLMSeededLFBOSearch::getPerfConfigBatch(ArrayRef<BenchmarkResult> prevResults) {
  if (!llm)
    return lfbo->getPerfConfigBatch(prevResults);

  llvm::append_range(seedResults, prevResults);
  if (std::vector<PerfConfigString> batch =
          llm->getPerfConfigBatch(prevResults);
      !batch.empty())
    return batch;

  // The model has nothing further to offer, so the second stage takes over
  // from everything the first found out.
  llm.reset();
  LLVM_DEBUG(llvm::dbgs() << "LLM-LFBO: handing " << seedResults.size()
                          << " measured configs to LFBO\n");
  LFBOOptions seeded = lfboOptions;
  seeded.seedResults = std::move(seedResults);
  lfbo = createLFBOSearchStrategy(mod, seeded);
  // The results of the stage that just ended went in as seeds, so as far as
  // the new one is concerned this is its first call, and the protocol says a
  // first call carries no results.
  return lfbo->getPerfConfigBatch({});
}

} // namespace

std::unique_ptr<TuningSearchStrategy>
mlir::rock::createLLMSeededLFBOSearchStrategy(ModuleOp mod,
                                              const LLMOptions &llmOptions,
                                              const LFBOOptions &lfboOptions) {
  return std::make_unique<LLMSeededLFBOSearch>(mod, llmOptions, lfboOptions);
}
