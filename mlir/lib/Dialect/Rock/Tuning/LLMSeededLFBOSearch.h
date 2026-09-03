//===- LLMSeededLFBOSearch.h - LLM proposals, then LFBO ---------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs `LLMSearch` and then `LFBOSearch` over the same space, handing the
// second everything the first measured.
//
// Ported from Helion's `LLMSeededLFBOTreeSearch` (llm_seeded_lfbo.py), whose
// case for the pairing is that the two searches are good at different things:
// the model knows what a GPU kernel wants and can jump to a plausible region
// of the space in one round, but it cannot measure anything and has no way to
// refine; the surrogate search cannot guess, but give it somewhere to start
// and it will walk downhill from there. So the model picks the neighbourhood
// and LFBO searches it.
//
// The handoff is two things, both of which already had homes:
//
// - Helion's `seed_training_data` is `LFBOOptions::seedResults`, which
//   `LFBOSearch::recordResults` folds into the surrogate's training set. The
//   failures go too: the surrogate classifies, so a config that could not be
//   run is a label like any other.
// - Helion's `FROM_BEST_AVAILABLE` is what `seedSearchCopies` already does,
//   since it ranks every config measured so far. Seeded, `LFBOSearch` skips
//   its initial population and starts its copies from the model's best, which
//   is Helion's `best_available_pad_random=False`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TUNING_LLMSEEDEDLFBOSEARCH_H
#define MLIR_LIB_DIALECT_ROCK_TUNING_LLMSEEDEDLFBOSEARCH_H

#include "LFBOSearch.h"
#include "LLMSearch.h"

#include <memory>

namespace mlir {
namespace rock {

/// Creates the two-stage search. `llmOptions` should be the quick profile
/// however much the run as a whole may spend, since the point of the first
/// stage is to find somewhere to start and not to search: Helion's hybrid
/// takes `QUICK_LLM_SEARCH_DEFAULTS` even under full effort.
std::unique_ptr<TuningSearchStrategy>
createLLMSeededLFBOSearchStrategy(ModuleOp mod, const LLMOptions &llmOptions,
                                  const LFBOOptions &lfboOptions);

} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TUNING_LLMSEEDEDLFBOSEARCH_H
