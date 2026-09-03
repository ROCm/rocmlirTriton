//===- LLMProposer.h - Running the config-proposing helper ------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transport for `LLMSearch`: hands a JSON description of the search state to a
// Python helper and reads back the configs a language model proposed.
//
// The split is deliberate. Helion talks to its providers over HTTP from
// Python (`helion/autotuner/llm/transport.py`), and everything the model
// actually needs -- the prompt text, the provider SDK, the salvaging of
// not-quite-JSON replies -- is far easier to write and to iterate on there
// than in C++. What stays here is the part that has to: the search state, and
// the loop that spends it.
//
// So this file knows nothing about prompts, models or providers. It runs a
// program, gives it a JSON object and expects a JSON object back.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TUNING_LLMPROPOSER_H
#define MLIR_LIB_DIALECT_ROCK_TUNING_LLMPROPOSER_H

#include "mlir/Support/LLVM.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <string>
#include <vector>

namespace mlir {
namespace rock {

/// Runs the helper that turns search state into proposed configs.
class LLMProposer {
public:
  /// `program` is the helper to run, or empty to use the one installed
  /// alongside the running executable. `sessionPath` is where the helper keeps
  /// the conversation between rounds, so that a later prompt can refer to an
  /// earlier one; it may be empty, in which case every round starts cold.
  /// `transcriptPath` is where the helper writes down what it asked and what
  /// it was told, for a person to read; empty means nowhere.
  LLMProposer(StringRef program, StringRef sessionPath,
              StringRef transcriptPath, unsigned timeoutSec);

  /// Asks for the configs `request` describes, as canonical named perf-config
  /// strings (`gemm:mPerBlock=128,...`) with every field spelled out.
  ///
  /// The model itself is asked for sparse configs -- Helion's contract too,
  /// "Only specify fields you want to change; unspecified = default" -- since
  /// that keeps a reply short enough to arrive well-formed and makes what a
  /// proposal is claiming legible. But a field left unspecified has to mean
  /// the exemplar the prompt was written around, and a partial perf-config
  /// string would instead pick up the schema defaults in RockAttrDefs.td,
  /// which are a serialization fallback rather than a config anyone would run.
  /// So the completion happens in the helper, against the exemplar the request
  /// carried, and what crosses back is a whole config in the one form
  /// anything in-tree accepts.
  ///
  /// Returns an error when the helper could not be run, took longer than the
  /// timeout, failed, or answered with something that is not a config list.
  ///
  /// An error here is fatal to the search that raised it, following Helion:
  /// "LLM failures are intentionally fatal: silently falling back to plain
  /// LFBO when the user opted into the LLM autotuner masks real config or
  /// connectivity bugs (e.g. wrong API key, missing mTLS cert)".
  llvm::Expected<std::vector<std::string>> propose(llvm::json::Object request);

  /// The helper this will run, resolved as the constructor would.
  static std::string resolveProgram(StringRef program);

private:
  std::string program;
  std::string sessionPath;
  std::string transcriptPath;
  unsigned timeoutSec;
};

} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TUNING_LLMPROPOSER_H
