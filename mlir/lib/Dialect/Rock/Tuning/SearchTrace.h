//===- SearchTrace.h - JSONL trace of a tuning search -----------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where an adaptive search records what it knew, what it proposed and what
// that cost, one JSON object per line, so that a run can be read back
// afterwards (see mlir/utils/performance/analysis/plotSearchTrace.py).
//
// Nothing a search does depends on this. A search that is not tracing is a
// search with a null writer, and every call here is then a no-op.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TUNING_SEARCHTRACE_H
#define MLIR_LIB_DIALECT_ROCK_TUNING_SEARCHTRACE_H

#include "mlir/Support/LLVM.h"
#include "llvm/Support/JSON.h"

#include <memory>

namespace mlir {
namespace rock {

/// Appends one JSON object per line to a file. A tuning run is long and is
/// often interrupted, so each line is flushed as it is written: a trace of the
/// iterations that did happen is worth having.
class TraceWriter {
public:
  explicit TraceWriter(StringRef path);

  /// Whether the file opened. `openTrace` drops a writer that says no, so a
  /// non-null `SharedTrace` is always one that records.
  bool enabled() const { return os != nullptr; }

  void write(llvm::json::Object record);

private:
  std::unique_ptr<llvm::raw_fd_ostream> os;
};

/// The trace a search writes to, or null when it is not tracing.
///
/// Shared rather than owned because a composite search is several searches
/// writing one file: `LLMSeededLFBO` runs an LLM stage and then an LFBO stage,
/// and were each to open the path for itself the second would truncate the
/// first's records on the way in.
using SharedTrace = std::shared_ptr<TraceWriter>;

/// Opens `path` for tracing, or returns null if `path` is empty or cannot be
/// written. A trace nobody can write is not worth failing a tuning run over,
/// so the failure is a warning on stderr rather than an error.
SharedTrace openTrace(StringRef path);

/// Whether `trace` records anything, so a caller can skip assembling a record
/// it would only throw away.
inline bool traceEnabled(const SharedTrace &trace) { return trace != nullptr; }

/// Writes `record` to `trace` when there is one.
inline void traceWrite(const SharedTrace &trace, llvm::json::Object record) {
  if (trace)
    trace->write(std::move(record));
}

/// A time that was never measured is `null` rather than a number, since JSON
/// has no infinity and a reader must not mistake one for a measurement.
llvm::json::Value finiteOrNull(double value);

} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TUNING_SEARCHTRACE_H
