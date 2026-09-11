//===- QuickTuningProblemKey.h - tuning problem identity --------*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The one serialization of a tuning problem, in the two spellings the codebase
// needs, plus the 64-bit hash of the quick-tuning spelling.
//
// Both spellings come out of the same emitter so that they cannot drift apart:
// the perf-database key that `getTuningProblemStr` hands to MIGraphX and
// perfRunner.py, and the quick-tuning problem key that the sharded
// quick-tuning database (QuickTuningShardDb.h) is probed with.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TUNING_QUICKTUNINGPROBLEMKEY_H
#define MLIR_DIALECT_ROCK_TUNING_QUICKTUNINGPROBLEMKEY_H

#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include <cstdint>

namespace mlir {
namespace rock {

/// Which fields a serialized tuning problem carries.
enum class TuningProblemFormat {
  /// The MIOpenDriver-flavoured perf-database key: architecture, compute-unit
  /// and chiplet counts, element types and problem shape. This is what
  /// `getTuningProblemStr` returns and what perfRunner.py parses, so its
  /// spelling is frozen.
  PerfDb,

  /// The quick-tuning problem key: the kernel type followed by the problem
  /// shape and the fusion/layout flags, with the architecture prefix and the
  /// element types dropped.
  ///
  /// Those are dropped because the shard the key is probed in already selects
  /// on them, and because dropping the data type is what lets a lookup that
  /// resolved its key through ParamLookupTable::findFallback (fp8 -> i8, say)
  /// still find its problem in the shard it landed on.
  ///
  /// The kernel type is spelled out even though shards are already kernel-type
  /// scoped, since `findFallback` also substitutes the kernel type: with it in
  /// the key, a gemm+gemm problem probing the attention shard misses by
  /// construction rather than by relying on two ops' shape fields never
  /// colliding.
  ///
  /// Unlike `PerfDb`, this spelling carries neither the surrounding fusions nor
  /// split-K legality. Quick-tuning problems are measured on standalone
  /// kernels, so folding the fusion identity in would make every fused kernel
  /// miss the per-problem lookup and fall back to the set cover, which is
  /// exactly the narrowing this key exists to provide. The LDS and
  /// register-pressure concern that puts fusions in the perf-database key is
  /// weaker here because a quick-tuning list is a sweep candidate list, not a
  /// verdict: a config the fusion cannot fit is rejected at compile time as
  /// `CompilationStatus::NotApplicable`, costing one candidate slot.
  QuickTuningKey,
};

/// Serializes `gemmIF`'s tuning problem into `out` in the requested format,
/// appending to whatever `out` already holds. Fails on an operation whose
/// element types or kernel type have no spelling; which operations fail does
/// not depend on `format`.
///
/// `PerfDb` emits a diagnostic when it fails and `QuickTuningKey` does not,
/// because the two callers disagree about whether failure is an error: a
/// problem the perf database cannot name is one it cannot tune, while a problem
/// the quick-tuning database cannot name is simply an unknown one, which gets
/// the set cover like any other. Staying silent is what lets that caller avoid
/// installing a diagnostic handler to discard the message -- a handler that,
/// since a pass pipeline runs functions in parallel over one context's
/// diagnostic engine, would also discard whatever another thread emits while it
/// is alive.
LogicalResult serializeTuningProblem(RockGemmWrapperInterface gemmIF,
                                     TuningProblemFormat format,
                                     SmallVectorImpl<char> &out);
LogicalResult serializeTuningProblem(RockGemmGemmWrapperInterface gemmGemmOp,
                                     TuningProblemFormat format,
                                     SmallVectorImpl<char> &out);

/// Hashes a `TuningProblemFormat::QuickTuningKey` string into the value stored
/// in a shard's `problems` array. Split out from the serialization so that
/// tests can pin the hash of a literal key, and so that a tool can hash a key
/// it read from a file.
uint64_t hashQuickTuningProblemKey(StringRef key);

/// Serializes the operation's quick-tuning problem key and hashes it. Fails
/// exactly when `serializeTuningProblem` does.
FailureOr<uint64_t> getQuickTuningProblemHash(RockGemmWrapperInterface gemmIF);
FailureOr<uint64_t>
getQuickTuningProblemHash(RockGemmGemmWrapperInterface gemmGemmOp);

/// As above, for `mod`'s primary tunable operation. Also fails when `mod` has
/// no such operation.
FailureOr<uint64_t> getQuickTuningProblemHash(ModuleOp mod);

/// Applies `fn` to `mod`'s primary tunable operation: its first
/// `RockGemmWrapperInterface` operation or, if it has none, its first
/// `RockGemmGemmWrapperInterface` operation. Returns `failure()` when it has
/// neither, so `ResultT` has to be constructible from `LogicalResult`.
///
/// The two tuning-problem entry points that take a whole module go through
/// this, so that they agree on which operation a module's tuning identity is
/// taken from.
template <typename ResultT, typename Fn>
ResultT visitPrimaryTuningOp(ModuleOp mod, Fn &&fn) {
  RockGemmWrapperInterface gemmIF;
  if (mod->walk([&](RockGemmWrapperInterface op) {
           gemmIF = op;
           return WalkResult::interrupt();
         })
          .wasInterrupted())
    return fn(gemmIF);

  RockGemmGemmWrapperInterface gemmGemmOp;
  if (mod->walk([&](RockGemmGemmWrapperInterface op) {
           gemmGemmOp = op;
           return WalkResult::interrupt();
         })
          .wasInterrupted())
    return fn(gemmGemmOp);

  return failure();
}

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TUNING_QUICKTUNINGPROBLEMKEY_H
