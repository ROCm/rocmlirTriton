//===--------- RockTuning.h - MLIR tuning parameter generation ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MLIR base types for tuning
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
#define MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H

#include "mlir-c/Dialect/Rock.h"
#include "mlir-c/Dialect/RockEnums.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/RWMutex.h"

#include <memory>
#include <vector>

namespace mlir {
namespace rock {

// The available sets of tuning parameters.
enum class TuningParamSetKind : uint32_t {
  // A short (around 10-15) list of tuning entries that should be tried to
  // quickly obtain reasonable performance on an unknown configuration.
  Quick = 0,
  // A full tuning space suitable for most offline tuning tasks which omits
  // configurations a heuristic expects not to perform well (see
  // `PopulateParams::couldBePerformant`).
  Full = 1,
  // A wider version of `Full` meant for tuning experiments: it stretches a few
  // of the axes -- numWaves up to the hardware's workgroup limit, larger
  // K/block tiles -- and drops the performance heuristic above, rather than
  // adding parameters. Despite the name, it is still not every combination of
  // every value a parameter can take: the checks that rule out configs no
  // kernel could run (the LDS blacklist, Triton's per-tensor element cap, ...)
  // prune it just as they prune `Full`.
  Exhaustive = 2,
};

// A serialized perf config, e.g. "gemm:mPerBlock=64,nPerBlock=64,...". This is
// the currency of the tuning interfaces: unlike the attributes the configs are
// built from, a string is self-contained, so it stays valid once the
// MLIRContext that produced it is gone and can be handed to a client that
// compiles in a context of its own.
using PerfConfigString = SmallString<ROCMLIR_TUNING_PARAM_STRING_BUFSZ>;

// The same config as the numbers it is made of, ordered as the perf config
// lists them (`RockTuningParamAttrInterface::getParamValues`). What a search
// works in, since moving one parameter is an index away rather than a reparse.
// Inline capacity covers the parameters a GEMM config has today, and nothing
// breaks if a parameter is added to, or removed from, a perf config.
using ConfigValues = SmallVector<int64_t, 24>;

// Reads a serialized config back through the attribute that owns the format,
// which is the only thing that knows how a config is spelled. Returns null if
// neither perf-config attribute recognizes the string.
RockTuningParamAttrInterface parsePerfConfig(MLIRContext *ctx,
                                             StringRef perfConfig);

// Parameter container holding one serialized tuning parameter
struct ParamEntry {
  PerfConfigString param;
  KernelType primaryOpType;
};

// Total tuning space
struct TuningParamSet {
  std::vector<PerfConfigString> tuningRange;
  KernelType primaryOpType;
};

// The checks `isFeasible` can turn a config down on.
//
// An enumeration rather than a message, because a rejection is counted and
// grouped as often as it is read: the LLM search tallies them per round and
// the trace records the tally. Their names reach a prompt, so they are short
// and stable -- a model shown "ldsBlacklist" twice should recognize it as the
// same wall.
enum class FeasibilityCheck : uint32_t {
  // A parameter holds a value its axis does not list, and which is not the
  // `kKnobDefault` a knob is allowed to carry.
  NotOnAxis = 0,
  // The values do not spell a perf config for this kernel at all.
  MalformedConfig = 1,
  // A tile would need more elements in one tensor than Triton allows.
  TritonTensorCap = 2,
  // A known-bad (m, n, k) tiling; see LdsBlacklist.h.
  LdsBlacklist = 3,
  // Compiling it would take long enough to stall the search.
  CompileCostBudget = 4,
  // `wavesPerEU` asks for more registers than an EU has.
  RegisterBudget = 5,
  // `useBufferAtomics` without `useBufferOps`.
  BufferKnobsDisagree = 6,
  // In the space, but the heuristic expects it to be slow; `Full` drops these
  // and `Exhaustive` keeps them.
  NotPerformant = 7,
};

// The short name of a check, for a prompt, a trace or a diagnostic.
StringRef getFeasibilityCheckName(FeasibilityCheck check);

// The same tuning space described by its axes instead of by their product: the
// values each parameter may take, plus the predicate deciding which of their
// combinations the space contains. A search can explore a space far larger than
// memory this way, since nothing is enumerated, which `createTunableParamSpace`
// necessarily does.
class TuningParamAxes {
public:
  virtual ~TuningParamAxes();

  // The values of each parameter that are worth trying, one list per
  // parameter, ordered as a perf config lists them (see
  // `RockTuningParamAttrInterface::getParamValues`).
  //
  // Where one parameter's legal values depend on another's, the list is
  // their union over the whole space, so the product of the axes is wider
  // than the space and `isFeasible` decides which combinations are in it.
  //
  // A value can be legal without being listed: a knob's `kKnobDefault`
  // resolves to off or on, so trying it too would only re-time whichever
  // it means here. `isFeasible` accepts such a config anyway, which is how
  // a search can start from one -- a quick-list config, say -- and step
  // onto the axes.
  virtual ArrayRef<std::vector<int64_t>> getAxes() const = 0;

  // Whether `values` is a config the space admits. When it is not,
  // `refusedOn` is set to the check that turned it down.
  //
  // That a caller can ask which check matters because the axes cannot express
  // what rules a config out: they hold each parameter's values, while
  // feasibility is a question about their combination. A search that proposes
  // blindly -- an LLM one -- has to be told which of its proposals died and
  // why, or it will keep proposing them; the searches that walk the space a
  // value at a time never ask.
  virtual bool isFeasible(ArrayRef<int64_t> values,
                          FeasibilityCheck *refusedOn) const = 0;

  bool isFeasible(ArrayRef<int64_t> values) const {
    return isFeasible(values, /*refusedOn=*/nullptr);
  }

  // The perf-config key of each parameter, ordered as `getAxes`.
  virtual void getParamNames(SmallVectorImpl<StringRef> &names) const = 0;

  // Which parameters are the tri-state knobs, ordered as `getAxes`. A knob's
  // axis holds 0 and 1, but `kKnobDefault` (-1, "let the compiler decide") is
  // legal too and is what every config the tuning space hands out spells, so
  // the axis understates what `isFeasible` accepts by exactly this much.
  //
  // A search that steps from one listed value to the next never has to know.
  // One that describes the space to somebody else -- to a language model, say
  // -- does, or it will offer a boolean where the interesting answer is "you
  // decide".
  virtual void getKnobParams(SmallVectorImpl<bool> &isKnob) const = 0;

  // Spells `values` as the perf config string the space would have emitted.
  virtual void serialize(ArrayRef<int64_t> values,
                         PerfConfigString &out) const = 0;
};

// Returns nullptr when the module holds no op with a tuning space.
std::unique_ptr<TuningParamAxes>
createTunableParamAxes(ModuleOp mod, TuningParamSetKind kind);

TuningParamSet *createTunableParamSpace(ModuleOp mod, TuningParamSetKind kind);
// Get a parameters from the set of tunable parameters.
bool tuningGetParam(TuningParamSet *tuningSpace, unsigned pos,
                    ParamEntry *paramEntry);
bool tuningSetStr(ModuleOp &mod, StringRef perfConfig);

// A tuning table for rocmlirTriton.
// Note that this table carries its own reader-writer lock so that it can be
// used from multiple client threads without requiring StringMap to be
// thread-safe.
struct TuningTable {
  llvm::sys::SmartRWMutex<true> lock;
  llvm::StringMap<
      std::pair<SmallString<ROCMLIR_TUNING_PARAM_STRING_BUFSZ>, float>>
      tuningMap;
};

TuningTable *tuningTableCreate();
size_t getTuningHash(ModuleOp &mod);
LogicalResult getTuningProblemStr(ModuleOp mod, SmallVectorImpl<char> &out);
bool tuningTableUpdate(TuningTable *perfTable, StringRef problem,
                       StringRef perfConfig, float time);
LogicalResult tuningTableLookup(TuningTable *perfTable, ModuleOp &mod,
                                SmallVectorImpl<char> &out);
LogicalResult tuningTableLookupByKey(TuningTable *perfTable,
                                     SmallVectorImpl<char> &out);

bool isSplitKRequested(ModuleOp mod, StringRef perfConfig);
bool isSplitKRequested(StringAttr perfConfig);
int64_t retrieveSplitKValue(StringAttr perfConfig);

// This method checks a given fused module is actually fusible
// for the given perfConfig
bool isModuleFusible(ModuleOp module, StringRef perfConfig);

} // namespace rock
} // namespace mlir
#endif // MLIR_DIALECT_ROCK_ROCKTUNINGTYPE_H
