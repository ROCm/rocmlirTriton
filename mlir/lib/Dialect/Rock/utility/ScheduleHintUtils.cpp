//===- ScheduleHintUtils.cpp - Parse Triton scheduleHint strings ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/ScheduleHintUtils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

#include <array>
#include <utility>

using namespace mlir;

namespace mlir {
namespace rock {

namespace {

// Sentinel value emitted by the tuner for the scheduleHint perfConfig
// field. Mirrors `rock::kKnobDefault` in
// `mlir/Dialect/Rock/Pipelines/Pipelines.h`; duplicated here to keep
// `MLIRRockUtility` free of a dependency on the Pipelines header.
constexpr int64_t kKnobDefault = -1;

// Stable bit -> variant name table. Append-only; never reorder.
constexpr std::array<std::pair<int64_t, llvm::StringRef>, 2>
    kScheduleHintBitTable = {{
        {kScheduleHintAttention, "attention"},
        {kScheduleHintMemoryBoundAttention, kMemoryBoundAttentionHint},
    }};

// Returns the union of all bits known to the table. Bits set in a
// bitfield outside this mask are unrecognised and should be rejected by
// `expandScheduleHintBitfield` / `scheduleHintBitfieldToString`.
constexpr int64_t allKnownBits() {
  int64_t mask = 0;
  for (auto [bit, name] : kScheduleHintBitTable)
    mask |= bit;
  return mask;
}

} // namespace

// We could just reuse Triton's upstream symbolizeSchedHint to validate
// the tokens. The downside is that for memory-bound-attention there is
// no validation, which is a bit weak. Therefore, in rocmlirTriton we
// use upstream's symbolizeSchedHint, but we also validate the
// memory-bound-attention token.
LogicalResult parseScheduleHint(StringRef raw,
                                SmallVectorImpl<std::string> &hints) {
  hints.clear();
  if (raw.equals_insensitive("none"))
    return success();

  llvm::SmallVector<StringRef, 2> tokens;
  raw.split(tokens, ',');
  for (StringRef token : tokens) {
    StringRef trimmed = token.trim();
    if (trimmed.empty()) {
      llvm::errs() << "rocmlir: empty token in scheduleHint '" << raw << "'\n";
      return failure();
    }
    std::string lowered = trimmed.lower();

    // Token is acceptable if it is either:
    //   (a) a TTGIR variant recognised by Triton's TableGen-generated
    //       `symbolizeSchedHint` (today: `attention`); or
    //   (b) the LLIR-only literal `memory-bound-attention` that
    //       upstream Triton matches via a string compare in compiler.py.
    bool isTtgirVariant =
        triton::amdgpu::symbolizeSchedHint(lowered).has_value();
    bool isLlirVariant = (lowered == kMemoryBoundAttentionHint);
    if (!isTtgirVariant && !isLlirVariant) {
      llvm::errs() << "rocmlir: unknown scheduleHint variant '" << trimmed
                   << "'; expected `none`, `" << kMemoryBoundAttentionHint
                   << "`, or one of Triton's `SchedHint` enum values\n";
      return failure();
    }
    hints.push_back(std::move(lowered));
  }
  return success();
}

FailureOr<int64_t> scheduleHintToBitfield(StringRef raw) {
  llvm::SmallVector<std::string, 2> hints;
  if (failed(parseScheduleHint(raw, hints)))
    return failure();
  int64_t bitfield = kScheduleHintNone;
  for (StringRef hint : hints) {
    bool matched = false;
    for (auto [bit, name] : kScheduleHintBitTable) {
      if (hint == name) {
        bitfield |= bit;
        matched = true;
        break;
      }
    }
    if (!matched) {
      // A future TTGIR variant recognised by `symbolizeSchedHint` but
      // not yet wired into the bit table reaches here. Surface it as a
      // hard error so we don't silently drop it; the fix is to extend
      // `kScheduleHintBitTable` and bump the documentation in
      // `RockAttrDefs.td`.
      llvm::errs() << "rocmlir: scheduleHint variant '" << hint
                   << "' is valid but has no bit assignment; extend "
                      "kScheduleHintBitTable in ScheduleHintUtils.cpp\n";
      return failure();
    }
  }
  return bitfield;
}

LogicalResult
expandScheduleHintBitfield(int64_t bitfield,
                           SmallVectorImpl<std::string> &hints) {
  hints.clear();
  if (bitfield == kKnobDefault || bitfield == kScheduleHintNone)
    return success();
  int64_t unknown = bitfield & ~allKnownBits();
  if (unknown != 0) {
    llvm::errs() << "rocmlir: scheduleHint bitfield " << bitfield
                 << " has unknown bits set (" << unknown << ")\n";
    return failure();
  }
  for (auto [bit, name] : kScheduleHintBitTable) {
    if (bitfield & bit)
      hints.push_back(name.str());
  }
  return success();
}

FailureOr<std::string> scheduleHintBitfieldToString(int64_t bitfield) {
  if (bitfield == kKnobDefault || bitfield == kScheduleHintNone)
    return std::string("none");
  llvm::SmallVector<std::string, 2> hints;
  if (failed(expandScheduleHintBitfield(bitfield, hints)))
    return failure();
  std::string out;
  for (size_t i = 0; i < hints.size(); ++i) {
    if (i > 0)
      out += ",";
    out += hints[i];
  }
  return out;
}

} // namespace rock
} // namespace mlir
