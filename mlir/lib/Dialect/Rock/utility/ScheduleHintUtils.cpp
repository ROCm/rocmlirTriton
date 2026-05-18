//===- ScheduleHintUtils.cpp - Parse Triton scheduleHint strings ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/ScheduleHintUtils.h"

#include "llvm/Support/raw_ostream.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

using namespace mlir;

namespace mlir {
namespace rock {

// We could just reuse Triton's upstream symbolizeSchedHint to validate the
// tokens. The downside is that for memory-bound-attention there is no
// validation, which is a bit weak. Therefore, in rocmlirTriton we use
// upstream's symbolizeSchedHint, but we also validate the
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
    //   (b) the LLIR-only literal `memory-bound-attention` that upstream
    //       Triton matches via a string compare in compiler.py.
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

} // namespace rock
} // namespace mlir
