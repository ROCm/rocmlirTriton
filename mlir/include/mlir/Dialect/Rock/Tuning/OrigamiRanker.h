//===- OrigamiRanker.h - Rank quick-tune configs with Origami ----*- C++
//-*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright Advanced Micro Devices, Inc.
//===----------------------------------------------------------------------===//
//
// Orders the quick-tuning perf-config list by predicted performance using
// Origami (external/origami), AMD's analytical latency model.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TUNING_ORIGAMIRANKER_H
#define MLIR_DIALECT_ROCK_TUNING_ORIGAMIRANKER_H

#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"

#include <vector>

namespace mlir {
namespace rock {

/// Whether Origami can model `arch`: it has both a latency model for the chip
/// and an entry in this bridge's hardware table (see OrigamiRanker.cpp).
bool origamiSupportsArch(StringRef arch);

/// Reorder `params` so the configs Origami predicts to be fastest for
/// `gemmOp`'s problem shape come first.
///
/// Every element survives: configs Origami rejects, or that cannot be
/// described to it, keep their relative order at the back of the list. The
/// list is left untouched when Origami cannot model the kernel at all --
/// an unsupported `arch`, a non-accelerated (FMA) kernel, which has no matrix
/// instruction to describe, or fewer than two configs.
void rankGemmParamsByOrigami(RockGemmWrapperInterface gemmOp,
                             const PopulateParamsInfo &info,
                             std::vector<GemmParamsAttr> &params);

/// Reorder `params` so the configs Origami predicts to be fastest for
/// `gemmGemmOp`'s attention problem come first, using Origami's flash-attention
/// model rather than its GEMM one.
///
/// Same contract as rankGemmParamsByOrigami: no config is ever dropped, and the
/// list is left untouched when Origami cannot model the kernel.
void rankAttentionParamsByOrigami(RockGemmGemmWrapperInterface gemmGemmOp,
                                  std::vector<GemmGemmParamsAttr> &params);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TUNING_ORIGAMIRANKER_H
