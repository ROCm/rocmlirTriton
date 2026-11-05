//===- ReduceUtils.h - Common helpers for the reduce passes --------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TRANSFORMS_REDUCEUTILS_H
#define MLIR_DIALECT_ROCK_TRANSFORMS_REDUCEUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace rock {

/// Waves per workgroup for a standalone reduce kernel.
///
/// This is not a free choice. The Triton pipeline a reduce kernel is finished
/// off by decides the real launch block size, as `numWarps * warpSize` in
/// `compileUtils.cpp`, and `numWarps` is never filled from a perf config for a
/// kernel with no GEMM op, so it keeps the `init(4)` default of
/// `TritonOptions::numWarps`. The tile is chosen against the thread count that
/// implies, so this constant has to track that default; it is not a knob.
///
/// Phase C of `docs/reduce-kernel-plan.md` is what would connect the two
/// instead of keeping them in step by hand.
constexpr int64_t kReduceNumWaves = 4;

/// Whether `funcOp` is a standalone reduce kernel, the kind the reduce
/// pipeline owns.
bool isStandaloneReduceKernel(func::FuncOp funcOp);

/// The schedule to lower `reduceOp` with, for a reduction of `numRows` rows of
/// `numCols` elements each.
///
/// The tile shape is the whole schedule, because `tritongpu-coalesce` derives
/// the blocked layout of the load from it and that layout is what decides how
/// the `tt.reduce` combines. Picking
///
///   colsPerBlock = min(nextPow2(numCols), maxTileCols)
///   rowsPerBlock = clamp(nextPow2(threads * vecWidth / colsPerBlock),
///                        1, nextPow2(numRows))
///
/// keeps the elements each thread holds near a full-width load, and falls out
/// as one workgroup per row for a long reduction and many rows per workgroup
/// for a short one. See `docs/reduce-kernel-plan.md`.
ReduceParamsAttr chooseReduceParams(ReduceOp reduceOp, int64_t numRows,
                                    int64_t numCols, int64_t waveSize,
                                    int64_t maxTileCols);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TRANSFORMS_REDUCEUTILS_H
