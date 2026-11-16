//===- NonGemmUtils.h - Common helpers for the nongemm passes -----------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// A nongemm kernel is a graph of views and elementwise ops, optionally ending
// in a single `rock.reduce`, with no GEMM/conv/attention root. These are the
// helpers the passes that lower one share.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TRANSFORMS_NONGEMMUTILS_H
#define MLIR_DIALECT_ROCK_TRANSFORMS_NONGEMMUTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace rock {

/// Waves per workgroup for a standalone nongemm kernel.
///
/// This is not a free choice. The Triton pipeline a nongemm kernel is
/// finished off by decides the real launch block size, as
/// `numWarps * warpSize` in `compileUtils.cpp`, and `numWarps` is never filled
/// from a perf config for a kernel with no GEMM op, so it keeps the `init(4)`
/// default of `TritonOptions::numWarps`. The tile is chosen against the thread
/// count that implies, so this constant has to track that default; it is not a
/// knob. The same goes for `kNonGemmNumStages` and `TritonOptions::numStages`,
/// which decides whether the loop over a long reduction is software-pipelined.
///
/// Recording both in `rock.nongemm_params` and then wiring that attribute into
/// `TritonOptions` is what would connect the two instead of keeping them in
/// step by hand; see `docs/nongemm-kernel-pipeline.md`.
constexpr int64_t kNonGemmNumWaves = 4;
constexpr int64_t kNonGemmNumStages = 2;

/// Whether `funcOp` is a standalone nongemm kernel, the kind the nongemm
/// pipeline owns.
bool isStandaloneNonGemmKernel(func::FuncOp funcOp);

/// Whether this nongemm kernel ends in a reduction. The two cases share a
/// pipeline but not a tile shape: a reduce grids over rows and walks the
/// reduced axis in a loop, a pure map grids over the flattened elements.
bool hasReduce(func::FuncOp funcOp);

/// The schedule to lower a nongemm kernel with, over its canonical `M x N`
/// view: `numRows` rows of `numCols` elements, where a kernel with no reduce
/// is `1 x totalElements`.
///
/// The tile shape is the whole schedule, because `tritongpu-coalesce` derives
/// the blocked layout of the load from it and that layout is what decides how
/// a `tt.reduce` combines. Picking
///
///   colsPerBlock = min(nextPow2(numCols), maxTileCols)
///   rowsPerBlock = clamp(nextPow2(threads * vecWidth / colsPerBlock),
///                        1, nextPow2(numRows))
///
/// keeps the elements each thread holds near a full-width load, and falls out
/// as one workgroup per row for a long reduction and many rows per workgroup
/// for a short one. A kernel with no reduce has `numRows == 1`, so the clamp
/// hands it `rowsPerBlock == 1` with no special case: with no cross-lane
/// combine to pay for, rows buy nothing and only cap the vector width.
NonGemmParamsAttr chooseNonGemmParams(MLIRContext *ctx, Type elemType,
                                      int64_t numRows, int64_t numCols,
                                      int64_t waveSize, int64_t maxTileCols);

/// Tile size used by the standalone elementwise lowering.
/// Falls back to 256 when rock.block_size is absent.
///
/// TODO(nongemm): this conflates two different quantities. `rock.block_size`
/// means threads per workgroup everywhere else (`AffixTuningParameters`,
/// `AddTritonMetadata`, `RockEmitGpuBinary`), while elements per block is
/// `rowsPerBlock * colsPerBlock`. It goes away once the elementwise lowering
/// is merged into `rock-lower-nongemm-to-blockwise` and takes its tile from
/// `chooseNonGemmParams` like the reduce path does.
int64_t getElementwiseTileSize(func::FuncOp funcOp);

/// Returns leaf tensor inputs of the elementwise fusion DAG.
/// Leaves are block arguments or arith constants reached via transform chains.
llvm::SetVector<Value> getElementwiseKernelInputs(func::FuncOp funcOp);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_TRANSFORMS_NONGEMMUTILS_H
