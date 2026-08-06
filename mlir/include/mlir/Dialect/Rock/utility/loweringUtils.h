//===- loweringUtils.h - functions that often come up during lowering or turing
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ROCK_UTILITY_LOWERINGUTILS_H
#define ROCK_UTILITY_LOWERINGUTILS_H

#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

namespace mlir {
class Operation;
class Type;

namespace rock {
struct ConvolutionDims;
struct GemmSize;

namespace layout {
/// Struct containing the {g,m,n} block coordinates of a block
/// with a given bid. I.e., block bid will compute C[g_block, m_block, n_block]
/// output
struct GridCoordinates {
  Value g_block;
  Value m_block;
  Value n_block;
};

/// Struct containing the {g,m,n,split} block coordinates of a block
/// with a given bid.
struct AttnGridCoordinates : GridCoordinates {
  Value split_block;
};
} // namespace layout

// This function will create views of the register buffer of the loaded tile
// of a matrix in global memory. Those views will provide sub-tiles of the
// respective hierarchy within the GPU.
FailureOr<ArrayAttr> getLoadRegsAsTileViews(OpBuilder &b, Location loc,
                                            Value globalBuffer, StringRef dName,
                                            ArrayRef<int64_t> bidGridLengths,
                                            int64_t kPerBlock,
                                            int64_t dPerBlock, bool isKFirst);

bool isWrWAtomicKernel(StringRef arch, Type dataType, bool requiredPadding);

// Return true if this shaped type will occupy more than 4 GB (2 ^ 32 bytes)
// in memory.
bool is4GBMemoryType(ShapedType type);

// Returns true if `kBlocks` is a structurally valid backward-weight K-blocks
// value for batch dimension `N`: strictly positive and an exact divisor of
// `N`. The bwd-weight atomic-add lowering in `ConvToGemm` splits the batch
// dimension into (kBlocks, N / kBlocks), so violating either constraint would
// silently truncate the tensor.
bool isValidKBlocks(int64_t kBlocks, int64_t N);

// Heuristic logic to compute KBlock for backward weight atomic add kernel.
// The logic is adopted from MIOpen.
//
// The logic searches within the range of [1, 20 * number of CUs / gridSize],
// where gridSize is the original number of workgroups required for the
// convolution, and find the largest KBlock number which preserves the 2
// contraints:
// - GemmK (before splitting) = KBlock * KPerBlock * KPack * GemmK (after
// splitting).
// - n (batch size) is divisible by KBlock.
//
// 20 is a magic number obtained in MIOpen after empirical testing. It offers a
// reasonable reduction of GemmK after splitting, without incurring too much
// overheads on atomic adds. One potential future work is to make this value be
// tunable.
LogicalResult calculateKBlockNum(const int64_t batchSize,
                                 const GemmSize &gemmSize, int64_t MPerBlock,
                                 int64_t NPerBlock, int64_t KPerBlock,
                                 int64_t KPack, int64_t num_cu,
                                 int64_t &nKBlock);

// Heuristic to determine if every element in the output would be written by the
// backward data convolution algorithm.
bool isEveryElementWrittenBwdData(ArrayRef<int64_t> strideDims,
                                  ArrayRef<int64_t> dilationDims,
                                  ArrayRef<int64_t> filterDims);

/// Populate a vector of kernel IDs to be used by a backward data convolution
/// algorithm. Several GEMMs may be needed to realize a complete backward data
/// convolution (all created within a single kernel function).
///
/// A kernel ID denotes an actual implicit GEMM kernels to
/// partipate the backward data convolution.
SmallVector<int64_t> backwardDataKernelIds(ArrayRef<int64_t> strideDims,
                                           ArrayRef<int64_t> dilationDims,
                                           ArrayRef<int64_t> filterDims);

/// Apply tile-alignment padding to a matrix in its `firstDim` and `secondDim`
/// if applicable. Use this only for padding introduced to satisfy tile
/// alignment constraints; it marks the padding as tile alignment and must not
/// be used for semantic/user-requested padding, such as rocmlir-gen
/// `--padding_h N`.
Value padMatrixForTileAlignment(Value matrix, OpBuilder &b, Location loc,
                                StringRef firstDim, int64_t firstDimPad,
                                StringRef secondDim, int64_t secondDimPad);

/// Apply tile-alignment padding to a vector in its `firstDim` if applicable.
/// Use this only for padding introduced to satisfy tile alignment constraints;
/// it marks the padding as tile alignment and must not be used for
/// semantic/user-requested padding, such as rocmlir-gen `--padding_h N`.
Value padVectorForTileAlignment(Value vector, OpBuilder &b, Location loc,
                                StringRef firstDim, int64_t firstDimPad);

/// Normalize the argument into the form requested.
/// If a group dimension is not present, add one.
/// If doTranspose is true, meaning the user's transpose requests don't match
/// what the underlying gridwise gemm expects, transpose the matrix to match,
/// using firstDim as the name of the first dimension in the new value and
/// secondDim as the name of the second dimesion.
Value normalizeMatrix(Value matrix, OpBuilder &b, Location loc,
                      bool doTranspose, StringRef firstDim,
                      StringRef secondDim);

// Get gridSize
FailureOr<IntegerAttr> getGridSize(Operation *op);

// Get blockSize
FailureOr<IntegerAttr> getBlockSize(Operation *op);

FailureOr<SetVector<StoreOp>> traceRootOutputToStoreOps(Value output);

// Check that `newStoreMethod` is compatible with the store's current method,
// then set a prefill attribute on the function argument that the store
// destination traces back to.  AtomicAdd -> zero, AtomicMax -> -inf/INT_MIN.
LogicalResult setStoreMethodAndPrefill(OpBuilder &builder, StoreOp storeOp,
                                       StoreMethod newStoreMethod);

// Trace root output back to its function arguments by
// tracing through rock.store operations to find the
// destination tensor, then traces that back to function arguments.
FailureOr<SmallVector<BlockArgument>> traceRootOutputToArgs(Value output,
                                                            func::FuncOp func);

// Trace value to a block argument, going through view-like operations
FailureOr<BlockArgument> findBlockArgument(Value value);

llvm::FailureOr<ArrayAttr>
computeOutputTransforms(OpBuilder &b, Location loc, int64_t mPerBlock,
                        int64_t nPerBlock, ArrayRef<int64_t> bidGridLengths);

ArrayAttr computeOutputLseTransforms(OpBuilder &b, Location loc,
                                     int64_t mPerBlock,
                                     ArrayRef<int64_t> bidGridLengths);

Type getAccType(Type elemA, Type elemB);

Value loadTile(PatternRewriter &rewriter, Location loc, Value in, Value kIter,
               StringRef dName, rock::layout::GridCoordinates gridCoords,
               int64_t kPerBlock, int64_t dPerBlock, bool isKFirst,
               SmallVector<int64_t, 3> &bidGridLengths,
               mlir::rock::CacheModifier cache);

Value createZeroAccBuffer(PatternRewriter &rewriter, Location loc,
                          ArrayRef<int64_t> shape, Type accType);

/// Insert rock.transform with Broadcast dims to expand dimensions of
/// `inp` to match `outShape`. Returns `inp` unchanged if no broadcast needed.
Value insertBroadcast(OpBuilder &b, Location loc, Value inp,
                      ArrayRef<int64_t> outShape);

bool isFusionOp(Operation *op);

// Returns true if `op` should be followed during forward tracing from a
// FusionRoot result through the fusion chain. This includes fusion ops
// (arith/math elementwise), view-like ops (rock.transform, etc.), and
// rock.reduce.
bool isForwardTraceOp(Operation *op);

/// Canonical description of a ReduceOp connected to a StoreOp through
/// view-only transform chains. Transform lists are ordered producer to
/// consumer. `fusionRoots` contains every distinct upstream FusionRoot.
struct ReductionStorePath {
  ReduceOp reduceOp;
  StoreOp storeOp;
  SmallVector<TransformOp> preReduceTransforms;
  SmallVector<TransformOp> postReduceTransforms;
  Value tileSource;
  SmallVector<Operation *> fusionRoots;
};

/// Match a single-use ReduceOp -> TransformOp* -> StoreOp path.
FailureOr<ReductionStorePath> getReductionStorePath(ReduceOp reduceOp);

/// Match a StoreOp fed by a ReduceOp through TransformOps. Returns nullopt for
/// an ordinary store and failure for a malformed reduction-store path.
FailureOr<std::optional<ReductionStorePath>>
getReductionStorePath(StoreOp storeOp);

/// Invert a producer-to-consumer TransformOp chain into the transform-stack
/// order accepted by rock::transform.
ArrayAttr invertTransformChain(OpBuilder &builder, Location loc,
                               ArrayRef<TransformOp> transforms);

/// Walk the fusion chain from `root` (the FusionRoot result) and collect
/// operands of fusion ops that are NOT in the FusionRoot-result chain. These
/// are "extra inputs" to output fusions (e.g., the second operand of
/// `arith.addf %gemm_result, %extra_input`). Returns a map from original
/// value to itself; callers update the mapped values after normalize+pad.

struct FusionInfo {
  DenseMap<Value, Value> extraInputs;
  DenseSet<Value> chainValues;
  SmallVector<Operation *> fusionOps;
};

FusionInfo collectFusionInfo(Value root);
DenseMap<Value, Value> collectFusionExtraInputs(Value root);

/// After propagateOutputType has updated the FusionRoot-chain operands and
/// result types in fusion ops, this replaces the extra input operands with
/// their padded versions using the provided map (original -> padded).
void replaceFusionExtraInputs(Value root,
                              const DenseMap<Value, Value> &inputMap);

/// Propagate a new output type through the fusion chain.
/// Replaces uses of `oldRoot` in fusion ops (arith.*, math.*) with `newRoot`,
/// and updates each fusion op's result type to carry the new shape while
/// preserving its original element type. Continues recursively through the
/// fusion chain. This is needed when the root value's type changes (e.g., due
/// to padding in GemmToGridwise) and downstream fusion ops need their operands
/// and result types updated accordingly.
void propagateOutputType(Value oldRoot, Value newRoot);

/// Result of tracing a fusion root output: stores, output views, and extra
/// fusion inputs collected from the fusion chain.
struct OutputsAndFusionInputs {
  SetVector<StoreOp> stores;
  SmallVector<Value> outputViews;
  DenseMap<Value, Value> fusionInputMap;
};

/// Trace a fusion root output to its store ops, collecting the output views
/// (store destinations) and any extra fusion inputs (operands of fusion ops
/// that are not part of the gemm-result chain, e.g. the bias in arith.addf).
FailureOr<OutputsAndFusionInputs> traceOutputsAndFusionInputs(Value rootOut);

/// Create a NarrowTypeEmulationConverter targeting i8 with memref conversions
/// and unrealized_conversion_cast materializations.  Used by both
/// RockConvertNarrowTypeSignaturesPass and RockEmulateNarrowTypesPass.
arith::NarrowTypeEmulationConverter create4BitTypeConverter();

/// Mark the module containing `op` (or `op` itself if it is a ModuleOp) with
/// the `rock.not_applicable` attribute, signaling that the current
/// (kernel x perf-config x hardware) combination cannot be compiled for a
/// structural reason (e.g. LDS overflow, unsupported fusion+SplitK, ...).
/// The pass should still call `signalPassFailure()` so the pipeline halts;
/// consumers (tuning driver) inspect this attribute on PM failure to
/// distinguish a "not-applicable" config from a real compilation bug.
void markAsNotApplicable(Operation *op);

} // end namespace rock
} // end namespace mlir
#endif
