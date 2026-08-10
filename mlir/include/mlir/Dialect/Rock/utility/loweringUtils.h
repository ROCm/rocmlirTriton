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
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
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

// Return true if this shaped type will occupy more than 4 GB (2 ^ 32 bytes)
// in memory.
bool is4GBMemoryType(ShapedType type);

/// Validate every field shared by Rock GEMM tuning parameter attributes.
/// `requirePow2MN` and `requirePow2K` select the stricter tile constraints
/// required by gemm+gemm, scaled GEMMs, and targets without non-power-of-two K
/// support. Emits an error on `op` for the first invalid field.
LogicalResult validatePerfConfig(Operation *op,
                                 RockTuningParamAttrInterface params,
                                 bool requirePow2MN, bool requirePow2K);

// Whether a GEMM's M/N and K tiles have to be powers of two. A plain GEMM's
// M/N tiles are taken apart by `rock-decompose-nonpow2-tiles` and its K tile by
// `rock-gridwise-gemm-to-blockwise`, neither of which handles a scaled GEMM,
// and the K decomposition also needs an arch that does not miscompile the
// peeled loop. `AffixTuningParameters` rejects a perf config that violates
// this, and the tuning space stays inside it.
struct Pow2TileRequirement {
  bool mn;
  bool k;
};
Pow2TileRequirement pow2TilesRequired(RockGemmWrapperInterface gemmOp);

// Whether `name` is how a conv layout spells input spatial dim `dim`. The dims
// are numbered ("0i", "1i", ...), and the first two carry the legacy spellings
// "hi" and "wi", which parts of the pipeline still emit.
bool isInputSpatialDimName(StringRef name, size_t dim);

// The factor a `kPerBlock` should be a multiple of for the GEMM's K index
// computation to be cheap to advance. Returns 1 when K carries no such
// structure, i.e. on a plain GEMM or a 1x1 conv.
//
// A forward conv merges the input's channel and spatial dims into gemmK, in the
// order the input layout lists them, and only the merge's outermost dim
// advances without carrying into the ones below it. Of those carries, only one
// into a spatial dim costs anything: it moves the padded input window, leaving
// the validity mask dependent on the K loop's induction variable. A carry into
// the channel dim is just a uniform address step.
//
// So the factor exists only for a channels-first input (Merge(c, y, x)), where
// pinning the spatial dims takes a multiple of the product of the filter's
// spatial extents. Any other order puts a spatial dim outermost --
// channels-last Merge(y, x, c), interleaved Merge(y, c, x) -- and then a
// spatial dim moves on every step whatever the tile size, so no alignment can
// buy anything.
int64_t kPerBlockAlignmentFactor(RockGemmWrapperInterface gemmOp);

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

/// Apply padding to a matrix in its `firstDim` and `secondDim` if applicable.
Value padMatrix(Value matrix, OpBuilder &b, Location loc, StringRef firstDim,
                int64_t firstDimPad, StringRef secondDim, int64_t secondDimPad);

// Apply padding to a vector in its `firstDim` if applicable.
Value padVector(Value vector, OpBuilder &b, Location loc, StringRef firstDim,
                int64_t firstDimPad);

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

Value loadTile(OpBuilder &b, Location loc, Value in, Value kIter,
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
