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

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

namespace gpu {
enum class AddressSpace : uint32_t;
}

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
                                            int64_t dPerBlock);

bool isWrWAtomicKernel(StringRef arch, Type dataType, bool requiredPadding);

// Return true if this shaped type will occupy more than 4 GB (2 ^ 32 bytes)
// in memory.
bool is4GBMemoryType(ShapedType type);

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
/// algorithm. In the current v4r1 algorithm, several kernels may be needed to
/// realize a complete backward data convolution.
///
/// A kernel ID denotes an actual implicit GEMM kernels to
/// partipate the backward data convolution.
SmallVector<int64_t> backwardDataKernelIds(ArrayRef<int64_t> strideDims,
                                           ArrayRef<int64_t> dilationDims,
                                           ArrayRef<int64_t> filterDims,
                                           bool usesV4R1);

/// Return a vector type of length `len` if `len` is more than 1, otherwise,
/// return `type`.
Type vectorTypeOrSelf(Type elementType, int64_t len);

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

// Given a `value` traverses its "views" until it finds the real
// `memref::AllocOp` or fails.
FailureOr<memref::AllocOp> findMemrefAlloc(Value value);

// Get gridSize
FailureOr<IntegerAttr> getGridSize(Operation *op);

// Get blockSize
FailureOr<IntegerAttr> getBlockSize(Operation *op);

// helper to create ReassociationIndices for flattening
ReassociationIndices getReassociationForFlattening(ShapedType srcTp);

// helper to obtain a flattened memref
Value getFlattenedMemref(OpBuilder &b, Value nonFlatMemRef);

/// Construct a `memref.view` operation that interprets the buffer `buffer`,
/// whose elements are bytes, as a buffer of `type`.
TypedValue<MemRefType> viewBufferAs(OpBuilder &b, Value buffer,
                                    Type elementType);

/// Same as above but the user provides output dimensions.
TypedValue<MemRefType> viewBufferAs(OpBuilder &b, Value buffer,
                                    Type elementType,
                                    ArrayRef<int64_t> dimensions);

// Trace gemm output back to its function arguments by
// tracing through rock.store operations to find the
// destination tensor, then traces that back to function arguments.
FailureOr<SmallVector<BlockArgument>>
traceGemmOutputToArgs(Value matC, func::FuncOp func);

// Trace value to a block argument, going through view-like operations
FailureOr<BlockArgument> findBlockArgument(Value value);

// Trace gemm output to all linalg.generic that happen after it (output fusions)
// TODO(roctriton):this returns an empty list as there are no linalg.generic
// output fusions in the tensor-based IR flow.
FailureOr<SmallVector<OpOperand *>>
traceGemmOutputToGenericOps(Value matC, func::FuncOp func);

llvm::FailureOr<ArrayAttr>
computeOutputTransforms(OpBuilder &b, Location loc, int64_t mPerBlock,
                        int64_t nPerBlock, ArrayRef<int64_t> bidGridLengths);

ArrayAttr computeOutputLseTransforms(OpBuilder &b, Location loc,
                                     int64_t mPerBlock,
                                     ArrayRef<int64_t> bidGridLengths);

Type getAccType(Type elemA, Type elemB);

Value loadTile(PatternRewriter &rewriter, Location loc, Value in, Value kIter,
               StringRef dName, rock::layout::GridCoordinates gridCoords,
               int64_t kPerBlock, int64_t dPerBlock,
               SmallVector<int64_t, 3> &bidGridLengths);

Value createZeroAccBuffer(PatternRewriter &rewriter, Location loc,
                          ArrayRef<int64_t> shape, Type accType);

} // end namespace rock
} // end namespace mlir
#endif
