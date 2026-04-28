//===- ConvToGemm.cpp - Convert linalg conv generics to matmul ------------===//
//
// Copyright 2026 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This pass converts linalg.generic operations representing convolutions into
// linalg.generic operations representing matrix multiplications (im2col + GEMM).
//
// The transformation follows the im2col approach:
// 1. Identify linalg.generic with convolution semantics
// 2. Create an im2col tensor that rearranges the input
// 3. Replace the convolution with a matmul operation
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace cpu {
#define GEN_PASS_DEF_CPUCONVTOGEMMPASS
#include "mlir/Conversion/CPU/Passes.h.inc"
} // namespace cpu
} // namespace mlir

#define DEBUG_TYPE "cpu-conv-to-gemm"

using namespace mlir;
using namespace mlir::cpu;

namespace {

/// Describes the structure of a backward data convolution identified
/// from a linalg.generic operation.
struct ConvBwdDataInfo {
  // Dimension indices in the iteration space (8 dimensions total)
  // Parallel dimensions (output shape)
  unsigned batchDim;       // d0: N - batch dimension
  unsigned groupDim;       // d1: G - group dimension
  unsigned outChannelDim;  // d2: C - output channel (input channel in forward)
  unsigned outHeightDim;   // d3: Ho - output height
  unsigned outWidthDim;    // d4: Wo - output width
  // Reduction dimensions (contraction)
  unsigned inChannelDim;   // d5: K - input channel (output channel in forward)
  unsigned filterHeightDim; // d6: Fh - filter height
  unsigned filterWidthDim;  // d7: Fw - filter width

  // Shapes extracted from tensors
  int64_t batchSize;      // N
  int64_t groupSize;      // G
  int64_t outChannels;    // C
  int64_t outHeight;      // Ho
  int64_t outWidth;       // Wo
  int64_t inChannels;     // K
  int64_t filterHeight;   // Fh
  int64_t filterWidth;    // Fw
  int64_t inputHeight;    // H_in (padded input)
  int64_t inputWidth;     // W_in (padded input)

  // Operands
  Value input;  // Padded/strided input gradient [N, H_in, W_in, G, K]
  Value filter; // Rotated filter [G, K, Fh, Fw, C]
  Value output; // Output (input gradient result) [N, Ho, Wo, G, C]

  // Element type
  Type elementType;
};

/// Check if a linalg.generic has the expected body for a contraction:
/// mul followed by add (or just yield for copy-like ops).
static bool hasContractionBody(linalg::GenericOp op) {
  Block &body = op.getRegion().front();

  // Expected pattern: mul + add + yield (3 ops in body) or similar
  if (body.getOperations().size() < 2)
    return false;

  // Find the yield op
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp)
    return false;

  // Check for mul + add pattern
  // The yielded value should come from an add
  Value yieldedValue = yieldOp.getOperand(0);
  auto addOp = yieldedValue.getDefiningOp<arith::AddFOp>();
  if (!addOp)
    return false;

  // One operand of add should be a mul, the other should be the accumulator
  auto mulOp = addOp.getLhs().getDefiningOp<arith::MulFOp>();
  if (!mulOp)
    mulOp = addOp.getRhs().getDefiningOp<arith::MulFOp>();

  return mulOp != nullptr;
}

/// Analyze the indexing maps to determine if this is a backward data convolution.
///
/// Expected pattern for backward data convolution:
///   input:  (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3 + d6, d4 + d7, d1, d5)
///   filter: (d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5, d6, d7, d2)
///   output: (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d3, d4, d1, d2)
///
/// Iterator types: [parallel, parallel, parallel, parallel, parallel,
///                  reduction, reduction, reduction]
///
/// This represents:
///   output[n, ho, wo, g, c] = Σ input[n, ho+fh, wo+fw, g, k] × filter[g, k, fh, fw, c]
///                            k,fh,fw
static bool isBackwardDataConvolution(linalg::GenericOp op,
                                      ConvBwdDataInfo &info) {
  // Check iterator types: expect 5 parallel + 3 reduction
  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();

  if (iteratorTypes.size() != 8) {
    LLVM_DEBUG(llvm::dbgs() << "  Not a bwd_data conv: expected 8 dims, got "
                            << iteratorTypes.size() << "\n");
    return false;
  }

  // Count parallel and reduction dimensions
  unsigned numParallel = 0, numReduction = 0;
  for (auto it : iteratorTypes) {
    if (it == utils::IteratorType::parallel)
      ++numParallel;
    else if (it == utils::IteratorType::reduction)
      ++numReduction;
  }

  if (numParallel != 5 || numReduction != 3) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Not a bwd_data conv: expected 5 parallel + 3 "
               << "reduction, got " << numParallel << " parallel + "
               << numReduction << " reduction\n");
    return false;
  }

  // Check that we have exactly 2 inputs and 1 output
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Not a bwd_data conv: expected 2 inputs + 1 output\n");
    return false;
  }

  // Check for contraction body
  if (!hasContractionBody(op)) {
    LLVM_DEBUG(llvm::dbgs() << "  Not a bwd_data conv: body is not mul+add\n");
    return false;
  }

  // Get indexing maps
  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();
  AffineMap inputMap = indexingMaps[0];
  AffineMap filterMap = indexingMaps[1];
  AffineMap outputMap = indexingMaps[2];

  LLVM_DEBUG({
    llvm::dbgs() << "  Analyzing indexing maps:\n";
    llvm::dbgs() << "    Input:  " << inputMap << "\n";
    llvm::dbgs() << "    Filter: " << filterMap << "\n";
    llvm::dbgs() << "    Output: " << outputMap << "\n";
  });

  // Verify output map has pure dimension expressions (no additions)
  for (AffineExpr expr : outputMap.getResults()) {
    if (!isa<AffineDimExpr>(expr)) {
      LLVM_DEBUG(llvm::dbgs() << "  Not a bwd_data conv: output map has "
                              << "non-dimension expression\n");
      return false;
    }
  }

  // The output map tells us which dimensions are batch, spatial, group, channel
  SmallVector<unsigned> outputDimIndices;
  for (AffineExpr expr : outputMap.getResults()) {
    outputDimIndices.push_back(cast<AffineDimExpr>(expr).getPosition());
  }

  if (outputDimIndices.size() != 5) {
    LLVM_DEBUG(llvm::dbgs() << "  Not a bwd_data conv: output has "
                            << outputDimIndices.size() << " dims, expected 5\n");
    return false;
  }

  // Extract dimension assignments from output map
  // Output layout: [N, Ho, Wo, G, C]
  info.batchDim = outputDimIndices[0];
  info.outHeightDim = outputDimIndices[1];
  info.outWidthDim = outputDimIndices[2];
  info.groupDim = outputDimIndices[3];
  info.outChannelDim = outputDimIndices[4];

  // The reduction dimensions should be the remaining ones
  SmallVector<unsigned> reductionDims;
  for (unsigned i = 0; i < iteratorTypes.size(); ++i) {
    if (iteratorTypes[i] == utils::IteratorType::reduction)
      reductionDims.push_back(i);
  }

  if (reductionDims.size() != 3) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Not a bwd_data conv: expected 3 reduction dims\n");
    return false;
  }

  info.inChannelDim = reductionDims[0];
  info.filterHeightDim = reductionDims[1];
  info.filterWidthDim = reductionDims[2];

  // Store operands
  info.input = op.getDpsInputOperand(0)->get();
  info.filter = op.getDpsInputOperand(1)->get();
  info.output = op.getDpsInitOperand(0)->get();

  // Extract shapes from the output tensor [N, Ho, Wo, G, C]
  auto outputType = cast<RankedTensorType>(info.output.getType());
  ArrayRef<int64_t> outputShape = outputType.getShape();

  info.batchSize = outputShape[0];
  info.outHeight = outputShape[1];
  info.outWidth = outputShape[2];
  info.groupSize = outputShape[3];
  info.outChannels = outputShape[4];
  info.elementType = outputType.getElementType();

  // Extract input dimensions [N, H_in, W_in, G, K]
  auto inputType = cast<RankedTensorType>(info.input.getType());
  ArrayRef<int64_t> inputShape = inputType.getShape();
  info.inputHeight = inputShape[1];
  info.inputWidth = inputShape[2];

  // Extract filter dimensions [G, K, Fh, Fw, C]
  auto filterType = cast<RankedTensorType>(info.filter.getType());
  ArrayRef<int64_t> filterShape = filterType.getShape();

  info.inChannels = filterShape[1];
  info.filterHeight = filterShape[2];
  info.filterWidth = filterShape[3];

  LLVM_DEBUG({
    llvm::dbgs() << "  Identified backward data convolution:\n";
    llvm::dbgs() << "    Batch: " << info.batchSize << "\n";
    llvm::dbgs() << "    Groups: " << info.groupSize << "\n";
    llvm::dbgs() << "    Output channels: " << info.outChannels << "\n";
    llvm::dbgs() << "    Output spatial: " << info.outHeight << " x "
                 << info.outWidth << "\n";
    llvm::dbgs() << "    Input channels: " << info.inChannels << "\n";
    llvm::dbgs() << "    Filter spatial: " << info.filterHeight << " x "
                 << info.filterWidth << "\n";
    llvm::dbgs() << "    Input spatial (padded): " << info.inputHeight << " x "
                 << info.inputWidth << "\n";
  });

  return true;
}

/// Create the im2col linalg.generic operation.
/// This extracts input patches into a high-dimensional tensor, then collapses it.
///
/// Input: [N, H_in, W_in, G, K]
/// Im2col intermediate: [G, N, Ho, Wo, K, Fh, Fw]
/// Im2col final: [G, N * Ho * Wo, K * Fh * Fw]
///
/// The indexing for im2col:
///   im2col[g, n, ho, wo, k, fh, fw] = input[n, ho + fh, wo + fw, g, k]
static Value createIm2ColOp(PatternRewriter &rewriter, Location loc,
                            const ConvBwdDataInfo &info) {
  MLIRContext *ctx = rewriter.getContext();

  // Step 1: Create a 7D im2col tensor [G, N, Ho, Wo, K, Fh, Fw]
  // G is first to become the batch dimension for batched GEMM
  SmallVector<int64_t> im2col7DShape = {
      info.groupSize,    // G
      info.batchSize,    // N
      info.outHeight,    // Ho
      info.outWidth,     // Wo
      info.inChannels,   // K
      info.filterHeight, // Fh
      info.filterWidth   // Fw
  };
  auto im2col7DType = RankedTensorType::get(im2col7DShape, info.elementType);
  Value im2col7DTensor =
      tensor::EmptyOp::create(rewriter, loc, im2col7DShape, info.elementType);

  // Build affine expressions for dimension mapping
  // Iteration space: (g, n, ho, wo, k, fh, fw)
  // Input access: input[n, ho + fh, wo + fw, g, k]
  // Output access: im2col[g, n, ho, wo, k, fh, fw] (identity)
  AffineExpr g, n, ho, wo, k, fh, fw;
  bindDims(ctx, g, n, ho, wo, k, fh, fw);

  // Input indexing: [n, ho + fh, wo + fw, g, k]
  SmallVector<AffineExpr> inputExprs = {n, ho + fh, wo + fw, g, k};
  AffineMap inputMap = AffineMap::get(7, 0, inputExprs, ctx);

  // Output indexing: identity [g, n, ho, wo, k, fh, fw]
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(7, ctx);

  // All dimensions are parallel (we're just copying/gathering data)
  SmallVector<utils::IteratorType> iteratorTypes(7,
                                                  utils::IteratorType::parallel);

  SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};

  auto im2col7DOp = linalg::GenericOp::create(
      rewriter, loc, im2col7DType,
      /*inputs=*/ValueRange{info.input},
      /*outputs=*/ValueRange{im2col7DTensor}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // Just copy the input value to the output
        linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
      });

  Value im2col7DResult = im2col7DOp.getResult(0);

  // Step 2: Collapse [G, N, Ho, Wo, K, Fh, Fw] to [G, N * Ho * Wo, K * Fh * Fw]
  int64_t mDim =
      info.batchSize * info.outHeight * info.outWidth; // N * Ho * Wo
  int64_t kDim =
      info.inChannels * info.filterHeight * info.filterWidth; // K * Fh * Fw

  SmallVector<int64_t> im2col3DShape = {info.groupSize, mDim, kDim};
  auto im2col3DType = RankedTensorType::get(im2col3DShape, info.elementType);

  // Reassociation: [[0], [1, 2, 3], [4, 5, 6]]
  // [G] -> [G]
  // [N, Ho, Wo] -> [N * Ho * Wo]
  // [K, Fh, Fw] -> [K * Fh * Fw]
  SmallVector<ReassociationIndices> reassociation = {{0}, {1, 2, 3}, {4, 5, 6}};

  Value im2col3D = tensor::CollapseShapeOp::create(
      rewriter, loc, im2col3DType, im2col7DResult, reassociation);

  LLVM_DEBUG({
    llvm::dbgs() << "  Im2col shapes: [" << info.groupSize << ", "
                 << info.batchSize << ", " << info.outHeight << ", "
                 << info.outWidth << ", " << info.inChannels << ", "
                 << info.filterHeight << ", " << info.filterWidth << "] -> ["
                 << info.groupSize << ", " << mDim << ", " << kDim << "]\n";
  });

  return im2col3D;
}

/// Reshape the filter from [G, K, Fh, Fw, C] to [G, K * Fh * Fw, C]
static Value reshapeFilterForGemm(PatternRewriter &rewriter, Location loc,
                                  const ConvBwdDataInfo &info) {
  // Filter: [G, K, Fh, Fw, C] -> [G, K * Fh * Fw, C]
  int64_t kDim =
      info.inChannels * info.filterHeight * info.filterWidth; // K * Fh * Fw
  int64_t nDim = info.outChannels;                            // C

  SmallVector<int64_t> newShape = {info.groupSize, kDim, nDim};
  auto newType = RankedTensorType::get(newShape, info.elementType);

  // Create reassociation indices: [[0], [1, 2, 3], [4]]
  // [G] -> [G]
  // [K, Fh, Fw] -> [K * Fh * Fw]
  // [C] -> [C]
  SmallVector<ReassociationIndices> reassociation = {{0}, {1, 2, 3}, {4}};

  return tensor::CollapseShapeOp::create(rewriter, loc, newType, info.filter,
                                         reassociation);
}

/// Create a batched GEMM operation as a linalg.generic.
/// im2col: [G, M, K] where G = groups, M = N * Ho * Wo, K = K * Fh * Fw
/// filter: [G, K, N] where G = groups, K = K * Fh * Fw, N = C
/// output: [G, M, N] where G = groups, M = N * Ho * Wo, N = C
static Value createGemmOp(PatternRewriter &rewriter, Location loc,
                          Value im2col, Value filter, Type elementType) {
  MLIRContext *ctx = rewriter.getContext();

  auto im2colType = cast<RankedTensorType>(im2col.getType());
  auto filterType = cast<RankedTensorType>(filter.getType());

  int64_t G = im2colType.getShape()[0]; // Groups
  int64_t M = im2colType.getShape()[1]; // N * Ho * Wo
  int64_t K = im2colType.getShape()[2]; // K * Fh * Fw
  int64_t N = filterType.getShape()[2]; // C

  LLVM_DEBUG({
    llvm::dbgs() << "  Creating batched GEMM: [" << G << ", " << M << ", " << K
                 << "] x [" << G << ", " << K << ", " << N << "] = [" << G
                 << ", " << M << ", " << N << "]\n";
  });

  // Create output tensor filled with zeros [G, M, N]
  SmallVector<int64_t> outputShape = {G, M, N};
  auto outputType = RankedTensorType::get(outputShape, elementType);
  Value outputTensor =
      tensor::EmptyOp::create(rewriter, loc, outputShape, elementType);

  // Fill with zero
  Value zero = arith::ConstantOp::create(
      rewriter, loc, elementType, rewriter.getZeroAttr(elementType));
  Value filledOutput =
      linalg::FillOp::create(rewriter, loc, zero, outputTensor).getResult(0);

  // Create batched GEMM as linalg.generic
  // C[g, m, n] = sum_k A[g, m, k] * B[g, k, n]
  // Iteration space: (g, m, n, k)
  AffineExpr gExpr, m, n, k;
  bindDims(ctx, gExpr, m, n, k);

  AffineMap aMap = AffineMap::get(4, 0, {gExpr, m, k}, ctx); // A[g, m, k]
  AffineMap bMap = AffineMap::get(4, 0, {gExpr, k, n}, ctx); // B[g, k, n]
  AffineMap cMap = AffineMap::get(4, 0, {gExpr, m, n}, ctx); // C[g, m, n]

  SmallVector<AffineMap> indexingMaps = {aMap, bMap, cMap};
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel,  // g (batch/group)
      utils::IteratorType::parallel,  // m
      utils::IteratorType::parallel,  // n
      utils::IteratorType::reduction  // k
  };

  auto gemmOp = linalg::GenericOp::create(
      rewriter, loc, outputType,
      /*inputs=*/ValueRange{im2col, filter},
      /*outputs=*/ValueRange{filledOutput}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value a = args[0];
        Value b = args[1];
        Value c = args[2];
        Value mul = arith::MulFOp::create(nestedBuilder, nestedLoc, a, b);
        Value add = arith::AddFOp::create(nestedBuilder, nestedLoc, c, mul);
        linalg::YieldOp::create(nestedBuilder, nestedLoc, add);
      });

  return gemmOp.getResult(0);
}

/// Reshape the GEMM result from [G, M, N] back to [N, Ho, Wo, G, C]
/// where G = groups, M = N * Ho * Wo (batch * spatial), N = C (channels)
static Value reshapeGemmResult(PatternRewriter &rewriter, Location loc,
                               Value gemmResult, const ConvBwdDataInfo &info) {
  // GEMM result: [G, N * Ho * Wo, C] -> [N, Ho, Wo, G, C]
  // Step 1: Expand [G, M, C] to [G, N, Ho, Wo, C]
  SmallVector<int64_t> expandedShape = {info.groupSize, info.batchSize,
                                        info.outHeight, info.outWidth,
                                        info.outChannels};
  auto expandedType = RankedTensorType::get(expandedShape, info.elementType);

  // Reassociation: [[0], [1, 2, 3], [4]]
  // [G] -> [G]
  // [M] -> [N, Ho, Wo]
  // [C] -> [C]
  SmallVector<ReassociationIndices> expandReassoc = {{0}, {1, 2, 3}, {4}};

  Value expanded = tensor::ExpandShapeOp::create(rewriter, loc, expandedType,
                                                 gemmResult, expandReassoc);

  // Step 2: Transpose from [G, N, Ho, Wo, C] to [N, Ho, Wo, G, C]
  // This requires a linalg.generic to permute dimensions
  SmallVector<int64_t> finalShape = {info.batchSize, info.outHeight,
                                     info.outWidth, info.groupSize,
                                     info.outChannels};
  auto finalType = RankedTensorType::get(finalShape, info.elementType);

  Value finalTensor =
      tensor::EmptyOp::create(rewriter, loc, finalShape, info.elementType);

  MLIRContext *ctx = rewriter.getContext();

  // Iteration space: (n, ho, wo, g, c)
  // Input: [G, N, Ho, Wo, C] -> access as (g, n, ho, wo, c)
  // Output: [N, Ho, Wo, G, C] -> access as (n, ho, wo, g, c) (identity)
  AffineExpr n, ho, wo, g, c;
  bindDims(ctx, n, ho, wo, g, c);

  // Input map: (n, ho, wo, g, c) -> (g, n, ho, wo, c)
  AffineMap inputMap = AffineMap::get(5, 0, {g, n, ho, wo, c}, ctx);
  // Output map: identity (n, ho, wo, g, c)
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(5, ctx);

  SmallVector<utils::IteratorType> iteratorTypes(5,
                                                  utils::IteratorType::parallel);
  SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};

  auto transposeOp = linalg::GenericOp::create(
      rewriter, loc, finalType,
      /*inputs=*/ValueRange{expanded},
      /*outputs=*/ValueRange{finalTensor}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
      });

  LLVM_DEBUG({
    llvm::dbgs() << "  Reshape GEMM result: [" << info.groupSize << ", "
                 << info.batchSize * info.outHeight * info.outWidth << ", "
                 << info.outChannels << "] -> [" << info.batchSize << ", "
                 << info.outHeight << ", " << info.outWidth << ", "
                 << info.groupSize << ", " << info.outChannels << "]\n";
  });

  return transposeOp.getResult(0);
}

/// Pattern to match and convert backward data convolution linalg.generic
/// to im2col + matmul.
struct ConvBwdDataToGemmPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Checking linalg.generic for conv pattern: "
                            << op.getLoc() << "\n");

    ConvBwdDataInfo info;
    if (!isBackwardDataConvolution(op, info)) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Converting backward data convolution to GEMM\n");

    Location loc = op.getLoc();

    // Step 1: Create im2col tensor
    LLVM_DEBUG(llvm::dbgs() << "  Step 1: Creating im2col...\n");
    Value im2col = createIm2ColOp(rewriter, loc, info);

    // Step 2: Reshape filter for GEMM
    LLVM_DEBUG(llvm::dbgs() << "  Step 2: Reshaping filter...\n");
    Value reshapedFilter = reshapeFilterForGemm(rewriter, loc, info);

    // Step 3: Create GEMM operation
    LLVM_DEBUG(llvm::dbgs() << "  Step 3: Creating GEMM...\n");
    Value gemmResult =
        createGemmOp(rewriter, loc, im2col, reshapedFilter, info.elementType);

    // Step 4: Reshape result back to output shape
    LLVM_DEBUG(llvm::dbgs() << "  Step 4: Reshaping result...\n");
    Value result = reshapeGemmResult(rewriter, loc, gemmResult, info);

    // Replace the original convolution with the new result
    rewriter.replaceOp(op, result);

    LLVM_DEBUG(llvm::dbgs() << "  Successfully converted conv to GEMM!\n");

    return success();
  }
};

struct CpuConvToGemmPass
    : public cpu::impl::CpuConvToGemmPassBase<CpuConvToGemmPass> {
  using CpuConvToGemmPassBase::CpuConvToGemmPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvBwdDataToGemmPattern>(ctx);

    // Apply patterns greedily
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace
