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
// linalg.generic operations representing matrix multiplications (im2col +
// GEMM). It handles both forward and backward-data convolutions; the lowering
// is identical for both directions because the im2col + matmul formulation is
// symmetric (only which tensor plays the role of "input" / "filter" / "output"
// changes). It also handles any operand layout permutation -- the matcher
// recovers the dim roles from the affine maps. Stride and dilation must be 1.
//
// The transformation follows the im2col approach:
// 1. Identify linalg.generic with convolution semantics
// 2. Create an im2col tensor that rearranges the input
// 3. Replace the convolution with a batched matmul
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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

/// Describes the structure of a 2D convolution-shaped `linalg.generic` op
/// after we have parsed its iterator types and indexing maps. The lowering
/// applies equally to forward and backward-data convolutions, so the field
/// names below intentionally describe the matcher's *canonical* role of each
/// dim (i.e. its position in the iteration space and which operands it
/// appears in), not the user-facing direction. The mapping to the usual
/// conv terminology is:
///
///   matcher field     | forward conv     | backward-data conv
///   ------------------|------------------|----------------------------
///   batchDim       N  | batch            | batch
///   groupDim       G  | group            | group
///   outChannelDim  C  | output channel K | "output" channel  (= fwd C)
///   inChannelDim   K  | input channel  C | "input"  channel  (= fwd K)
///   outHeight/Wo  Ho/Wo | output spatial | output spatial    (= fwd in spat)
///   filterH/W    Fh/Fw | filter spatial | filter spatial
///
/// In other words, "outChannelDim" is the parallel channel that appears in
/// the output and the filter (and is *absent* from the input), and
/// "inChannelDim" is the reduction channel that appears as a pure dim in
/// both the input and the filter -- which side of the conv that is depends
/// on direction, but the lowering is identical either way (im2col gathers
/// along Fh/Fw/inChannel, and the GEMM contracts along K = inChannel*Fh*Fw).
///
/// The fields in this struct are layout-agnostic: each `*Dim` is a position
/// in the iteration space, and each `*OperandPos` array records where that
/// iteration-space dim shows up among the operand's tensor dimensions.
struct ConvInfo {
  // Iteration-space positions (5 parallel + 3 reduction = 8 dims total).
  unsigned batchDim;        // N: parallel, in output and input.
  unsigned groupDim;        // G: parallel, in all three operands.
  unsigned outChannelDim;   // C: parallel, in output and filter, NOT in input.
  unsigned outHeightDim;    // Ho: parallel, in output; appears in input only
                            //     as part of a (Ho + Fh) sum.
  unsigned outWidthDim;     // Wo: parallel, in output; (Wo + Fw) sum in input.
  unsigned inChannelDim;    // K: reduction, pure dim in input and filter.
  unsigned filterHeightDim; // Fh: reduction, in filter; (Ho + Fh) sum in input.
  unsigned filterWidthDim;  // Fw: reduction, in filter; (Wo + Fw) sum in input.

  // For each operand, the tensor-dim position of each logical conv dim.
  // E.g. outputDimPos[batchDim] tells us where N lives in the output's
  // shape. Entries for dims that don't appear in that operand are ~0u.
  // Indexed by iteration-space position (0..7).
  SmallVector<unsigned, 8> outputDimPos;
  SmallVector<unsigned, 8> inputDimPos;
  SmallVector<unsigned, 8> filterDimPos;

  // Shapes (read from the operand tensors via the *DimPos lookups).
  int64_t batchSize;     // N
  int64_t groupSize;     // G
  int64_t outChannels;   // C (matcher's parallel-channel-in-output-and-filter)
  int64_t outHeight;     // Ho
  int64_t outWidth;      // Wo
  int64_t inChannels;    // K (matcher's reduction-channel-in-input-and-filter)
  int64_t filterHeight;  // Fh
  int64_t filterWidth;   // Fw

  // Operands (in the user's original layout).
  Value input;
  Value filter;
  Value output;

  Type elementType;
};

constexpr unsigned kInvalidDim = ~0u;

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

/// Try to interpret a `linalg.generic` as a 2D convolution contraction (any
/// layout, stride 1, dilation 1). Both forward and backward-data convs match
/// this pattern -- they are mirror images that produce the same im2col +
/// batched-GEMM lowering, only with different operands playing the
/// "input"/"filter" roles. On success, populate `info` with the
/// iteration-space dim positions and per-operand dim positions, and return
/// success(); otherwise return failure() and leave `info` untouched.
///
/// Most of the heavy lifting -- iterator-type counts, projected-permutation
/// checks on output/filter, the unconvolved-vs-convolved input-dim walk
/// that distinguishes batch/depth/output-channel/input-channel/filter-loop --
/// is delegated to `linalg::inferConvolutionDims`. We then map the upstream
/// classification to our role names:
///
///   inferConvolutionDims | matcher field    | conv role
///   ---------------------|------------------|------------------
///   batch[0]             | batchDim         | N
///   depth[0]             | groupDim         | G
///   outputChannel[0]     | outChannelDim    | C
///   inputChannel[0]      | inChannelDim     | K
///   outputImage          | (Ho, Wo) iter pos | output spatial
///   filterLoop           | (Fh, Fw) iter pos | filter spatial
///
/// The (Ho, Wo) order we want is *output-tensor* order, not iter-space order,
/// so we recover it by walking the output map's results. Pairing each output
/// spatial dim with its filter spatial dim requires one short walk over the
/// input map's `(parDim + redDim)` sums.
///
/// Stride-1 / dilation-1 is enforced via `dims->strides` / `dims->dilations`
/// (anything else, e.g. fwd conv with stride > 1, falls through cleanly with
/// a "NOT CONVERTED" diagnostic instead of crashing the linalg verifier).
/// The `mul + add` body shape is verified by `hasContractionBody` (the
/// upstream conv interface checks indexing maps but not the body).
static LogicalResult matchConvolutionLikeGeneric(linalg::GenericOp op,
                                                 ConvInfo &info) {
  FailureOr<linalg::ConvolutionDimensions> dims =
      linalg::inferConvolutionDims(cast<linalg::LinalgOp>(op.getOperation()));
  if (failed(dims)) {
    LLVM_DEBUG(llvm::dbgs() << "  Not a conv: inferConvolutionDims failed\n");
    return failure();
  }

  // We only handle the canonical 2D, single-group, single-channel-pair case.
  if (dims->batch.size() != 1 || dims->depth.size() != 1 ||
      dims->outputChannel.size() != 1 || dims->inputChannel.size() != 1 ||
      dims->outputImage.size() != 2 || dims->filterLoop.size() != 2) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Not a conv: unsupported dim layout (got "
               << dims->batch.size() << "B/" << dims->depth.size() << "D/"
               << dims->outputChannel.size() << "OC/"
               << dims->inputChannel.size() << "IC/"
               << dims->outputImage.size() << "OI/" << dims->filterLoop.size()
               << "FL)\n");
    return failure();
  }

  if (!llvm::all_of(dims->strides, [](int64_t s) { return s == 1; }) ||
      !llvm::all_of(dims->dilations, [](int64_t d) { return d == 1; })) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Not a conv: only stride-1 / dilation-1 supported\n");
    return failure();
  }

  if (!hasContractionBody(op)) {
    LLVM_DEBUG(llvm::dbgs() << "  Not a conv: body is not mul+add\n");
    return failure();
  }

  info.batchDim = dims->batch[0];
  info.groupDim = dims->depth[0];
  info.outChannelDim = dims->outputChannel[0];
  info.inChannelDim = dims->inputChannel[0];

  // Operands.
  info.input = op.getDpsInputOperand(0)->get();
  info.filter = op.getDpsInputOperand(1)->get();
  info.output = op.getDpsInitOperand(0)->get();
  auto outType = cast<RankedTensorType>(info.output.getType());
  auto filType = cast<RankedTensorType>(info.filter.getType());
  info.elementType = outType.getElementType();

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  AffineMap inputMap = maps[0];
  AffineMap filterMap = maps[1];
  AffineMap outputMap = maps[2];

  LLVM_DEBUG({
    llvm::dbgs() << "  Analyzing indexing maps:\n";
    llvm::dbgs() << "    Input:  " << inputMap << "\n";
    llvm::dbgs() << "    Filter: " << filterMap << "\n";
    llvm::dbgs() << "    Output: " << outputMap << "\n";
  });

  // Recover (Ho, Wo) in *output-tensor* order by walking outputMap's results.
  // Anything that isn't N/G/C is a spatial dim.
  SmallVector<unsigned, 2> outputSpatial;
  for (AffineExpr expr : outputMap.getResults()) {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    if (pos != info.batchDim && pos != info.groupDim &&
        pos != info.outChannelDim)
      outputSpatial.push_back(pos);
  }
  if (outputSpatial.size() != 2)
    return failure();
  info.outHeightDim = outputSpatial[0];
  info.outWidthDim = outputSpatial[1];

  // Pair each output spatial dim with its filter spatial dim by walking the
  // (parDim + redDim) sums in the input map.
  auto isDim = [](AffineExpr e, unsigned &pos) {
    if (auto d = dyn_cast<AffineDimExpr>(e)) {
      pos = d.getPosition();
      return true;
    }
    return false;
  };
  auto isPar = [&](unsigned d) {
    return op.getIteratorTypesArray()[d] == utils::IteratorType::parallel;
  };
  auto findRedFor = [&](unsigned parDim) -> std::optional<unsigned> {
    for (AffineExpr expr : inputMap.getResults()) {
      auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
      if (!bin || bin.getKind() != AffineExprKind::Add)
        continue;
      unsigned a, b;
      if (!isDim(bin.getLHS(), a) || !isDim(bin.getRHS(), b))
        continue;
      unsigned p = isPar(a) ? a : b;
      unsigned r = isPar(a) ? b : a;
      if (p == parDim)
        return r;
    }
    return std::nullopt;
  };
  std::optional<unsigned> fh = findRedFor(info.outHeightDim);
  std::optional<unsigned> fw = findRedFor(info.outWidthDim);
  if (!fh || !fw)
    return failure();
  info.filterHeightDim = *fh;
  info.filterWidthDim = *fw;

  // Build per-operand dim-position tables (iter-space pos -> tensor-dim pos).
  unsigned numIters = op.getNumLoops();
  info.outputDimPos.assign(numIters, kInvalidDim);
  info.inputDimPos.assign(numIters, kInvalidDim);
  info.filterDimPos.assign(numIters, kInvalidDim);
  for (auto [i, expr] : llvm::enumerate(outputMap.getResults()))
    info.outputDimPos[cast<AffineDimExpr>(expr).getPosition()] = i;
  for (auto [i, expr] : llvm::enumerate(filterMap.getResults()))
    info.filterDimPos[cast<AffineDimExpr>(expr).getPosition()] = i;
  // For the input map, pure dims map directly; sums map both their parallel
  // and their reduction operand to the same tensor position (we only ever
  // look up the parallel side for output spatial dims).
  for (auto [i, expr] : llvm::enumerate(inputMap.getResults())) {
    unsigned pos;
    if (isDim(expr, pos)) {
      info.inputDimPos[pos] = i;
    } else {
      auto bin = cast<AffineBinaryOpExpr>(expr);
      info.inputDimPos[cast<AffineDimExpr>(bin.getLHS()).getPosition()] = i;
      info.inputDimPos[cast<AffineDimExpr>(bin.getRHS()).getPosition()] = i;
    }
  }

  ArrayRef<int64_t> oShape = outType.getShape();
  ArrayRef<int64_t> fShape = filType.getShape();
  info.batchSize = oShape[info.outputDimPos[info.batchDim]];
  info.groupSize = oShape[info.outputDimPos[info.groupDim]];
  info.outChannels = oShape[info.outputDimPos[info.outChannelDim]];
  info.outHeight = oShape[info.outputDimPos[info.outHeightDim]];
  info.outWidth = oShape[info.outputDimPos[info.outWidthDim]];
  info.inChannels = fShape[info.filterDimPos[info.inChannelDim]];
  info.filterHeight = fShape[info.filterDimPos[info.filterHeightDim]];
  info.filterWidth = fShape[info.filterDimPos[info.filterWidthDim]];

  LLVM_DEBUG({
    llvm::dbgs() << "  Identified convolution:\n";
    llvm::dbgs() << "    N=" << info.batchSize << " G=" << info.groupSize
                 << " C=" << info.outChannels << " K=" << info.inChannels
                 << "\n";
    llvm::dbgs() << "    Out spatial: " << info.outHeight << "x"
                 << info.outWidth << "  Filter spatial: " << info.filterHeight
                 << "x" << info.filterWidth << "\n";
  });
  return success();
}

/// Walk back through reshape-like ops to a `tensor.generate` whose body
/// contains a `subi(constant, blockArg)` -- that's the bwd-data filter
/// rotation `f'[fh, fw] = f[Fh-1-fh, Fw-1-fw]` materialised by rocmlir-gen
/// for backward-data convolutions. Forward conv filters do not have this
/// producer chain.
///
/// This is *purely* a debug-output helper: the matcher and the lowering are
/// direction-agnostic and behave identically whether or not the filter has
/// this rotation. Used only to print "forward" vs "backward data" in the
/// debug log so it's easier to correlate with rocmlir-gen invocations.
static bool hasBwdDataFilterRotation(Value filter) {
  Value v = filter;
  while (v) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    if (auto gen = dyn_cast<tensor::GenerateOp>(def)) {
      bool found = false;
      gen.getBody().walk([&](arith::SubIOp sub) {
        if (mlir::isa<BlockArgument>(sub.getRhs()))
          found = true;
      });
      return found;
    }
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp,
            tensor::ReshapeOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    return false;
  }
  return false;
}

/// Build the im2col gather as a single 7D `linalg.generic`, then collapse to
/// the [G, N*Ho*Wo, K*Fh*Fw] shape consumed by the batched GEMM.
///
/// We always materialise the im2col tensor in the canonical
///   [G, N, Ho, Wo, K, Fh, Fw]
/// layout (g first so the GEMM's batch dim is g). The trick is the input
/// indexing map: it must address the user's `info.input` in *its* layout,
/// not the canonical one. We build it by walking the input's tensor dims
/// in order and emitting the right iter-space expression for each.
static Value createIm2ColOp(PatternRewriter &rewriter, Location loc,
                            const ConvInfo &info) {
  MLIRContext *ctx = rewriter.getContext();

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

  // Iteration space (7D, all parallel): (g, n, ho, wo, k, fh, fw).
  AffineExpr g, n, ho, wo, k, fh, fw;
  bindDims(ctx, g, n, ho, wo, k, fh, fw);

  // Input access in *user* layout: walk the input's tensor dims and pick the
  // right iter-space expression for each. The user's `info.inputDimPos[d]`
  // tells us which tensor dim of `info.input` corresponds to logical conv
  // dim `d`. For the two spatial axes, both the parallel and reduction iter
  // dims map to the same tensor dim (the sum), so we read either off the
  // table and then construct `parIter + redIter`.
  auto inputType = cast<RankedTensorType>(info.input.getType());
  SmallVector<AffineExpr> inputExprs(inputType.getRank());
  inputExprs[info.inputDimPos[info.batchDim]] = n;
  inputExprs[info.inputDimPos[info.groupDim]] = g;
  inputExprs[info.inputDimPos[info.inChannelDim]] = k;
  inputExprs[info.inputDimPos[info.outHeightDim]] = ho + fh;
  inputExprs[info.inputDimPos[info.outWidthDim]] = wo + fw;
  AffineMap inputMap = AffineMap::get(7, 0, inputExprs, ctx);
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(7, ctx);

  SmallVector<utils::IteratorType> iteratorTypes(7,
                                                 utils::IteratorType::parallel);
  SmallVector<AffineMap> indexingMaps = {inputMap, outputMap};

  auto im2col7DOp = linalg::GenericOp::create(
      rewriter, loc, im2col7DType,
      /*inputs=*/ValueRange{info.input},
      /*outputs=*/ValueRange{im2col7DTensor}, indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        linalg::YieldOp::create(nestedBuilder, nestedLoc, args[0]);
      });

  Value im2col7DResult = im2col7DOp.getResult(0);

  int64_t mDim = info.batchSize * info.outHeight * info.outWidth;
  int64_t kDim = info.inChannels * info.filterHeight * info.filterWidth;

  SmallVector<int64_t> im2col3DShape = {info.groupSize, mDim, kDim};
  auto im2col3DType = RankedTensorType::get(im2col3DShape, info.elementType);

  // [G, N, Ho, Wo, K, Fh, Fw] -> [G, N*Ho*Wo, K*Fh*Fw]
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

/// Reshape the user's filter (in any 5D permutation of {G, K, Fh, Fw, C})
/// into the canonical batched-GEMM-B layout [G, K*Fh*Fw, C]. We do this with
/// a `linalg.generic` transpose to the canonical [G, K, Fh, Fw, C] shape and
/// then a `tensor.collapse_shape` to fold (K, Fh, Fw) into the GEMM's K dim.
static Value reshapeFilterForGemm(PatternRewriter &rewriter, Location loc,
                                  const ConvInfo &info) {
  MLIRContext *ctx = rewriter.getContext();

  // Step 1: transpose to canonical [G, K, Fh, Fw, C].
  SmallVector<int64_t> canonShape = {info.groupSize, info.inChannels,
                                     info.filterHeight, info.filterWidth,
                                     info.outChannels};
  auto canonType = RankedTensorType::get(canonShape, info.elementType);
  Value canonInit =
      tensor::EmptyOp::create(rewriter, loc, canonShape, info.elementType);

  // Iteration space (5D parallel): (g, k, fh, fw, c).
  AffineExpr g, k, fh, fw, c;
  bindDims(ctx, g, k, fh, fw, c);
  AffineExpr canonExprs[5] = {g, k, fh, fw, c};

  // Build the input map: walk the user's filter dims in order.
  auto filterType = cast<RankedTensorType>(info.filter.getType());
  SmallVector<AffineExpr> filterExprs(filterType.getRank());
  filterExprs[info.filterDimPos[info.groupDim]] = g;
  filterExprs[info.filterDimPos[info.inChannelDim]] = k;
  filterExprs[info.filterDimPos[info.filterHeightDim]] = fh;
  filterExprs[info.filterDimPos[info.filterWidthDim]] = fw;
  filterExprs[info.filterDimPos[info.outChannelDim]] = c;
  AffineMap inMap = AffineMap::get(5, 0, filterExprs, ctx);
  AffineMap outMap = AffineMap::get(
      5, 0, ArrayRef<AffineExpr>(canonExprs, 5), ctx);

  SmallVector<utils::IteratorType> iters(5, utils::IteratorType::parallel);
  auto transpose = linalg::GenericOp::create(
      rewriter, loc, canonType,
      /*inputs=*/ValueRange{info.filter},
      /*outputs=*/ValueRange{canonInit},
      SmallVector<AffineMap>{inMap, outMap}, iters,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        linalg::YieldOp::create(nb, nl, args[0]);
      });

  // Step 2: collapse (K, Fh, Fw) into the GEMM's K dim.
  int64_t kDim = info.inChannels * info.filterHeight * info.filterWidth;
  SmallVector<int64_t> newShape = {info.groupSize, kDim, info.outChannels};
  auto newType = RankedTensorType::get(newShape, info.elementType);
  SmallVector<ReassociationIndices> reassociation = {{0}, {1, 2, 3}, {4}};
  return tensor::CollapseShapeOp::create(rewriter, loc, newType,
                                         transpose.getResult(0), reassociation);
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

/// Take the canonical [G, N*Ho*Wo, C] GEMM result and reshape it back to the
/// user's output layout (any permutation of {N, G, C, Ho, Wo}).
///
/// Step 1 expands the GEMM's M dim into [N, Ho, Wo], giving [G, N, Ho, Wo, C].
/// Step 2 transposes to whatever permutation the original output uses.
static Value reshapeGemmResult(PatternRewriter &rewriter, Location loc,
                               Value gemmResult, const ConvInfo &info) {
  MLIRContext *ctx = rewriter.getContext();

  SmallVector<int64_t> expandedShape = {info.groupSize, info.batchSize,
                                        info.outHeight, info.outWidth,
                                        info.outChannels};
  auto expandedType = RankedTensorType::get(expandedShape, info.elementType);
  SmallVector<ReassociationIndices> expandReassoc = {{0}, {1, 2, 3}, {4}};
  Value expanded = tensor::ExpandShapeOp::create(rewriter, loc, expandedType,
                                                 gemmResult, expandReassoc);

  // Iteration space: same shape as the user's output. We reuse the user's
  // output layout, build an identity output map, and an input map that pulls
  // from the expanded tensor's [G, N, Ho, Wo, C] layout.
  auto userOutType = cast<RankedTensorType>(info.output.getType());
  SmallVector<int64_t> finalShape(userOutType.getShape().begin(),
                                  userOutType.getShape().end());
  Value finalTensor =
      tensor::EmptyOp::create(rewriter, loc, finalShape, info.elementType);

  // Bind iter dims in the user's output order so the output map is identity.
  // Build a small mapping: for each iter-space slot (which corresponds to
  // tensor dim `i` of the output), we know which logical conv dim lives
  // there via `info.outputDimPos`. Invert that to "iter slot -> logical".
  SmallVector<unsigned, 5> iterToLogical(5, kInvalidDim);
  for (unsigned d : {info.batchDim, info.groupDim, info.outChannelDim,
                     info.outHeightDim, info.outWidthDim}) {
    iterToLogical[info.outputDimPos[d]] = d;
  }

  SmallVector<AffineExpr, 5> iterDims(5);
  for (unsigned i = 0; i < 5; ++i)
    iterDims[i] = getAffineDimExpr(i, ctx);

  // Helper: get the iter-space expr that corresponds to logical dim `logical`.
  auto exprForLogical = [&](unsigned logical) -> AffineExpr {
    return iterDims[info.outputDimPos[logical]];
  };

  // Expanded tensor layout is [G, N, Ho, Wo, C] -- read each axis from the
  // iter-space expression for its logical dim.
  SmallVector<AffineExpr, 5> inputExprs = {
      exprForLogical(info.groupDim),
      exprForLogical(info.batchDim),
      exprForLogical(info.outHeightDim),
      exprForLogical(info.outWidthDim),
      exprForLogical(info.outChannelDim),
  };
  AffineMap inputMap = AffineMap::get(5, 0, inputExprs, ctx);
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(5, ctx);

  SmallVector<utils::IteratorType> iters(5, utils::IteratorType::parallel);
  auto transposeOp = linalg::GenericOp::create(
      rewriter, loc, userOutType,
      /*inputs=*/ValueRange{expanded},
      /*outputs=*/ValueRange{finalTensor},
      SmallVector<AffineMap>{inputMap, outputMap}, iters,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        linalg::YieldOp::create(nb, nl, args[0]);
      });

  LLVM_DEBUG({
    llvm::dbgs() << "  Reshape GEMM result: [" << info.groupSize << ", "
                 << info.batchSize * info.outHeight * info.outWidth << ", "
                 << info.outChannels << "] -> user output layout\n";
  });

  return transposeOp.getResult(0);
}

/// Pattern to match a 2D convolution-shaped `linalg.generic` (forward or
/// backward-data; the lowering is the same im2col + batched GEMM either way --
/// see the `matchConvolutionLikeGeneric` and `ConvInfo` comments for the
/// forward/bwd-data <-> matcher-role mapping) and rewrite it to a batched
/// matmul. The matcher handles any layout permutation of the operand tensors;
/// non-unit stride / dilation are rejected by `matchConvolutionLikeGeneric`
/// so they fall through cleanly rather than crashing the linalg verifier.
struct ConvToGemmPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Checking linalg.generic for conv pattern: "
                            << op.getLoc() << "\n");

    ConvInfo info;
    if (failed(matchConvolutionLikeGeneric(op, info)))
      return failure();

    LLVM_DEBUG({
      bool bwd = hasBwdDataFilterRotation(info.filter);
      llvm::dbgs() << "Converting "
                   << (bwd ? "backward data" : "forward")
                   << " convolution to GEMM\n";
    });

    Location loc = op.getLoc();

    LLVM_DEBUG(llvm::dbgs() << "  Step 1: Creating im2col...\n");
    Value im2col = createIm2ColOp(rewriter, loc, info);

    LLVM_DEBUG(llvm::dbgs() << "  Step 2: Reshaping filter...\n");
    Value reshapedFilter = reshapeFilterForGemm(rewriter, loc, info);

    LLVM_DEBUG(llvm::dbgs() << "  Step 3: Creating GEMM...\n");
    Value gemmResult =
        createGemmOp(rewriter, loc, im2col, reshapedFilter, info.elementType);

    LLVM_DEBUG(llvm::dbgs() << "  Step 4: Reshaping result...\n");
    Value result = reshapeGemmResult(rewriter, loc, gemmResult, info);

    rewriter.replaceOp(op, result);
    LLVM_DEBUG(llvm::dbgs() << "  Successfully converted conv to GEMM!\n");
    return success();
  }
};

/// Heuristic: a `linalg.generic` is considered convolution-like (i.e. a
/// candidate for this pass) when it has a mul+add contraction body and an
/// iteration space larger than the canonical batched matmul (4 dims:
/// `[g, m, n, k]`). A plain matmul / batched matmul is intentionally excluded
/// so we do not mistakenly flag it as "unconverted convolution".
static bool isConvolutionLikeGeneric(linalg::GenericOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
    return false;
  if (!hasContractionBody(op))
    return false;
  return op.getNumLoops() > 4;
}

/// Print a one-line tag identifying the `linalg.generic` op for use in
/// diagnostic messages. Includes the source location and the indexing maps,
/// which is what we need to figure out why a conv was not converted.
static void printConvOpTag(llvm::raw_ostream &os, linalg::GenericOp op) {
  os << "    loc: " << op.getLoc() << "\n";
  os << "    iterators: " << op.getNumLoops() << " (";
  llvm::interleaveComma(op.getIteratorTypesArray(), os,
                        [&](utils::IteratorType it) {
                          os << (it == utils::IteratorType::parallel
                                     ? "par"
                                     : "red");
                        });
  os << ")\n";
  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  for (auto [idx, map] : llvm::enumerate(maps)) {
    os << "    map[" << idx << "]: " << map << "\n";
  }
}

struct CpuConvToGemmPass
    : public cpu::impl::CpuConvToGemmPassBase<CpuConvToGemmPass> {
  using CpuConvToGemmPassBase::CpuConvToGemmPassBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Collect convolution-like linalg.generic ops inside cpu_verifier funcs
    // before we run the rewrites, so we can report which ones got converted
    // and which ones did not. We key on the op pointer; that pointer is also
    // valid to compare against the post-pass walk because converted ops are
    // erased (their pointers won't show up again).
    struct ConvCandidate {
      Operation *op;
      StringRef funcName;
      Location loc;
      std::string description;
    };
    SmallVector<ConvCandidate> candidates;

    module.walk([&](func::FuncOp func) {
      if (!func->hasAttr("rock.cpu_verifier"))
        return;
      func.walk([&](linalg::GenericOp generic) {
        if (!isConvolutionLikeGeneric(generic))
          return;
        std::string desc;
        llvm::raw_string_ostream os(desc);
        printConvOpTag(os, generic);
        candidates.push_back(
            {generic.getOperation(), func.getName(), generic.getLoc(),
             std::move(desc)});
      });
    });

    RewritePatternSet patterns(ctx);
    patterns.add<ConvToGemmPattern>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Determine which candidates survived. A surviving candidate is one whose
    // op pointer is still reachable inside its parent cpu_verifier function.
    DenseSet<Operation *> surviving;
    module.walk([&](func::FuncOp func) {
      if (!func->hasAttr("rock.cpu_verifier"))
        return;
      func.walk([&](linalg::GenericOp generic) {
        surviving.insert(generic.getOperation());
      });
    });

    unsigned numConverted = 0, numUnconverted = 0;
    for (const ConvCandidate &c : candidates) {
      bool converted = !surviving.contains(c.op);
      if (converted) {
        ++numConverted;
        llvm::errs() << "[cpu-conv-to-gemm] CONVERTED conv in @" << c.funcName
                     << "\n"
                     << c.description;
      } else {
        ++numUnconverted;
        llvm::errs() << "[cpu-conv-to-gemm] NOT CONVERTED conv in @"
                     << c.funcName << " (no matching pattern)\n"
                     << c.description;
      }
    }

    if (!candidates.empty()) {
      llvm::errs() << "[cpu-conv-to-gemm] summary: " << numConverted
                   << " converted, " << numUnconverted << " not converted (of "
                   << candidates.size() << " convolution-like ops in "
                   << "cpu_verifier funcs)\n";
    }
  }
};

} // end anonymous namespace
