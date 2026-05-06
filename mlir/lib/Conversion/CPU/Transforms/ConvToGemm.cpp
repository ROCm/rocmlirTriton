//===- ConvToGemm.cpp - Rewrite linalg conv generics to a fused form ------===//
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
// This pass rewrites convolution-shaped `linalg.generic` ops in CPU verifier
// functions into a single canonical "fused conv" `linalg.generic` whose body
// is a mul+add contraction and whose iteration space directly indexes the
// padded input via affine sums (no materialised im2col tensor). The fused op
// keeps the original 8-D iteration space (5 parallel + 3 reduction) and is
// equivalent to the previous im2col + batched-GEMM form for both forward and
// backward-data convolutions. Stride and dilation must be 1.
//
// Compared to the historical im2col + GEMM lowering (which constructed an
// explicit `tensor<G x N x Ho x Wo x K x Fh x Fw>` intermediate and then a
// 3-D batched matmul), this form does not materialise the im2col tensor at
// all -- the gather lives inside the contraction's input affine map. That
// avoids the multi-hundred-GB intermediate that the old form produced for
// ResNet50-scale shapes.
//
// The transformation tags the rewritten op with `rock.cpu_fused_conv` so
// downstream survival/diagnostic checks can identify it, and so the matcher
// here doesn't try to rewrite it a second time.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/CPU/Passes.h"

#include "Schedules/ScheduleUtils.h"

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

/// Return `true` when `op`'s nearest enclosing `func::FuncOp` carries the
/// `rock.cpu_verifier` attribute. This pass is only meant to rewrite convs
/// that live in CPU verifier functions emitted by rocmlir-gen; gating the
/// rewrite pattern on this predicate keeps any convolution-shaped
/// `linalg.generic` in the host module (or in user code that happens to
/// flow through this pipeline) untouched, even though the greedy driver
/// is invoked at module scope.
static bool isInsideCpuVerifierFunc(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  return func && func->hasAttr("rock.cpu_verifier");
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

/// Build the fused 8-D conv-as-matmul `linalg.generic`. The iteration space
/// is `(n, g, c, ho, wo, kc, fh, fw)` (5 parallel + 3 reduction); each
/// operand's indexing map is in the user's original tensor layout, with the
/// two spatial axes of the input expressed as `(ho + fh)` and `(wo + fw)`.
/// The body is the standard mul+add accumulation, and the output is freshly
/// zero-filled so the contraction starts from a clean accumulator.
///
/// The result is type-equivalent to the original conv `linalg.generic`: same
/// iter types, same per-operand affine maps, same body shape -- but emitted
/// from this pass under our control so downstream schedules can reliably
/// match it without depending on the upstream conv generator.
static Value createFusedConvOp(PatternRewriter &rewriter, Location loc,
                               const ConvInfo &info) {
  MLIRContext *ctx = rewriter.getContext();

  // Iteration space (matcher-canonical order): n, g, c, ho, wo, kc, fh, fw.
  // The position of each iter dim within the iter-space (`*Dim` fields on
  // `info`) is whatever the user's original op chose; we deliberately fix a
  // fresh canonical order here so the schedule on the next pass can match
  // by iter-type signature alone without caring about the input op's quirks.
  AffineExpr n, g, c, ho, wo, kc, fh, fw;
  bindDims(ctx, n, g, c, ho, wo, kc, fh, fw);

  auto inputType = cast<RankedTensorType>(info.input.getType());
  auto filterType = cast<RankedTensorType>(info.filter.getType());
  auto outputType = cast<RankedTensorType>(info.output.getType());

  // Input map: walk the input's tensor dims in order and pick the right
  // iter-space expr for each. Spatial axes use the (par + red) sum.
  SmallVector<AffineExpr> inputExprs(inputType.getRank());
  inputExprs[info.inputDimPos[info.batchDim]] = n;
  inputExprs[info.inputDimPos[info.groupDim]] = g;
  inputExprs[info.inputDimPos[info.inChannelDim]] = kc;
  inputExprs[info.inputDimPos[info.outHeightDim]] = ho + fh;
  inputExprs[info.inputDimPos[info.outWidthDim]] = wo + fw;

  // Filter map: walk the filter's tensor dims in order.
  SmallVector<AffineExpr> filterExprs(filterType.getRank());
  filterExprs[info.filterDimPos[info.groupDim]] = g;
  filterExprs[info.filterDimPos[info.inChannelDim]] = kc;
  filterExprs[info.filterDimPos[info.filterHeightDim]] = fh;
  filterExprs[info.filterDimPos[info.filterWidthDim]] = fw;
  filterExprs[info.filterDimPos[info.outChannelDim]] = c;

  // Output map: walk the output's tensor dims in order.
  SmallVector<AffineExpr> outputExprs(outputType.getRank());
  outputExprs[info.outputDimPos[info.batchDim]] = n;
  outputExprs[info.outputDimPos[info.groupDim]] = g;
  outputExprs[info.outputDimPos[info.outChannelDim]] = c;
  outputExprs[info.outputDimPos[info.outHeightDim]] = ho;
  outputExprs[info.outputDimPos[info.outWidthDim]] = wo;

  AffineMap inputMap = AffineMap::get(8, 0, inputExprs, ctx);
  AffineMap filterMap = AffineMap::get(8, 0, filterExprs, ctx);
  AffineMap outputMap = AffineMap::get(8, 0, outputExprs, ctx);

  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel,  // n
      utils::IteratorType::parallel,  // g
      utils::IteratorType::parallel,  // c
      utils::IteratorType::parallel,  // ho
      utils::IteratorType::parallel,  // wo
      utils::IteratorType::reduction, // kc
      utils::IteratorType::reduction, // fh
      utils::IteratorType::reduction, // fw
  };

  // Fresh, zero-filled init in the user's output layout. We never reuse the
  // original op's init operand: callers may pass an undef `tensor.empty`,
  // and even in the bwd-data case the matcher accepts ops whose body simply
  // overwrites the init -- so the only safe accumulator start is zero.
  Value emptyOut = tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                           info.elementType);
  Value zero = arith::ConstantOp::create(
      rewriter, loc, info.elementType, rewriter.getZeroAttr(info.elementType));
  Value initOut =
      linalg::FillOp::create(rewriter, loc, zero, emptyOut).getResult(0);

  auto fused = linalg::GenericOp::create(
      rewriter, loc, outputType,
      /*inputs=*/ValueRange{info.input, info.filter},
      /*outputs=*/ValueRange{initOut},
      SmallVector<AffineMap>{inputMap, filterMap, outputMap}, iteratorTypes,
      [&](OpBuilder &nb, Location nl, ValueRange args) {
        Value mul = arith::MulFOp::create(nb, nl, args[0], args[1]);
        Value add = arith::AddFOp::create(nb, nl, args[2], mul);
        linalg::YieldOp::create(nb, nl, add);
      });
  fused->setAttr(kFusedConvAttrName, UnitAttr::get(ctx));

  return fused.getResult(0);
}

/// Pattern to match a 2D convolution-shaped `linalg.generic` (forward or
/// backward-data) and rewrite it to a single fused `linalg.generic` whose
/// indexing maps inline the im2col gather. See
/// `matchConvolutionLikeGeneric` / `ConvInfo` for the shared matcher-role
/// terminology that covers both directions, and `createFusedConvOp` for
/// the emitted form. Non-unit stride / dilation are rejected up-front by
/// the matcher so they fall through cleanly rather than crashing the
/// linalg verifier.
struct ConvToGemmPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Skip ops we have already rewritten -- otherwise the greedy driver
    // would re-match the fused result (it has the same iter-type signature
    // as the input) and loop forever.
    if (op->hasAttr(kFusedConvAttrName))
      return failure();

    // Only rewrite convolutions inside CPU verifier funcs. The greedy
    // driver runs at module scope, so without this gate any conv-shaped
    // `linalg.generic` in the host module would also be rewritten.
    if (!isInsideCpuVerifierFunc(op))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Checking linalg.generic for conv pattern: "
                            << op.getLoc() << "\n");

    ConvInfo info;
    if (failed(matchConvolutionLikeGeneric(op, info)))
      return failure();

    LLVM_DEBUG({
      bool bwd = hasBwdDataFilterRotation(info.filter);
      llvm::dbgs() << "Rewriting "
                   << (bwd ? "backward data" : "forward")
                   << " convolution to fused linalg.generic\n";
    });

    Value result = createFusedConvOp(rewriter, op.getLoc(), info);
    rewriter.replaceOp(op, result);
    LLVM_DEBUG(llvm::dbgs() << "  Successfully rewrote conv to fused form\n");
    return success();
  }
};

/// Heuristic: a `linalg.generic` is considered convolution-like (i.e. a
/// candidate for this pass) when it lives inside a `rock.cpu_verifier`
/// function, has a mul+add contraction body, and an iteration space larger
/// than the canonical batched matmul (4 dims: `[g, m, n, k]`). A plain
/// matmul / batched matmul is intentionally excluded so we do not
/// mistakenly flag it as "unconverted convolution". Already-rewritten ops
/// (carrying `rock.cpu_fused_conv`) are excluded too: they also have a
/// mul+add body and >4 iter dims, but they are the *output* of this pass,
/// not unconverted candidates.
static bool isConvolutionLikeGeneric(linalg::GenericOp op) {
  if (!isInsideCpuVerifierFunc(op))
    return false;
  if (op->hasAttr(kFusedConvAttrName))
    return false;
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
