//===- AttnToGridwise.cpp - Rock Attention implementation -------===//
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
// ============================================================
//
// This pass converts rock.attention and rock.gemm_elementwise_gemm into the
// appropriate rock.gridwise_attention adding padding and group dimensions if
// needed.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <sstream>
#include <tuple>
#include <utility>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKATTNTOGRIDWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-attn-to-gridwise"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockAttnToGridwisePass
    : public rock::impl::RockAttnToGridwisePassBase<RockAttnToGridwisePass> {
  void runOnOperation() override;
};

struct GemmElementwiseGemmRewritePattern
    : public OpConversionPattern<GemmElementwiseGemmOp> {
  using OpConversionPattern<GemmElementwiseGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(GemmElementwiseGemmOp op,
                                GemmElementwiseGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override;
};

struct AttentionRewritePattern : public OpConversionPattern<AttentionOp> {
  using OpConversionPattern<AttentionOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(AttentionOp op, AttentionOpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override;
};

// Move num_heads dimension to sequence length dimension. This is useful for the
// decoding phase, when batch=1, seq_len_q = 1 and GQA (example: num_heads_q=64,
// num_heads_kv=8), we can move numRepeat=num_heads_q/num_heads_kv = 8, to the
// seq_len_q dimension and use the tile size better (otherwise seq_len_q=1 and
// it will get padded to 32). This reduces the amount of workgroups by
// numRepeat. However, typically decoding phase will use split_kv anyway to
// increase the number of workgroups.
static Value moveNumHeadsToSeqLenQ(OpBuilder builder, Location loc,
                                   Value inputTensor, int64_t numRepeats) {
  ArrayRef<int64_t> inpShape =
      cast<ShapedType>(inputTensor.getType()).getShape();

  assert(inpShape.size() == 3 && "input must be 3D");
  assert(inpShape[0] % numRepeats == 0 &&
         "gemmG must be divisible by numRepeats");

  int64_t newGemmG = inpShape[0] / numRepeats;
  SmallVector<StringRef> startNames = {"gemmG", "seqLen", "headDim"};

  // (gemmG, seqLen, headDim) -> (gemmG / numRepeats, seqLen, headDim,
  // numRepeats)
  rock::BottomUpTMBuilder unmerge(builder, startNames, inpShape);
  unmerge.unmerge({"gemmG", "numRepeats"}, {0, 2}, "gemmG",
                  {newGemmG, numRepeats});
  unmerge.passThrough({"seqLen", "headDim"}, {1, 3}, {"seqLen", "headDim"});
  auto unmergeAttr = unmerge.get();
  Value matrixUnmerge =
      rock::TransformOp::create(builder, loc, inputTensor, unmergeAttr);

  // (gemmG / numRepeats, seqLen, headDim, numRepeats) -> (gemmG / numRepeats,
  // seqLen * numRepeats, headDim)
  auto merger = rock::BottomUpTMBuilder::above(unmerge, unmergeAttr);
  merger.merge("seqLen", 1, {"seqLen", "numRepeats"});
  merger.passThrough(ArrayRef<uint32_t>{0, 2}, ArrayRef<uint32_t>{0, 3});
  auto mergerAttr = merger.get();
  return rock::TransformOp::create(builder, loc, matrixUnmerge, mergerAttr);
}

// Same as moveNumHeadsToSeqLenQ() but for currSeqLen tensor (KV-Cache)
static Value moveNumHeadsToSeqLenCurrSeqLen(OpBuilder builder, Location loc,
                                            Value inputTensor,
                                            int64_t numRepeats) {
  ArrayRef<int64_t> inpShape =
      cast<ShapedType>(inputTensor.getType()).getShape();

  assert(inpShape.size() == 1 && "input must be 1D");
  assert(inpShape[0] % numRepeats == 0 &&
         "gemmG must be divisible by numRepeats");

  int64_t newGemmG = inpShape[0] / numRepeats;
  SmallVector<StringRef> startNames = {"gemmG"};

  // (gemmG) -> (gemmG / numRepeats, numRepeats)
  rock::BottomUpTMBuilder unmerge(builder, startNames, inpShape);
  unmerge.unmerge({"gemmG", "numRepeats"}, {0, 1}, "gemmG",
                  {newGemmG, numRepeats});
  auto unmergeAttr = unmerge.get();
  Value matrixUnmerge =
      rock::TransformOp::create(builder, loc, inputTensor, unmergeAttr);

  // slice numRepeats to 1
  auto slicer = rock::BottomUpTMBuilder::above(unmerge, unmergeAttr);
  slicer.slice({"numRepeats"}, {"numRepeats"}, {0}, {1});
  slicer.passThrough(ArrayRef<uint32_t>{0}, ArrayRef<uint32_t>{0});
  auto slicerAttr = slicer.get();
  Value matrixSliced =
      rock::TransformOp::create(builder, loc, matrixUnmerge, slicerAttr);

  // (gemmG / numRepeats, headDim, seqLen, numRepeats) -> (gemmG / numRepeats,
  // headDim, seqLen * numRepeats)
  auto merger = rock::BottomUpTMBuilder::above(slicer, slicerAttr);
  merger.merge("seqLen", 0, {"gemmG", "numRepeats"});
  auto mergerAttr = merger.get();
  return rock::TransformOp::create(builder, loc, matrixSliced, mergerAttr);
}

// Same as moveNumHeadsToSeqLenQ() but for the output tensor
static Value moveNumHeadsToSeqLenOut(OpBuilder builder, Location loc,
                                     Value inputTensor, int64_t numRepeats,
                                     int64_t splitKV) {
  if (auto *defOp = inputTensor.getDefiningOp())
    builder.setInsertionPointAfter(defOp);

  ArrayRef<int64_t> inpShape =
      cast<ShapedType>(inputTensor.getType()).getShape();

  assert((inpShape.size() == 2 || inpShape.size() == 3) &&
         "input must be 2D or 3D");
  assert(inpShape[0] % numRepeats == 0 &&
         "gemmG must be divisible by numRepeats");
  assert(inpShape[0] % splitKV == 0 && "gemmG must be divisible by numRepeats");

  int64_t newGemmG = inpShape[0] / (numRepeats * splitKV);
  bool isLSE = inpShape.size() == 2;

  SmallVector<StringRef> startNamesAll = {"gemmG", "seqLen", "headDim"};
  ArrayRef<StringRef> startNames =
      ArrayRef<StringRef>(startNamesAll).take_front(inpShape.size());

  // Note that for LSE, there are only two dimensions (gemmG, seqLen)
  // (gemmG, seqLen, headDim) -> (gemmG / (splitKV*numRepeats), splitKV, seqLen,
  // numRepeats, headDim)
  rock::BottomUpTMBuilder unmerge(builder, startNames, inpShape);
  unmerge.unmerge({"gemmG", "numRepeats", "splitKV"}, {0, 3, 1}, "gemmG",
                  {newGemmG, numRepeats, splitKV});
  if (isLSE)
    unmerge.passThrough({"seqLen"}, {2}, {"seqLen"});
  else
    unmerge.passThrough({"seqLen", "headDim"}, {2, 4}, {"seqLen", "headDim"});
  auto unmergeAttr = unmerge.get();
  Value matrixUnmerge =
      rock::TransformOp::create(builder, loc, inputTensor, unmergeAttr);

  // (gemmG / (splitKV*numRepeats), splitKV, seqLen, numRepeats, headDim) ->
  // (gemmG / numRepeats, seqLen * numRepeats, headDim)
  auto merger = rock::BottomUpTMBuilder::above(unmerge, unmergeAttr);
  merger.merge("seqLen", 1, {"seqLen", "numRepeats"});
  merger.merge("gemmG", 0, {"gemmG", "splitKV"});
  if (!isLSE)
    merger.passThrough({"headDim"}, {2}, {"headDim"});
  auto mergerAttr = merger.get();
  return rock::TransformOp::create(builder, loc, matrixUnmerge, mergerAttr);
}

// Result of GQA (Grouped-Query Attention) processing
struct GQAResult {
  IntegerAttr numRepeats;
  Value queries;
  Value keys;
  Value values;
  Value currentSeqLen;
  Value prefixOffset;
};

template <typename Fn>
static void transformViewsAttn(OpBuilder &rw,
                               MutableArrayRef<Value> outputViews,
                               DenseMap<Value, Value> &fusionInputMap, Fn fn) {
  auto apply = [&](Value &view) {
    OpBuilder::InsertionGuard guard(rw);
    if (Operation *defOp = view.getDefiningOp())
      rw.setInsertionPointAfter(defOp);
    view = fn(view);
  };
  for (auto &outputView : outputViews)
    apply(outputView);
  for (auto &[orig, view] : fusionInputMap)
    apply(view);
}

// This function will implement GQA, moving numRepeat=num_heads_q/num_heads_kv
// to the seq_len_q dimension. See moveNumHeadsToSeqLenQ() comment for more
// details.
static GQAResult processGQA(ConversionPatternRewriter &rw, Location loc,
                            Value queries, Value keys, Value values,
                            SmallVector<Value> &outputViews,
                            DenseMap<Value, Value> &fusionInputMapOut,
                            SmallVector<Value> &lseViews,
                            DenseMap<Value, Value> &fusionInputMapLse,
                            Value currentSeqLen, Value prefixOffset,
                            int64_t numHeadsQ, int64_t numHeadsKV,
                            int64_t splitKV) {

  assert(numHeadsQ % numHeadsKV == 0);
  IntegerAttr numRepeatsAttr = nullptr;

  if (numHeadsQ != numHeadsKV) {
    int64_t numRepeats = numHeadsQ / numHeadsKV;

    numRepeatsAttr = rw.getIndexAttr(numRepeats);
    queries = moveNumHeadsToSeqLenQ(rw, loc, queries, numRepeats);
    if (currentSeqLen)
      currentSeqLen =
          moveNumHeadsToSeqLenCurrSeqLen(rw, loc, currentSeqLen, numRepeats);
    if (prefixOffset)
      prefixOffset =
          moveNumHeadsToSeqLenCurrSeqLen(rw, loc, prefixOffset, numRepeats);

    transformViewsAttn(rw, outputViews, fusionInputMapOut, [&](Value v) {
      return moveNumHeadsToSeqLenOut(rw, loc, v, numRepeats, splitKV);
    });
    if (!lseViews.empty()) {
      transformViewsAttn(rw, lseViews, fusionInputMapLse, [&](Value v) {
        return moveNumHeadsToSeqLenOut(rw, loc, v, numRepeats, splitKV);
      });
    }
  }

  return GQAResult{numRepeatsAttr, queries,       keys,
                   values,         currentSeqLen, prefixOffset};
}

template <typename Op>
static LogicalResult
computeGridSizeAttentionGemmElmtGemm(ConversionPatternRewriter &rw, Op op,
                                     Value a, Value b, Value c,
                                     int64_t splitKV) {
  GemmParamsAttr accelParams0 =
      cast<GemmParamsAttr>(op.getGemm0Params().value());

  SmallVector<int64_t, 3> aShape =
      llvm::to_vector<3>(cast<ShapedType>(a.getType()).getShape());

  SmallVector<int64_t, 3> bShape =
      llvm::to_vector<3>(cast<ShapedType>(b.getType()).getShape());

  SmallVector<int64_t, 3> cShape =
      llvm::to_vector<3>(cast<ShapedType>(c.getType()).getShape());

  GemmSize gemm0Size(/*g=*/aShape[0], /*m=*/aShape[1],
                     /*k=*/aShape[2],
                     /*n=*/bShape[2]);

  int64_t gridSize =
      (gemm0Size.m / accelParams0.getMPerBlock()) * gemm0Size.g * splitKV;

  IntegerAttr gridSizeAttr = rw.getI32IntegerAttr(gridSize);
  func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
  funcOp->setAttr(rock::GridSizeAttr::getMnemonic(), gridSizeAttr);
  return success();
}

static FailureOr<std::tuple<Value, Value, Value>>
arrangeGemmGemmSplitKTransform(OpBuilder &builder,
                               RockGemmGemmWrapperInterface op, Location loc,
                               int64_t splitNFactor, Value a, Value b, Value c,
                               MutableArrayRef<Value> outputViews,
                               DenseMap<Value, Value> &fusionInputMapOut) {
  Value out = op->getResult(0);
  // set the store method and prefill attribute on output store ops
  FailureOr<SetVector<StoreOp>> storeOps = traceRootOutputToStoreOps(out);
  if (failed(storeOps))
    return op->emitError("can't trace gemm output to store ops");
  for (StoreOp storeOp : storeOps.value()) {
    if (failed(
            setStoreMethodAndPrefill(builder, storeOp, StoreMethod::AtomicAdd)))
      return failure();
  }

  const int64_t origN = cast<RankedTensorType>(b.getType()).getShape()[2];
  const int64_t nPad = llvm::alignTo(origN, splitNFactor) - origN;

  b = padMatrix(b, builder, loc, "gemmK", 0, "gemmN", nPad);
  c = padMatrix(c, builder, loc, "gemmK", nPad, "gemmO", 0);

  // perform coordinate transformations
  Value aNew{nullptr}, bNew{nullptr}, cNew{nullptr};
  ArrayRef<int64_t> aShape = cast<RankedTensorType>(a.getType()).getShape();
  ArrayRef<int64_t> bShape = cast<RankedTensorType>(b.getType()).getShape();
  ArrayRef<int64_t> cShape = cast<RankedTensorType>(c.getType()).getShape();
  ArrayRef<int64_t> outShape =
      cast<RankedTensorType>(outputViews[0].getType()).getShape();
  for (auto outputView : outputViews) {
    if (cast<RankedTensorType>(outputView.getType()).getShape() != outShape)
      return op->emitError("all output views must have the same shape");
  }

  const int64_t N = bShape[2];

  struct GemmOperandsData {
    Value &in;
    Value &out;
    SmallVector<StringRef> inputDimNames;
    unsigned presevedDimIdx;
    unsigned splitDimIdx;
    ArrayRef<int64_t> inputShape;
  };

  llvm::SmallVector<GemmOperandsData, 2> gemmOperands{
      {b, bNew, {"gemmG", "gemmK", "gemmN"}, 1, 2, bShape},
      {c, cNew, {"gemmG", "gemmN", "gemmO"}, 2, 1, cShape}};
  for (auto &gemmOperand : gemmOperands) {
    // Prepare matrix B and C - i.e.,
    //    (gemmG, gemmK, gemmN) and (gemmG, gemmN, gemmO), respectively
    // Using bottom-up transformations
    // 1. unmerge (gemmN) -> (gemmNSplit, gemmN*)
    // 2. merge (gemmG, gemmNSplit) -> (gemmG*)

    StringRef preservedDimName =
        gemmOperand.inputDimNames[gemmOperand.presevedDimIdx];
    StringRef splitDimName = gemmOperand.inputDimNames[gemmOperand.splitDimIdx];
    assert(splitDimName == "gemmN");

    BottomUpTMBuilder unmergeTransform(builder, gemmOperand.inputDimNames,
                                       gemmOperand.inputShape, loc);

    unmergeTransform.passThrough({"gemmG", preservedDimName}, {0, 3},
                                 {"gemmG", preservedDimName});
    unmergeTransform.unmerge({"gemmNSplit", "gemmN"}, {1, 2}, "gemmN",
                             {splitNFactor, N / splitNFactor});
    auto unmergeTransformAttr = unmergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(unmergeTransformAttr);

    auto mergeTransform =
        BottomUpTMBuilder::above(unmergeTransform, unmergeTransformAttr);

    mergeTransform.merge("gemmG", 0, {"gemmG", "gemmNSplit"});
    mergeTransform.passThrough(
        {"gemmN", preservedDimName},
        {gemmOperand.splitDimIdx, gemmOperand.presevedDimIdx},
        {"gemmN", preservedDimName});

    auto mergeTransformAttr = mergeTransform.get();
    transformAttrs.push_back(mergeTransformAttr);

    std::reverse(transformAttrs.begin(), transformAttrs.end());
    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    gemmOperand.out =
        mlir::rock::transform(builder, gemmOperand.in, arrayTransformAttrs);
  }

  {
    // Prepare matrix A - i.e., (gemmG, gemmM, gemmK)
    // Using bottom-up transformations
    // 1. addDim (gemmNSplit)
    // 2. merge (gemmG, gemmNSplit) -> (gemmG*)
    BottomUpTMBuilder addDimTransform(builder, {"gemmG", "gemmM", "gemmK"},
                                      aShape, loc);

    addDimTransform.passThrough({"gemmG", "gemmM", "gemmK"});
    addDimTransform.addDim("gemmNSplit", 3, splitNFactor);
    auto addDimTransformAttr = addDimTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(addDimTransformAttr);

    auto mergeTransform =
        BottomUpTMBuilder::above(addDimTransform, addDimTransformAttr);

    mergeTransform.merge("gemmG", 0, {"gemmG", "gemmNSplit"});
    mergeTransform.passThrough({"gemmM", "gemmK"});

    auto mergeTransformAttr = mergeTransform.get();
    transformAttrs.push_back(mergeTransformAttr);

    std::reverse(transformAttrs.begin(), transformAttrs.end());
    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    aNew = mlir::rock::transform(builder, a, arrayTransformAttrs);
  }

  {
    // Prepare matrix out - i.e., (gemmG, gemmM, gemmO)
    // Using top-down transformations
    // 1. merge (gemmG * gemmNSplit, gemmM, gemmO) -> (gemmG, gemmNSplit, gemmM,
    // gemmO)
    // 2. ignore (gemmG, gemmNSplit, gemmM, gemmN) -> (gemmG, gemmM, gemmO)

    const int64_t G = outShape[0];
    const int64_t M = outShape[1];
    const int64_t O = outShape[2];

    TopDownTMBuilder mergeTransform(builder, {"gemmG", "gemmM", "gemmO"},
                                    {G * splitNFactor, M, O});

    mergeTransform.merge({"gemmG", "gemmNSplit"}, {0, 1}, "gemmG",
                         {G, splitNFactor});
    mergeTransform.passThrough({"gemmM", "gemmO"}, {2, 3}, {"gemmM", "gemmO"});
    auto mergeTransformAttr = mergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(mergeTransformAttr);

    TopDownTMBuilder ignoreTransform =
        TopDownTMBuilder::below(mergeTransform, mergeTransformAttr);

    ignoreTransform.ignore("gemmNSplit");
    ignoreTransform.passThrough({"gemmG", "gemmM", "gemmO"}, {0, 1, 2},
                                {"gemmG", "gemmM", "gemmO"});

    TransformMapAttr ignoreTransformAttr = ignoreTransform.get();
    transformAttrs.push_back(ignoreTransformAttr);

    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);

    transformViewsAttn(builder, outputViews, fusionInputMapOut, [&](Value v) {
      return rock::transform(builder, v, arrayTransformAttrs);
    });
  }
  return std::make_tuple(aNew, bNew, cNew);
}

static LogicalResult commonAttentionGemmElmtGemm(
    ConversionPatternRewriter &rw, RockGemmGemmWrapperInterface op, Value a,
    Value b, Value c, Value currentSeqLen, Value prefixOffset, UnitAttr causal,
    IntegerAttr splitKV, ValueRange elementwiseInputs,
    Region &preSecondOpRegion, bool enableSoftmax, TypeAttr softmaxType,
    int64_t numHeadsQ, int64_t numHeadsKV,
    BoolAttr preSoftmaxHasSplitKVTransforms) {
  Location loc = op->getLoc();

  if (!op.getGemm0Params().has_value()) {
    return op.emitError("gemm0 params is missing and it should've been "
                        "assigned by affix-tuning-params");
  }
  GemmParamsAttr params0 = cast<GemmParamsAttr>(op.getGemm0Params().value());
  if (!op.getGemm1Params().has_value()) {
    return op.emitError("gemm1 params is missing and it should've been "
                        "assigned by affix-tuning-params");
  }
  GemmParamsAttr params1 = cast<GemmParamsAttr>(op.getGemm1Params().value());
  bool hasLse = enableSoftmax && op->getNumResults() >= 2;
  Value out = op->getResult(0);
  Value lse = nullptr;
  if (hasLse)
    lse = op->getResult(1);

  auto outViewsResult = rock::traceOutputsAndFusionInputs(out);
  if (failed(outViewsResult))
    return failure();
  auto &[outputStores, outputViews, fusionInputMapOut] = outViewsResult.value();
  SetVector<StoreOp> lseStores;
  SmallVector<Value> lseViews;
  DenseMap<Value, Value> fusionInputMapLse;
  if (hasLse) {
    auto lseViewsResult = rock::traceOutputsAndFusionInputs(lse);
    if (failed(lseViewsResult))
      return failure();
    auto &lseInfo = lseViewsResult.value();
    lseStores = std::move(lseInfo.stores);
    lseViews = std::move(lseInfo.outputViews);
    fusionInputMapLse = std::move(lseInfo.fusionInputMap);
  }

  // Note: the gridwise ops take M x K, K x N and K x N
  a = normalizeMatrix(a, rw, loc, op.getTransposedA(), "gemm0M", "gemm0K");
  b = normalizeMatrix(b, rw, loc, op.getTransposedB(), "gemm0K", "gemm0N");
  c = normalizeMatrix(c, rw, loc, op.getTransposedC(), "gemm1K", "gemm1N");
  transformViewsAttn(rw, outputViews, fusionInputMapOut, [&](Value v) {
    return normalizeMatrix(v, rw, loc, op.getTransposedOut(), "gemm1M",
                           "gemm1N");
  });

  const int64_t splitKFactor = params1.getSplitKFactor();
  if (splitKFactor > 1) {
    if (enableSoftmax)
      return op.emitError("split-k is not supported for attention");

    auto maybeSplitk = arrangeGemmGemmSplitKTransform(
        rw, op, loc, splitKFactor, a, b, c, outputViews, fusionInputMapOut);
    if (failed(maybeSplitk))
      return op.emitError("split-k set up failed");

    std::tie(a, b, c) = maybeSplitk.value();
  }

  int64_t splitKVNum = splitKV.getInt();

  // Grouped-Query Attention (GQA)
  IntegerAttr numRepeatsGQA = nullptr;
  if (enableSoftmax) {
    GQAResult gqa =
        processGQA(rw, op.getLoc(), a, b, c, outputViews, fusionInputMapOut,
                   lseViews, fusionInputMapLse, currentSeqLen, prefixOffset,
                   numHeadsQ, numHeadsKV, splitKVNum);
    numRepeatsGQA = gqa.numRepeats;
    a = gqa.queries;
    b = gqa.keys;
    c = gqa.values;
    currentSeqLen = gqa.currentSeqLen;
    prefixOffset = gqa.prefixOffset;
  }

  // Note, matrix dimension correctness is handled in the verifier
  ArrayRef<int64_t> aShape = cast<ShapedType>(a.getType()).getShape();
  ArrayRef<int64_t> bShape = cast<ShapedType>(b.getType()).getShape();
  ArrayRef<int64_t> cShape = cast<ShapedType>(c.getType()).getShape();
  assert(aShape[0] == bShape[0]);
  assert(cShape[0] == bShape[0]);
  assert(aShape[2] == bShape[1]);
  assert(cShape[1] == bShape[2]);

  GemmSize gemm0Size(/*g=*/aShape[0], /*m=*/aShape[1],
                     /*k=*/aShape[2],
                     /*n=*/bShape[2]);
  GemmSize gemm1Size(/*g=*/aShape[0], /*m=*/aShape[1],
                     /*k=*/cShape[1],
                     /*n=*/cShape[2]);
  GemmSize gemm0ExtraPad = requiredPadding(params0, gemm0Size, 1, 1, splitKVNum)
                               .value_or(GemmSize{0, 0, 0, 0});
  GemmSize gemm1ExtraPad = requiredPadding(params1, gemm1Size, splitKVNum)
                               .value_or(GemmSize{0, 0, 0, 0});

  a = padMatrix(a, rw, loc, "gemm0M", gemm0ExtraPad.m, "gemm0K",
                gemm0ExtraPad.k);
  b = padMatrix(b, rw, loc, "gemm0K", gemm0ExtraPad.k, "gemm0N",
                gemm0ExtraPad.n);
  c = padMatrix(c, rw, loc, "gemm1K", gemm1ExtraPad.k, "gemm1N",
                gemm1ExtraPad.n);
  transformViewsAttn(rw, outputViews, fusionInputMapOut, [&](Value v) {
    return padMatrix(v, rw, loc, "gemm1M", gemm1ExtraPad.m, "gemm1N",
                     gemm1ExtraPad.n);
  });
  if (hasLse) {
    transformViewsAttn(rw, lseViews, fusionInputMapLse, [&](Value v) {
      return padVector(v, rw, loc, "gemm1M", gemm1ExtraPad.m);
    });
  }

  if (failed(
          computeGridSizeAttentionGemmElmtGemm(rw, op, a, b, c, splitKVNum))) {
    return op.emitError("failed to compute the grid size of "
                        "`GemmElementwiseGemmOp`/`AttentionOp`");
  }

  IntegerAttr prePadG0MAttr;
  if (gemm0ExtraPad.m) {
    prePadG0MAttr = rw.getIndexAttr(gemm0Size.m);
  }
  IntegerAttr prePadG0NAttr;
  if (gemm0ExtraPad.n) {
    prePadG0NAttr = rw.getIndexAttr(gemm0Size.n);
  }

  auto newOutputType = RankedTensorType::get(
      cast<ShapedType>(outputViews[0].getType()).getShape(),
      cast<ShapedType>(op.getOutType()).getElementType());
  RankedTensorType newLseType = nullptr;
  if (hasLse)
    newLseType = RankedTensorType::get(
        cast<ShapedType>(lseViews[0].getType()).getShape(),
        cast<ShapedType>(lse.getType()).getElementType());
  auto newOp = GridwiseAttentionOp::create(
      rw, loc, newOutputType, newLseType, a, b, c, elementwiseInputs,
      currentSeqLen, prefixOffset, causal, splitKV,
      /*disableQBypassLDS=*/nullptr, prePadG0MAttr, prePadG0NAttr,
      numRepeatsGQA, softmaxType, params0, params1,
      rw.getBoolAttr(enableSoftmax), preSoftmaxHasSplitKVTransforms);
  bool fusionsFound = rock::gemmGemmHasPreSecondGemmFusion(op);
  if (fusionsFound) {
    rw.inlineRegionBefore(preSecondOpRegion, newOp.getPreSoftmaxBody(),
                          newOp.getPreSoftmaxBody().begin());
  }

  auto postProcessOutput =
      [](ConversionPatternRewriter &rw, Value rootOut, Value newRootOut,
         SmallVector<Value> viewsIn, DenseMap<Value, Value> &fusionInputMap,
         SetVector<StoreOp> &stores, int64_t splitKFactor) {
        rock::propagateOutputType(rootOut, newRootOut);
        rock::replaceFusionExtraInputs(newRootOut, fusionInputMap);
        assert(stores.size() == viewsIn.size() &&
               "stores and viewsIn must have the same size");
        for (size_t i = 0; i < stores.size(); ++i) {
          StoreOp storeOp = stores[i];
          Value view = viewsIn[i];
          // adjust the store method
          StoreMethodAttr storeMethod = storeOp.getStoreMethodAttr();
          if (splitKFactor > 1)
            storeMethod =
                rw.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::AtomicAdd);
          // If the store's source is the gemm result directly (no fusions),
          // use the gridwise result. Otherwise, propagateOutputType has already
          // updated the fusion chain, and storeOp.getSource() has the correct
          // type.
          Value source = storeOp.getSource();
          if (source == rootOut)
            source = newRootOut;
          rw.setInsertionPoint(storeOp);
          auto newStoreOp = rock::StoreOp::create(rw, storeOp.getLoc(),
                                                  storeOp.getResult().getType(),
                                                  source, view, storeMethod);
          rw.replaceOp(storeOp, newStoreOp.getResult());
        }
      };
  postProcessOutput(rw, out, newOp->getResult(0), outputViews,
                    fusionInputMapOut, outputStores, splitKFactor);
  if (hasLse)
    postProcessOutput(rw, lse, newOp->getResult(1), lseViews, fusionInputMapLse,
                      lseStores, splitKFactor);

  rw.replaceOp(op, newOp);
  return success();
}
} // end namespace

LogicalResult
AttentionRewritePattern::matchAndRewrite(AttentionOp op,
                                         AttentionOpAdaptor adaptor,
                                         ConversionPatternRewriter &rw) const {
  return commonAttentionGemmElmtGemm(
      rw, op, adaptor.getQueries(), adaptor.getKeys(), adaptor.getValues(),
      adaptor.getCurrentSeqLen(), adaptor.getPrefixOffset(),
      adaptor.getCausalAttr(), adaptor.getSplitKVAttr(),
      adaptor.getPreSoftmaxElemWiseInputs(), op.getPreSoftmaxBody(),
      /*enableSoftmax=*/true, op.getSoftmaxTypeAttr(), adaptor.getNumHeadsQ(),
      adaptor.getNumHeadsKV(), adaptor.getPreSoftmaxHasSplitKVTransformsAttr());
}

LogicalResult GemmElementwiseGemmRewritePattern::matchAndRewrite(
    GemmElementwiseGemmOp op, GemmElementwiseGemmOpAdaptor adaptor,
    ConversionPatternRewriter &rw) const {

  auto splitKV = rw.getI32IntegerAttr(1);
  return commonAttentionGemmElmtGemm(
      rw, op, adaptor.getA(), adaptor.getB(), adaptor.getC(),
      /*currentSeqLen=*/nullptr, /*prefixOffset=*/nullptr, /*causal=*/nullptr,
      splitKV, adaptor.getElemwiseInputs(), op.getPreSecondGemmBody(),
      /*enableSoftmax=*/false, /*softmaxType=*/nullptr, /*numHeadsQ=*/1,
      /*numHeadsKV=*/1,
      /*preSoftmaxHasSplitKVTransforms=*/rw.getBoolAttr(false));
}

void RockAttnToGridwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::AttentionOp, rock::GemmElementwiseGemmOp>();
  target.addLegalOp<rock::TransformOp, rock::GridwiseAttentionOp, rock::StoreOp,
                    arith::TruncIOp, arith::ExtFOp, arith::ExtSIOp,
                    arith::TruncFOp>();

  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GemmElementwiseGemmRewritePattern, AttentionRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
