//===- GemmToGridwise.cpp - Rock GEMM implementation ------------===//
//
// Copyright 2022 Advanced Micro Devices.
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
// This pass converts rock.gemm into the appropriate rock.gridwise_gemm
// adding padding and group dimensions if needed.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
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
#define GEN_PASS_DEF_ROCKGEMMTOGRIDWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gemm-to-gridwise"

using namespace mlir;
using namespace mlir::rock;

namespace {
/// Reconcile the unit group dimension introduced by GEMM normalization with
/// the original rank-two GEMM-space value feeding a reduction transform chain.
static LogicalResult bridgeNormalizedReductionInput(PatternRewriter &rewriter,
                                                    ReduceOp reduceOp) {
  Value value = reduceOp.getIn();
  TransformOp firstTransform;
  while (auto transformOp = value.getDefiningOp<TransformOp>()) {
    firstTransform = transformOp;
    value = transformOp.getInput();
  }
  if (!firstTransform)
    return success();

  auto actualType = cast<RankedTensorType>(firstTransform.getInput().getType());
  ArrayRef<int64_t> expectedShape =
      firstTransform.getTransform().getLowerBounds();
  if (actualType.getShape() == expectedShape)
    return success();

  ArrayRef<int64_t> actualShape = actualType.getShape();
  if (actualShape.size() != 3 || expectedShape.size() != 2 ||
      actualShape[0] != 1 || actualShape.drop_front() != expectedShape)
    return reduceOp.emitError(
        "unsupported GEMM normalization for explicit fused reduction");

  rewriter.setInsertionPoint(firstTransform);
  TopDownTMBuilder removeGroup(rewriter, {"gemmM", "gemmN"}, expectedShape,
                               reduceOp.getLoc());
  removeGroup.constDim("gemmG", 0, /*constantVal=*/0,
                       /*lowerSize=*/actualShape[0]);
  removeGroup.passThrough({"gemmM", "gemmN"}, {1, 2}, {"gemmM", "gemmN"});
  Value bridged =
      TransformOp::create(rewriter, reduceOp.getLoc(),
                          firstTransform.getInput(), removeGroup.get());
  firstTransform.getInputMutable().assign(bridged);
  return success();
}

class RockGemmToGridwisePass
    : public rock::impl::RockGemmToGridwisePassBase<RockGemmToGridwisePass> {
  void runOnOperation() override;
};

struct GemmRewritePattern : public OpConversionPattern<GemmOp> {
  using OpConversionPattern<GemmOp>::OpConversionPattern;

  struct SplitKTransformedOperands {
    Value a;
    Value b;
    SmallVector<Value> outputViews;
    DenseMap<Value, Value> fusionInputMap;
    Value scaleA;
    Value scaleB;
  };

  LogicalResult matchAndRewrite(GemmOp op, GemmOpAdaptor adaptor,
                                ConversionPatternRewriter &rw) const override;

  LogicalResult computeGridSize(ConversionPatternRewriter &rw, GemmOp op,
                                Value a, Value b) const;

  FailureOr<SplitKTransformedOperands>
  arrangeSplitKTransform(OpBuilder &builder, GemmOp op, Location loc,
                         int64_t splitKFactor, Value a, Value b,
                         ArrayRef<Value> outputViews,
                         const DenseMap<Value, Value> &fusionInputMap,
                         Value scaleA, Value scaleB) const;
};
} // end namespace

LogicalResult
GemmRewritePattern::matchAndRewrite(GemmOp op, GemmOpAdaptor adaptor,
                                    ConversionPatternRewriter &rw) const {
  Location loc = op->getLoc();

  Attribute params = op.getParams().value_or(nullptr);
  if (!params) {
    return op.emitOpError("cannot lower gemm without tuning parameters");
  }

  Value a = adaptor.getA(), b = adaptor.getB();

  auto maybeViews = rock::traceOutputsAndFusionInputs(op.getResult());
  if (failed(maybeViews))
    return op.emitOpError("cannot trace to rock::StoreOp");
  // Use plain references rather than structured bindings so the lambda below
  // can capture them without requiring -Wc++20-extensions.
  auto &views = maybeViews.value();
  SetVector<StoreOp> &allStores = views.stores;
  SmallVector<Value> outputViews;
  SmallVector<StoreOp> stores;
  SmallVector<ReduceOp> reductions;
  for (auto [store, outputView] :
       llvm::zip_equal(allStores, views.outputViews)) {
    FailureOr<std::optional<ReductionStorePath>> maybePath =
        getReductionStorePath(store);
    if (failed(maybePath))
      return op.emitOpError("malformed blockwise reduction store path");
    if (*maybePath && (**maybePath).reduceOp.getBlockwise()) {
      reductions.push_back((**maybePath).reduceOp);
    } else {
      stores.push_back(store);
      outputViews.push_back(outputView);
    }
  }
  DenseMap<Value, Value> &fusionInputMap = views.fusionInputMap;
  SmallVector<Type> storeResultTypes;
  storeResultTypes.reserve(stores.size());
  for (StoreOp store : stores)
    storeResultTypes.push_back(store.getResult().getType());

  Value scaleA = adaptor.getScaleA(), scaleB = adaptor.getScaleB();

  ShapedType typeA = cast<ShapedType>(a.getType());
  ShapedType typeB = cast<ShapedType>(b.getType());
  Type elemTypeA = typeA.getElementType();
  Type elemTypeB = typeB.getElementType();
  ArrayRef<int64_t> aShape = typeA.getShape();
  ArrayRef<int64_t> bShape = typeB.getShape();

  auto elemAWidth = elemTypeA.getIntOrFloatBitWidth();
  auto elemBWidth = elemTypeB.getIntOrFloatBitWidth();

  // TODO: use AmdArchDb to figure out when we need to convert A and B instead
  // of hardcoding it! Extend input types to the highest-precision type among
  // the inputs
  if (elemTypeA != elemTypeB &&
      (!isa<FloatType>(elemTypeA) || !isa<FloatType>(elemTypeB) ||
       elemAWidth != 8 || elemBWidth != 8)) {
    if (elemTypeA.getIntOrFloatBitWidth() > elemTypeB.getIntOrFloatBitWidth()) {
      auto newBType = RankedTensorType::get(bShape, elemTypeA);
      b = createTypeConversionOp(rw, loc, b, newBType);
    } else {
      auto newAType = RankedTensorType::get(aShape, elemTypeB);
      a = createTypeConversionOp(rw, loc, a, newAType);
    }
  }
  SmallVector<int64_t> scaleAShape, scaleBShape;

  // Note: the gridwise ops take M x K and K x N
  a = normalizeMatrix(a, rw, loc, op.getATransposed(), "gemmM", "gemmK");
  b = normalizeMatrix(b, rw, loc, op.getBTransposed(), "gemmK", "gemmN");
  auto transformViews = [&](auto fn) {
    for (auto &outputView : outputViews)
      outputView = fn(outputView);
    for (auto &[orig, view] : fusionInputMap)
      view = fn(view);
  };
  transformViews([&](Value v) {
    return normalizeMatrix(v, rw, loc, op.getOTransposed(), "gemmM", "gemmN");
  });
  aShape = cast<ShapedType>(a.getType()).getShape();
  bShape = cast<ShapedType>(b.getType()).getShape();
  if (scaleA && scaleB) {
    bool transposeScaleA = op.getAScaleTransposed();
    scaleA =
        normalizeMatrix(scaleA, rw, loc, transposeScaleA, "gemmM", "gemmK");
    bool transposeScaleB = op.getBScaleTransposed();
    scaleB =
        normalizeMatrix(scaleB, rw, loc, transposeScaleB, "gemmN", "gemmK");
  }

  const int64_t splitKFactor = op.getParams()->getSplitKFactor();
  if (splitKFactor > 1 && !reductions.empty())
    return op.emitOpError(
        "split-K is not yet supported with an explicit fused reduction");
  if (splitKFactor > 1) {
    auto maybeSplitk =
        arrangeSplitKTransform(rw, op, loc, splitKFactor, a, b, outputViews,
                               fusionInputMap, scaleA, scaleB);
    if (failed(maybeSplitk))
      return maybeSplitk;

    auto &transformed = maybeSplitk.value();
    a = transformed.a;
    b = transformed.b;
    outputViews = transformed.outputViews;
    fusionInputMap = std::move(transformed.fusionInputMap);
    scaleA = transformed.scaleA;
    scaleB = transformed.scaleB;
    aShape = cast<ShapedType>(a.getType()).getShape();
    bShape = cast<ShapedType>(b.getType()).getShape();
  }

  // Note, matrix dimension correctness is handled in the verifier
  GemmSize size(/*g=*/aShape[0], /*m=*/aShape[1], /*k=*/aShape[2],
                /*n=*/bShape[2]);

  GemmSize extraPad =
      requiredPadding(params, size).value_or(GemmSize{0, 0, 0, 0});
  if (!reductions.empty() &&
      (extraPad.g != 0 || extraPad.m != 0 || extraPad.n != 0))
    return op.emitOpError(
        "output padding is not yet supported with an explicit fused reduction");

  a = padMatrixForTileAlignment(a, rw, loc, "gemmM", extraPad.m, "gemmK",
                                extraPad.k);
  b = padMatrixForTileAlignment(b, rw, loc, "gemmK", extraPad.k, "gemmN",
                                extraPad.n);
  transformViews([&](Value v) {
    return padMatrixForTileAlignment(v, rw, loc, "gemmM", extraPad.m, "gemmN",
                                     extraPad.n);
  });
  if (scaleA && scaleB) {
    int64_t quantBlockSize = op.getQuantBlockSize().value();
    int64_t newK = size.k + extraPad.k;
    // this should never happen as long as both quantBlockSize <= kPerBlock
    // (GemmOp verifier checks this) and both are powers of two
    assert(newK % quantBlockSize == 0 &&
           "newK is not divisible by quantBlockSize");
    int64_t newScaleK = newK / quantBlockSize;
    int64_t padScaleK =
        newScaleK - cast<ShapedType>(scaleA.getType()).getDimSize(2);
    scaleA = padMatrixForTileAlignment(scaleA, rw, loc, "gemmM", extraPad.m,
                                       "gemmK", padScaleK);
    scaleB = padMatrixForTileAlignment(scaleB, rw, loc, "gemmN", extraPad.n,
                                       "gemmK", padScaleK);
  }

  if (failed(computeGridSize(rw, op, a, b))) {
    return op.emitError("failed to compute the grid size of `GemmOp`");
  }

  auto paddedAShape = cast<ShapedType>(a.getType()).getShape();
  auto paddedBShape = cast<ShapedType>(b.getType()).getShape();
  auto newOutputType = RankedTensorType::get(
      {paddedAShape[0], paddedAShape[1], paddedBShape[2]}, op.getCType());
  auto gridwiseOp = GridwiseGemmOp::create(rw, loc, newOutputType, a, b, scaleA,
                                           scaleB, op.getQuantBlockSizeAttr(),
                                           cast<GemmParamsAttr>(params));
  Value result = gridwiseOp.getResult();

  // Propagate the new (potentially padded) output type through any fusion ops
  // between the gemm result and the store ops. This replaces uses of the old
  // gemm result inside fusion ops with the gridwise result and updates their
  // result types to match the new shape.
  rock::propagateOutputType(op.getResult(), result);
  for (ReduceOp reduceOp : reductions)
    if (failed(bridgeNormalizedReductionInput(rw, reduceOp)))
      return failure();

  // Replace extra fusion input operands with their padded versions.
  rock::replaceFusionExtraInputs(result, fusionInputMap);

  for (size_t i = 0; i < stores.size(); ++i) {
    StoreOp storeOp = stores[i];
    Value view = outputViews[i];
    // adjust the store method
    StoreMethodAttr storeMethod = storeOp.getStoreMethodAttr();
    if (splitKFactor > 1)
      storeMethod =
          rw.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::AtomicAdd);

    // If the store's source is the gemm result directly (no fusions),
    // use the gridwise result. Otherwise, propagateOutputType has already
    // updated the fusion chain, and storeOp.getSource() has the correct type.
    Value source = storeOp.getSource();
    if (source == op.getResult()) {
      source = result;
    }
    rw.setInsertionPoint(storeOp);
    auto newStoreOp =
        rock::StoreOp::create(rw, storeOp.getLoc(), storeResultTypes[i], source,
                              view, storeOp.getResultAlias(), storeMethod);
    rw.replaceOp(storeOp, newStoreOp.getResult());
  }

  rw.replaceOp(op, result);
  return success();
}

FailureOr<GemmRewritePattern::SplitKTransformedOperands>
GemmRewritePattern::arrangeSplitKTransform(
    OpBuilder &builder, GemmOp op, Location loc, int64_t splitKFactor, Value a,
    Value b, ArrayRef<Value> outputViews,
    const DenseMap<Value, Value> &fusionInputMap, Value scaleA,
    Value scaleB) const {
  // Set the store method and prefill attribute on output store ops
  Value matC = op.getResult();
  FailureOr<SetVector<StoreOp>> storeOps = traceRootOutputToStoreOps(matC);
  if (failed(storeOps))
    return op->emitError("can't trace gemm output to store ops");
  for (StoreOp storeOp : storeOps.value()) {
    if (failed(
            setStoreMethodAndPrefill(builder, storeOp, StoreMethod::AtomicAdd)))
      return failure();
  }

  const int64_t origK = cast<RankedTensorType>(a.getType()).getShape()[2];
  int64_t kPad = 0;
  int64_t blockSize = 0;
  if (scaleA && scaleB) {
    blockSize = op.getQuantBlockSize().value();
    kPad = llvm::alignTo(origK, splitKFactor * blockSize) - origK;
  } else {
    kPad = llvm::alignTo(origK, splitKFactor) - origK;
  }

  a = padMatrixForTileAlignment(a, builder, loc, "gemmM", 0, "gemmK", kPad);
  b = padMatrixForTileAlignment(b, builder, loc, "gemmK", kPad, "gemmN", 0);
  if (scaleA && scaleB) {
    assert(kPad % blockSize == 0 &&
           "kPad must be a multiple of quantBlockSize");
    int64_t scaleKPad = kPad / blockSize;
    scaleA = padMatrixForTileAlignment(scaleA, builder, loc, "gemmM", 0,
                                       "gemmK", scaleKPad);
    // scaleB is [G, N, K] after normalizeMatrix(scaleB, ..., "gemmN", "gemmK")
    scaleB = padMatrixForTileAlignment(scaleB, builder, loc, "gemmN", 0,
                                       "gemmK", scaleKPad);
  }

  // perform coordinate transformations
  Value aNew{nullptr}, bNew{nullptr};
  SmallVector<Value> outputViewsNew;
  DenseMap<Value, Value> fusionInputMapNew(fusionInputMap);
  Value scaleANew{nullptr}, scaleBNew{nullptr};
  ArrayRef<int64_t> aShape = cast<RankedTensorType>(a.getType()).getShape();
  ArrayRef<int64_t> bShape = cast<RankedTensorType>(b.getType()).getShape();
  ArrayRef<int64_t> cShape =
      cast<RankedTensorType>(outputViews[0].getType()).getShape();
  for (auto outputView : outputViews) {
    if (cast<RankedTensorType>(outputView.getType()).getShape() != cShape)
      return op->emitError("all output views must have the same shape");
  }

  const int64_t K = aShape[2];

  struct GemmOperandsData {
    Value &in;
    Value &out;
    SmallVector<StringRef> inputDimNames;
    ArrayRef<int64_t> inputShape;
    uint32_t nonKDim;
    uint32_t kDim;
    uint32_t newNonKDim;
    int64_t kLen;
  };

  llvm::SmallVector<GemmOperandsData, 4> gemmOperands{
      {a, aNew, {"gemmG", "gemmM", "gemmK"}, aShape, 1, 2, 1, K},
      {b, bNew, {"gemmG", "gemmK", "gemmN"}, bShape, 2, 1, 3, K}};
  if (scaleA && scaleB) {
    ArrayRef<int64_t> scaleAShape =
        cast<RankedTensorType>(scaleA.getType()).getShape();
    ArrayRef<int64_t> scaleBShape =
        cast<RankedTensorType>(scaleB.getType()).getShape();
    int64_t scaleAK = scaleAShape[2];
    int64_t scaleBK = scaleBShape[2];
    gemmOperands.push_back({scaleA,
                            scaleANew,
                            {"gemmG", "gemmM", "gemmK"},
                            scaleAShape,
                            1,
                            2,
                            1,
                            scaleAK});
    // After normalizeMatrix(scaleB, ..., "gemmN", "gemmK"), scaleB is [G, N, K]
    gemmOperands.push_back({scaleB,
                            scaleBNew,
                            {"gemmG", "gemmN", "gemmK"},
                            scaleBShape,
                            1,
                            2,
                            1,
                            scaleBK});
  }
  for (auto &gemmOperand : gemmOperands) {
    // Prepare matrix A and B - i.e.,
    //    (gemmG, gemmK, gemmM) and (gemmG, gemmK, gemmN), respectively
    // Using bottom-up transformations
    // 1. unmerge (gemmK) -> (gemmKSplit, gemmK*)
    // 2. merge (gemmG, gemmKSplit) -> (gemmG*)

    StringRef preservedDimName;
    for (auto &dimName : gemmOperand.inputDimNames) {
      if ((dimName != "gemmK") && (dimName != "gemmG"))
        preservedDimName = dimName;
    }

    int64_t operandK = gemmOperand.kLen;
    assert(operandK % splitKFactor == 0 &&
           "operandK must be divisible by splitKFactor after padding");
    BottomUpTMBuilder unmergeTransform(builder, gemmOperand.inputDimNames,
                                       gemmOperand.inputShape, loc);

    unmergeTransform.passThrough({"gemmG", preservedDimName},
                                 {0, gemmOperand.newNonKDim},
                                 {"gemmG", preservedDimName});
    unmergeTransform.unmerge({"gemmKSplit", "gemmK"},
                             {gemmOperand.kDim, gemmOperand.kDim + 1}, "gemmK",
                             {splitKFactor, operandK / splitKFactor});

    auto unmergeTransformAttr = unmergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(unmergeTransformAttr);

    auto mergeTransform =
        BottomUpTMBuilder::above(unmergeTransform, unmergeTransformAttr);

    mergeTransform.merge("gemmG", 0, {"gemmG", "gemmKSplit"});
    mergeTransform.passThrough({"gemmK", preservedDimName},
                               {gemmOperand.kDim, gemmOperand.nonKDim},
                               {"gemmK", preservedDimName});

    auto mergeTransformAttr = mergeTransform.get();
    transformAttrs.push_back(mergeTransformAttr);

    std::reverse(transformAttrs.begin(), transformAttrs.end());
    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    gemmOperand.out =
        mlir::rock::transform(builder, gemmOperand.in, arrayTransformAttrs);
  }

  {
    // Prepare matrix C - i.e., (gemmG, gemmM, gemmN)
    // Using top-down transformations
    // 1. merge (gemmG * gemmKSplit, gemmM, gemmN) -> (gemmG, gemmKSplit, gemmM,
    // gemmN)
    // 2. ignore (gemmG, gemmKSplit, gemmM, gemmN) -> (gemmG, gemmM, gemmN)

    const int64_t G = cShape[0];
    const int64_t M = cShape[1];
    const int64_t N = cShape[2];

    TopDownTMBuilder mergeTransform(builder, {"gemmG", "gemmM", "gemmN"},
                                    {G * splitKFactor, M, N});

    mergeTransform.merge({"gemmG", "gemmKSplit"}, {0, 1}, "gemmG",
                         {G, splitKFactor});
    mergeTransform.passThrough({"gemmM", "gemmN"}, {2, 3}, {"gemmM", "gemmN"});
    auto mergeTransformAttr = mergeTransform.get();

    SmallVector<Attribute> transformAttrs;
    transformAttrs.push_back(mergeTransformAttr);

    TopDownTMBuilder ignoreTransform =
        TopDownTMBuilder::below(mergeTransform, mergeTransformAttr);

    ignoreTransform.ignore("gemmKSplit");
    ignoreTransform.passThrough({"gemmG", "gemmM", "gemmN"}, {0, 1, 2},
                                {"gemmG", "gemmM", "gemmN"});

    TransformMapAttr ignoreTransformAttr = ignoreTransform.get();
    transformAttrs.push_back(ignoreTransformAttr);

    ArrayAttr arrayTransformAttrs = builder.getArrayAttr(transformAttrs);
    for (auto view : outputViews) {
      OpBuilder::InsertionGuard guard(builder);
      if (Operation *defOp = view.getDefiningOp())
        builder.setInsertionPointAfter(defOp);
      outputViewsNew.push_back(
          mlir::rock::transform(builder, view, arrayTransformAttrs));
    }
    // Apply the same transforms to fusion extra inputs.
    // Set insertion point after each value's defining op to maintain dominance.
    for (auto &[orig, view] : fusionInputMapNew) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(view.getDefiningOp());
      view = mlir::rock::transform(builder, view, arrayTransformAttrs);
    }
  }
  return SplitKTransformedOperands{
      aNew, bNew, outputViewsNew, fusionInputMapNew, scaleANew, scaleBNew};
}

LogicalResult GemmRewritePattern::computeGridSize(ConversionPatternRewriter &rw,
                                                  GemmOp op, Value a,
                                                  Value b) const {
  Attribute params = op.getParams().value();

  const auto aShape = cast<RankedTensorType>(a.getType()).getShape();
  const auto bShape = cast<RankedTensorType>(b.getType()).getShape();

  const int64_t G = aShape[0];
  const int64_t M = aShape[1];
  const int64_t N = bShape[2];

  auto tuningParams = cast<GemmParamsAttr>(params);
  auto mPerBlock = tuningParams.getMPerBlock();
  auto nPerBlock = tuningParams.getNPerBlock();

  const auto gridSize = (M / mPerBlock) * (N / nPerBlock) * G;
  assert(gridSize > 0);

  func::FuncOp funcOp = cast<func::FuncOp>(op->getParentOp());
  funcOp->setAttr(rock::GridSizeAttr::getMnemonic(),
                  rw.getI32IntegerAttr(gridSize));
  return success();
}

void RockGemmToGridwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);

  target.addIllegalOp<rock::GemmOp>();
  target.addLegalOp<rock::TransformOp, rock::GridwiseGemmOp, rock::StoreOp,
                    arith::TruncIOp, arith::ExtFOp, arith::ExtSIOp,
                    arith::TruncFOp>();

  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GemmRewritePattern>(ctx);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
