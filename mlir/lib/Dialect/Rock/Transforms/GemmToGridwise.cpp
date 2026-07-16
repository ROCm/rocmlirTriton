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

/// Stream-K padding for a gemm being lowered to a gridwise_gemm: when the
/// perf-config requests stream-K (streamKMultiple >= 1, mutually exclusive with
/// split-K and unsupported for scaled gemms), choose the partition dimension
/// and apply all the padding RockStreamKDecompose relies on, in place on
/// `a`/`b` (and the output/fusion views via `transformViews`):
///   (1) pad the partition dim up to whole extra tiles so a balanced wave +
///       split-K-remainder decomposition always exists, and
///   (2) pad K (zero-fill) up to a multiple of splitK*kPerBlock so the
///       remainder's K-split folds evenly (usually a no-op).
/// Returns the chosen partition dim (to be stashed on the gridwise op for the
/// pass), or nullopt when stream-K is off or not applicable. See the shared
/// chooseStreamKPlan / StreamKDecompose.cpp.
static std::optional<StreamKPartDim> applyStreamKPadding(
    OpBuilder &rw, Location loc, GemmOp op, Attribute params,
    int64_t splitKFactor, Value scaleA, Value scaleB, Value &a, Value &b,
    ArrayRef<int64_t> &aShape, ArrayRef<int64_t> &bShape,
    llvm::function_ref<void(llvm::function_ref<Value(Value)>)> transformViews) {
  auto gwParams = cast<GemmParamsAttr>(params);
  const int64_t streamKMultiple = gwParams.getStreamKMultiple();
  if (streamKMultiple < 1 || splitKFactor > 1 || (scaleA && scaleB))
    return std::nullopt;

  int64_t mPerBlock = gwParams.getMPerBlock();
  int64_t nPerBlock = gwParams.getNPerBlock();
  int64_t kPerBlock = gwParams.getKPerBlock();
  int64_t curG = cast<ShapedType>(a.getType()).getShape()[0];
  int64_t curM = cast<ShapedType>(a.getType()).getShape()[1];
  int64_t curN = cast<ShapedType>(b.getType()).getShape()[2];
  int64_t numCU = rock::getNumCUValue(op);
  auto maybePlan = rock::chooseStreamKPlan(
      curG, curM / mPerBlock, curN / nPerBlock, numCU, streamKMultiple);
  if (failed(maybePlan))
    return std::nullopt;

  if (maybePlan->padBlocks > 0) {
    if (maybePlan->partDim == StreamKPartDim::M) {
      int64_t padM = maybePlan->padBlocks * mPerBlock;
      a = padMatrix(a, rw, loc, "gemmM", padM, "gemmK", 0);
      transformViews([&](Value v) {
        return padMatrix(v, rw, loc, "gemmM", padM, "gemmN", 0);
      });
    } else if (maybePlan->partDim == StreamKPartDim::N) {
      int64_t padN = maybePlan->padBlocks * nPerBlock;
      b = padMatrix(b, rw, loc, "gemmK", 0, "gemmN", padN);
      transformViews([&](Value v) {
        return padMatrix(v, rw, loc, "gemmM", 0, "gemmN", padN);
      });
    }
  }
  // K carries no output view, so only A and B need padding.
  int64_t curK = cast<ShapedType>(a.getType()).getShape()[2];
  int64_t padK = llvm::alignTo(curK, maybePlan->splitK * kPerBlock) - curK;
  if (padK > 0) {
    a = padMatrix(a, rw, loc, "gemmM", 0, "gemmK", padK);
    b = padMatrix(b, rw, loc, "gemmK", padK, "gemmN", 0);
  }
  aShape = cast<ShapedType>(a.getType()).getShape();
  bShape = cast<ShapedType>(b.getType()).getShape();
  return maybePlan->partDim;
}

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
  SetVector<StoreOp> &stores = views.stores;
  SmallVector<Value> &outputViews = views.outputViews;
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

  // TODO: use AmdArchDb to figure out when we need to convert A and B instead of hardcoding it!
  // Extend input types to the highest-precision type among the inputs
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

  a = padMatrix(a, rw, loc, "gemmM", extraPad.m, "gemmK", extraPad.k);
  b = padMatrix(b, rw, loc, "gemmK", extraPad.k, "gemmN", extraPad.n);
  transformViews([&](Value v) {
    return padMatrix(v, rw, loc, "gemmM", extraPad.m, "gemmN", extraPad.n);
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
    scaleA =
        padMatrix(scaleA, rw, loc, "gemmM", extraPad.m, "gemmK", padScaleK);
    scaleB =
        padMatrix(scaleB, rw, loc, "gemmN", extraPad.n, "gemmK", padScaleK);
  }

  // Stream-K: choose the partition dim and apply its padding (see
  // applyStreamKPadding). The chosen dim, if any, is stashed on the gridwise op
  // below for RockStreamKDecompose.
  std::optional<StreamKPartDim> streamKPartDim =
      applyStreamKPadding(rw, loc, op, params, splitKFactor, scaleA, scaleB, a,
                          b, aShape, bShape, transformViews);

  if (failed(computeGridSize(rw, op, a, b))) {
    return op.emitError("failed to compute the grid size of `GemmOp`");
  }

  auto newOutputType = RankedTensorType::get(
      cast<ShapedType>(outputViews[0].getType()).getShape(), op.getCType());
  auto gridwiseOp =
      GridwiseGemmOp::create(rw, loc, newOutputType, a, b, scaleA, scaleB, op.getQuantBlockSizeAttr(), 
                             cast<GemmParamsAttr>(params));
  if (streamKPartDim)
    gridwiseOp->setAttr(
        StreamKPartDimAttr::getMnemonic(),
        StreamKPartDimAttr::get(rw.getContext(), *streamKPartDim));
  Value result = gridwiseOp.getResult();

  // Propagate the new (potentially padded) output type through any fusion ops
  // between the gemm result and the store ops. This replaces uses of the old
  // gemm result inside fusion ops with the gridwise result and updates their
  // result types to match the new shape.
  rock::propagateOutputType(op.getResult(), result);

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

  a = padMatrix(a, builder, loc, "gemmM", 0, "gemmK", kPad);
  b = padMatrix(b, builder, loc, "gemmK", kPad, "gemmN", 0);
  if (scaleA && scaleB) {
    assert(kPad % blockSize == 0 &&
           "kPad must be a multiple of quantBlockSize");
    int64_t scaleKPad = kPad / blockSize;
    scaleA = padMatrix(scaleA, builder, loc, "gemmM", 0, "gemmK", scaleKPad);
    // scaleB is [G, N, K] after normalizeMatrix(scaleB, ..., "gemmN", "gemmK")
    scaleB = padMatrix(scaleB, builder, loc, "gemmN", 0, "gemmK", scaleKPad);
  }

  // perform coordinate transformations
  Value aNew{nullptr}, bNew{nullptr};
  SmallVector<Value> outputViewsNew;
  DenseMap<Value, Value> fusionInputMapNew(fusionInputMap);
  Value scaleANew{nullptr}, scaleBNew{nullptr};
  ArrayRef<int64_t> cShape =
      cast<RankedTensorType>(outputViews[0].getType()).getShape();
  for (auto outputView : outputViews) {
    if (cast<RankedTensorType>(outputView.getType()).getShape() != cShape)
      return op->emitError("all output views must have the same shape");
  }

  struct GemmOperandsData {
    Value &in;
    Value &out;
    StringRef preservedName;
    uint32_t nonKDim;
    uint32_t kDim;
  };

  llvm::SmallVector<GemmOperandsData, 4> gemmOperands{{a, aNew, "gemmM", 1, 2},
                                                      {b, bNew, "gemmN", 2, 1}};
  if (scaleA && scaleB) {
    gemmOperands.push_back({scaleA, scaleANew, "gemmM", 1, 2});
    // After normalizeMatrix(scaleB, ..., "gemmN", "gemmK"), scaleB is [G, N, K]
    gemmOperands.push_back({scaleB, scaleBNew, "gemmN", 1, 2});
  }
  for (auto &gemmOperand : gemmOperands) {
    // Fold the split-K factor into G:
    //   1. unmerge (gemmK) -> (gemmKSplit, gemmK*)
    //   2. merge (gemmG, gemmKSplit) -> (gemmG*)
    gemmOperand.out = splitKFoldOperand(
        builder, loc, gemmOperand.in, splitKFactor, gemmOperand.kDim,
        gemmOperand.nonKDim, gemmOperand.preservedName);
  }

  // Map the C side ([G*splitK, M, N]) back onto the real [G, M, N] destination,
  // ignoring the split dimension so every K-split accumulates into the same
  // tile. Set the insertion point after each value's defining op to maintain
  // dominance.
  for (auto view : outputViews) {
    OpBuilder::InsertionGuard guard(builder);
    if (Operation *defOp = view.getDefiningOp())
      builder.setInsertionPointAfter(defOp);
    outputViewsNew.push_back(
        splitKFoldOutputView(builder, loc, view, splitKFactor));
  }
  for (auto &[orig, view] : fusionInputMapNew) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(view.getDefiningOp());
    view = splitKFoldOutputView(builder, loc, view, splitKFactor);
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
