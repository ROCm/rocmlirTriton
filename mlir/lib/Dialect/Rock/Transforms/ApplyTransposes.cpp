//===- ApplyTransposes.cpp - Detect and apply transpose attributes --------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//
//
// This pass detects transpose patterns in rock.transform chains on inputs and
// outputs of AttentionOp, GemmElementwiseGemmOp, and ConvElementwiseGemmOp,
// and folds them into the ops' transpose attributes. This brings the MIGraphX
// pipeline representation in line with what rocmlir-gen produces.
//
// These ops are created by TosaToRockPass before TransposeRewritePattern runs,
// so their transposes are never folded at the TOSA level. Standalone GemmOp and
// ConvOp are created after TransposeRewritePattern and don't need this pass.
//
// For inputs, the pass uses stride analysis (untransform + getLowerSubDimensions)
// to detect when the last two non-batch dims are swapped relative to physical
// memory layout. When a transpose is found, the operand is replaced with the
// pre-transpose value (stripping the rock.transform) and the transpose
// attribute is set.
//
// For outputs, the pass checks whether the result flows into a single-use
// rock.transform that swaps the last two dims. If so, the op is recreated with
// the transposed result type and the output transpose attribute is set. The
// rock.transform is removed since the op now directly produces the transposed
// shape.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <limits>
#include <numeric>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKAPPLYTRANSPOSESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-apply-transposes"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockApplyTransposesPass
    : public rock::impl::RockApplyTransposesPassBase<RockApplyTransposesPass> {
  void runOnOperation() override;
};

// Check if a rock.transform op is a simple passthrough swap of the last two
// non-batch dims (i.e., a transpose of the form [0, 2, 1] for 3D tensors).
static bool isTransposeTransform(TransformOp transformOp) {
  TransformMapAttr mapAttr = transformOp.getTransform();
  ArrayRef<int64_t> upperBounds = mapAttr.getUpperBounds();
  ArrayRef<int64_t> lowerBounds = mapAttr.getLowerBounds();
  int64_t rank = upperBounds.size();
  if (rank < 2 || rank != (int64_t)lowerBounds.size())
    return false;

  ArrayRef<TransformAttr> ops = mapAttr.getOps();
  if (ops.size() != (size_t)rank)
    return false;

  // All ops must be PassThrough, and the last two must swap indices.
  for (int64_t i = 0; i < rank; ++i) {
    TransformAttr op = ops[i];
    if (op.getType() != TransformType::PassThrough)
      return false;
    ArrayRef<uint32_t> upperDims = op.getUpperDims();
    ArrayRef<uint32_t> lowerDims = op.getLowerDims();
    if (upperDims.size() != 1 || lowerDims.size() != 1)
      return false;

    uint32_t upperIdx = upperDims[0];
    uint32_t lowerIdx = lowerDims[0];
    if (i < rank - 2) {
      if (upperIdx != lowerIdx)
        return false;
    } else if (i == rank - 2) {
      if (upperIdx != (uint32_t)(rank - 2) || lowerIdx != (uint32_t)(rank - 1))
        return false;
    } else {
      if (upperIdx != (uint32_t)(rank - 1) || lowerIdx != (uint32_t)(rank - 2))
        return false;
    }
  }
  return true;
}

// Detect whether an input operand is transposed by analyzing strides through
// its rock.transform chain. If transposed and the defining op is a simple
// transpose transform, returns the pre-transpose value. Otherwise falls back
// to adding a sort transform.
//
// Returns {newValue, transposedAttr} where transposedAttr is non-null if
// a transpose was detected.
static std::pair<Value, UnitAttr>
detectInputTranspose(Value tensor, PatternRewriter &b) {
  auto tensorType = cast<ShapedType>(tensor.getType());
  int64_t rank = tensorType.getRank();
  if (rank < 2)
    return {tensor, nullptr};

  // Get the transform chain
  SmallVector<TransformMapAttr> transformsList;
  auto [source, needs64Bit] = untransform(tensor, transformsList);
  if (transformsList.empty())
    return {tensor, nullptr};

  // Compute strides for each upper dimension
  ArrayAttr transforms = b.getArrayAttr(
      SmallVector<Attribute>(transformsList.begin(), transformsList.end()));
  TransformMapAttr firstCoordTransform = transformsList[0];
  int64_t upperRank = firstCoordTransform.getUpperBounds().size();

  SmallVector<uint32_t> strides(upperRank);
  for (int64_t idx = 0; idx < upperRank; idx++) {
    FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>>>
        maybeLowerSubDims = getLowerSubDimensions(b, transforms, idx);
    if (failed(maybeLowerSubDims))
      return {tensor, nullptr};

    auto lowerSubDims = maybeLowerSubDims.value();
    uint32_t minStride =
        lowerSubDims.empty() ? 1 : std::numeric_limits<uint32_t>::max();
    for (auto &[dim, subDimInfos] : lowerSubDims) {
      for (auto subDim : subDimInfos)
        minStride = std::min(minStride, static_cast<uint32_t>(subDim.stride));
    }
    strides[idx] = minStride;
  }

  LLVM_DEBUG(llvm::dbgs() << "strides=");
  LLVM_DEBUG(llvm::interleaveComma(strides, llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  // Check if the last two non-batch dims are swapped.
  // For a non-transposed tensor, the second-to-last dim should have a larger
  // stride than the last dim. If reversed, the tensor is transposed.
  int64_t lastIdx = upperRank - 1;
  int64_t prevLastIdx = upperRank - 2;
  if (strides[prevLastIdx] >= strides[lastIdx])
    return {tensor, nullptr};

  // Transpose detected. Try to strip the defining transform if it's a simple
  // transpose.
  if (auto transformOp = tensor.getDefiningOp<TransformOp>()) {
    if (isTransposeTransform(transformOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Stripping transpose transform: " << transformOp << "\n");
      return {transformOp.getInput(), b.getUnitAttr()};
    }
  }

  // Fallback: add a dimension-swapping transform on top to present the
  // physical layout (same approach as upstream's sortByMemoryLayout).
  SmallVector<uint32_t> startIndices(upperRank);
  std::iota(startIndices.begin(), startIndices.end(), 0);
  std::swap(startIndices[prevLastIdx], startIndices[lastIdx]);

  SmallVector<uint32_t> endIndices(upperRank);
  std::iota(endIndices.begin(), endIndices.end(), 0);

  BottomUpTMBuilder sortDims(b, firstCoordTransform.getUpperBounds(),
                             tensor.getLoc());
  sortDims.passThrough(endIndices, startIndices);

  SmallVector<Attribute> transformAttrs{sortDims.get()};
  Value sorted = rock::transform(b, tensor, b.getArrayAttr(transformAttrs));

  LLVM_DEBUG(llvm::dbgs() << "Added sort transform (fallback): " << sorted
                          << "\n");
  return {sorted, b.getUnitAttr()};
}

// Detect whether the result of an op flows into a single-use rock.transform
// that transposes the last two dims. If so, returns the transposed result type
// and the transform op.
static std::pair<RankedTensorType, TransformOp>
detectOutputTranspose(Value result) {
  if (!result.hasOneUse())
    return {nullptr, nullptr};

  OpOperand &use = *result.getUses().begin();
  auto transformOp = dyn_cast<TransformOp>(use.getOwner());
  if (!transformOp)
    return {nullptr, nullptr};

  if (!isTransposeTransform(transformOp))
    return {nullptr, nullptr};

  auto transposedType =
      cast<RankedTensorType>(transformOp.getOutput().getType());
  LLVM_DEBUG(llvm::dbgs() << "Detected output transpose: " << transformOp
                          << "\n");
  return {transposedType, transformOp};
}

template <typename OpT>
static SmallVector<Operation *> getOperations(func::FuncOp &func) {
  SmallVector<Operation *, 4> ops;
  func.walk([&ops](OpT operation) { ops.push_back(operation); });
  return ops;
}

struct AttentionRewritePattern : public OpRewritePattern<AttentionOp> {
  using OpRewritePattern<AttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AttentionOp op,
                                PatternRewriter &b) const final {
    auto [newQ, transQ] = detectInputTranspose(op.getQueries(), b);
    auto [newK, transK] = detectInputTranspose(op.getKeys(), b);
    auto [newV, transV] = detectInputTranspose(op.getValues(), b);

    // Detect output transpose
    auto [transposedResultType, outputTransformOp] =
        detectOutputTranspose(op.getResult());
    UnitAttr transO =
        transposedResultType ? b.getUnitAttr() : op.getOTransposedAttr();
    RankedTensorType resultType =
        transposedResultType
            ? transposedResultType
            : cast<RankedTensorType>(op.getResult().getType());

    if (!transQ && !transK && !transV && !transposedResultType)
      return failure();

    // Preserve existing transpose attrs if we didn't detect a new one
    UnitAttr finalTransQ = transQ ? transQ : op.getQTransposedAttr();
    UnitAttr finalTransK = transK ? transK : op.getKTransposedAttr();
    UnitAttr finalTransV = transV ? transV : op.getVTransposedAttr();

    // Use original values if no transpose detected for that operand
    if (!transQ)
      newQ = op.getQueries();
    if (!transK)
      newK = op.getKeys();
    if (!transV)
      newV = op.getValues();

    Type lseType = op.getLse() ? op.getLse().getType() : nullptr;

    auto newOp = AttentionOp::create(
        b, op->getLoc(), resultType, lseType, newQ, newK, newV,
        op.getPreSoftmaxElemWiseInputs(), op.getCurrentSeqLen(),
        op.getPrefixOffset(), op.getNumHeadsQAttr(), op.getNumHeadsKVAttr(),
        finalTransQ, finalTransK, finalTransV, transO, op.getCausalAttr(),
        op.getSplitKVAttr(), op.getSoftmaxTypeAttr(), op.getParams0Attr(),
        op.getParams1Attr(), op.getFirstGemmIndicesAttr(),
        op.getPreSoftmaxHasSplitKVTransformsAttr());

    if (!op.getPreSoftmaxBody().empty()) {
      b.inlineRegionBefore(op.getPreSoftmaxBody(), newOp.getPreSoftmaxBody(),
                           newOp.getPreSoftmaxBody().begin());
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newOp->setAttr("perf_config", attr);

    if (transposedResultType) {
      // Replace uses of the output transform's result with the new op's result
      b.replaceAllUsesWith(outputTransformOp.getOutput(), newOp.getResult());
      b.eraseOp(outputTransformOp);
    }
    b.replaceOp(op, newOp);
    return success();
  }
};

struct GemmElementwiseGemmRewritePattern
    : public OpRewritePattern<GemmElementwiseGemmOp> {
  using OpRewritePattern<GemmElementwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmElementwiseGemmOp op,
                                PatternRewriter &b) const final {
    auto [newA, transA] = detectInputTranspose(op.getA(), b);
    auto [newB, transB] = detectInputTranspose(op.getB(), b);
    auto [newC, transC] = detectInputTranspose(op.getC(), b);

    auto [transposedResultType, outputTransformOp] =
        detectOutputTranspose(op.getResult());
    UnitAttr transO =
        transposedResultType ? b.getUnitAttr() : op.getOTransposedAttr();
    RankedTensorType resultType =
        transposedResultType
            ? transposedResultType
            : cast<RankedTensorType>(op.getResult().getType());

    if (!transA && !transB && !transC && !transposedResultType)
      return failure();

    UnitAttr finalTransA = transA ? transA : op.getATransposedAttr();
    UnitAttr finalTransB = transB ? transB : op.getBTransposedAttr();
    UnitAttr finalTransC = transC ? transC : op.getCTransposedAttr();

    if (!transA)
      newA = op.getA();
    if (!transB)
      newB = op.getB();
    if (!transC)
      newC = op.getC();

    auto newOp = GemmElementwiseGemmOp::create(
        b, op->getLoc(), resultType, newA, newB, newC, op.getElemwiseInputs(),
        finalTransA, finalTransB, finalTransC, transO, op.getParams0Attr(),
        op.getParams1Attr(), op.getFirstGemmIndicesAttr());

    if (!op.getPreSecondGemmBody().empty()) {
      b.inlineRegionBefore(op.getPreSecondGemmBody(),
                           newOp.getPreSecondGemmBody(),
                           newOp.getPreSecondGemmBody().begin());
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newOp->setAttr("perf_config", attr);

    if (transposedResultType) {
      b.replaceAllUsesWith(outputTransformOp.getOutput(), newOp.getResult());
      b.eraseOp(outputTransformOp);
    }
    b.replaceOp(op, newOp);
    return success();
  }
};

struct ConvElementwiseGemmRewritePattern
    : public OpRewritePattern<ConvElementwiseGemmOp> {
  using OpRewritePattern<ConvElementwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvElementwiseGemmOp op,
                                PatternRewriter &b) const final {
    auto [newC, transC] = detectInputTranspose(op.getC(), b);

    // ConvElementwiseGemmOp may or may not have a result
    Value mainResult = op.getResult();
    RankedTensorType transposedResultType = nullptr;
    TransformOp outputTransformOp = nullptr;
    UnitAttr transO = op.getOTransposedAttr();

    if (mainResult) {
      std::tie(transposedResultType, outputTransformOp) =
          detectOutputTranspose(mainResult);
      if (transposedResultType)
        transO = b.getUnitAttr();
    }

    if (!transC && !transposedResultType)
      return failure();

    UnitAttr finalTransC = transC ? transC : op.getCTransposedAttr();
    if (!transC)
      newC = op.getC();

    // Determine result type
    TypeRange resultTypes = op->getResultTypes();
    SmallVector<Type> newResultTypes;
    if (transposedResultType) {
      newResultTypes.push_back(transposedResultType);
      resultTypes = newResultTypes;
    }

    auto newOp = ConvElementwiseGemmOp::create(
        b, op->getLoc(), resultTypes.empty() ? TypeRange{} : resultTypes,
        op.getFilter(), op.getInput(), newC, op.getElemwiseInputs(),
        op.getOut(), finalTransC, transO, op.getPaddingAttr(),
        op.getStridesAttr(), op.getDilationsAttr(), op.getParams0Attr(),
        op.getParams1Attr(), op.getFirstGemmIndicesAttr());

    if (!op.getPreSecondGemmBody().empty()) {
      b.inlineRegionBefore(op.getPreSecondGemmBody(),
                           newOp.getPreSecondGemmBody(),
                           newOp.getPreSecondGemmBody().begin());
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      newOp->setAttr("perf_config", attr);

    if (transposedResultType && outputTransformOp) {
      b.replaceAllUsesWith(outputTransformOp.getOutput(), newOp.getResult());
      b.eraseOp(outputTransformOp);
    }
    b.replaceOp(op, newOp);
    return success();
  }
};

} // namespace

void RockApplyTransposesPass::runOnOperation() {
  auto func = getOperation();
  if (!func->hasAttr(KernelAttr::getMnemonic()))
    return;

  auto &ctx = getContext();
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingOps);

  RewritePatternSet patternsAttention(&ctx);
  patternsAttention.add<AttentionRewritePattern>(&ctx);
  if (failed(applyOpPatternsGreedily(getOperations<AttentionOp>(func),
                                     std::move(patternsAttention), config)))
    return signalPassFailure();

  RewritePatternSet patternsGemmElemGemm(&ctx);
  patternsGemmElemGemm.add<GemmElementwiseGemmRewritePattern>(&ctx);
  if (failed(applyOpPatternsGreedily(
          getOperations<GemmElementwiseGemmOp>(func),
          std::move(patternsGemmElemGemm), config)))
    return signalPassFailure();

  RewritePatternSet patternsConvElemGemm(&ctx);
  patternsConvElemGemm.add<ConvElementwiseGemmRewritePattern>(&ctx);
  if (failed(applyOpPatternsGreedily(
          getOperations<ConvElementwiseGemmOp>(func),
          std::move(patternsConvElemGemm), config)))
    return signalPassFailure();
}
