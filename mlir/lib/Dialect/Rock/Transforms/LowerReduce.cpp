//===- LowerReduce.cpp - Lower rock.reduce to broadcast + atomic store ===//
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
// This pass converts rock.reduce ops into a Broadcast transform on the
// rock.store destination, changing the store method to atomic_add (for sum).
//
// Before:
//   %fused : tensor<1x100x100xf16>
//   %reduced = rock.reduce sum %fused {axis = 1}
//   ...
//   %result = rock.store %reduced to %dest by set
//
// After:
//   %fused : tensor<1x100x100xf16>
//   %dest_bc = rock.transform %dest by <Broadcast on reduced axis>
//   %result = rock.store %fused to %dest_bc by atomic_add
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERREDUCEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-reduce"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Build a TransformMapAttr that inverts `trMap`.
///
/// The original trMap maps upper→lower (source=lower, result=upper in a
/// TransformOp).  The inverse maps in the opposite direction: its source has
/// shape `srcShape` (the original upper bounds) and its result has the
/// original lower bounds, with dimension names taken from `targetNames`.
///
/// Currently handles Merge (→Unmerge), PassThrough, and AddDim (→drop).
static FailureOr<TransformMapAttr>
invertTransformMap(OpBuilder &b, TransformMapAttr trMap,
                   ArrayRef<int64_t> srcShape,
                   ArrayRef<StringRef> targetNames, Location loc) {
  SmallVector<StringAttr> srcNameStorage;
  SmallVector<StringRef> srcNames;
  for (int64_t i = 0, e = srcShape.size(); i < e; ++i) {
    srcNameStorage.push_back(b.getStringAttr(Twine("inv") + Twine(i)));
    srcNames.push_back(srcNameStorage.back().getValue());
  }

  BottomUpTMBuilder inv(b, srcNames, srcShape, loc);
  for (TransformAttr tAttr : trMap.getOps()) {
    switch (tAttr.getType()) {
    case TransformType::Merge: {
      SmallVector<StringRef> upperNames;
      SmallVector<uint32_t> upperDims;
      for (uint32_t d : tAttr.getLowerDims()) {
        upperNames.push_back(targetNames[d]);
        upperDims.push_back(d);
      }
      inv.unmerge(upperNames, upperDims,
                  srcNames[tAttr.getUpperDims()[0]], tAttr.getParams());
      break;
    }
    case TransformType::PassThrough:
      for (size_t k = 0; k < tAttr.getUpperDims().size(); ++k)
        inv.passThrough({targetNames[tAttr.getLowerDims()[k]]},
                        {tAttr.getLowerDims()[k]},
                        {srcNames[tAttr.getUpperDims()[k]]});
      break;
    case TransformType::AddDim:
      for (uint32_t ud : tAttr.getUpperDims())
        inv.dropDimAtIndex(srcNames[ud], 0);
      break;
    default:
      return failure();
    }
  }
  return inv.get();
}

/// Build a broadcast TransformMapAttr that expands the reduced `axis`
/// from size 1 back to `targetShape[axis]`, passing all other dims through.
static TransformMapAttr
buildBroadcastMap(OpBuilder &b, ArrayRef<StringRef> dimNames,
                  ArrayRef<int64_t> reducedShape,
                  ArrayRef<int64_t> targetShape, int64_t axis, Location loc) {
  BottomUpTMBuilder bc(b, dimNames, reducedShape, loc);
  for (int64_t i = 0, e = reducedShape.size(); i < e; ++i) {
    if (i == axis)
      bc.broadcast({static_cast<uint32_t>(i)}, {targetShape[i]});
    else
      bc.passThrough(dimNames[i]);
  }
  return bc.get();
}

/// Walk a single-use chain of TransformOps starting from `start`, collecting
/// them into `chain`.  Returns the final value (the first non-TransformOp use).
static Value collectTransformChain(Value start,
                                   SmallVectorImpl<TransformOp> &chain) {
  Value cur = start;
  while (cur.hasOneUse()) {
    auto *user = *cur.user_begin();
    if (auto tOp = dyn_cast<TransformOp>(user)) {
      chain.push_back(tOp);
      cur = tOp.getResult();
    } else {
      break;
    }
  }
  return cur;
}

struct ReduceToStoreRewritePattern : public OpRewritePattern<rock::ReduceOp> {
  using OpRewritePattern<rock::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();
    Value reduceInput = reduceOp.getIn();
    int64_t axis = reduceOp.getAxis().getSExtValue();

    auto inputType = cast<RankedTensorType>(reduceInput.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    StoreMethod storeMethod;
    switch (reduceOp.getReduceMethod()) {
    case ReduceMethod::Sum: storeMethod = StoreMethod::AtomicAdd; break;
    case ReduceMethod::Max: storeMethod = StoreMethod::AtomicMax; break;
    }

    Value reduceResult = reduceOp.getResult();
    if (reduceResult.getNumUses() != 1)
      return reduceOp.emitError("Expected exactly one use of the ReduceOp");

    // Walk from the reduce result through any intermediate TransformOps
    // (e.g. Merge flattening the output to match the function signature)
    // to the store.
    SmallVector<TransformOp> intermediateOps;
    Value storeSource = collectTransformChain(reduceResult, intermediateOps);

    if (!storeSource.hasOneUse())
      return reduceOp.emitError(
          "Expected single-use chain from reduce to store");
    auto storeOp = dyn_cast<StoreOp>(*storeSource.user_begin());
    if (!storeOp)
      return reduceOp.emitError(
          "rock.reduce result does not feed into a rock.store");

    // Canonical dimension names for the reduce output rank.
    SmallVector<StringAttr> dimNameStorage;
    SmallVector<StringRef> dimNames;
    for (int64_t i = 0; i < rank; ++i) {
      dimNameStorage.push_back(rewriter.getStringAttr(Twine("d") + Twine(i)));
      dimNames.push_back(dimNameStorage.back().getValue());
    }

    ArrayRef<int64_t> reduceOutputShape =
        cast<RankedTensorType>(reduceResult.getType()).getShape();

    rewriter.setInsertionPoint(storeOp);
    Value transformedDest = storeOp.getDest();

    // Undo intermediate transforms on the store destination so its shape
    // matches the reduce output (e.g. Unmerge a prior Merge).
    for (TransformOp tOp : llvm::reverse(intermediateOps)) {
      auto destShape =
          cast<RankedTensorType>(transformedDest.getType()).getShape();
      FailureOr<TransformMapAttr> invMap = invertTransformMap(
          rewriter, tOp.getTransform(), destShape, dimNames, loc);
      if (failed(invMap))
        return reduceOp.emitError(
            "Cannot invert intermediate transform between reduce and store");
      transformedDest =
          TransformOp::create(rewriter, loc, transformedDest, *invMap);
    }

    // Broadcast the reduced axis back to the unreduced input size.
    TransformMapAttr broadcastMap = buildBroadcastMap(
        rewriter, dimNames, reduceOutputShape, inputShape, axis, loc);
    transformedDest =
        TransformOp::create(rewriter, loc, transformedDest, broadcastMap);

    auto newStore =
        StoreOp::create(rewriter, loc, storeOp.getResult().getType(),
                        reduceInput, transformedDest, storeOp.getStoreMethod());
    if (failed(setStoreMethodAndPrefill(rewriter, newStore, storeMethod)))
      return storeOp.emitError("failed to set store method and prefill");

    rewriter.replaceOp(storeOp, newStore);
    for (TransformOp tOp : llvm::reverse(intermediateOps))
      rewriter.eraseOp(tOp);
    rewriter.eraseOp(reduceOp);
    return success();
  }
};

struct RockLowerReduce
    : public rock::impl::RockLowerReducePassBase<RockLowerReduce> {
  void runOnOperation() override;
};

} // namespace

void RockLowerReduce::runOnOperation() {
  MLIRContext *ctx = &getContext();

  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::ReduceOp>();
  target.addLegalDialect<rock::RockDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<math::MathDialect>();
  target.addLegalDialect<func::FuncDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<ReduceToStoreRewritePattern>(ctx);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}
