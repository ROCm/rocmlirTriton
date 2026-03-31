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

/// Walk a single-use chain of TransformOps from `start` to a StoreOp,
/// collecting intermediate TransformOps into `chain`.  Returns the StoreOp,
/// or failure if the chain is broken by a multi-use value or an unexpected op.
static FailureOr<StoreOp>
collectTransformChain(Value start, SmallVectorImpl<TransformOp> &chain) {
  Value cur = start;
  while (cur.hasOneUse()) {
    Operation *user = *cur.user_begin();
    if (auto tOp = dyn_cast<TransformOp>(user)) {
      chain.push_back(tOp);
      cur = tOp.getResult();
    } else if (auto store = dyn_cast<StoreOp>(user)) {
      return store;
    } else {
      user->emitError("unexpected op between rock.reduce and rock.store: ")
          << user->getName();
      return failure();
    }
  }

  emitError(cur.getLoc(), "expected single-use chain from reduce to store");
  return failure();
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
    case ReduceMethod::Sum:
      storeMethod = StoreMethod::AtomicAdd;
      break;
    case ReduceMethod::Max:
      storeMethod = StoreMethod::AtomicMax;
      break;
    }

    Value reduceResult = reduceOp.getResult();
    if (reduceResult.getNumUses() != 1)
      return reduceOp.emitError("Expected exactly one use of the ReduceOp");

    // Walk from the reduce result through any intermediate TransformOps
    // (e.g. Merge flattening the output to match the function signature)
    // to the store.
    SmallVector<TransformOp> intermediateOps;
    FailureOr<StoreOp> maybeStore =
        collectTransformChain(reduceResult, intermediateOps);
    if (failed(maybeStore))
      return failure();
    StoreOp storeOp = *maybeStore;

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
      TransformMapAttr invMap =
          rock::invertTransformMap(rewriter, tOp.getTransform(), loc);
      if (!invMap)
        return reduceOp.emitError(
            "Cannot invert intermediate transform between reduce and store");
      transformedDest =
          TransformOp::create(rewriter, loc, transformedDest, invMap);
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
