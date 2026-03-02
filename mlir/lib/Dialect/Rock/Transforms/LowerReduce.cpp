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
//   %reduced = rock.reduce sum %fused into %out {axis = 1}
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

struct ReduceToStoreRewritePattern : public OpRewritePattern<rock::ReduceOp> {
  using OpRewritePattern<rock::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    Location loc = reduceOp.getLoc();
    Value reduceInput = reduceOp.getIn();
    int64_t axis = reduceOp.getAxis().getSExtValue();
    ReduceMethod method = reduceOp.getReduceMethod();

    auto inputType = cast<RankedTensorType>(reduceInput.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputShape.size();

    StoreMethod storeMethod;
    switch (method) {
    case ReduceMethod::Sum:
      storeMethod = StoreMethod::AtomicAdd;
      break;
    case ReduceMethod::Max:
      storeMethod = StoreMethod::AtomicMax;
      break;
    }

    // Find the rock.store that directly consumes the reduce result.
    Value reduceResult = reduceOp.getResult();
    SmallVector<StoreOp> storeOps;
    for (OpOperand &use : reduceResult.getUses()) {
      if (auto store = dyn_cast<StoreOp>(use.getOwner())) {
        if (use.getOperandNumber() == 0) {
          storeOps.push_back(store);
        }
      }
    }

    if (storeOps.empty())
      return reduceOp.emitError(
          "rock.reduce result does not feed into a rock.store");

    if (storeOps.size() != 1)
      return reduceOp.emitError(
          "Expected exactly one rock.store to consume the reduce result");

    StoreOp storeOp = storeOps[0];

    // Build a Broadcast transform on the store destination.
    // The destination has the reduced shape (axis dim = 1).
    // We broadcast that axis to match the unreduced input shape.
    Value storeDest = storeOp.getDest();
    auto destType = cast<RankedTensorType>(storeDest.getType());
    ArrayRef<int64_t> destShape = destType.getShape();

    SmallVector<StringAttr> dimNameAttrs;
    SmallVector<StringRef> dimNames;
    for (int64_t i = 0; i < rank; ++i) {
      dimNameAttrs.push_back(rewriter.getStringAttr(Twine("d") + Twine(i)));
      dimNames.push_back(dimNameAttrs.back().getValue());
    }

    BottomUpTMBuilder tmBuilder(rewriter, dimNames, destShape, loc);
    for (int64_t i = 0; i < rank; ++i) {
      if (i == axis) {
        tmBuilder.broadcast({static_cast<uint32_t>(i)}, {inputShape[i]});
      } else {
        tmBuilder.passThrough(dimNames[i]);
      }
    }
    TransformMapAttr broadcastMap = tmBuilder.get();

    rewriter.setInsertionPoint(storeOp);
    ArrayAttr transformsArr = rewriter.getArrayAttr({broadcastMap});
    Value transformedDest = rock::transform(rewriter, storeDest, transformsArr);

    // Create a new store with the old store method, then update it
    // via setStoreMethodAndPrefill which sets the atomic method + prefill.
    auto newStore =
        StoreOp::create(rewriter, loc, storeOp.getResult().getType(),
                        reduceInput, transformedDest, storeOp.getStoreMethod());

    if (failed(setStoreMethodAndPrefill(rewriter, newStore, storeMethod)))
      return storeOp.emitError("failed to set store method and prefill");

    rewriter.replaceOp(storeOp, newStore);
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
