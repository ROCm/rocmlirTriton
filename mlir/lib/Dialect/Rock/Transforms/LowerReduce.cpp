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

    Value reduceResult = reduceOp.getResult();
    if (reduceResult.getNumUses() != 1)
      return reduceOp.emitError("Expected exactly one use of the ReduceOp");

    // Walk from reduceResult through any intermediate TransformOps to the store.
    // Earlier passes may insert transforms (e.g. Merge) between the reduce
    // output and the store to match the function signature's flat tensor shape.
    Value current = reduceResult;
    SmallVector<TransformOp> intermediateOps;
    while (current.hasOneUse()) {
      Operation *user = *current.user_begin();
      if (auto transformOp = dyn_cast<TransformOp>(user)) {
        intermediateOps.push_back(transformOp);
        current = transformOp.getResult();
      } else {
        break;
      }
    }

    if (!current.hasOneUse())
      return reduceOp.emitError(
          "Expected single-use chain from reduce to store");

    auto storeOp = dyn_cast<StoreOp>(*current.user_begin());
    if (!storeOp)
      return reduceOp.emitError(
          "rock.reduce result does not feed into a rock.store");

    Value storeDest = storeOp.getDest();
    auto reduceOutputType = cast<RankedTensorType>(reduceResult.getType());
    ArrayRef<int64_t> reduceOutputShape = reduceOutputType.getShape();

    SmallVector<StringAttr> dimNameAttrs;
    SmallVector<StringRef> dimNames;
    for (int64_t i = 0; i < rank; ++i) {
      dimNameAttrs.push_back(rewriter.getStringAttr(Twine("d") + Twine(i)));
      dimNames.push_back(dimNameAttrs.back().getValue());
    }

    rewriter.setInsertionPoint(storeOp);
    Value transformedDest = storeDest;

    // If there are intermediate transforms (e.g. Merge reshaping the reduce
    // output to match the flat function argument), construct inverse transforms
    // on the store destination to get back to the reduce output shape.
    if (!intermediateOps.empty()) {
      for (TransformOp tOp : llvm::reverse(intermediateOps)) {
        TransformMapAttr trMap = tOp.getTransform();
        auto currentType = cast<RankedTensorType>(transformedDest.getType());
        ArrayRef<int64_t> currentShape = currentType.getShape();

        SmallVector<StringAttr> curNameAttrs;
        SmallVector<StringRef> curNames;
        for (int64_t j = 0; j < (int64_t)currentShape.size(); ++j) {
          curNameAttrs.push_back(
              rewriter.getStringAttr(Twine("inv") + Twine(j)));
          curNames.push_back(curNameAttrs.back().getValue());
        }

        BottomUpTMBuilder invBuilder(rewriter, curNames, currentShape, loc);
        for (TransformAttr tAttr : trMap.getOps()) {
          switch (tAttr.getType()) {
          case TransformType::Merge: {
            StringRef lowerName = curNames[tAttr.getUpperDims()[0]];
            SmallVector<StringRef> upperNames;
            SmallVector<uint32_t> upperDims;
            for (uint32_t d : tAttr.getLowerDims()) {
              upperNames.push_back(dimNames[d]);
              upperDims.push_back(d);
            }
            invBuilder.unmerge(upperNames, upperDims, lowerName,
                               tAttr.getParams());
            break;
          }
          case TransformType::PassThrough: {
            for (size_t k = 0; k < tAttr.getUpperDims().size(); ++k) {
              invBuilder.passThrough(
                  ArrayRef<StringRef>{dimNames[tAttr.getLowerDims()[k]]},
                  ArrayRef<uint32_t>{tAttr.getLowerDims()[k]},
                  ArrayRef<StringRef>{curNames[tAttr.getUpperDims()[k]]});
            }
            break;
          }
          case TransformType::AddDim: {
            for (uint32_t ud : tAttr.getUpperDims())
              invBuilder.dropDimAtIndex(curNames[ud], 0);
            break;
          }
          default:
            return reduceOp.emitError("Cannot invert intermediate transform "
                                      "between reduce and store");
          }
        }
        TransformMapAttr invMap = invBuilder.get();
        transformedDest =
            TransformOp::create(rewriter, loc, transformedDest, invMap);
      }
    }

    // Build broadcast from the reduce output shape to the unreduced input shape.
    BottomUpTMBuilder bcBuilder(rewriter, dimNames, reduceOutputShape, loc);
    for (int64_t i = 0; i < rank; ++i) {
      if (i == axis)
        bcBuilder.broadcast({static_cast<uint32_t>(i)}, {inputShape[i]});
      else
        bcBuilder.passThrough(dimNames[i]);
    }
    TransformMapAttr broadcastMap = bcBuilder.get();
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
