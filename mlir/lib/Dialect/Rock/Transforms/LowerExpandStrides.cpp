//===------------------- LowerExpandStrides.cpp ---------------------------===//
//
// Copyright 2026 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers rock.expand_strides by moving the stride expansion from
// the source chain of rock.store to the dest chain, using Unmerge + Slice
// transforms.
//
// Before:
//   %expanded = rock.expand_strides %input : 4x24x24 -> 4x48x24
//   %flat = rock.transform %expanded by <Merge> : 4x48x24 -> 4608
//   rock.store %flat to %output_arg : 4608 -> 4608
//
// After:
//   %unmerged = rock.transform %output_arg by <Unmerge> : 4608 -> 4x48x24
//   %sliced = rock.transform %unmerged by <Slice> : 4x48x24 -> 4x24x24
//   rock.store %input to %sliced : 4x24x24 -> 4608
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKEXPANDSTRIDESLOWERINGPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

// Trace forward from a value through rock.transform ops to find a rock.store
// that uses the (possibly transformed) value as its source.
// Returns the store and the collected transform attributes between the
// starting value and the store source.
static FailureOr<StoreOp>
traceToStore(Value start, SmallVectorImpl<TransformMapAttr> &transforms) {
  Value current = start;
  while (true) {
    if (!current.hasOneUse())
      return failure();
    Operation *user = *current.getUsers().begin();
    if (auto storeOp = dyn_cast<StoreOp>(user)) {
      if (storeOp.getSource() == current)
        return storeOp;
      return failure();
    }
    if (auto transformOp = dyn_cast<TransformOp>(user)) {
      transforms.push_back(transformOp.getTransform());
      current = transformOp.getOutput();
      continue;
    }
    return failure();
  }
}

struct ExpandStridesLoweringPattern
    : public OpRewritePattern<rock::ExpandStridesOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ExpandStridesOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value expandedResult = op.getResult();

    auto inputType = cast<RankedTensorType>(input.getType());
    auto expandedType = cast<RankedTensorType>(expandedResult.getType());

    SmallVector<TransformMapAttr> forwardTransforms;
    auto maybeStore = traceToStore(expandedResult, forwardTransforms);
    if (failed(maybeStore))
      return rewriter.notifyMatchFailure(
          op, "expand_strides result doesn't lead to a rock.store");

    StoreOp storeOp = *maybeStore;
    Value storeDest = storeOp.getDest();
    auto destType = cast<RankedTensorType>(storeDest.getType());

    ArrayRef<int64_t> expandedShape = expandedType.getShape();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputType.getRank();

    // Build dest-side transforms: storeDest -> Unmerge -> expanded -> Slice -> input
    Value newDest = storeDest;

    // Step 1: If the dest is flat (1D) and the expanded shape is N-D,
    // create an Unmerge to reshape dest from flat to expanded shape.
    if (destType.getShape() != expandedShape) {
      if (destType.getRank() != 1)
        return rewriter.notifyMatchFailure(
            op, "expected 1-D store dest for multi-dim expanded shape");

      SmallVector<StringRef> upperNames;
      SmallVector<uint32_t> upperDims;
      SmallVector<int64_t> lengths;
      for (int64_t i = 0; i < rank; ++i) {
        upperNames.push_back(
            rewriter.getStringAttr("dim" + std::to_string(i)).getValue());
        upperDims.push_back(i);
        lengths.push_back(expandedShape[i]);
      }

      BottomUpTMBuilder unmergeBuilder(rewriter, {"flat"},
                                       destType.getShape(), loc);
      unmergeBuilder.unmerge(upperNames, upperDims, "flat", lengths);
      TransformMapAttr unmergeAttr = unmergeBuilder.get();
      newDest = TransformOp::create(rewriter, loc, newDest, unmergeAttr);
    }

    // Step 2: Slice from expanded shape to input shape on dimensions
    // that were expanded.
    SmallVector<StringRef> dimNames;
    for (int64_t i = 0; i < rank; ++i)
      dimNames.push_back(
          rewriter.getStringAttr("dim" + std::to_string(i)).getValue());

    BottomUpTMBuilder sliceBuilder(rewriter, dimNames, expandedShape, loc);
    for (int64_t i = 0; i < rank; ++i) {
      StringRef dimName = dimNames[i];
      if (inputShape[i] == expandedShape[i]) {
        sliceBuilder.passThrough(dimName);
      } else {
        StringRef sliceName =
            rewriter.getStringAttr("slice" + std::to_string(i)).getValue();
        sliceBuilder.slice({sliceName}, {dimName}, {0}, {inputShape[i]});
      }
    }
    TransformMapAttr sliceAttr = sliceBuilder.get();
    Value slicedDest =
        TransformOp::create(rewriter, loc, newDest, sliceAttr);

    // Create the new store: source = original input, dest = sliced dest
    auto newStore = StoreOp::create(rewriter, loc, storeOp.getResult().getType(),
                                    input, slicedDest, storeOp.getStoreMethod());

    rewriter.replaceOp(storeOp, newStore.getResult());

    // Erase the now-dead forward transform chain and expand_strides
    Value current = expandedResult;
    SmallVector<Operation *> toErase;
    for (size_t i = 0; i < forwardTransforms.size(); ++i) {
      for (Operation *user : current.getUsers()) {
        if (auto trOp = dyn_cast<TransformOp>(user)) {
          current = trOp.getOutput();
          toErase.push_back(trOp);
          break;
        }
      }
    }
    for (Operation *deadOp : llvm::reverse(toErase))
      rewriter.eraseOp(deadOp);
    rewriter.eraseOp(op);

    return success();
  }
};

struct RockExpandStridesLoweringPass
    : public rock::impl::RockExpandStridesLoweringPassBase<
          RockExpandStridesLoweringPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ExpandStridesLoweringPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
