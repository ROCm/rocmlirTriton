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
// the source chain of rock.store to the dest chain, using Slice (and
// optionally Unmerge) transforms.
//
// Case 1: expand_strides followed by Merge then store (flat dest)
//
//   Before:
//     %expanded = rock.expand_strides %input : 4x24x24 -> 4x48x24
//     %flat = rock.transform %expanded by <Merge> : 4x48x24 -> 4608
//     rock.store %flat to %output_arg : 4608 -> 4608
//
//   After:
//     %unmerged = rock.transform %output_arg by <Unmerge> : 4608 -> 4x48x24
//     %sliced = rock.transform %unmerged by <Slice> : 4x48x24 -> 4x24x24
//     rock.store %input to %sliced : 4x24x24 -> 4608
//
// Case 2: expand_strides directly feeds store (dest already has expanded shape)
//
//   Before:
//     %expanded = rock.expand_strides %input : 4x24x24 -> 4x48x24
//     rock.store %expanded to %output_arg : 4x48x24 -> 4x48x24
//
//   After:
//     %sliced = rock.transform %output_arg by <Slice> : 4x48x24 -> 4x24x24
//     rock.store %input to %sliced : 4x24x24 -> 4x48x24
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
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

    auto maybeStores = traceRootOutputToStoreOps(expandedResult);
    if (failed(maybeStores) || maybeStores->size() != 1)
      return rewriter.notifyMatchFailure(
          op, "expand_strides result doesn't lead to exactly one rock.store");

    StoreOp storeOp = *maybeStores->begin();

    // Collect intermediate TransformOps between expand_strides and the store.
    SmallVector<TransformOp> forwardChainOps;
    Value cur = expandedResult;
    while (cur != storeOp.getSource()) {
      if (!cur.hasOneUse())
        return rewriter.notifyMatchFailure(
            op, "non-single-use value in forward chain");
      Operation *user = *cur.getUsers().begin();
      auto trOp = dyn_cast<TransformOp>(user);
      if (!trOp)
        return rewriter.notifyMatchFailure(
            op, "non-transform op between expand_strides and store");
      forwardChainOps.push_back(trOp);
      cur = trOp.getOutput();
    }

    // Validate that the forward chain is one of the two supported forms:
    //   (a) empty, expand_strides feeds store directly, or
    //   (b) a single TransformOp whose only transform is a Merge
    //       (i.e., a canonical flatten to 1-D).
    // Any other chain (transpose, slice, multi-step reassociation, etc.)
    // would be silently miscompiled by the hard-coded unmerge+slice below.
    if (!forwardChainOps.empty()) {
      if (forwardChainOps.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "expected at most one merge transform between "
                "expand_strides and store");

      TransformMapAttr fwdMap = forwardChainOps[0].getTransform();
      ArrayRef<TransformAttr> fwdOps = fwdMap.getOps();
      if (fwdOps.size() != 1 ||
          fwdOps[0].getType() != TransformType::Merge)
        return rewriter.notifyMatchFailure(
            op, "forward chain transform is not a single Merge; "
                "cannot safely rewrite");
    }

    Value storeDest = storeOp.getDest();
    auto destType = cast<RankedTensorType>(storeDest.getType());

    ArrayRef<int64_t> expandedShape = expandedType.getShape();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t rank = inputType.getRank();

    // Build dest-side transforms: storeDest -> Unmerge -> expanded -> Slice ->
    // input
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

      BottomUpTMBuilder unmergeBuilder(rewriter, {"flat"}, destType.getShape(),
                                       loc);
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
    Value slicedDest = TransformOp::create(rewriter, loc, newDest, sliceAttr);

    // Create the new store: source = original input, dest = sliced dest
    auto newStore =
        StoreOp::create(rewriter, loc, storeOp.getResult().getType(), input,
                        slicedDest, storeOp.getStoreMethod());

    rewriter.replaceOp(storeOp, newStore.getResult());

    // Erase the now-dead forward transform chain and expand_strides
    for (auto trOp : llvm::reverse(forwardChainOps))
      rewriter.eraseOp(trOp);
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
