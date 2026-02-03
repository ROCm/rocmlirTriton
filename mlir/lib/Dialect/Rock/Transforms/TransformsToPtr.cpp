//===- TransformsToPtr.cpp - Lower blockwise ops to pointer form ----------===//
//
// Copyright 2026 The MLIR Authors.
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
//===----------------------------------------------------------------------===//
//
// This pass converts BlockwiseLoadTileOp and BlockwiseStoreTileOp into their
// pointer-based variants (BlockwiseLoadTilePtrOp and BlockwiseStoreTilePtrOp)
// by first computing pointer and mask tensors via TransformsToPtrOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTRANSFORMSTOPTRPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-transforms-to-ptr"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockTransformsToPtrPass
    : public rock::impl::RockTransformsToPtrPassBase<
          RockTransformsToPtrPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

namespace {

//===----------------------------------------------------------------------===//
// BlockwiseLoadTileOp lowering.
//===----------------------------------------------------------------------===//
struct BlockwiseLoadTileRewritePattern
    : public OpRewritePattern<BlockwiseLoadTileOp> {
  using OpRewritePattern<BlockwiseLoadTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseLoadTileOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value source = op.getSource();
    auto sourceIndices = op.getSourceIndices();

    // Get the shape from the result type
    auto resultTensorType = cast<RankedTensorType>(op.getResult().getType());
    auto shape = resultTensorType.getShape();
    Type elementType = resultTensorType.getElementType();

    // Create pointer tensor type (i32) and mask tensor type (i1)
    auto pointerTensorType = RankedTensorType::get(shape, b.getI32Type());
    auto maskTensorType = RankedTensorType::get(shape, b.getI1Type());

    // Create rock.transforms_to_ptr operation (returns pointer and mask tensors)
    auto transformsToPtrOp = TransformsToPtrOp::create(
        b, loc, pointerTensorType, maskTensorType, source, sourceIndices);
    Value pointerTensor = transformsToPtrOp.getPointers();
    Value maskTensor = transformsToPtrOp.getMask();

    // Create rock.blockwise_load_tile_ptr operation (returns loaded tensor)
    auto resultType = RankedTensorType::get(shape, elementType);
    auto loadOp = BlockwiseLoadTilePtrOp::create(
        b, loc, resultType, pointerTensor, maskTensor);

    b.replaceOp(op, loadOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseStoreTileOp lowering.
//===----------------------------------------------------------------------===//
struct BlockwiseStoreTileRewritePattern
    : public OpRewritePattern<BlockwiseStoreTileOp> {
  using OpRewritePattern<BlockwiseStoreTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseStoreTileOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value source = op.getSource();
    Value dest = op.getDest();
    auto extraIndices = op.getExtraIndices();
    auto storeMethod = op.getStoreMethod();

    // Get the shape from the source tensor
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto shape = sourceType.getShape();

    // Create pointer tensor type (i32) and mask tensor type (i1)
    auto pointerTensorType = RankedTensorType::get(shape, b.getI32Type());
    auto maskTensorType = RankedTensorType::get(shape, b.getI1Type());

    // Create rock.transforms_to_ptr operation (returns pointer and mask tensors)
    auto transformsToPtrOp = TransformsToPtrOp::create(
        b, loc, pointerTensorType, maskTensorType, dest, extraIndices);
    Value pointerTensor = transformsToPtrOp.getPointers();
    Value maskTensor = transformsToPtrOp.getMask();

    // Create rock.blockwise_store_tile_ptr operation (returns stored tensor)
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto storeOp = BlockwiseStoreTilePtrOp::create(
        b, loc, resultType, pointerTensor, maskTensor, source, storeMethod);

    b.replaceOp(op, storeOp.getResult());
    return success();
  }
};

} // end anonymous namespace

void RockTransformsToPtrPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<BlockwiseLoadTileOp, BlockwiseStoreTileOp>();
  target.addLegalOp<BlockwiseLoadTilePtrOp, BlockwiseStoreTilePtrOp,
                    TransformsToPtrOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<BlockwiseLoadTileRewritePattern, BlockwiseStoreTileRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
