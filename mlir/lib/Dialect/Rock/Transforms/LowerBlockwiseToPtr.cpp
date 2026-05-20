//===- LowerBlockwiseToPtr.cpp - Lower blockwise ops to pointer form ------===//
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
// This pass converts BlockwiseLoadOp and BlockwiseStoreOp into their
// pointer-based variants (BlockwiseLoadPtrOp and BlockwiseStorePtrOp)
// by first computing pointer and mask tensors via TransformsToPtrOp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERBLOCKWISETOPTRPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-blockwise-to-ptr"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockLowerBlockwiseToPtrPass
    : public rock::impl::RockLowerBlockwiseToPtrPassBase<
          RockLowerBlockwiseToPtrPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

namespace {

static FailureOr<Value> buildStoreResultAlias(PatternRewriter &b,
                                              BlockwiseStoreOp op) {
  RankedTensorType resultType =
      cast<RankedTensorType>(op.getResult().getType());
  ArrayRef<int64_t> resultShape = resultType.getShape();

  SmallVector<TransformMapAttr> transforms;
  Value root = std::get<0>(rock::untransform(op.getDest(), transforms));
  FailureOr<BlockArgument> maybeRoot = rock::findBlockArgument(root);
  if (failed(maybeRoot))
    return failure();

  root = maybeRoot.value();
  ArrayRef<int64_t> rootShape = cast<ShapedType>(root.getType()).getShape();
  if (rootShape == resultShape)
    return root;

  SmallVector<Attribute> aliasTransforms;
  ArrayRef<int64_t> currentShape = rootShape;
  for (auto transform : llvm::reverse(transforms)) {
    if (transform.getLowerBounds().asArrayRef() != currentShape)
      return failure();

    aliasTransforms.insert(aliasTransforms.begin(), transform);
    currentShape = transform.getUpperBounds().asArrayRef();
    if (currentShape == resultShape) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(op);
      return rock::transform(b, root, b.getArrayAttr(aliasTransforms));
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// BlockwiseLoadOp lowering.
//===----------------------------------------------------------------------===//
struct BlockwiseLoadRewritePattern : public OpRewritePattern<BlockwiseLoadOp> {
  using OpRewritePattern<BlockwiseLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseLoadOp op,
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

    // Create rock.transforms_to_ptr operation (returns pointer and mask
    // tensors)
    auto transformsToPtrOp = TransformsToPtrOp::create(
        b, loc, pointerTensorType, maskTensorType, source, sourceIndices);
    Value pointerTensor = transformsToPtrOp.getPointers();
    Value maskTensor = transformsToPtrOp.getMask();

    // Create rock.blockwise_load_ptr operation (returns loaded tensor)
    auto resultType = RankedTensorType::get(shape, elementType);
    auto loadOp = BlockwiseLoadPtrOp::create(b, loc, resultType, pointerTensor,
                                             maskTensor);

    b.replaceOp(op, loadOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseStoreOp lowering.
//===----------------------------------------------------------------------===//
struct BlockwiseStoreRewritePattern
    : public OpRewritePattern<BlockwiseStoreOp> {
  using OpRewritePattern<BlockwiseStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BlockwiseStoreOp op,
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

    // Create rock.transforms_to_ptr operation (returns pointer and mask
    // tensors)
    auto transformsToPtrOp = TransformsToPtrOp::create(
        b, loc, pointerTensorType, maskTensorType, dest, extraIndices);
    Value pointerTensor = transformsToPtrOp.getPointers();
    Value maskTensor = transformsToPtrOp.getMask();

    // Create rock.blockwise_store_ptr operation (returns stored tensor)
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto storeOp = BlockwiseStorePtrOp::create(
        b, loc, resultType, pointerTensor, maskTensor, source, storeMethod);

    FailureOr<Value> destAlias = buildStoreResultAlias(b, op);
    for (OpOperand &use :
         llvm::make_early_inc_range(op.getResult().getUses())) {
      if (isa<func::ReturnOp>(use.getOwner())) {
        use.set(storeOp.getResult());
        continue;
      }
      if (failed(destAlias))
        return op.emitError("can't trace store destination to function "
                            "argument");
      use.set(destAlias.value());
    }
    b.eraseOp(op);
    return success();
  }
};

} // end anonymous namespace

void RockLowerBlockwiseToPtrPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<BlockwiseLoadOp, BlockwiseStoreOp>();
  target
      .addLegalOp<BlockwiseLoadPtrOp, BlockwiseStorePtrOp, TransformsToPtrOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<BlockwiseLoadRewritePattern, BlockwiseStoreRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
