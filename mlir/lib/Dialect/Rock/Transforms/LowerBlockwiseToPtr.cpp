//===- LowerBlockwiseToPtr.cpp - Lower blockwise ops to pointer form ------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

    auto transformsToPtrOp =
        TransformsToPtrOp::create(b, loc, source, sourceIndices);
    Value pointerTensor = transformsToPtrOp.getPointers();
    Value maskTensor = transformsToPtrOp.getMask();

    auto loadOp = BlockwiseLoadPtrOp::create(b, loc, op.getResult().getType(),
                                             pointerTensor, maskTensor,
                                             op.getCacheModifier());

    b.replaceOp(op, loadOp.getResult());
    return success();
  }
};

static void lowerBlockwiseStore(BlockwiseStoreOp op) {
  OpBuilder b(op);
  Location loc = op.getLoc();

  Value source = op.getSource();
  Value dest = op.getDest();
  auto extraIndices = op.getExtraIndices();
  auto storeMethod = op.getStoreMethod();

  op.getResult().replaceAllUsesWith(op.getResultAliasOrDest());

  auto transformsToPtrOp =
      TransformsToPtrOp::create(b, loc, dest, extraIndices);
  Value pointerTensor = transformsToPtrOp.getPointers();
  Value maskTensor = transformsToPtrOp.getMask();
  BlockwiseStorePtrOp::create(b, loc, pointerTensor, maskTensor, source,
                              storeMethod);
  op.erase();
}

//===----------------------------------------------------------------------===//
// ReturnOpRewritePattern - Update return ops to return nothing and update
// the parent function signature to return void
//===----------------------------------------------------------------------===//
struct ReturnOpRewritePattern : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    // Only convert return ops that have operands
    if (returnOp.getOperands().empty())
      return failure();

    // Update the parent function's signature to return void
    auto funcOp = returnOp->getParentOfType<func::FuncOp>();
    if (funcOp && funcOp.getFunctionType().getNumResults() > 0) {
      FunctionType newFuncType = FunctionType::get(
          rewriter.getContext(), funcOp.getFunctionType().getInputs(),
          /*results=*/{});
      rewriter.modifyOpInPlace(funcOp, [&]() {
        funcOp.setFunctionType(newFuncType);
        funcOp.setAllResultAttrs(ArrayRef<DictionaryAttr>{});
      });
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);
    return success();
  }
};

} // end anonymous namespace

void RockLowerBlockwiseToPtrPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  // Only operate on kernel functions; non-kernel funcs may legitimately return
  // tensors and must not have their signatures rewritten to void.
  auto funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // Step 1: rewrite returns to void before lowering stores. This must happen
  // first because BlockwiseStorePtrOp has no result, so any remaining use of
  // a BlockwiseStoreOp's result (typically by func.return) would block
  // erasure during the conversion below.
  {
    ConversionTarget target(*ctx);
    target.addLegalOp<func::FuncOp>();
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [](func::ReturnOp op) { return op.getOperands().empty(); });

    RewritePatternSet patterns(ctx);
    patterns.add<ReturnOpRewritePattern>(ctx);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  // Step 2: lower blockwise_store to its pointer-based variant in reverse
  // program order. Store results can be threaded through later stores via
  // resultAlias, so rewrite users only after their consuming stores are gone.
  SmallVector<BlockwiseStoreOp> storeOps;
  funcOp.walk([&](BlockwiseStoreOp storeOp) { storeOps.push_back(storeOp); });
  for (BlockwiseStoreOp storeOp : llvm::reverse(storeOps))
    lowerBlockwiseStore(storeOp);

  // Step 3: lower blockwise_load to its pointer-based variant.
  ConversionTarget target(*ctx);
  target.addIllegalOp<BlockwiseLoadOp, BlockwiseStoreOp>();
  target
      .addLegalOp<BlockwiseLoadPtrOp, BlockwiseStorePtrOp, TransformsToPtrOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<BlockwiseLoadRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
