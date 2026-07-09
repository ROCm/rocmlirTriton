//===- TransformsToPointerArith.cpp - Expand transform maps to arith ------===//
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
// This pass lowers TransformsToPtrOp by expanding transform map chains into
// arithmetic operations that compute pointer offsets and validity masks. The
// coordinate/validity expansion engine lives here (its only consumer); it is
// built on the small shared arith/broadcast helpers in PointerArithExpand.cpp,
// which the carry-based LICM path also uses to rebuild the per-iteration mask.
//
//===----------------------------------------------------------------------===//

#include "PointerArithExpand.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTRANSFORMSTOPOINTERARITHPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-transforms-to-pointer-arith"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {

struct RockTransformsToPointerArithPass
    : public rock::impl::RockTransformsToPointerArithPassBase<
          RockTransformsToPointerArithPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

namespace {

/// Tail of the TransformsToPtrOp lowering. Given the chain `root` buffer, the
/// remaining `transformVec`, the seeded `initValues` (extra indices + per-tile
/// ranges), and the result tile `shape`, expand to the linearized offset +
/// mask, prepend the base pointer, and replace `op`.
static LogicalResult lowerToPointer(PatternRewriter &b, Operation *op,
                                    Location loc, Value buffer,
                                    ArrayRef<TransformMapAttr> transformVec,
                                    ValueRange initValues,
                                    ArrayRef<int64_t> shape) {
  // After regularize-input, the root of any transform chain must be either
  // a block argument (kernel input tensor) or an arith.constant (splat).
  if (!isa<BlockArgument>(buffer) &&
      !buffer.getDefiningOp<arith::ConstantOp>()) {
    return op->emitOpError("expected transform chain root to be a block "
                           "argument or arith.constant, but got: ")
           << *buffer.getDefiningOp();
  }

  FailureOr<OffsetAndMask> expanded =
      expandCoordsToOffsetAndMask(b, loc, transformVec, initValues, shape);
  if (failed(expanded))
    return op->emitOpError("Transforms are not well formed");

  // Hoist pointer extraction to function entry to avoid redundant extractions
  // when the op is inside loops or other control flow.
  // For constant buffers (like fakeTensor used for index calculations),
  // we use a base pointer of 0 since the actual pointer value doesn't matter.
  Value baseAddr;
  {
    OpBuilder::InsertionGuard guard(b);
    bool isConstantBuffer =
        buffer.getDefiningOp<arith::ConstantOp>() != nullptr;
    if (isConstantBuffer) {
      baseAddr = arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(rock::getPtrGlueType(b.getContext()), 0));
    } else {
      auto parentFunc = op->getParentOfType<func::FuncOp>();
      b.setInsertionPointToStart(&parentFunc.front());
      baseAddr = rock::ExtractPtrOp::create(b, loc, buffer);
    }
  }
  auto splatType =
      RankedTensorType::get(shape, rock::getPtrGlueType(b.getContext()));
  Value baseAddrSplat = triton::SplatOp::create(b, loc, splatType, baseAddr);

  // baseAddr is i32 which might be too narrow in some cases.
  // This is intentional: rock-to-ttir replaces this i32 tensor with a
  // tt.ptr-based tt.addptr, so the actual address width is handled there.
  Value pointerTensor =
      arith::AddIOp::create(b, loc, baseAddrSplat, expanded->offset);

  b.replaceOp(op, {pointerTensor, expanded->mask});
  return success();
}

//===----------------------------------------------------------------------===//
// TransformsToPtrOp lowering.
//===----------------------------------------------------------------------===//
struct TransformsToPtrRewritePattern
    : public OpRewritePattern<TransformsToPtrOp> {
  using OpRewritePattern<TransformsToPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransformsToPtrOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();
    auto pointerResultType = cast<RankedTensorType>(op.getPointers().getType());
    ArrayRef<int64_t> shape = pointerResultType.getShape();

    Value source = isolateTransforms(b, op.getSource());
    auto [buffer, transforms, _] = untransform(b, source);

    // Seed the chain with the extra (scalar) indices followed by one range
    // tensor per result tile dimension.
    SmallVector<Value> initValues(op.getExtraIndices());
    for (size_t dimension = 0; dimension < shape.size(); ++dimension)
      initValues.push_back(
          makeRange(b, loc, 0, shape[dimension], shape.size(), dimension));

    SmallVector<TransformMapAttr> transformVec =
        llvm::to_vector(transforms.getAsRange<TransformMapAttr>());
    return lowerToPointer(b, op, loc, buffer, transformVec, initValues, shape);
  }
};

} // end anonymous namespace

void RockTransformsToPointerArithPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::TransformsToPtrOp>();
  // Note: We don't mark TransformOp as illegal. After TransformsToPtrOp
  // conversion, transform chains become dead code (each transform only used
  // by the next transform in the chain). These will be cleaned up by
  // Canonicalizer.
  target.addLegalDialect<rock::RockDialect, arith::ArithDialect,
                         triton::TritonDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<TransformsToPtrRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
