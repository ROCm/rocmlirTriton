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
#include "llvm/Support/MathExtras.h"

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
/// remaining `transformVec`, the original `source` view, the seeded
/// `initValues` (extra indices + per-tile ranges), the result tile `shape`, and
/// the offset element type `indexType` (i32 or i64), expand to the linearized
/// offset + mask, prepend the base pointer, and replace `op`.
static LogicalResult lowerToPointer(PatternRewriter &b, Operation *op,
                                    Location loc, Value buffer,
                                    Value source, size_t numExtraIndices,
                                    ArrayRef<TransformMapAttr> transformVec,
                                    ValueRange initValues,
                                    ArrayRef<int64_t> shape, Type indexType) {
  // After regularize-input, the root of any transform chain must be either
  // a block argument (kernel input tensor) or an arith.constant (splat).
  if (!isa<BlockArgument>(buffer) &&
      !buffer.getDefiningOp<arith::ConstantOp>()) {
    return op->emitOpError("expected transform chain root to be a block "
                           "argument or arith.constant, but got: ")
           << *buffer.getDefiningOp();
  }

  FailureOr<OffsetAndMask> expanded = expandCoordsToOffsetAndMask(
      b, loc, transformVec, initValues, shape, indexType);
  if (failed(expanded))
    return op->emitOpError("Transforms are not well formed");

  // Hoist pointer extraction to function entry to avoid redundant extractions
  // when the op is inside loops or other control flow.
  // For constant buffers (like fakeTensor used for index calculations),
  // we use a base pointer of 0 since the actual pointer value doesn't matter.
  Value baseAddr;
  bool isConstantBuffer =
      buffer.getDefiningOp<arith::ConstantOp>() != nullptr;
  {
    OpBuilder::InsertionGuard guard(b);
    if (isConstantBuffer) {
      baseAddr =
          arith::ConstantOp::create(b, loc, b.getIntegerAttr(indexType, 0));
    } else {
      auto parentFunc = op->getParentOfType<func::FuncOp>();
      b.setInsertionPointToStart(&parentFunc.front());
      // The base pointer placeholder shares the offset width so the base+offset
      // add type-checks. RockTensorToTritonPtr later discards this and
      // re-splats the real !tt.ptr, so the width is irrelevant to the final
      // address.
      baseAddr = rock::ExtractPtrOp::create(b, loc, indexType, buffer);
    }
  }
  auto splatType = RankedTensorType::get(shape, indexType);
  Value baseAddrSplat = triton::SplatOp::create(b, loc, splatType, baseAddr);

  // baseAddrSplat and the offset share indexType, so the add type-checks
  // directly. Its value is a placeholder: rock-to-ttir replaces this splat with
  // a genuine !tt.ptr tensor and turns this add into a tt.addptr, so the actual
  // address width is handled there.
  Value pointerTensor =
      arith::AddIOp::create(b, loc, baseAddrSplat, expanded->offset);

  // Attach a vectorization hint so the global load gets widened. Triton's
  // AxisInfoAnalysis can't see through the flattened im2col address math
  // (divui/remui) and would scalarize the load to contiguity=1. Here the
  // transform chain is still intact, so getMaxVectorization can prove the
  // contiguous run length per tile dim (capped at 128 bits, i.e. 4 for f32).
  // We stamp it as tt.contiguity/tt.divisibility on the pointer op (which
  // rock-tensor-to-triton-ptr turns into tt.addptr); the coalescer then
  // widens the load to a single 128-bit buffer load. LDS staging keeps this
  // decoupled from the MFMA operand layout, so correctness is unaffected.
  //
  // Divisibility is vecLen*elemBytes bytes (the widened load only needs
  // element alignment). Constant index-calc buffers never load, so skip them.
  if (!isConstantBuffer) {
    auto sourceType = cast<ShapedType>(source.getType());
    // `source` is a higher-rank view: leading `numExtraIndices` dims are the
    // block coordinates fixed by extra indices; the trailing `shape.size()`
    // dims are the per-thread tile that becomes the pointer tensor. Map tile
    // dim `d` to source dim `numExtraIndices + d`.
    if (sourceType.getRank() ==
        static_cast<int64_t>(numExtraIndices + shape.size())) {
      int64_t elemBytes =
          llvm::divideCeil(sourceType.getElementTypeBitWidth(), 8);
      SmallVector<int32_t> contigPerDim(shape.size(), 1);
      SmallVector<int32_t> divPerDim(shape.size(), 1);
      bool haveHint = false;
      for (uint32_t d = 0; d < shape.size(); ++d) {
        int64_t vecLen =
            getMaxVectorization(source,
                                static_cast<uint32_t>(numExtraIndices + d))
                .max;
        contigPerDim[d] = static_cast<int32_t>(vecLen);
        divPerDim[d] = static_cast<int32_t>(vecLen * elemBytes);
        if (vecLen > 1)
          haveHint = true;
      }
      if (haveHint) {
        auto hintTy = RankedTensorType::get(
            {static_cast<int64_t>(shape.size())}, b.getI32Type());
        pointerTensor.getDefiningOp()->setDiscardableAttr(
            "tt.contiguity", DenseIntElementsAttr::get(hintTy, contigPerDim));
        pointerTensor.getDefiningOp()->setDiscardableAttr(
            "tt.divisibility", DenseIntElementsAttr::get(hintTy, divPerDim));
      }
    }
  }

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
    // The offset element width (i32 or i64) is decided by
    // TransformsToPtrOp::inferReturnTypes based on whether the transform chain
    // requires 64-bit indexing. All coordinate/offset arithmetic below is
    // produced in this width so it cannot overflow before reaching tt.addptr.
    Type indexElemType = pointerResultType.getElementType();

    Value source = isolateTransforms(b, op.getSource());
    auto [buffer, transforms, isBig] = untransform(b, source);

    // Defensive check: the op verifier already rejects a result type that
    // disagrees with inferReturnTypes, but guard against that and this pass
    // diverging so we never emit i32 offsets for a chain needing 64-bit
    // indexing.
    if (isBig != indexElemType.isInteger(64)) {
      return op.emitOpError("offset element type ")
             << indexElemType
             << " disagrees with the transform chain's 64-bit indexing "
                "requirement (expected "
             << (isBig ? "i64" : "i32") << ")";
    }

    // Seed the chain with the extra (scalar) indices followed by one range
    // tensor per result tile dimension. The extra indices arrive as i32 and are
    // widened to the index width so they compose with the make_range
    // coordinates.
    SmallVector<Value> initValues;
    for (Value extraIndex : op.getExtraIndices()) {
      if (extraIndex.getType() != indexElemType)
        extraIndex = arith::ExtSIOp::create(b, loc, indexElemType, extraIndex);
      initValues.push_back(extraIndex);
    }
    for (size_t dimension = 0; dimension < shape.size(); ++dimension)
      initValues.push_back(makeRange(b, loc, 0, shape[dimension], shape.size(),
                                     dimension, indexElemType));

    SmallVector<TransformMapAttr> transformVec =
        llvm::to_vector(transforms.getAsRange<TransformMapAttr>());
    return lowerToPointer(b, op, loc, buffer, source,
                          op.getExtraIndices().size(), transformVec, initValues,
                          shape, indexElemType);
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
