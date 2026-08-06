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
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

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

/// Select only reductions fully covered by the initial blockwise lowering.
/// Every rejected form continues through the legacy atomic-store rewrite.
static bool isBlockwiseReductionCandidate(ReduceOp reduceOp,
                                          ReductionStorePath &path,
                                          OpBuilder &builder) {
  if (reduceOp.getReduceMethod() != ReduceMethod::Sum)
    return false;

  auto inputType = dyn_cast<RankedTensorType>(reduceOp.getIn().getType());
  auto resultType = dyn_cast<RankedTensorType>(reduceOp.getResult().getType());
  if (!inputType || !resultType || !inputType.hasStaticShape() ||
      inputType.getRank() != 3 || resultType.getRank() != 3 ||
      reduceOp.getAxis().getSExtValue() != 2)
    return false;

  // Keep the initial contract intentionally narrow: one invertible output view
  // to a rank-one store and at least one input view from GEMM space.
  if (path.postReduceTransforms.size() != 1 || path.preReduceTransforms.empty())
    return false;
  auto storeType =
      dyn_cast<RankedTensorType>(path.storeOp.getSource().getType());
  if (!storeType || storeType.getRank() != 1)
    return false;
  auto toTransformStack = [&](ArrayRef<TransformOp> transforms) {
    return builder.getArrayAttr(
        llvm::map_to_vector(transforms, [](TransformOp transformOp) {
          return static_cast<Attribute>(transformOp.getTransform());
        }));
  };
  if (!invertTransforms(builder, reduceOp.getLoc(),
                        toTransformStack(path.preReduceTransforms)) ||
      !invertTransforms(builder, reduceOp.getLoc(),
                        toTransformStack(path.postReduceTransforms)))
    return false;

  // Ambiguous multi-root fusions and direct rank-two GEMM consumers need a
  // richer normalization plan. Route them through the legacy path.
  if (path.fusionRoots.size() != 1 || path.tileSource.getDefiningOp<GemmOp>())
    return false;
  auto gemmOp = dyn_cast<GemmOp>(path.fusionRoots.front());
  if (!gemmOp || !gemmOp.getParams())
    return false;
  auto params = *gemmOp.getParams();
  if (gemmOp.getOTransposed() || params.getSplitKFactor() != 1 ||
      !llvm::isPowerOf2_64(params.getMPerBlock()) ||
      !llvm::isPowerOf2_64(params.getNPerBlock()))
    return false;

  GemmSize size = gemmOp.getGemmSize();
  return size.m % params.getMPerBlock() == 0 &&
         size.n % params.getNPerBlock() == 0;
}

struct ReduceToStoreRewritePattern : public OpRewritePattern<rock::ReduceOp> {
  using OpRewritePattern<rock::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getBlockwise())
      return failure();

    Location loc = reduceOp.getLoc();
    Value reduceInput = reduceOp.getIn();

    auto inputType = cast<RankedTensorType>(reduceInput.getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();

    StoreMethod storeMethod;
    switch (reduceOp.getReduceMethod()) {
    case ReduceMethod::Sum:
      storeMethod = StoreMethod::AtomicAdd;
      break;
    case ReduceMethod::Max:
      storeMethod = StoreMethod::AtomicMax;
      break;
    }

    FailureOr<ReductionStorePath> maybePath = getReductionStorePath(reduceOp);
    if (failed(maybePath))
      return reduceOp.emitError(
          "could not find rock.store along single-use chain from reduce");
    ReductionStorePath path = std::move(*maybePath);
    StoreOp storeOp = path.storeOp;

    rewriter.setInsertionPoint(storeOp);
    Value transformedDest = storeOp.getDest();

    // Undo intermediate transforms on the store destination so its shape
    // matches the reduce output (e.g. Unmerge a prior Merge).
    if (!path.postReduceTransforms.empty()) {
      ArrayAttr stack = rewriter.getArrayAttr(llvm::map_to_vector(
          path.postReduceTransforms, [](TransformOp transformOp) {
            return static_cast<Attribute>(transformOp.getTransform());
          }));
      ArrayAttr inverted = invertTransforms(rewriter, loc, stack);
      if (!inverted)
        return reduceOp.emitError(
            "Cannot invert intermediate transform between reduce and store");
      transformedDest = rock::transform(rewriter, transformedDest, inverted);
    }

    // Broadcast the reduced axis back to the unreduced input size.
    transformedDest =
        rock::insertBroadcast(rewriter, loc, transformedDest, inputShape);

    // Create a new store with the old store method, then update it
    // via setStoreMethodAndPrefill which sets the atomic method + prefill.
    auto newStore = StoreOp::create(
        rewriter, loc, storeOp.getResult().getType(), reduceInput,
        transformedDest, storeOp.getResultAlias(), storeOp.getStoreMethod());
    if (failed(setStoreMethodAndPrefill(rewriter, newStore, storeMethod)))
      return storeOp.emitError("failed to set store method and prefill");

    // The post-reduction transforms and reduceOp form a single-use chain that
    // is now dead. All erasures are deferred by the ConversionPatternRewriter
    // and committed together, so mark every op in the chain for removal —
    // otherwise the framework inserts unrealized_conversion_casts for
    // the still-live intermediate references.
    rewriter.replaceOp(storeOp, newStore);
    for (TransformOp tOp : path.postReduceTransforms)
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
  OpBuilder builder(ctx);

  // Selection is performed exactly once. Downstream passes consume the marker
  // instead of independently guessing why a ReduceOp survived this pass.
  getOperation().walk([&](ReduceOp reduceOp) {
    reduceOp.removeBlockwiseAttr();
    FailureOr<ReductionStorePath> path = getReductionStorePath(reduceOp);
    if (succeeded(path) &&
        isBlockwiseReductionCandidate(reduceOp, *path, builder))
      reduceOp.setBlockwiseAttr(builder.getUnitAttr());
  });

  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<rock::ReduceOp>(
      [](rock::ReduceOp op) { return static_cast<bool>(op.getBlockwise()); });
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
