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

#include "llvm/ADT/DenseSet.h"
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

static GemmOp findUpstreamGemm(Value value, DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return nullptr;
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;
  if (auto gemmOp = dyn_cast<GemmOp>(defOp))
    return gemmOp;
  if (auto transformOp = dyn_cast<TransformOp>(defOp))
    return findUpstreamGemm(transformOp.getInput(), visited);
  if (!rock::isFusionOp(defOp))
    return nullptr;
  for (Value operand : defOp->getOperands())
    if (GemmOp gemmOp = findUpstreamGemm(operand, visited))
      return gemmOp;
  return nullptr;
}

/// Return true only for the initially supported blockwise-reduction form.
/// Everything else continues through the legacy broadcast + atomic lowering.
static bool isBlockwiseReductionCandidate(ReduceOp reduceOp) {
  if (reduceOp.getReduceMethod() != ReduceMethod::Sum)
    return false;

  auto inputType = dyn_cast<RankedTensorType>(reduceOp.getIn().getType());
  auto resultType = dyn_cast<RankedTensorType>(reduceOp.getResult().getType());
  if (!inputType || !resultType || !inputType.hasStaticShape() ||
      inputType.getRank() != 3 || resultType.getRank() != 3 ||
      reduceOp.getAxis().getSExtValue() != 2)
    return false;

  // The supported global result is a single transform from the keep-dims
  // reduction result to a rank-one store.
  if (!reduceOp.getResult().hasOneUse())
    return false;
  auto postTransform =
      dyn_cast<TransformOp>(*reduceOp.getResult().user_begin());
  if (!postTransform || !postTransform.getResult().hasOneUse())
    return false;
  auto storeOp = dyn_cast<StoreOp>(*postTransform.getResult().user_begin());
  if (!storeOp ||
      cast<RankedTensorType>(storeOp.getSource().getType()).getRank() != 1)
    return false;

  // Require an explicit view from GEMM space to the logical reduction input.
  // This is the form LowerStores knows how to map back onto a local tile.
  Value producer = reduceOp.getIn();
  bool hasPreTransform = false;
  while (auto transformOp = producer.getDefiningOp<TransformOp>()) {
    hasPreTransform = true;
    producer = transformOp.getInput();
  }
  if (!hasPreTransform)
    return false;

  DenseSet<Value> visited;
  GemmOp gemmOp = findUpstreamGemm(producer, visited);
  if (!gemmOp || !gemmOp.getParams())
    return false;
  auto params = *gemmOp.getParams();
  if (params.getSplitKFactor() != 1)
    return false;

  ArrayRef<int64_t> gemmShape =
      cast<RankedTensorType>(gemmOp.getResult().getType()).getShape();
  if (gemmShape.size() < 2)
    return false;
  int64_t m = gemmShape[gemmShape.size() - 2];
  int64_t n = gemmShape[gemmShape.size() - 1];
  return m % params.getMPerBlock() == 0 && n % params.getNPerBlock() == 0;
}

/// Walk a single-use chain of TransformOps from `start` to a StoreOp,
/// collecting intermediate TransformOps into `chain`.  Returns the StoreOp,
/// or failure if the chain is broken by a multi-use value or an unexpected op.
static FailureOr<StoreOp>
collectTransformChain(Value start, SmallVectorImpl<TransformOp> &chain) {
  Value cur = start;
  while (cur.hasOneUse()) {
    Operation *user = *cur.user_begin();
    if (auto tOp = dyn_cast<TransformOp>(user)) {
      chain.push_back(tOp);
      cur = tOp.getResult();
    } else if (auto store = dyn_cast<StoreOp>(user)) {
      return store;
    } else {
      user->emitError("unexpected op between rock.reduce and rock.store: ")
          << user->getName();
      return failure();
    }
  }

  return emitError(cur.getLoc(),
                   "expected single-use chain from reduce to store");
}

struct ReduceToStoreRewritePattern : public OpRewritePattern<rock::ReduceOp> {
  using OpRewritePattern<rock::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (isBlockwiseReductionCandidate(reduceOp))
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

    Value reduceResult = reduceOp.getResult();
    if (reduceResult.getNumUses() != 1)
      return reduceOp.emitError("Expected exactly one use of the ReduceOp");

    // Walk from the reduce result through any intermediate TransformOps
    // (e.g. Merge flattening the output to match the function signature)
    // to the store.
    SmallVector<TransformOp> intermediateOps;
    FailureOr<StoreOp> maybeStore =
        collectTransformChain(reduceResult, intermediateOps);
    if (failed(maybeStore))
      return reduceOp.emitError(
          "could not find rock.store along single-use chain from reduce");
    StoreOp storeOp = *maybeStore;

    rewriter.setInsertionPoint(storeOp);
    Value transformedDest = storeOp.getDest();

    // Undo intermediate transforms on the store destination so its shape
    // matches the reduce output (e.g. Unmerge a prior Merge).
    if (!intermediateOps.empty()) {
      ArrayAttr inverted = rock::invertTransforms(
          rewriter, loc,
          rewriter.getArrayAttr(llvm::map_to_vector(
              intermediateOps, [](TransformOp tOp) -> Attribute {
                return tOp.getTransform();
              })));
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

    // The intermediate transforms and reduceOp form a single-use chain
    // (enforced by collectTransformChain and the getNumUses check above)
    // that is now dead.  All erasures are deferred by the
    // ConversionPatternRewriter (applyPartialConversion) and committed
    // together, so we must mark every op in the chain for removal —
    // otherwise the framework inserts unrealized_conversion_casts for
    // the still-live intermediate references.
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
  target.addDynamicallyLegalOp<rock::ReduceOp>(
      [](rock::ReduceOp op) { return isBlockwiseReductionCandidate(op); });
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
