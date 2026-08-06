//===- RegularizeOutput.cpp - Move output fusions before transforms ---===//
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

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEOUTPUTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-regularize-output"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RegularizeContext {
  OpBuilder &builder;
  Location loc;
  Operation *rootOp;
  RankedTensorType rootType;
  DenseSet<Value> &rootReachable;
  DenseMap<Value, SmallVector<Attribute>> &transformsFromRoot;
  DenseMap<Value, Value> &gemmEquiv;

  ArrayAttr getInverse(const SmallVector<Attribute> &attrs) {
    SmallVector<Attribute> reversed(attrs.rbegin(), attrs.rend());
    ArrayAttr arr = builder.getArrayAttr(reversed);
    return invertTransforms(builder, loc, arr);
  }
};

// Recursively resolve a rootReachable value to its gemm-space equivalent.
// TransformOps are looked through (transparent). Fusion ops are cloned with
// resolved operands. External operands get inverse transforms applied.
// Results are cached in ctx.gemmEquiv so each value is resolved at most once.
static FailureOr<Value> getGemmSpaceEquiv(Value v, RegularizeContext &ctx) {
  auto it = ctx.gemmEquiv.find(v);
  if (it != ctx.gemmEquiv.end())
    return it->second;

  // Look through TransformOps — the source is the same data, just
  // with a different shape view.
  if (auto tOp = v.getDefiningOp<TransformOp>()) {
    FailureOr<Value> result = getGemmSpaceEquiv(tOp.getInput(), ctx);
    if (failed(result))
      return failure();
    ctx.gemmEquiv[v] = result.value();
    return result.value();
  }

  // Must be a fusion op if it's rootReachable and not root/transform.
  Operation *defOp = v.getDefiningOp();
  assert(defOp && isFusionOp(defOp) &&
         "rootReachable value must be root, TransformOp, or fusion op");

  auto tfIt = ctx.transformsFromRoot.find(v);
  assert(tfIt != ctx.transformsFromRoot.end());
  const SmallVector<Attribute> &accTfAttrs = tfIt->second;
  bool hasTransforms = !accTfAttrs.empty();

  // Resolve each operand to its gemm-space equivalent.
  SmallVector<Value> newOperands;
  for (OpOperand &operand : defOp->getOpOperands()) {
    Value orig = operand.get();

    if (ctx.rootReachable.contains(orig)) {
      FailureOr<Value> resolved = getGemmSpaceEquiv(orig, ctx);
      if (failed(resolved))
        return failure();
      newOperands.push_back(*resolved);
      continue;
    }

    // No transforms accumulated → use the original operand as-is.
    if (!hasTransforms) {
      newOperands.push_back(orig);
      continue;
    }

    // Splat constants: just recreate with gemm-space shape.
    if (auto cstOp = orig.getDefiningOp<arith::ConstantOp>()) {
      auto splatAttr = dyn_cast<SplatElementsAttr>(cstOp.getValue());
      if (!splatAttr)
        return ctx.rootOp->emitError(
            "cannot regularize: non-splat constant extra "
            "operand is not supported");
      auto newType = RankedTensorType::get(ctx.rootType.getShape(),
                                           splatAttr.getElementType());
      auto newSplat =
          SplatElementsAttr::get(newType, splatAttr.getSplatValue<Attribute>());
      ctx.builder.setInsertionPointAfter(cstOp);
      Value newCst = arith::ConstantOp::create(ctx.builder, ctx.loc, newSplat);
      newOperands.push_back(newCst);
      continue;
    }

    // External tensor: apply inverse transforms to bring to gemm space.
    ArrayAttr inv = ctx.getInverse(accTfAttrs);
    if (!inv)
      return ctx.rootOp->emitError(
          "cannot regularize: transforms are not invertible "
          "and extra operand is not a splat constant");
    ctx.builder.setInsertionPointAfterValue(orig);
    Value invOrig = rock::transform(ctx.builder, orig, inv);
    newOperands.push_back(invOrig);
  }

  // Clone the fusion op right after its original, with gemm-space
  // operands and result type.
  auto oldType = cast<RankedTensorType>(v.getType());
  auto newType =
      RankedTensorType::get(ctx.rootType.getShape(), oldType.getElementType());

  ctx.builder.setInsertionPointAfter(defOp);
  Operation *cloned = ctx.builder.clone(*defOp);
  for (auto [idx, newOp] : llvm::enumerate(newOperands))
    cloned->setOperand(idx, newOp);
  assert(cloned->getNumResults() == 1 &&
         "fusion ops are expected to have a single result");
  cloned->getResult(0).setType(newType);

  Value result = cloned->getResult(0);
  ctx.gemmEquiv[v] = result;
  return result;
}

/// Phase 1: Flood-fill from root through TransformOps and fusion ops.
/// Discovers all reachable values, their accumulated transform stacks from
/// root, and any StoreOps that consume reachable values.
static void
floodFillFromRoot(Value root, DenseSet<Value> &rootReachable,
                  DenseMap<Value, SmallVector<Attribute>> &transformsFromRoot,
                  SmallVector<StoreOp> &stores) {
  struct FloodItem {
    Value val;
    SmallVector<Attribute> transforms;
  };
  SmallVector<FloodItem> work;
  work.push_back({root, {}});

  while (!work.empty()) {
    auto [v, tfAttrs] = work.pop_back_val();
    if (!rootReachable.insert(v).second)
      continue;
    transformsFromRoot[v] = tfAttrs;

    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      if (auto tOp = dyn_cast<TransformOp>(owner)) {
        SmallVector<Attribute> newTf(tfAttrs);
        newTf.push_back(tOp.getTransform());
        work.push_back({tOp.getResult(), newTf});
      } else if (isFusionOp(owner)) {
        for (Value res : owner->getResults())
          work.push_back({res, tfAttrs});
      } else if (auto reduceOp = dyn_cast<ReduceOp>(owner)) {
        // Keep reductions as semantic boundaries, but continue the trace so
        // stores after the reduction are visible to this pass. The transform
        // stack is only meaningful for the reduction input; reduced results
        // are never converted to GEMM space.
        work.push_back({reduceOp.getResult(), tfAttrs});
      } else if (auto storeOp = dyn_cast<StoreOp>(owner);
                 storeOp && use.getOperandNumber() == 0) {
        stores.push_back(storeOp);
      }
    }
  }
}

/// Return the reduction feeding `value` through a post-reduction transform
/// chain, or null when this is an ordinary elementwise store source.
static ReduceOp findUpstreamReduce(Value value) {
  while (auto transformOp = value.getDefiningOp<TransformOp>())
    value = transformOp.getInput();
  return value.getDefiningOp<ReduceOp>();
}

/// Move the elementwise producer of each reduction input to GEMM space while
/// preserving the explicit ReduceOp and its logical input shape. The original
/// transform stack is reapplied immediately before the reduction.
static LogicalResult regularizeReductionInputs(ArrayRef<StoreOp> stores,
                                               RegularizeContext &ctx) {
  DenseSet<Operation *> visited;
  for (StoreOp storeOp : stores) {
    ReduceOp reduceOp = findUpstreamReduce(storeOp.getSource());
    if (!reduceOp || !visited.insert(reduceOp).second)
      continue;

    Value oldInput = reduceOp.getIn();
    auto tfIt = ctx.transformsFromRoot.find(oldInput);
    if (tfIt == ctx.transformsFromRoot.end())
      return reduceOp.emitError(
          "reduction input is not reachable from the fusion root");

    FailureOr<Value> gemmInput = getGemmSpaceEquiv(oldInput, ctx);
    if (failed(gemmInput))
      return failure();

    Value newInput = *gemmInput;
    if (!tfIt->second.empty()) {
      ctx.builder.setInsertionPoint(reduceOp);
      // transform() consumes maps in upper-view-to-root order and applies
      // them in reverse. transformsFromRoot records the opposite order.
      SmallVector<Attribute> transforms(tfIt->second.rbegin(),
                                        tfIt->second.rend());
      newInput = rock::transform(ctx.builder, newInput,
                                 ctx.builder.getArrayAttr(transforms));
    }
    reduceOp.getInMutable().assign(newInput);
  }
  return success();
}

/// Phase 2: Rewire stores to use gemm-space fused values.
/// Lazily resolves fusion ops to gemm space via getGemmSpaceEquiv, then
/// updates each store's source and (if needed) destination.
static LogicalResult rewireStoresToGemmSpace(ArrayRef<StoreOp> stores,
                                             RegularizeContext &ctx) {
  for (StoreOp storeOp : stores) {
    // Reduction stores retain their rank-reduced source and destination. Their
    // input producer was regularized separately above, and LowerStores will
    // lower the explicit reduction after the block-local tile is available.
    if (findUpstreamReduce(storeOp.getSource()))
      continue;

    Value storeSource = storeOp.getSource();
    FailureOr<Value> newSource = getGemmSpaceEquiv(storeSource, ctx);
    if (failed(newSource))
      return failure();

    storeOp.getSourceMutable().assign(*newSource);

    auto tfIt = ctx.transformsFromRoot.find(storeSource);
    if (tfIt != ctx.transformsFromRoot.end() && !tfIt->second.empty()) {
      ArrayAttr inv = ctx.getInverse(tfIt->second);
      if (!inv) {
        ctx.rootOp->emitError(
            "cannot regularize: transforms are not invertible "
            "for store destination rewrite");
        return failure();
      }
      ctx.builder.setInsertionPoint(storeOp);
      Value dest = storeOp.getDest();
      Value newDest = rock::transform(ctx.builder, dest, inv);
      storeOp.getDestMutable().assign(newDest);
    }
  }
  return success();
}

/// Phase 3: Erase dead original ops left behind by the cloning in Phase 2.
/// Ops that define rootReachable values (excluding root and live gemmEquiv
/// results) are erased in reverse program order.
static void eraseDeadOps(const DenseSet<Value> &rootReachable, Value root,
                         const DenseMap<Value, Value> &gemmEquiv) {
  DenseSet<Operation *> liveOps;
  for (auto &[_, liveVal] : gemmEquiv)
    if (Operation *op = liveVal.getDefiningOp())
      liveOps.insert(op);

  SmallVector<Operation *> deadOps;
  for (Value v : rootReachable) {
    if (v == root)
      continue;
    Operation *op = v.getDefiningOp();
    if (op && !liveOps.contains(op))
      deadOps.push_back(op);
  }
  llvm::sort(deadOps,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *op : llvm::reverse(deadOps)) {
    if (op->use_empty())
      op->erase();
  }
}

struct RockRegularizeOutput
    : public rock::impl::RockRegularizeOutputPassBase<RockRegularizeOutput> {
  void runOnOperation() override;
};

} // namespace

// This pass moves output fusions (arith/math element-wise ops) to operate
// directly in the FusionRoot's output space (e.g. gemm space), eliminating
// any rock.transform ops that sit between the root and the fusions.
//
// It uses a clone-based approach: instead of mutating ops in place, it builds
// new fusion ops in gemm space via recursive resolution, then rewires stores to
// use the new values. The original ops become dead and are cleaned up by DCE.
//
// Before:
//   %gemm = rock.gemm ...                         [1,100,100]
//   %t = rock.transform %gemm by T                [1,100,10,10]
//   %fused = arith.addf %t, %extra                [1,100,10,10]
//   rock.store %fused to %dest
//
// After:
//   %gemm = rock.gemm ...                         [1,100,100]
//   %extra_inv = rock.transform %extra by inv(T)  [1,100,100]
//   %fused' = arith.addf %gemm, %extra_inv        [1,100,100]
//   %dest_inv = rock.transform %dest by inv(T)    [1,100,100]
//   rock.store %fused' to %dest_inv
//
// The pass handles complex DAGs where fusion ops have multiple operands
// that all trace back to the root (possibly through different transform
// paths). A DenseMap cache ensures each value is resolved exactly once,
// so identical operands (e.g. arith.addf %x, %x) share the same result.
void RockRegularizeOutput::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(funcOp.getContext());

  SmallVector<Operation *> fusionRoots;
  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::rock::FusionRoot>())
      fusionRoots.push_back(op);
  });

  for (Operation *rootOp : fusionRoots) {
    Location loc = rootOp->getLoc();

    for (Value root : rootOp->getResults()) {
      auto rootType = cast<RankedTensorType>(root.getType());

      // Phase 1: Discover reachable values, transform stacks, and stores.
      DenseSet<Value> rootReachable;
      DenseMap<Value, SmallVector<Attribute>> transformsFromRoot;
      SmallVector<StoreOp> stores;
      floodFillFromRoot(root, rootReachable, transformsFromRoot, stores);

      // we assume all root ops must lead to a rock.store
      if (stores.empty()) {
        rootOp->emitError("No stores found for fusion root");
        return signalPassFailure();
      }

      // if there are no transforms, no work to do!
      bool anyTransforms = llvm::any_of(
          transformsFromRoot, [](auto &kv) { return !kv.second.empty(); });
      if (!anyTransforms)
        continue;

      // Phase 2: Resolve gemm-space equivalents and rewire stores.
      DenseMap<Value, Value> gemmEquiv;
      gemmEquiv[root] = root;

      RegularizeContext ctx{builder,  loc,           rootOp,
                            rootType, rootReachable, transformsFromRoot,
                            gemmEquiv};

      if (failed(regularizeReductionInputs(stores, ctx)))
        return signalPassFailure();

      if (failed(rewireStoresToGemmSpace(stores, ctx)))
        return signalPassFailure();

      // Phase 3: Erase dead original ops left behind by cloning.
      eraseDeadOps(rootReachable, root, gemmEquiv);
    } // for each result of rootOp
  }
}
