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
      } else if (auto storeOp = dyn_cast<StoreOp>(owner);
                 storeOp && use.getOperandNumber() == 0) {
        stores.push_back(storeOp);
      }
    }
  }
}

/// Phase 2: Rewire stores to use gemm-space fused values.
/// Lazily resolves fusion ops to gemm space via getGemmSpaceEquiv, then
/// updates each store's source and (if needed) destination.
static LogicalResult rewireStoresToGemmSpace(ArrayRef<StoreOp> stores,
                                             RegularizeContext &ctx) {
  for (StoreOp storeOp : stores) {
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

/// Push rock.transform ops that sit on intermediate (non-block-arg) values
/// back through the elementwise ops that produced them, until every
/// transform is directly on a block argument or has been absorbed into a
/// splat constant.  This is the body-region analogue of the external-IR
/// flood-fill in floodFillFromRoot / getGemmSpaceEquiv.
static LogicalResult sinkTransformsToLeaves(Operation *op, Block &block) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &bodyOp :
         llvm::make_early_inc_range(block.without_terminator())) {
      auto transformOp = dyn_cast<TransformOp>(&bodyOp);
      if (!transformOp)
        continue;

      Value input = transformOp.getInput();
      if (isa<BlockArgument>(input))
        continue;

      Operation *defOp = input.getDefiningOp();
      if (!defOp)
        continue;

      // Part of a block-arg transform chain. collectArgTransformChains
      // will handle these after sinking completes.
      if (isa<TransformOp>(defOp))
        continue;

      TransformMapAttr attr = transformOp.getTransform();
      Location loc = transformOp.getLoc();

      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        auto splatAttr = dyn_cast<SplatElementsAttr>(constOp.getValue());
        if (!splatAttr)
          return op->emitOpError()
                 << "cannot sink transform through non-splat constant";
        auto newType =
            cast<RankedTensorType>(transformOp.getResult().getType());
        OpBuilder builder(transformOp);
        Value newConst = arith::ConstantOp::create(
            builder, loc,
            SplatElementsAttr::get(newType,
                                   splatAttr.getSplatValue<Attribute>()));
        transformOp.getResult().replaceAllUsesWith(newConst);
        transformOp->erase();
        if (constOp->use_empty())
          constOp->erase();
        changed = true;
        break;
      }

      if (!isFusionOp(defOp))
        return op->emitOpError()
               << "rock.transform on non-elementwise, non-constant "
                  "intermediate value is not supported";

      auto resultType =
          cast<RankedTensorType>(transformOp.getResult().getType());

      OpBuilder builder(transformOp);
      SmallVector<Value> newOperands;
      for (Value operand : defOp->getOperands()) {
        if (auto cOp = operand.getDefiningOp<arith::ConstantOp>()) {
          auto splatVal = dyn_cast<SplatElementsAttr>(cOp.getValue());
          if (!splatVal)
            return op->emitOpError()
                   << "cannot sink transform through non-splat constant";
          auto cType =
              cast<RankedTensorType>(operand.getType()).getElementType();
          auto newCType = RankedTensorType::get(resultType.getShape(), cType);
          Value newC = arith::ConstantOp::create(
              builder, loc,
              SplatElementsAttr::get(newCType,
                                     splatVal.getSplatValue<Attribute>()));
          newOperands.push_back(newC);
        } else {
          Value t = TransformOp::create(builder, loc, operand, attr);
          newOperands.push_back(t);
        }
      }

      Operation *newElemwise = builder.clone(*defOp);
      for (unsigned i = 0; i < newOperands.size(); ++i)
        newElemwise->setOperand(i, newOperands[i]);
      newElemwise->getResult(0).setType(resultType);

      transformOp.getResult().replaceAllUsesWith(newElemwise->getResult(0));
      transformOp->erase();
      if (defOp->use_empty())
        defOp->erase();

      changed = true;
      break;
    }
  }
  return success();
}

/// Linear chain of rock.transform ops rooted at a block argument.
struct ArgTransformChain {
  SmallVector<TransformMapAttr> transforms;
  SmallVector<TransformOp> transformOps;
  Value chainEnd;
};

/// Walk each block argument's single-use transform chain. Returns failure
/// if any argument has a branching (multi-use) transform chain.
static FailureOr<SmallVector<ArgTransformChain>>
collectArgTransformChains(Operation *op, Block &block) {
  SmallVector<ArgTransformChain> chains(block.getNumArguments());

  for (unsigned i = 0; i < block.getNumArguments(); ++i) {
    Value chainEnd = block.getArgument(i);
    while (chainEnd.hasOneUse()) {
      Operation *user = *chainEnd.getUsers().begin();
      auto transformOp = dyn_cast<TransformOp>(user);
      if (!transformOp)
        break;
      chains[i].transforms.push_back(transformOp.getTransform());
      chains[i].transformOps.push_back(transformOp);
      chainEnd = transformOp.getResult();
    }
    if (!chainEnd.use_empty()) {
      for (Operation *user : chainEnd.getUsers()) {
        if (isa<TransformOp>(user))
          return op->emitOpError()
                 << "elementwise body block argument " << i
                 << " has a non-linear transform chain "
                    "(multi-use or branching); this is not supported";
      }
    }
    chains[i].chainEnd = chainEnd;
  }
  return chains;
}

/// Move transform chains from inside the body to the external inputs.
/// Arg0's transforms are removed (its chain end rewired to the block arg).
/// For extra-input args, the body transforms plus the inverse of arg0's
/// transforms are applied to the corresponding external operand.
static LogicalResult
externalizeBodyTransforms(OpBuilder &builder, Operation *op, Block &block,
                          MutableOperandRange mutableInputs,
                          SmallVectorImpl<ArgTransformChain> &chains) {
  Location loc = op->getLoc();
  auto rootType = cast<RankedTensorType>(block.getArgument(0).getType());

  SmallVector<Attribute> invertedArg0Transforms;
  if (!chains[0].transforms.empty()) {
    SmallVector<Attribute> arg0Attrs(chains[0].transforms.begin(),
                                     chains[0].transforms.end());
    ArrayAttr inverted =
        invertTransforms(builder, loc, builder.getArrayAttr(arg0Attrs));
    if (!inverted)
      return op->emitOpError()
             << "failed to invert first-GEMM argument transforms";
    invertedArg0Transforms.append(inverted.begin(), inverted.end());
    chains[0].chainEnd.replaceAllUsesWith(block.getArgument(0));
  }

  builder.setInsertionPoint(op);

  for (unsigned i = 1; i < block.getNumArguments(); ++i) {
    auto &chain = chains[i];

    SmallVector<Attribute> allTransforms;
    for (auto t : chain.transforms)
      allTransforms.push_back(t);
    allTransforms.append(invertedArg0Transforms.begin(),
                         invertedArg0Transforms.end());

    if (!allTransforms.empty()) {
      Value current = mutableInputs[i - 1].get();
      for (Attribute t : allTransforms)
        current = TransformOp::create(builder, loc, current,
                                      cast<TransformMapAttr>(t));
      mutableInputs.slice(i - 1, 1).assign(current);
    }

    auto argElemType =
        cast<RankedTensorType>(block.getArgument(i).getType()).getElementType();
    block.getArgument(i).setType(
        RankedTensorType::get(rootType.getShape(), argElemType));
    chain.chainEnd.replaceAllUsesWith(block.getArgument(i));
  }

  // Erase dead transform ops (reverse order to satisfy use-def).
  for (auto &chain : chains)
    for (auto it = chain.transformOps.rbegin(); it != chain.transformOps.rend();
         ++it)
      (*it)->erase();

  return success();
}

/// Replace references to external splat constants with inline copies so
/// the body is self-contained.  Called unconditionally — even when the
/// body has no transforms — so that downstream passes never need to
/// handle captured constants.
static LogicalResult inlineExternalConstants(OpBuilder &builder, Operation *op,
                                             Block &block) {
  Location loc = op->getLoc();

  for (Operation &bodyOp : block.without_terminator()) {
    for (OpOperand &operand : bodyOp.getOpOperands()) {
      Value val = operand.get();
      if (val.getParentBlock() == &block)
        continue;
      auto valType = dyn_cast<RankedTensorType>(val.getType());
      if (!valType)
        continue;
      auto *defOp = val.getDefiningOp();
      auto extConst = defOp ? dyn_cast<arith::ConstantOp>(defOp) : nullptr;
      if (!extConst)
        return op->emitOpError()
               << "non-constant external value in elementwise body is "
                  "not supported";
      auto splatAttr = dyn_cast<SplatElementsAttr>(extConst.getValue());
      if (!splatAttr)
        return op->emitOpError()
               << "non-splat external constant in elementwise body is "
                  "not supported";
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(&bodyOp);
      Value newConst = arith::ConstantOp::create(
          builder, loc,
          SplatElementsAttr::get(valType, splatAttr.getSplatValue<Attribute>()));
      operand.set(newConst);
    }
  }
  return success();
}

/// Reshape all inline splat constants in the body to match `targetShape`.
static LogicalResult reshapeBodyConstants(Operation *op, Block &block,
                                          ArrayRef<int64_t> targetShape) {
  for (Operation &bodyOp : block.without_terminator()) {
    auto constOp = dyn_cast<arith::ConstantOp>(&bodyOp);
    if (!constOp)
      continue;
    auto origType = dyn_cast<RankedTensorType>(constOp.getType());
    if (!origType || origType.getShape() == targetShape)
      continue;
    auto splatAttr = dyn_cast<SplatElementsAttr>(constOp.getValue());
    if (!splatAttr)
      return op->emitOpError()
             << "non-splat constant in elementwise body cannot be reshaped";
    auto newType =
        RankedTensorType::get(targetShape, origType.getElementType());
    constOp.setValueAttr(
        SplatElementsAttr::get(newType, splatAttr.getSplatValue<Attribute>()));
    constOp.getResult().setType(newType);
  }
  return success();
}

/// Propagate `targetShape` through elementwise op results whose shapes
/// have become stale after block argument types changed.
static void propagateBodyResultTypes(Block &block) {
  for (Operation &bodyOp : block.without_terminator()) {
    if (bodyOp.getNumResults() != 1 || bodyOp.getNumOperands() == 0)
      continue;
    auto operandTy =
        dyn_cast<RankedTensorType>(bodyOp.getOperand(0).getType());
    auto resultTy =
        dyn_cast<RankedTensorType>(bodyOp.getResult(0).getType());
    if (!operandTy || !resultTy)
      continue;
    if (operandTy.getShape() != resultTy.getShape())
      bodyOp.getResult(0).setType(
          RankedTensorType::get(operandTy.getShape(),
                                resultTy.getElementType()));
  }
}

/// Regularize the body region of a gemm-gemm-like op so that it contains
/// only elementwise operations (no rock.transform ops).
/// Transforms on block arguments are pushed outside the body by applying
/// them to the corresponding external inputs (via `mutableInputs`).
/// If arg0 (the first GEMM product) has transforms, they are eliminated
/// by applying their inverse to all other block arguments' transform chains
/// so that the entire body operates in arg0's original shape.
static LogicalResult regularizeGemmGemmBody(OpBuilder &builder,
                                            Operation *op, Region &body,
                                            MutableOperandRange mutableInputs) {
  if (body.empty())
    return success();

  Block &block = body.front();
  if (block.without_terminator().empty())
    return success();

  if (block.getNumArguments() == 0)
    return op->emitOpError(
        "elementwise body block must have at least one argument");

  if (failed(inlineExternalConstants(builder, op, block)))
    return failure();

  if (failed(sinkTransformsToLeaves(op, block)))
    return failure();

  auto maybeChains = collectArgTransformChains(op, block);
  if (failed(maybeChains))
    return failure();
  auto &chains = *maybeChains;

  bool anyTransforms = llvm::any_of(
      chains, [](const ArgTransformChain &c) { return !c.transforms.empty(); });
  if (!anyTransforms)
    return success();

  if (failed(externalizeBodyTransforms(builder, op, block, mutableInputs,
                                       chains)))
    return failure();

  ArrayRef<int64_t> targetShape =
      cast<RankedTensorType>(block.getArgument(0).getType()).getShape();

  if (failed(reshapeBodyConstants(op, block, targetShape)))
    return failure();

  propagateBodyResultTypes(block);

  return success();
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

  // Regularize elementwise body regions in gemm-gemm-like ops, pushing
  // any rock.transform ops out of the body and onto the external inputs.
  funcOp.walk([&](RockGemmGemmWrapperInterface ggOp) {
    if (failed(regularizeGemmGemmBody(
            builder, ggOp, ggOp.getPreSecondGemmRegion(),
            ggOp.getPreSecondGemmElemwiseInputsMutable())))
      return signalPassFailure();
  });

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

      if (failed(rewireStoresToGemmSpace(stores, ctx)))
        return signalPassFailure();

      // Phase 3: Erase dead original ops left behind by cloning.
      eraseDeadOps(rootReachable, root, gemmEquiv);
    } // for each result of rootOp
  }
}
