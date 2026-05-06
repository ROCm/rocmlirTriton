//===- RegularizeInterGemmFusion.cpp - Regularize inter-GEMM fusions ---===//
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
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEINTERGEMMFUSIONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-regularize-inter-gemm-fusion"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// If the value yielded from the body is produced by a `rock.transform`
/// whose own input is not a block argument, that transform is a "yield-
/// boundary" view that the upstream lowering inserted to convert the
/// body's working shape back into arg0's shape. Rewire the yield to
/// consume the transform's input directly and erase the transform.
/// Repeat for any stack of such boundary transforms (rare in current
/// migraphx output, but the loop costs nothing if there's only one).
///
/// Example. Suppose arg0's shape is <1x4x4xf32> but the body operates in
/// a "working shape" of <16xf32>. The example shows a single working
/// shape because that is all this function reasons about. Any other
/// intermediate working shapes in the body are handled later by
/// `sinkTransformsToLeaves`:
///
///   ^bb0(%arg0: tensor<1x4x4xf32>, %arg1: tensor<1x4x4xf32>):
///     %a = rock.transform %arg0 by T_arg0 : tensor<1x4x4xf32> to tensor<16xf32>
///     %b = rock.transform %arg1 by T_arg1 : tensor<1x4x4xf32> to tensor<16xf32>
///     %s = arith.addf %a, %b              : tensor<16xf32>
///     %y = rock.transform %s by T_yield   : tensor<16xf32> to tensor<1x4x4xf32>
///     rock.yield %y                       : tensor<1x4x4xf32>
///
/// `%y`'s defining transform is the yield boundary. Provided the round-
/// trip `T_yield` ∘ inv(T_arg0) is identity on arg0's shape (so `%s` is
/// per-element equivalent to a value already in arg0's shape), this
/// function rewires the yield to consume `%s` directly and erases
/// `T_yield`. Downstream steps then push T_arg0/T_arg1 outward and
/// retype the body so it reads `%arg0`/`%arg1` directly in <1x4x4xf32>.
///
/// Without this step, `sinkTransformsToLeaves` would push the yield-
/// boundary transform inward through every body op until it reaches a
/// leaf. When the body is a DAG (a value with multiple consumers),
/// sinking clones the subtree it passes through but leaves the original
/// alive (because the original still has consumers via its other use(s)),
/// ending up with the same block-arg-rooted transform consumed by both
/// the original and each clone. `collectArgTransformChains` then sees a
/// multi-use chain end and rejects the body with "non-linear transform
/// chain".
///
/// Erasing the yield-boundary transform up front sidesteps the entire
/// problem. The sinker has nothing left to push inward; the only
/// remaining transforms are the (already-linear) block-arg chains. The
/// body is then converted to arg0's shape entirely by type substitution,
/// in three coordinated steps that together re-work every value's type:
///   1. `externalizeBodyTransforms` retypes the block-arg leaves. For
///      arg0, it drops the in-body chain so the body reads `%arg0`
///      directly (already in arg0's shape). For each extra block arg, it
///      hoists the in-body chain plus inv(T_arg0) onto the corresponding
///      external operand and retypes the block arg itself to arg0's
///      shape.
///   2. `reshapeBodyConstants` retypes the constant leaves. Every inline
///      splat in the body is rewritten to arg0's shape with the same
///      value.
///   3. `propagateBodyResultTypes` walks the body in program order and
///      retypes each intermediate op's result based on operand 0's type.
///      This step is only sound if every operand source of every visited
///      op has already been retyped by step 1 (block args), step 2
///      (constants), or by an earlier visit in this same walk
///      (intermediate body ops). For pure arith/math elementwise bodies,
///      the only kind this pass accepts, that condition holds, so the
///      new shape flows down the DAG to the value the yield was rewired
///      to consume.
static LogicalResult eraseYieldBoundaryTransform(Operation *op, Block &block) {
  auto yieldOp = cast<rock::YieldOp>(block.getTerminator());

  // arg0 is the first GEMM's result (e.g. QK^T for attention, A*B for
  // gemm-elementwise-gemm). Bind it to a named value to make the
  // intent of the walks below easier to follow.
  Value firstGemmRes = block.getArgument(0);

  // Cheap-rejection shape guard: if the outermost boundary transform's
  // output shape isn't already arg0's shape, the inversion check below
  // is guaranteed to fail (different result count from rootShape), so
  // we may as well bail out before walking arg0's chain.
  auto rootShape = cast<RankedTensorType>(firstGemmRes.getType()).getShape();
  auto firstResShape =
      cast<RankedTensorType>(yieldOp.getOperand(0).getType()).getShape();
  if (firstResShape != rootShape)
    return success();

  // Walk the (possibly stacked) boundary transforms in walking order from
  // yield inward, without modifying the IR. Each link must be single-
  // use (so erasing it later is safe and Op::erase() can't assert) and
  // rooted at something other than a block arg (so the link isn't part
  // of an arg-rooted chain that collectArgTransformChains will handle).
  SmallVector<TransformMapAttr> boundaryAttrs;
  {
    Value cur = yieldOp.getOperand(0);
    while (auto tOp = cur.getDefiningOp<TransformOp>()) {
      // Multi-use boundary links aren't supported: the inversion check
      // below only justifies the rewrite if nothing else observes this
      // view, and a surviving multi-use link would later be rejected by
      // collectArgTransformChains as a non-linear transform chain.
      if (!tOp.getResult().hasOneUse())
        break;

      // Reaching a block arg means the entire chain from yield back to
      // that arg is single-use, i.e. the arg has no other in-body uses.
      // For non-arg0 args (extra elementwise inputs like biases/scales)
      // that would mean the body discards the first GEMM result entirely,
      // which no current producer emits. Codify that invariant: the only
      // block arg the yield-rooted chain can reach is arg0 itself, whose
      // chain is then picked up by collectArgTransformChains.
      if (isa<BlockArgument>(tOp.getInput())) {
        if (tOp.getInput() != firstGemmRes)
          return op->emitOpError()
                 << "yield-rooted single-use transform chain must land on "
                    "arg0 (the first GEMM result)";
        break;
      }

      boundaryAttrs.push_back(tOp.getTransform());
      cur = tOp.getInput();
    }
  }

  if (boundaryAttrs.empty())
    return success();

  // Walk arg0's in-body transform chain (linear single-use chain of
  // TransformOps) and collect its attributes in walking order from arg0
  // outward.
  SmallVector<TransformMapAttr> arg0Attrs;
  {
    Value cur = firstGemmRes;
    while (cur.hasOneUse()) {
      auto chainTOp = dyn_cast<TransformOp>(*cur.getUsers().begin());
      if (!chainTOp)
        break;
      arg0Attrs.push_back(chainTOp.getTransform());
      cur = chainTOp.getResult();
    }
  }

  // Inversion check. composeTransforms(L) applies L[0] first, then L[1],
  // ..., then L[N-1]. The boundary stack maps arg0_shape -> working_shape
  // (apply outer first, so boundaryAttrs is already in the right order).
  // Arg0's chain maps working_shape -> arg0_shape via right-to-left
  // composition of its attrs (each rock.transform reads at its top by
  // applying its map down to the underlying data), so we pass them
  // reversed. The composed round-trip must land on identity (modulo unit
  // dims) for the rewrite to preserve per-element values.
  SmallVector<TransformMapAttr> roundTrip(boundaryAttrs);
  for (auto attr : llvm::reverse(arg0Attrs))
    roundTrip.push_back(attr);
  AffineMap composed = composeTransforms(roundTrip);
  if (!isIdentityOnShape(composed, rootShape))
    return success();

  // All checks passed, perform the erasure. Same loop structure as the
  // walking step above; we re-derive each tOp from yield's current
  // operand because each erase changes what yield points at.
  while (true) {
    auto tOp = yieldOp.getOperand(0).getDefiningOp<TransformOp>();
    if (!tOp || isa<BlockArgument>(tOp.getInput()) ||
        !tOp.getResult().hasOneUse())
      return success();
    yieldOp.setOperand(0, tOp.getInput());
    tOp.erase();
  }
}

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
      if (newElemwise->getNumResults() != 1)
        return op->emitOpError() << "expected single-result elementwise op";
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

  if (mutableInputs.size() != block.getNumArguments() - 1)
    return op->emitOpError()
           << "expected " << (block.getNumArguments() - 1)
           << " extra elementwise inputs (one per non-root block arg), but got "
           << mutableInputs.size();

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

    // When the body has no transforms on this arg, chainEnd is the block
    // argument itself; updating the block-arg type above is sufficient.
    if (chain.chainEnd != block.getArgument(i))
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
          SplatElementsAttr::get(valType,
                                 splatAttr.getSplatValue<Attribute>()));
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
    auto operandTy = dyn_cast<RankedTensorType>(bodyOp.getOperand(0).getType());
    auto resultTy = dyn_cast<RankedTensorType>(bodyOp.getResult(0).getType());
    if (!operandTy || !resultTy)
      continue;
    if (operandTy.getShape() != resultTy.getShape())
      bodyOp.getResult(0).setType(RankedTensorType::get(
          operandTy.getShape(), resultTy.getElementType()));
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
                                            RockGemmGemmWrapperInterface ggOp) {
  if (!rock::gemmGemmHasPreSecondGemmFusion(ggOp))
    return success();

  Operation *op = ggOp;
  Region &body = ggOp.getPreSecondGemmRegion();
  Block &block = body.front();
  MutableOperandRange mutableInputs =
      ggOp.getPreSecondGemmElemwiseInputsMutable();

  if (block.getNumArguments() == 0)
    return op->emitOpError(
        "elementwise body block must have at least one argument");

  if (failed(inlineExternalConstants(builder, op, block)))
    return failure();

  if (failed(eraseYieldBoundaryTransform(op, block)))
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

  if (failed(
          externalizeBodyTransforms(builder, op, block, mutableInputs, chains)))
    return failure();

  ArrayRef<int64_t> targetShape =
      cast<RankedTensorType>(block.getArgument(0).getType()).getShape();

  if (failed(reshapeBodyConstants(op, block, targetShape)))
    return failure();

  propagateBodyResultTypes(block);

  return success();
}

struct RockRegularizeInterGemmFusion
    : public rock::impl::RockRegularizeInterGemmFusionPassBase<
          RockRegularizeInterGemmFusion> {
  void runOnOperation() override;
};

} // namespace

void RockRegularizeInterGemmFusion::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(funcOp.getContext());

  funcOp.walk([&](RockGemmGemmWrapperInterface ggOp) {
    if (failed(regularizeGemmGemmBody(builder, ggOp)))
      return signalPassFailure();
  });
}
