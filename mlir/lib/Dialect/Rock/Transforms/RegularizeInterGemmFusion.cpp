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
static LogicalResult
regularizeGemmGemmBody(OpBuilder &builder,
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
