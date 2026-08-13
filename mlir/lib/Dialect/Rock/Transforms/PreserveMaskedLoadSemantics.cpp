//===- PreserveMaskedLoadSemantics.cpp - Re-mask OOB after fusions -------===//
//
// Copyright Advanced Micro Devices, Inc.
// Copyright 2026 The MLIR Authors.
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
// This pass runs AFTER RockLowerBlockwiseToPtrPass and BEFORE
// RockTransformsToPointerArithPass.
//
// When a BlockwiseLoadPtrOp has Pad or Embed transforms, its mask tensor marks
// out-of-bounds (OOB) positions. tt.load returns 0 for those positions. If the
// loaded tile feeds into a non-zero-preserving fusion (e.g. arith.addf %tile,
// 1.0), OOB positions become non-zero and can corrupt consumers that do not
// carry the original mask. Even zero-preserving fusions need special handling
// when they feed a max reduction, because zero is not the neutral element for
// max.
//
// This pass detects such cases and inserts:
//   %safe = arith.select %mask, %fused_result, %fill
// at non-fusion uses of the fusion chain. The fill value is chosen for the
// consumer: zero by default, or the type's smallest representable value (e.g.
// -inf for IEEE floats, signed INT_MIN for integers) for `rock.blockwise_reduce
// max` consumers, so masked-out lanes do not influence the reduction.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPRESERVEMASKEDLOADSEMANTICSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-preserve-masked-load-semantics"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Check if a mask tensor is trivial (constant splat of true).
/// A trivial mask means no Pad/Embed validity constraints exist.
static bool isTrivialMask(Value mask) {
  DenseElementsAttr attr;
  if (!matchPattern(mask, m_Constant(&attr)))
    return false;
  if (!attr.isSplat())
    return false;
  return attr.getSplatValue<bool>();
}

/// Return true for shape-only operations inserted when RegularizeInput
/// reconstructs a narrowed broadcast load. The load's validity mask must follow
/// the same shape changes before it can be applied to the reconstructed fusion
/// result.
static bool isNarrowLoadShapeOp(Operation *op) {
  return isa<triton::ExpandDimsOp, triton::BroadcastOp>(op);
}

static bool isFusionOrNarrowLoadShapeOp(Operation *op) {
  return rock::isFusionOp(op) || isNarrowLoadShapeOp(op);
}

/// Starting from a load result, follow uses through fusion ops and the
/// expand/broadcast pair used to reconstruct narrowed loads. Collect
/// fusion leaves together with the shape-op path their masks must follow.
/// Mask operations are materialized later, after zero-preservation analysis
/// determines that the leaf actually needs re-masking.
static SmallVector<std::pair<Value, SmallVector<Operation *>>>
collectFusionChainLeaves(Value loadResult) {
  SmallVector<std::pair<Value, SmallVector<Operation *>>> leaves;
  SmallVector<std::pair<Value, SmallVector<Operation *>>> worklist;
  DenseSet<std::pair<Operation *, Operation *>> visited;
  worklist.push_back({loadResult, {}});

  while (!worklist.empty()) {
    auto [value, shapeOps] = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      // Each shape op has one input, so its identity uniquely identifies the
      // shape path. Keep converging paths distinct when their masks differ.
      Operation *lastShapeOp = shapeOps.empty() ? nullptr : shapeOps.back();
      if (!isFusionOrNarrowLoadShapeOp(owner) ||
          !visited.insert({owner, lastShapeOp}).second)
        continue;

      Value result = owner->getResult(0);
      SmallVector<Operation *> resultShapeOps = shapeOps;
      if (isNarrowLoadShapeOp(owner))
        resultShapeOps.push_back(owner);
      bool hasNonFusionUse = false;
      bool hasFusionUse = false;
      for (OpOperand &resultUse : result.getUses()) {
        if (isFusionOrNarrowLoadShapeOp(resultUse.getOwner()))
          hasFusionUse = true;
        else
          hasNonFusionUse = true;
      }
      if (hasNonFusionUse && rock::isFusionOp(owner))
        leaves.push_back({result, resultShapeOps});
      if (hasFusionUse)
        worklist.push_back({result, std::move(resultShapeOps)});
    }
  }
  return leaves;
}

/// Clone the recorded shape-op path with the validity mask as its input.
static Value materializeMaskPath(OpBuilder &builder, Value mask,
                                 ArrayRef<Operation *> shapeOps) {
  for (Operation *op : shapeOps) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(op);
    IRMapping mapping;
    mapping.map(op->getOperand(0), mask);
    Operation *maskOp = builder.clone(*op, mapping);
    auto resultType = cast<RankedTensorType>(maskOp->getResult(0).getType());
    maskOp->getResult(0).setType(
        resultType.cloneWith(resultType.getShape(), builder.getI1Type()));
    mask = maskOp->getResult(0);
  }
  return mask;
}

/// Trace backwards from a value through fusion ops and narrowed-load shape
/// operations, and collect all BlockwiseLoadPtrOp ops that feed into it
/// (directly or indirectly).
static void collectContributingLoads(Value val,
                                     SmallVectorImpl<BlockwiseLoadPtrOp> &loads,
                                     DenseSet<Operation *> &visited) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return;
  if (auto loadOp = dyn_cast<BlockwiseLoadPtrOp>(defOp)) {
    loads.push_back(loadOp);
    return;
  }
  if (!rock::isFusionOp(defOp) && !isNarrowLoadShapeOp(defOp))
    return;
  if (!visited.insert(defOp).second)
    return;
  for (Value operand : defOp->getOperands())
    collectContributingLoads(operand, loads, visited);
}

/// Test if a fusion chain preserves zero. Clones the chain, replaces all
/// values that trace back to a BlockwiseLoadPtrOp with zero constants,
/// constant-folds each op, and checks if the final result is zero.
///
/// Inspired by knownToPreserveZero in LinalgAlign.cpp (rocMLIR)
static bool fusionChainPreservesZero(Value leaf, OpBuilder &builder) {
  Operation *leafOp = leaf.getDefiningOp();
  if (!leafOp || !rock::isFusionOp(leafOp))
    return true;

  // Collect all fusion ops in the chain by walking backwards from the leaf.
  SmallVector<Operation *> chainOps;
  DenseSet<Operation *> inChain;
  SmallVector<Operation *> backWorklist;
  backWorklist.push_back(leafOp);

  while (!backWorklist.empty()) {
    Operation *op = backWorklist.pop_back_val();
    if (!inChain.insert(op).second)
      continue;
    chainOps.push_back(op);
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (defOp && rock::isFusionOp(defOp))
        backWorklist.push_back(defOp);
    }
  }

  // Reverse so we process in topological order (producers before consumers).
  std::reverse(chainOps.begin(), chainOps.end());

  // Clone the chain, replace load-originating inputs with zeros, fold.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(leafOp);
  IRMapping mapping;
  SmallVector<Operation *> toErase;
  bool anyFoldFailed = false;

  for (Operation *op : chainOps) {
    for (Value operand : op->getOperands()) {
      if (mapping.contains(operand))
        continue;
      if (auto *defOp = operand.getDefiningOp())
        if (inChain.count(defOp))
          continue;
      // Trace back: only replace with zero if it originates from a load.
      SmallVector<BlockwiseLoadPtrOp> loads;
      DenseSet<Operation *> traceVisited;
      collectContributingLoads(operand, loads, traceVisited);
      if (loads.empty())
        continue;
      auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
      if (!tensorType)
        continue;
      auto zeroAttr = builder.getZeroAttr(tensorType);
      Value zero = arith::ConstantOp::create(builder, op->getLoc(), tensorType,
                                             zeroAttr);
      mapping.map(operand, zero);
      toErase.push_back(zero.getDefiningOp());
    }

    Operation *cloned = builder.clone(*op, mapping);
    toErase.push_back(cloned);

    SmallVector<OpFoldResult> foldResults;
    if (succeeded(cloned->fold(foldResults)) && foldResults.size() == 1) {
      if (auto attr = dyn_cast<Attribute>(foldResults[0])) {
        Value constVal = arith::ConstantOp::create(builder, cloned->getLoc(),
                                                   cast<TypedAttr>(attr));
        mapping.map(op->getResult(0), constVal);
        toErase.push_back(constVal.getDefiningOp());
        continue;
      }
      if (auto val = dyn_cast<Value>(foldResults[0])) {
        mapping.map(op->getResult(0), val);
        continue;
      }
    } else {
      anyFoldFailed = true;
    }
    mapping.map(op->getResult(0), cloned->getResult(0));
  }

  Value clonedLeaf = mapping.lookup(leaf);
  bool isZero = matchPattern(clonedLeaf, m_AnyZeroFloat()) ||
                matchPattern(clonedLeaf, m_Zero());

  LLVM_DEBUG(llvm::dbgs() << "  Zero-preservation test: "
                          << (isZero ? "PRESERVES"
                                     : (anyFoldFailed ? "Fold failed!"
                                                      : "DOES NOT preserve"))
                          << " zero\n");

  for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
    if (*it && (*it)->use_empty())
      (*it)->erase();
  }

  return isZero;
}

static bool hasMaxReduceConsumer(Value value) {
  SmallVector<Value> worklist{value};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      auto reduceOp = dyn_cast<BlockwiseReduceOp>(owner);
      if (reduceOp && reduceOp.getReduceMethod() == ReduceMethod::Max)
        return true;

      if (isa<ViewLikeOpInterface>(owner) || isNarrowLoadShapeOp(owner)) {
        for (Value result : owner->getResults())
          worklist.push_back(result);
      }
    }
  }

  return false;
}

static bool useNeedsMaxNeutralFill(Operation *useOwner) {
  auto reduceOp = dyn_cast<BlockwiseReduceOp>(useOwner);
  if (reduceOp)
    return reduceOp.getReduceMethod() == ReduceMethod::Max;

  if (!isa<ViewLikeOpInterface>(useOwner) && !isNarrowLoadShapeOp(useOwner))
    return false;

  for (Value result : useOwner->getResults()) {
    if (hasMaxReduceConsumer(result))
      return true;
  }
  return false;
}

/// Choose the neutral fill value for masked-out lanes of a fusion-chain leaf
/// based on what `useOwner` is going to do with the tensor.
/// For `rock.blockwise_reduce max`, masked-out lanes must not influence the
/// reduction. The neutral element of max is the smallest representable value
/// of the element type, which depends on the type:
///   - IEEE-like floats (f32, f16, bf16, f64, f8E5M2, ...): -inf
///   - FP formats without inf (f4E2M1FN, f8E4M3FN, f8E4M3FNUZ): the most
///     negative finite value (`APFloat::getLargest(..., negative=true)`)
///   - Signless / signed integers: the signed minimum of the bit width.
///     Unsigned integers can't reach this helper (`BlockwiseLoadPtrOp`'s
///     NativeMemoryOpTypes only contains signless integers and arith doesn't
///     accept unsigned operands), so we don't special-case them here.
/// All other consumers (stores, sum reductions, etc.) get plain zero, which is
/// the neutral element for sum and a safe default for ops whose semantics
/// preserve zero across OOB lanes.
static Value createMaskFillValue(OpBuilder &builder, Location loc,
                                 RankedTensorType type, Operation *useOwner) {
  auto zeroFill = [&]() {
    return arith::ConstantOp::create(builder, loc, type,
                                     builder.getZeroAttr(type));
  };

  if (!useNeedsMaxNeutralFill(useOwner))
    return zeroFill();

  Type elemType = type.getElementType();
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    const llvm::fltSemantics &semantics = floatType.getFloatSemantics();
    APFloat fillValue = APFloat::semanticsHasInf(semantics)
                            ? APFloat::getInf(semantics, /*Negative=*/true)
                            : APFloat::getLargest(semantics, /*Negative=*/true);
    return arith::ConstantOp::create(
        builder, loc, type,
        SplatElementsAttr::get(type,
                               builder.getFloatAttr(elemType, fillValue)));
  }

  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    APInt fillValue = APInt::getSignedMinValue(intType.getWidth());
    return arith::ConstantOp::create(
        builder, loc, type,
        SplatElementsAttr::get(type,
                               builder.getIntegerAttr(elemType, fillValue)));
  }

  // Fallback to filling with zero.
  return zeroFill();
}

struct RockPreserveMaskedLoadSemanticsPass
    : public rock::impl::RockPreserveMaskedLoadSemanticsPassBase<
          RockPreserveMaskedLoadSemanticsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockPreserveMaskedLoadSemanticsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // ---- Phase 1: Analysis ----
  // Process each load independently. For each load with a non-trivial mask,
  // follow its fusion chain to find leaves. Skip leaves whose chains preserve
  // zero unless a max-reduction consumer needs the type's minimum value instead
  // of zero. Record the mask contribution for the rest.
  // Result: leafMasks[leaf] = paths from contributing non-trivial masks to the
  //         leaf shape for values that need re-masking.
  DenseMap<Value, SmallVector<std::pair<Value, SmallVector<Operation *>>>>
      leafMasks;
  DenseSet<Value> zeroPreserving;
  OpBuilder builder(funcOp.getContext());

  auto recordMaskCandidate = [&](Value candidate, Value mask,
                                 SmallVector<Operation *> shapeOps) {
    bool needsMaxNeutralFill = hasMaxReduceConsumer(candidate);
    if (!needsMaxNeutralFill && zeroPreserving.contains(candidate))
      return;

    if (!needsMaxNeutralFill && !leafMasks.count(candidate) &&
        fusionChainPreservesZero(candidate, builder)) {
      zeroPreserving.insert(candidate);
      return;
    }

    auto &paths = leafMasks[candidate];
    for (const auto &[existingMask, existingShapeOps] : paths)
      if (existingMask == mask && existingShapeOps == shapeOps)
        return;
    paths.push_back({mask, std::move(shapeOps)});
  };

  funcOp.walk([&](BlockwiseLoadPtrOp loadOp) {
    if (isTrivialMask(loadOp.getMaskTensor()))
      return;

    Value mask = loadOp.getMaskTensor();
    Value loadResult = loadOp.getResult();
    if (hasMaxReduceConsumer(loadResult))
      recordMaskCandidate(loadResult, mask, {});

    auto leaves = collectFusionChainLeaves(loadResult);

    for (auto &[leaf, shapeOps] : leaves)
      recordMaskCandidate(leaf, mask, std::move(shapeOps));
  });

  if (leafMasks.empty())
    return;

  LLVM_DEBUG(llvm::dbgs() << "Found " << leafMasks.size()
                          << " values needing OOB re-masking\n");

  // ---- Phase 2: Emit masking IR ----
  // For each remaining leaf, insert arith.select to restore a neutral value at
  // OOB positions.
  for (auto &[leaf, maskPaths] : leafMasks) {
    LLVM_DEBUG(llvm::dbgs() << "Inserting OOB mask for: " << leaf << "\n");

    Location loc = leaf.getLoc();
    auto leafType = cast<RankedTensorType>(leaf.getType());

    // Gather non-fusion uses before rewriting so that newly inserted selects do
    // not perturb the use-list walk.
    SmallVector<std::pair<Operation *, unsigned>> usesToReplace;
    for (OpOperand &use : leaf.getUses()) {
      Operation *owner = use.getOwner();
      if (!rock::isFusionOp(owner))
        usesToReplace.emplace_back(owner, use.getOperandNumber());
    }

    SmallVector<Value> masks;
    masks.reserve(maskPaths.size());
    for (const auto &[mask, shapeOps] : maskPaths)
      masks.push_back(materializeMaskPath(builder, mask, shapeOps));

    // Combine masks with `and` once, right after the leaf. Each materialized
    // mask path dominates every non-fusion use of the leaf, so the `and` can be
    // shared across them.
    builder.setInsertionPointAfterValue(leaf);
    auto it = masks.begin();
    Value combinedMask = *it;
    for (++it; it != masks.end(); ++it)
      combinedMask = arith::AndIOp::create(builder, loc, combinedMask, *it);

    // Insert one arith.select per non-fusion use, choosing the fill for the
    // consuming op (for example, -inf for floating-point max reductions).
    for (auto [owner, operandNumber] : usesToReplace) {
      builder.setInsertionPoint(owner);
      Value fill = createMaskFillValue(builder, loc, leafType, owner);
      Value safe =
          arith::SelectOp::create(builder, loc, combinedMask, leaf, fill);
      owner->setOperand(operandNumber, safe);
    }
  }
}
