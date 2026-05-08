//===- MaskNonZeroPreservingFusions.cpp - Re-mask OOB after fusions -------===//
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
// This pass runs AFTER RockLowerBlockwiseToPtrPass and BEFORE
// RockTransformsToPointerArithPass.
//
// When a BlockwiseLoadPtrOp has Pad or Embed transforms, its mask tensor marks
// out-of-bounds (OOB) positions. tt.load returns 0 for those positions. If the
// loaded tile feeds into a non-zero-preserving fusion (e.g. arith.addf %tile,
// 1.0), OOB positions become non-zero and corrupt the GEMM result (tt.dot has
// no mask).
//
// This pass detects such cases and inserts:
//   %safe = arith.select %mask, %fused_result, %zero
// at the end of the fusion chain, ensuring OOB positions remain zero.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <limits>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKMASKNONZEROPRESERVINGFUSIONSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-mask-non-zero-preserving-fusions"

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

/// Starting from a load result, follow uses through fusion ops and collect
/// "leaf" values — fusion op results that have at least one non-fusion use.
static void collectFusionChainLeaves(Value loadResult,
                                     SmallVectorImpl<Value> &leaves) {
  SmallVector<Value> worklist;
  DenseSet<Operation *> visited;
  worklist.push_back(loadResult);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (!rock::isFusionOp(owner) || visited.count(owner))
        continue;
      visited.insert(owner);

      Value result = owner->getResult(0);
      bool hasNonFusionUse = false;
      bool hasFusionUse = false;
      for (OpOperand &resultUse : result.getUses()) {
        if (rock::isFusionOp(resultUse.getOwner()))
          hasFusionUse = true;
        else
          hasNonFusionUse = true;
      }
      if (hasNonFusionUse)
        leaves.push_back(result);
      if (hasFusionUse)
        worklist.push_back(result);
    }
  }
}

/// Trace backwards from a value through fusion ops and collect all
/// BlockwiseLoadPtrOp ops that feed into it (directly or indirectly).
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
  if (!rock::isFusionOp(defOp))
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
                          << (isZero ? "PRESERVES" : (anyFoldFailed ? "Fold failed!" : "DOES NOT preserve"))
                          << " zero\n");

  for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
    if (*it && (*it)->use_empty())
      (*it)->erase();
  }

  return isZero;
}

static Value createMaskFillValue(OpBuilder &builder, Location loc,
                                 RankedTensorType type, Operation *useOwner) {
  if (auto reduceOp = dyn_cast<BlockwiseReduceOp>(useOwner)) {
    Type elemType = type.getElementType();
    if (reduceOp.getReduceMethod() == ReduceMethod::Max &&
        isa<FloatType>(elemType)) {
      return rock::createConstantFloatOp(
          builder, loc, type, elemType,
          -std::numeric_limits<float>::infinity(), APFloat::opOK);
    }
  }

  return arith::ConstantOp::create(builder, loc, type,
                                   builder.getZeroAttr(type));
}

struct RockMaskNonZeroPreservingFusionsPass
    : public rock::impl::RockMaskNonZeroPreservingFusionsPassBase<
          RockMaskNonZeroPreservingFusionsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockMaskNonZeroPreservingFusionsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // ---- Phase 1: Analysis ----
  // Process each load independently. For each load with a non-trivial mask,
  // follow its fusion chain to find leaves. Skip leaves whose chains preserve
  // zero. Record the mask contribution for the rest.
  // Result: leafMasks[leaf] = set of non-trivial masks from contributing loads
  //         that need re-masking.
  DenseMap<Value, SetVector<Value>> leafMasks;
  DenseSet<Value> zeroPreserving;
  OpBuilder builder(funcOp.getContext());

  funcOp.walk([&](BlockwiseLoadPtrOp loadOp) {
    if (isTrivialMask(loadOp.getMaskTensor()))
      return;

    Value mask = loadOp.getMaskTensor();
    SmallVector<Value> leaves;
    collectFusionChainLeaves(loadOp.getResult(), leaves);

    for (Value leaf : leaves) {
      if (zeroPreserving.contains(leaf))
        continue;
      if (!leafMasks.count(leaf) && fusionChainPreservesZero(leaf, builder)) {
        zeroPreserving.insert(leaf);
        continue;
      }
      leafMasks[leaf].insert(mask);
    }
  });

  if (leafMasks.empty())
    return;

  LLVM_DEBUG(llvm::dbgs() << "Found " << leafMasks.size()
                          << " non-zero-preserving fusion chain leaves\n");

  // ---- Phase 2: Emit masking IR ----
  // For each remaining leaf, insert arith.select to re-zero OOB positions.
  for (auto &[leaf, masks] : leafMasks) {
    LLVM_DEBUG(llvm::dbgs()
               << "Non-zero-preserving fusion chain, inserting mask for: "
               << leaf << "\n");

    Location loc = leaf.getLoc();
    auto leafType = cast<RankedTensorType>(leaf.getType());

    SmallVector<std::pair<Operation *, unsigned>> usesToReplace;
    for (OpOperand &use : leaf.getUses()) {
      Operation *owner = use.getOwner();
      if (!rock::isFusionOp(owner))
        usesToReplace.emplace_back(owner, use.getOperandNumber());
    }

    for (auto [owner, operandNumber] : usesToReplace) {
      builder.setInsertionPoint(owner);
      auto it = masks.begin();
      Value combinedMask = *it;
      for (++it; it != masks.end(); ++it)
        combinedMask = arith::AndIOp::create(builder, loc, combinedMask, *it);

      Value fill = createMaskFillValue(builder, loc, leafType, owner);
      Value safe =
          arith::SelectOp::create(builder, loc, combinedMask, leaf, fill);
      owner->setOperand(operandNumber, safe);
    }
  }
}
