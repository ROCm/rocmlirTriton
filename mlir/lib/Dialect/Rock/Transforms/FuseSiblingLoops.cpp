//===- FuseSiblingLoops.cpp - merge decomposed sibling K-loops ----------===//
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
// =============================================================================
//
// DecomposeNonPow2Tiles splits one non-power-of-two rock.gridwise_gemm into
// several power-of-two sub-gemms; GridwiseGemmToBlockwise then lowers each into
// its own scf.for K-loop (createMainLoop + rock.blockwise_gemm). All those
// sibling loops share identical K bounds (0 .. K/kPerBlock, step 1) and reload
// the shared operand (B for an M-split, A for an N-split).
//
// This pass fuses sibling scf.for loops that live in the same block, have
// identical lower/upper bounds and step, and are mutually independent, into a
// single loop. The fusion itself reuses mlir::fuseIndependentSiblingForLoops,
// which does no bound or legality check, so this pass owns both.
//
// Bounds are compared by *constant value* rather than SSA identity, because
// each decomposed sub-gemm builds its own 0 / 1 / kIterations bound constants.
//
// The fused loop is anchored at the first loop, so the later loop's operands
// (bound/accumulator constants, grid-coordinate computations, and the sliced
// A/B-view transform chains the decomposition leaves between the loops) must be
// defined before it. The pass hoists those pure ops above the anchor as needed.
// CSE run after this pass then deduplicates the now-co-located shared operand
// loads (and the duplicated grid-coordinate computations feeding them).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFUSESIBLINGLOOPSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fuse-sibling-loops"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockFuseSiblingLoopsPass
    : public rock::impl::RockFuseSiblingLoopsPassBase<
          RockFuseSiblingLoopsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

/// Two loops can be fused only if their lower/upper bounds and step are equal,
/// so they share an iteration space. Each bound matches if it is the same SSA
/// value (covers dynamic bounds computed once and reused by both loops) or has
/// the same constant value (the common case: each decomposed sub-gemm
/// materializes its own bound constants, so SSA identity does not hold).
static bool haveIdenticalBounds(scf::ForOp a, scf::ForOp b) {
  auto sameBound = [](Value x, Value y) -> bool {
    if (x == y)
      return true;
    std::optional<int64_t> cx = getConstantIntValue(x);
    std::optional<int64_t> cy = getConstantIntValue(y);
    return cx && cy && *cx == *cy;
  };
  return sameBound(a.getLowerBound(), b.getLowerBound()) &&
         sameBound(a.getUpperBound(), b.getUpperBound()) &&
         sameBound(a.getStep(), b.getStep());
}

/// Ensure `v` dominates `source` (the fusion anchor), hoisting if needed.
///
/// If `v`'s defining op sits after `source` in the same block, it is moved
/// above `source` -- but only if it is pure and its own operands can likewise
/// be made to dominate (recursively). This covers the loop-invariant constants
/// and the sliced A/B-view rock.transform chains the decomposition leaves
/// between the sibling loops. A side-effecting op, or a real SSA dependence on
/// `source`, makes fusion illegal and returns false. Same-block ordering uses
/// isBeforeInBlock so it stays correct as ops are moved.
static bool hoistToDominate(Value v, scf::ForOp source,
                            DominanceInfo &domInfo) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return true; // Block argument: dominates source.
  Operation *anchor = source.getOperation();
  if (defOp == anchor)
    return false; // Real dependence on source's result.
  if (defOp->getBlock() != anchor->getBlock())
    return domInfo.properlyDominates(defOp, anchor);
  if (defOp->isBeforeInBlock(anchor))
    return true;
  if (!isMemoryEffectFree(defOp))
    return false;
  for (Value operand : defOp->getOperands())
    if (!hoistToDominate(operand, source, domInfo))
      return false;
  defOp->moveBefore(anchor);
  return true;
}

/// Make every value `target` (which occurs after `source` in the same block)
/// consumes dominate `source` so the fused loop, created at `source`, is valid.
static bool makeTargetOperandsDominate(scf::ForOp target, scf::ForOp source,
                                       DominanceInfo &domInfo) {
  for (Value operand : target->getOperands())
    if (!hoistToDominate(operand, source, domInfo))
      return false;

  bool ok = true;
  visitUsedValuesDefinedAbove(target->getRegions(), [&](OpOperand *operand) {
    if (!hoistToDominate(operand->get(), source, domInfo))
      ok = false;
  });
  return ok;
}

/// Greedily fuse sibling scf.for loops within a single block. Loops are visited
/// in program order; the first loop of a compatible group becomes the anchor
/// (`source`) and every later loop with identical constant bounds that is legal
/// to fuse is merged into it.
static void fuseSiblingsInBlock(Block &block, IRRewriter &rewriter) {
  SmallVector<scf::ForOp> loops(block.getOps<scf::ForOp>());
  if (loops.size() < 2)
    return;

  // Track which loops were already consumed by an earlier fusion.
  llvm::SmallPtrSet<Operation *, 8> fused;

  for (size_t i = 0; i < loops.size(); ++i) {
    if (fused.contains(loops[i].getOperation()))
      continue;
    scf::ForOp source = loops[i];

    for (size_t j = i + 1; j < loops.size(); ++j) {
      scf::ForOp target = loops[j];
      if (fused.contains(target.getOperation()))
        continue;
      if (!haveIdenticalBounds(source, target)) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping fusion of loops " << i << " and "
                                << j << ": bounds differ\n");
        continue;
      }

      // Recompute dominance against the current (possibly already fused)
      // anchor before each merge, since previous fusions mutate the block.
      DominanceInfo domInfo(source->getParentOp());
      if (!makeTargetOperandsDominate(target, source, domInfo)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping fusion of loops " << i << " and " << j
                   << ": target operands cannot be made to dominate the "
                      "anchor (side effect or real dependence)\n");
        continue;
      }

      // fuseIndependentSiblingForLoops creates a new loop after `source`,
      // clones both bodies into it, and erases both originals. Record `target`
      // before the call, since it is a dangling pointer afterwards.
      LLVM_DEBUG(llvm::dbgs()
                 << "Fusing loop " << j << " into anchor loop " << i << "\n");
      fused.insert(target.getOperation());
      source = fuseIndependentSiblingForLoops(target, source, rewriter);
    }
  }
}

void RockFuseSiblingLoopsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping non-kernel func @" << func.getName() << "\n");
    return;
  }

  // Fusion interleaves the sibling loop bodies, which is only safe when the
  // loops are independent. We prove that purely from SSA dependences (see
  // makeTargetOperandsDominate), which is sufficient only while the kernel is
  // value-semantic, i.e. every op is memory-effect-free -- the case this early
  // in the pipeline (before bufferization, no memref/ptr ops). If any op has a
  // memory effect we can no longer rule out a memory dependence between the
  // loops, so error out rather than risk silently reordering memory operations.
  Operation *funcOp = func.getOperation();
  if (func.walk([&](Operation *op) {
            // Skip the func itself (FuncOp is not memory-effect-free) and check
            // only the ops it contains.
            if (op == funcOp || isMemoryEffectFree(op))
              return WalkResult::advance();
            op->emitError("rock-fuse-sibling-loops requires value-semantic IR "
                          "but found an op with memory effects");
            return WalkResult::interrupt();
          })
          .wasInterrupted())
    return signalPassFailure();

  IRRewriter rewriter(&getContext());

  // Process every block (loops are only fused with siblings in the same block).
  func.walk([&](Block *block) { fuseSiblingsInBlock(*block, rewriter); });
}
