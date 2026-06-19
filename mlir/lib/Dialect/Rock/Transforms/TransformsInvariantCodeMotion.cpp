//===- TransformsInvariantCodeMotion.cpp - Hoist in-loop pointer arith ----===//
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

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTRANSFORMSINVARIANTCODEMOTIONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-transforms-invariant-code-motion"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockTransformsInvariantCodeMotionPass
    : public rock::impl::RockTransformsInvariantCodeMotionPassBase<
          RockTransformsInvariantCodeMotionPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

namespace {

/// Returns true if `e` is affine-linear in dimension `dim`, i.e. `dim` only
/// appears scaled by constants inside additions and never flows through a
/// mod/floordiv/ceildiv or a multiplication by a non-constant. Such an
/// expression has a constant partial "slope" with respect to `dim`, which is
/// what makes a constant per-iteration stride valid.
static bool isAffineExprLinearInDim(AffineExpr e, unsigned dim) {
  switch (e.getKind()) {
  case AffineExprKind::Constant:
  case AffineExprKind::DimId:
  case AffineExprKind::SymbolId:
    return true;
  case AffineExprKind::Add: {
    auto bin = cast<AffineBinaryOpExpr>(e);
    return isAffineExprLinearInDim(bin.getLHS(), dim) &&
           isAffineExprLinearInDim(bin.getRHS(), dim);
  }
  case AffineExprKind::Mul: {
    auto bin = cast<AffineBinaryOpExpr>(e);
    // Affine multiplications always have a constant operand; `dim` may only
    // appear on the non-constant side, where it stays linearly scaled.
    if (isa<AffineConstantExpr>(bin.getRHS()))
      return isAffineExprLinearInDim(bin.getLHS(), dim);
    if (isa<AffineConstantExpr>(bin.getLHS()))
      return isAffineExprLinearInDim(bin.getRHS(), dim);
    return false;
  }
  case AffineExprKind::Mod:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    auto bin = cast<AffineBinaryOpExpr>(e);
    // `dim` must not flow through a div/mod: those are non-linear in `dim`.
    return !bin.getLHS().isFunctionOfDim(dim) &&
           !bin.getRHS().isFunctionOfDim(dim);
  }
  }
  return false;
}

/// Compute the constant change in `offset` when every dim in `ivPositions` is
/// incremented by one (all simultaneously, as the loop advances the iv).
/// Returns failure if the slope is not a compile-time constant (e.g. the offset
/// is non-linear in the iv after all).
static FailureOr<int64_t> ivUnitStride(AffineExpr offset,
                                       ArrayRef<unsigned> ivPositions,
                                       unsigned numDims, unsigned numSymbols) {
  MLIRContext *ctx = offset.getContext();
  SmallVector<AffineExpr> repAtZero, repAtOne;
  for (unsigned i = 0; i < numDims; ++i) {
    repAtZero.push_back(getAffineDimExpr(i, ctx));
    repAtOne.push_back(getAffineDimExpr(i, ctx));
  }
  for (unsigned p : ivPositions) {
    repAtZero[p] = getAffineConstantExpr(0, ctx);
    repAtOne[p] = getAffineConstantExpr(1, ctx);
  }
  AffineExpr atZero = offset.replaceDims(repAtZero);
  AffineExpr atOne = offset.replaceDims(repAtOne);
  AffineExpr diff = simplifyAffineExpr(atOne - atZero, numDims, numSymbols);
  if (auto c = dyn_cast<AffineConstantExpr>(diff))
    return c.getValue();
  return failure();
}

/// True if `v` is defined inside `loop` (its induction var, an iter_arg, or any
/// value produced by an op nested in the loop body).
static bool isDefinedInLoop(Value v, scf::ForOp loop) {
  Operation *owner = v.getDefiningOp();
  if (!owner)
    owner = cast<BlockArgument>(v).getOwner()->getParentOp();
  return owner == loop.getOperation() || loop->isProperAncestor(owner);
}

/// A LICM candidate is pair of:
///  - A transforms_to_ptr op inside a loop, which offset is affine-linear in the iv.
///  - The resulting pointer stride.
struct Candidate {
  TransformsToPtrOp op;
  int64_t unitStride;
};

/// Decide whether the passed transforms_to_ptr op can be LICM'ed.
static bool analyzeCandidate(TransformsToPtrOp op, scf::ForOp loop,
                             Candidate &cand) {
  auto bail = [&](const char *reason) {
    LLVM_DEBUG(llvm::dbgs() << "[analyzeCandidate] skip " << op.getLoc() << ": "
                            << reason << "\n");
    return false;
  };

  if (op->getBlock() != loop.getBody())
    return bail("op is not a direct child of the loop body");

  // Defensive: scf.for has a single induction variable by construction, but
  // some loop-like ops carry several. getSingleInductionVar() returns the iv
  // only when there is exactly one, so this also guards a future generalization
  // of the pass to other loop-like ops.
  std::optional<Value> maybeIv =
      cast<LoopLikeOpInterface>(loop.getOperation()).getSingleInductionVar();
  if (!maybeIv)
    return bail("loop does not have exactly one induction variable");
  Value iv = *maybeIv;

  // Prototype: only handle i32 loops, matching the i32 pointer offsets.
  if (!loop.getStep().getType().isInteger(32))
    return bail("loop induction variable is not i32");

  // Get the transforms_to_ptr indices and check whether they are the iv of the loop,
  // or loop-invariant.
  ValueRange extra = op.getExtraIndices();
  SmallVector<unsigned> ivPositions;
  for (auto [pos, idx] : llvm::enumerate(extra)) {
    if (idx == iv) {
      ivPositions.push_back(pos);
      continue;
    }
    if (isDefinedInLoop(idx, loop))
      return bail("an extra index is loop-variant but is not the iv");
  }
  if (ivPositions.empty())
    return bail("pointer does not depend on the iv (already loop-invariant)");

  // Walk the transform chain. A validity-impacting map (Pad / invalidatable
  // Embed) means the mask depends on coordinates and is not safe to hoist.
  SmallVector<TransformMapAttr> transforms;
  auto [root, isBig] = untransform(op.getSource(), transforms);
  (void)isBig;
  if (!isa<BlockArgument>(root))
    return bail("transform chain root is not a block argument");

  // Our rewrite computes the mask once before the loop and reuses it 
  // every iteration. This trick is only valid if the mask doesn't change
  // with the IV. Here we make sure that this is the case.
  for (TransformMapAttr t : transforms)
    if (mapImpactsValidity(t))
      return bail("transform chain has a validity-impacting map "
                  "(mask is not loop-invariant)");

  // Collapse the whole transform chain into one affine map.
  AffineMap composed = composeTransforms(transforms);
  if (!composed || composed.getNumResults() != 1)
    return bail("composed transform map is null or not single-result");
  AffineExpr offset = composed.getResult(0);

  // The composed map's leading input dims are the extra indices (in order),
  // followed by the per-output-dim ranges. Require linearity in every dim that
  // carries the induction variable.
  for (unsigned p : ivPositions)
    if (!isAffineExprLinearInDim(offset, p))
      return bail("offset is not affine-linear in the iv");

  FailureOr<int64_t> stride = ivUnitStride(offset, ivPositions,
                                           composed.getNumDims(),
                                           composed.getNumSymbols());
  if (failed(stride))
    return bail("offset has no compile-time-constant per-iteration stride");

  if (*stride == 0)
    return bail("iv stride is zero (load is loop-invariant; nothing to "
                "incrementalize)");

  cand.op = op;
  cand.unitStride = *stride;
  return true;
}

/// Clone, just before `loop`, the in-loop ops that define `v`, so that `v`
/// becomes available in the loop preheader. Ops defined outside the loop are
/// referenced directly. `map` caches and returns the rewired value.
static Value cloneSliceBeforeLoop(OpBuilder &b, Value v, scf::ForOp loop,
                                  IRMapping &map) {
  if (Value cached = map.lookupOrNull(v))
    return cached;
  if (!isDefinedInLoop(v, loop))
    return v;
  Operation *def = v.getDefiningOp();
  assert(def && "loop-defined value without a defining op should be the iv, "
                "which is excluded by analyzeCandidate");
  for (Value operand : def->getOperands())
    cloneSliceBeforeLoop(b, operand, loop, map);
  // clone() remaps operands via `map` and records result mappings into `map`.
  b.clone(*def, map);
  return map.lookup(v);
}

/// Per-candidate preheader values produced for the LICM'ed loop.
struct ReducedPtr {
  TransformsToPtrOp op; // original in-loop op (to be removed)
  Value basePtr;        // loop-invariant pointer tensor at iv == lb
  Value baseMask;       // loop-invariant mask
  Value strideSplat;    // constant per-iteration offset increment
  Value accInit;        // zero offset accumulator initializer
};

/// Try to LICM all eligible transforms_to_ptr ops in `loop`.
/// Returns true (and rewrites the loop) if at least one was reduced.
static bool tryHoistInvariantTransforms(scf::ForOp loop) {
  SmallVector<Candidate> candidates;
  for (Operation &o : loop.getBody()->without_terminator()) {
    if (auto tp = dyn_cast<TransformsToPtrOp>(&o)) {
      Candidate cand;
      if (analyzeCandidate(tp, loop, cand))
        candidates.push_back(cand);
    }
  }
  if (candidates.empty())
    return false;

  Location loc = loop.getLoc();
  Value iv = loop.getInductionVar();
  Value lb = loop.getLowerBound();
  Value step = loop.getStep();

  OpBuilder b(loop);

  // Build the base pointer/mask before the loop, the constant stride and the zero
  // accumulator init per candidate.
  SmallVector<ReducedPtr> reduced;
  for (Candidate &cand : candidates) {
    IRMapping cloneMap;
    Value srcPre = cloneSliceBeforeLoop(b, cand.op.getSource(), loop, cloneMap);

    SmallVector<Value> baseIdx;
    for (Value idx : cand.op.getExtraIndices())
      baseIdx.push_back(idx == iv ? lb : idx);

    auto ptrType = cast<RankedTensorType>(cand.op.getPointers().getType());
    Type maskType = cand.op.getMask().getType();
    auto baseOp =
        TransformsToPtrOp::create(b, loc, ptrType, maskType, srcPre, baseIdx);

    Value strideScalar = arith::MulIOp::create(
        b, loc, step,
        arith::ConstantOp::create(
            b, loc, b.getI32IntegerAttr(static_cast<int32_t>(cand.unitStride))));
    Value strideSplat =
        triton::SplatOp::create(b, loc, ptrType, strideScalar);
    Value accInit = arith::ConstantOp::create(
        b, loc, cast<TypedAttr>(b.getZeroAttr(ptrType)));

    reduced.push_back({cand.op, baseOp.getPointers(), baseOp.getMask(),
                       strideSplat, accInit});
  }

  // New iter_args: the original ones followed by one offset accumulator per
  // candidate, initialized to zero.
  unsigned numOrig = loop.getInitArgs().size();
  SmallVector<Value> newInits(loop.getInitArgs().begin(),
                              loop.getInitArgs().end());
  for (const ReducedPtr &r : reduced)
    newInits.push_back(r.accInit);

  auto newLoop = scf::ForOp::create(b, loc, lb, loop.getUpperBound(), step,
                                    newInits);

  // Map old body values into the new body.
  IRMapping bodyMap;
  bodyMap.map(iv, newLoop.getInductionVar());
  for (unsigned i = 0; i < numOrig; ++i)
    bodyMap.map(loop.getRegionIterArg(i), newLoop.getRegionIterArg(i));

  b.setInsertionPointToStart(newLoop.getBody());

  // Reconstruct each candidate pointer inside the body as basePtr + acc (a
  // single add that RockToTTIR/FuncToTritonFunc lower to tt.addptr), and route
  // the loop-invariant mask.
  // 
  // TODO: This is a bit dumb, we could do this without a new AddIOp op. However,
  // we need to do this because FuncToTritonFunc expects to have
  // a base pointer plus a loop-carried integer offset pointers in the loop:
  // - The iter_arg: a pure integer offset.
  // - The base pointer
  // The only way to collapse to a single addi (simpler, and might affect performance)
  // would be to adapt FuncToTritonFunc lowering to recognize a tt.addptr 
  // recurrence carried through an scf.for iter_arg. 
  llvm::SmallPtrSet<Operation *, 4> candidateOps;
  for (auto [j, r] : llvm::enumerate(reduced)) {
    candidateOps.insert(r.op.getOperation());
    Value acc = newLoop.getRegionIterArg(numOrig + j);
    Value ptr = arith::AddIOp::create(b, loc, r.basePtr, acc);
    bodyMap.map(r.op.getPointers(), ptr);
    bodyMap.map(r.op.getMask(), r.baseMask);
  }

  for (Operation &o : loop.getBody()->without_terminator()) {
    if (candidateOps.contains(&o))
      continue;
    b.clone(o, bodyMap);
  }

  // Build the new yield: original yields (remapped) plus the incremented
  // accumulator for each carried candidate.
  auto oldYield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<Value> newYields;
  for (Value y : oldYield.getResults())
    newYields.push_back(bodyMap.lookupOrDefault(y));
  for (auto [j, r] : llvm::enumerate(reduced)) {
    Value acc = newLoop.getRegionIterArg(numOrig + j);
    newYields.push_back(arith::AddIOp::create(b, loc, acc, r.strideSplat));
  }
  scf::YieldOp::create(b, loc, newYields);

  // Re-wire the original results and drop the old loop.
  for (unsigned i = 0; i < numOrig; ++i)
    loop.getResult(i).replaceAllUsesWith(newLoop.getResult(i));
  loop.erase();
  return true;
}

} // end anonymous namespace

void RockTransformsInvariantCodeMotionPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Re-walk after each rewrite: tryHoistInvariantTransforms replaces the loop op, so
  // collected handles would dangle. A reduced loop has no remaining candidates
  // (they become preheader ops + iter_args), so this terminates.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp f) { loops.push_back(f); });
    for (scf::ForOp f : loops) {
      if (tryHoistInvariantTransforms(f)) {
        changed = true;
        break;
      }
    }
  }
}
