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
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

/// Propagate a constant coordinate *diff* through a transform chain and return
/// the resulting constant change in the single linearized buffer offset, or
/// failure if it is not a compile-time constant.
///
/// This mirrors rocMLIR's index-diff rules (`IndexDiffUpdateRewritePattern` in
/// SugarToLoops.cpp): each transform maps an upper-space diff to a lower-space
/// diff. Because the input diff is a compile-time constant (a unit iv step) and
/// every rule is constant arithmetic, the whole propagation stays constant -
/// including through `Merge`/`Unmerge` reconstructions (e.g. the conv `gemmK`
/// packing) where a flattened affine map would keep opaque floordiv/mod.
///
/// `transforms` is ordered from the view (index 0) down to the buffer, as
/// produced by `untransform`. `diff` is indexed by the upper-space dims of
/// `transforms[0]`.
///
/// Carry-neutrality guard: rocMLIR keeps the running coordinate valid via carry
/// propagation on `Merge`. We instead want a single loop-invariant stride,
/// which is only valid when those carries do not change the linearized offset,
/// i.e. when the merged dim's lower dims are laid out contiguously underneath
/// (their buffer strides nest as the merge factors). When that does not hold
/// the offset is only piecewise-linear in the iv, and we bail.
static FailureOr<int64_t>
linearizedDiffStride(ArrayRef<TransformMapAttr> transforms,
                     DenseMap<unsigned, int64_t> diff) {
  if (transforms.empty())
    return failure();

  for (size_t mapIdx = 0; mapIdx < transforms.size(); ++mapIdx) {
    TransformMapAttr map = transforms[mapIdx];
    DenseMap<unsigned, int64_t> lower;
    auto upperDiff = [&](unsigned d) -> int64_t {
      auto it = diff.find(d);
      return it == diff.end() ? 0 : it->second;
    };

    for (TransformAttr t : map.getOps()) {
      ArrayRef<uint32_t> upperDims = t.getUpperDims();
      ArrayRef<uint32_t> lowerDims = t.getLowerDims();
      ArrayRef<int64_t> params = t.getParams();
      switch (t.getType()) {
      case TransformType::PassThrough:
      case TransformType::Pad:
      case TransformType::Slice:
        // Offset-preserving: a unit change upstairs is a unit change below.
        // (Pad only shifts validity, not the stride.)
        for (auto [pi, qi] : llvm::zip(upperDims, lowerDims))
          lower[qi] = upperDiff(pi);
        break;
      case TransformType::Embed: {
        // lowerDiff = sum_i params_i * upperDiff_i
        int64_t d = 0;
        for (auto [coef, pi] : llvm::zip(params, upperDims))
          d += coef * upperDiff(pi);
        lower[lowerDims[0]] = d;
        break;
      }
      case TransformType::Unmerge: {
        // lowerDiff = sum_i stride_i * upperDiff_i, where stride_i is the
        // product of the trailing bounds (the row-major stride of dim i).
        SmallVector<int64_t> strides = computeStrides(params);
        int64_t d = 0;
        for (size_t i = 0; i < params.size(); ++i)
          d += strides[i] * upperDiff(upperDims[i]);
        lower[lowerDims[0]] = d;
        break;
      }
      case TransformType::Merge: {
        // Decompose the merged-dim diff into its lower components using the
        // merge's own nested strides (P_j = product of trailing params).
        int64_t D = upperDiff(upperDims[0]);
        SmallVector<int64_t> P(params.size(), 1);
        for (int j = static_cast<int>(params.size()) - 2; j >= 0; --j)
          P[j] = P[j + 1] * params[j + 1];
        for (size_t j = 0; j < lowerDims.size(); ++j)
          lower[lowerDims[j]] = (D / P[j]) % params[j];

        if (D != 0) {
          // Carry-neutrality guard: the lower dims must nest contiguously in
          // the buffer (stride(q_j) == stride(q_{j+1}) * params_{j+1}). Compute
          // each lower dim's buffer stride by propagating its own unit diff
          // through the maps below this one.
          ArrayRef<TransformMapAttr> below = transforms.drop_front(mapIdx + 1);
          SmallVector<int64_t> s(lowerDims.size());
          for (size_t j = 0; j < lowerDims.size(); ++j) {
            DenseMap<unsigned, int64_t> unit;
            unit[lowerDims[j]] = 1;
            FailureOr<int64_t> sj = linearizedDiffStride(below, unit);
            if (failed(sj))
              return failure();
            s[j] = *sj;
          }
          for (size_t j = 0; j + 1 < lowerDims.size(); ++j)
            if (s[j] != s[j + 1] * params[j + 1])
              return failure(); // non-contiguous: offset is piecewise-linear
        }
        break;
      }
      case TransformType::ConstDim:
        for (unsigned qi : lowerDims)
          lower[qi] = 0;
        break;
      case TransformType::AddDim:
        // The upper dim has no lower counterpart and is dropped.
        break;
      case TransformType::Broadcast:
        // A nonzero diff into a broadcast dim wraps (position-dependent), so it
        // has no constant stride; a zero diff is harmless.
        for (auto [pi, qi] : llvm::zip(upperDims, lowerDims)) {
          if (upperDiff(pi) != 0)
            return failure();
          lower[qi] = 0;
        }
        break;
      default:
        return failure();
      }
    }
    diff = std::move(lower);
  }

  if (transforms.back().getLowerBounds().size() != 1)
    return failure();
  auto it = diff.find(0);
  return it == diff.end() ? 0 : it->second;
}

/// Returns true if the validity mask produced by `transforms` can vary with the
/// induction variable, i.e. some validity-impacting map (a Pad, or an Embed
/// that can go out of bounds) constrains a coordinate that is a function of an
/// iv-carrying input dim. When false, the mask is loop-invariant and may be
/// computed once before the loop and reused every iteration.
///
/// `transforms` is ordered from the upper view (index 0) down to the lower view
/// at the root, as produced by `untransform`. `ivPositions` are the upper-view
/// dims that carry the induction variable.
///
/// For a validity-impacting map at index `i`, its upper coordinates are
/// expressed as affine functions of the upper-view dims by composing the
/// transforms strictly above it (indices `[0, i)`); for the topmost map there
/// are no transforms above it, so its upper coordinates are the upper-view dims
/// directly (the identity). A padded `gemmM` while the iv lives in `gemmK`, for
/// example, leaves the mask invariant and is accepted.
static bool maskDependsOnIv(ArrayRef<TransformMapAttr> transforms,
                            ArrayRef<unsigned> ivPositions) {
  for (auto [i, t] : llvm::enumerate(transforms)) {
    if (!mapImpactsValidity(t))
      continue;
    SmallVector<unsigned> upperDims = validityImpactingUpperDims(t);
    if (upperDims.empty())
      continue;
    AffineMap above = composeTransforms(transforms.take_front(i));
    for (unsigned u : upperDims) {
      AffineExpr coord =
          above ? above.getResult(u) : getAffineDimExpr(u, t.getContext());
      for (unsigned p : ivPositions)
        if (coord.isFunctionOfDim(p))
          return true;
    }
  }
  return false;
}

/// True if `v` is defined inside `loop` (its induction var, an iter_arg, or any
/// value produced by an op nested in the loop body).
static bool isDefinedInLoop(Value v, scf::ForOp loop) {
  Operation *owner = v.getDefiningOp();
  if (!owner)
    owner = cast<BlockArgument>(v).getOwner()->getParentOp();
  return owner == loop.getOperation() || loop->isProperAncestor(owner);
}

/// True if `v` is `iv`, possibly reached through a chain of integer casts.
static bool isInductionVar(Value v, Value iv) {
  while (v != iv) {
    Operation *def = v.getDefiningOp();
    if (!def || !isa<arith::IndexCastOp, arith::IndexCastUIOp>(def))
      return false;
    v = def->getOperand(0);
  }
  return true;
}

/// Cast integer/index scalar `v` to the integer/index type `toTy`, inserting
/// the appropriate `arith` cast.
static Value castScalar(OpBuilder &b, Location loc, Value v, Type toTy) {
  Type fromTy = v.getType();
  if (fromTy == toTy)
    return v;
  if (isa<IndexType>(fromTy) || isa<IndexType>(toTy))
    return arith::IndexCastOp::create(b, loc, toTy, v);
  if (fromTy.getIntOrFloatBitWidth() > toTy.getIntOrFloatBitWidth())
    return arith::TruncIOp::create(b, loc, toTy, v);
  return arith::ExtSIOp::create(b, loc, toTy, v);
}

/// A LICM candidate is pair of:
///  - A transforms_to_ptr op inside a loop, which offset is affine-linear in
///  the iv.
///  - The resulting pointer stride.
///  - The root buffer (the block argument the transform chain bottoms out in),
///  used to detect candidates that alias the same base pointer.
struct Candidate {
  TransformsToPtrOp op;
  int64_t unitStride;
  Value rootBase;
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

  std::optional<Value> maybeIv =
      cast<LoopLikeOpInterface>(loop.getOperation()).getSingleInductionVar();
  if (!maybeIv)
    return bail("loop does not have exactly one induction variable");
  Value iv = *maybeIv;

  // Classify each transforms_to_ptr index: it must be either the loop iv (which
  // we will incrementalize) or loop-invariant (ignored). A third possibility is
  // pointer does not depend on the iv, which we skip.
  ValueRange extra = op.getExtraIndices();
  SmallVector<unsigned> ivPositions;
  for (auto [pos, idx] : llvm::enumerate(extra)) {
    if (isInductionVar(idx, iv)) {
      ivPositions.push_back(pos);
      continue;
    }
    // TODO: We may want to LICM in these cases too:
    // https://amd-hub.atlassian.net/browse/AIROCMLIR-1029
    if (isDefinedInLoop(idx, loop))
      return bail("an extra index is loop-variant but is not the iv");
  }
  if (ivPositions.empty())
    return bail("pointer does not depend on the iv (already loop-invariant)");

  // Walk the transform chain down to its root.
  SmallVector<TransformMapAttr> transforms;
  Value root;
  std::tie(root, std::ignore) = untransform(op.getSource(), transforms);
  if (!isa<BlockArgument>(root))
    return bail("transform chain root is not a block argument");
  cand.rootBase = root;

  // Our rewrite computes the mask once before the loop and reuses it every
  // iteration. That is only valid when the mask is loop-invariant, i.e. no
  // validity-impacting map (a Pad, or an Embed that can go out of bounds)
  // constrains a coordinate that depends on the iv. A Pad/Embed on a dimension
  // unrelated to the iv (e.g. a padded gemmM while the iv lives in gemmK) keeps
  // the mask invariant and is fine to hoist.
  if (maskDependsOnIv(transforms, ivPositions))
    return bail("validity mask depends on the iv (not loop-invariant)");

  // Compute the constant per-iteration (unit-iv) stride of the linearized
  // pointer offset by propagating a unit induction-variable step through the
  // transform chain (mirroring rocMLIR's index-diff rules). This is exact
  // through Merge/Unmerge reconstructions (e.g. the conv gemmK packing) where a
  // flattened affine map would keep opaque floordiv/mod, and it bails when a
  // Merge the iv flows through is not contiguous (offset only
  // piecewise-linear).
  DenseMap<unsigned, int64_t> ivDiff;
  for (unsigned p : ivPositions)
    ivDiff[p] = 1;
  FailureOr<int64_t> stride = linearizedDiffStride(transforms, ivDiff);
  if (failed(stride))
    return bail("offset has no compile-time-constant per-iteration stride "
                "(non-linear / non-contiguous in the iv)");

  // TODO: We may want to LICM in these cases too:
  // https://amd-hub.atlassian.net/browse/AIROCMLIR-1029
  if (*stride == 0)
    return bail("iv stride is zero (load is loop-invariant; nothing to "
                "incrementalize)");

  cand.op = op;
  cand.unitStride = stride.value();
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

  // Build the base pointer/mask before the loop, the constant stride and the
  // zero accumulator init per candidate.
  SmallVector<ReducedPtr> reduced;
  for (Candidate &cand : candidates) {
    // Rebuild the loop-invariant inputs in the preheader with the iv pinned to
    // the loop's lower bound. Seeding the clone map with iv -> lb means an
    // index that is the iv (or a cast of it, e.g. index_cast(iv) for an
    // index-typed loop) is re-expressed at lb, while loop-invariant indices are
    // referenced directly.
    IRMapping cloneMap;
    cloneMap.map(iv, lb);
    Value srcPre = cloneSliceBeforeLoop(b, cand.op.getSource(), loop, cloneMap);

    SmallVector<Value> baseIdx;
    for (Value idx : cand.op.getExtraIndices())
      baseIdx.push_back(cloneSliceBeforeLoop(b, idx, loop, cloneMap));

    auto ptrType = cast<RankedTensorType>(cand.op.getPointers().getType());
    Type maskType = cand.op.getMask().getType();
    auto baseOp =
        TransformsToPtrOp::create(b, loc, ptrType, maskType, srcPre, baseIdx);

    // The per-iteration offset increment is `step * unitStride`, expressed in
    // the pointer-offset element type. The offsets are integers independent of
    // the induction-variable type, so cast the
    // step to that type before multiplying by the constant stride.
    Type offsetTy = ptrType.getElementType();
    Value strideScalar = arith::MulIOp::create(
        b, loc, castScalar(b, loc, step, offsetTy),
        arith::ConstantOp::create(b, loc,
                                  b.getIntegerAttr(offsetTy, cand.unitStride)));
    Value strideSplat = triton::SplatOp::create(b, loc, ptrType, strideScalar);
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

  auto newLoop =
      scf::ForOp::create(b, loc, lb, loop.getUpperBound(), step, newInits);

  // Map old body values into the new body.
  IRMapping bodyMap;
  bodyMap.map(iv, newLoop.getInductionVar());
  for (unsigned i = 0; i < numOrig; ++i)
    bodyMap.map(loop.getRegionIterArg(i), newLoop.getRegionIterArg(i));

  b.setInsertionPointToStart(newLoop.getBody());

  // Reconstruct each candidate pointer inside the body as basePtr + acc (a
  // single add that RockToTTIR/RockTensorToTritonPtr lower to tt.addptr), and
  // route the loop-invariant mask.
  //
  // TODO: This is a bit dumb, we could do this without a new AddIOp op.
  // However, we need to do this because RockTensorToTritonPtr expects to have a
  // base pointer plus a loop-carried integer offset pointers in the loop:
  // - The iter_arg: a pure integer offset.
  // - The base pointer
  // The only way to collapse to a single addi (simpler, and might affect
  // performance) would be to adapt RockTensorToTritonPtr lowering to recognize a
  // tt.addptr recurrence carried through an scf.for iter_arg.
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
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // Re-walk after each rewrite: tryHoistInvariantTransforms replaces the loop
  // op, so collected handles would dangle. A reduced loop has no remaining
  // candidates (they become preheader ops + iter_args), so this terminates.
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
