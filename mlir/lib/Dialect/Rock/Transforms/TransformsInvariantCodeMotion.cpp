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

#include "PointerArithExpand.h"

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
#include "mlir/IR/Matchers.h"
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
/// Propagate an upper-space coordinate `diff` through a single transform `map`,
/// returning the resulting lower-space diff. This is the pure per-transform
/// index-diff rule set (no carry-neutrality guard); both `linearizedDiffStride`
/// and the carry-candidate analysis build on it. Returns failure for diffs that
/// are not compile-time constants (a nonzero diff into a Broadcast) or for
/// unsupported transform types.
static FailureOr<DenseMap<unsigned, int64_t>>
applyDiffOneMap(TransformMapAttr map, const DenseMap<unsigned, int64_t> &diff) {
  DenseMap<unsigned, int64_t> lower;
  auto upper = [&](unsigned d) -> int64_t {
    auto it = diff.find(d);
    return it == diff.end() ? 0 : it->second;
  };

  for (TransformAttr t : map.getOps()) {
    ArrayRef<uint32_t> p = t.getUpperDims();
    ArrayRef<uint32_t> q = t.getLowerDims();
    ArrayRef<int64_t> e = t.getParams();
    switch (t.getType()) {
    case TransformType::PassThrough:
    case TransformType::Pad:
    case TransformType::Slice:
      // Offset-preserving: a unit change upstairs is a unit change below.
      // (Pad only shifts validity, not the stride.)
      for (auto [pi, qi] : llvm::zip(p, q))
        lower[qi] = upper(pi);
      break;
    case TransformType::Embed: {
      // lowerDiff = sum_i e_i * upperDiff_i
      int64_t d = 0;
      for (auto [coef, pi] : llvm::zip(e, p))
        d += coef * upper(pi);
      lower[q[0]] = d;
      break;
    }
    case TransformType::Unmerge: {
      // lowerDiff = sum_i f_i * upperDiff_i, f_i = product of trailing bounds.
      int64_t d = 0, f = 1;
      for (int i = static_cast<int>(e.size()) - 1; i >= 0; --i) {
        d += f * upper(p[i]);
        f *= e[i];
      }
      lower[q[0]] = d;
      break;
    }
    case TransformType::Merge: {
      // Decompose the merged-dim diff into its lower components using the
      // merge's own nested strides (P_j = product of trailing params).
      int64_t D = upper(p[0]);
      SmallVector<int64_t> P(e.size(), 1);
      for (int j = static_cast<int>(e.size()) - 2; j >= 0; --j)
        P[j] = P[j + 1] * e[j + 1];
      for (size_t j = 0; j < q.size(); ++j)
        lower[q[j]] = (D / P[j]) % e[j];
      break;
    }
    case TransformType::ConstDim:
      for (unsigned qi : q)
        lower[qi] = 0;
      break;
    case TransformType::AddDim:
      // The upper dim has no lower counterpart and is dropped.
      break;
    case TransformType::Broadcast:
      // A nonzero diff into a broadcast dim wraps (position-dependent), so it
      // has no constant stride; a zero diff is harmless.
      for (auto [pi, qi] : llvm::zip(p, q)) {
        if (upper(pi) != 0)
          return failure();
        lower[qi] = 0;
      }
      break;
    default:
      return failure();
    }
  }
  return lower;
}

static FailureOr<int64_t>
linearizedDiffStride(ArrayRef<TransformMapAttr> transforms,
                     DenseMap<unsigned, int64_t> diff) {
  if (transforms.empty())
    return failure();

  for (size_t mapIdx = 0; mapIdx < transforms.size(); ++mapIdx) {
    TransformMapAttr map = transforms[mapIdx];
    auto upper = [&](unsigned d) -> int64_t {
      auto it = diff.find(d);
      return it == diff.end() ? 0 : it->second;
    };

    // Carry-neutrality guard: for any Merge the iv flows through (nonzero merged
    // diff), the lower dims must nest contiguously in the buffer
    // (stride(q_j) == stride(q_{j+1}) * e_{j+1}); otherwise the offset is only
    // piecewise-linear in the iv and there is no single constant stride.
    for (TransformAttr t : map.getOps()) {
      if (t.getType() != TransformType::Merge)
        continue;
      if (upper(t.getUpperDims()[0]) == 0)
        continue;
      ArrayRef<uint32_t> q = t.getLowerDims();
      ArrayRef<int64_t> e = t.getParams();
      ArrayRef<TransformMapAttr> below = transforms.drop_front(mapIdx + 1);
      SmallVector<int64_t> s, ext;
      for (size_t j = 0; j < q.size(); ++j) {
        // Size-1 dims are skipped: they contribute nothing to the offset,
        // making their buffer stride irrelevant.
        if (e[j] == 1)
          continue;
        DenseMap<unsigned, int64_t> unit;
        unit[q[j]] = 1;
        FailureOr<int64_t> sj = linearizedDiffStride(below, unit);
        if (failed(sj))
          return failure();
        s.push_back(*sj);
        ext.push_back(e[j]);
      }
      for (size_t j = 0; j + 1 < s.size(); ++j)
        if (s[j] != s[j + 1] * ext[j + 1])
          return failure(); // non-contiguous: offset is piecewise-linear
    }

    FailureOr<DenseMap<unsigned, int64_t>> lower = applyDiffOneMap(map, diff);
    if (failed(lower))
      return failure();
    diff = std::move(*lower);
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

/// How an in-loop `transforms_to_ptr` is incrementalized across iterations.
enum class CandKind {
  /// The linearized offset is affine in the iv and the mask is loop-invariant:
  /// carry a single integer accumulator (`base + k*stride`).
  Affine,
  /// The iv flows through a non-contiguous `Merge` and/or the mask depends on
  /// the iv: carry the merge's decomposed lower coordinates (one
  /// `tensor<tile x i32>` each) and rebuild the offset + mask every iteration.
  Carry,
};

/// A LICM candidate: an in-loop `transforms_to_ptr` op together with the
/// information needed to incrementalize it. `rootBase` is the block argument the
/// transform chain bottoms out in (used to detect aliasing of the same buffer).
struct Candidate {
  CandKind kind;
  TransformsToPtrOp op;
  Value rootBase;

  // --- Affine ---
  int64_t unitStride = 0;

  // --- Carry ---
  SmallVector<TransformMapAttr> transforms; // view -> root
  unsigned mergeIdx = 0;                     // index of the iv-traversed Merge
  // Lower-space positions (in the merge map's lower coordinate space) of the
  // merge's decomposed dims, in merge order (highest place value first). These
  // are the coordinates carried as iter_args.
  SmallVector<unsigned> variantPositions;
  SmallVector<int64_t> radices; // merge params, per carried dim
  SmallVector<int64_t> digits;  // per-loop-step carry digit, per carried dim
};

/// Carry fallback for `analyzeCandidate`: the iv flows through a single `Merge`
/// (e.g. the conv `gemmK` packing) whose decomposition is non-contiguous and/or
/// drives the validity mask. We carry that merge's decomposed lower coordinates
/// and rebuild offset + mask from the sub-chain below the merge each iteration.
static bool analyzeCarryCandidate(TransformsToPtrOp op, scf::ForOp loop,
                                  ArrayRef<unsigned> ivPositions,
                                  ArrayRef<TransformMapAttr> transforms,
                                  Value root, Candidate &cand) {
  auto bail = [&](const char *reason) {
    LLVM_DEBUG(llvm::dbgs() << "[analyzeCarryCandidate] skip " << op.getLoc()
                            << ": " << reason << "\n");
    return false;
  };

  // The per-iteration merged diff must be a compile-time constant, so the loop
  // step must be constant.
  APInt stepAP;
  if (!matchPattern(loop.getStep(), m_ConstantInt(&stepAP)))
    return bail("loop step is not a compile-time constant");
  int64_t stepConst = stepAP.getSExtValue();

  // Propagate the unit-iv diff down the chain, locating the single Merge whose
  // merged upper dim the iv reaches with a nonzero diff.
  DenseMap<unsigned, int64_t> diff;
  for (unsigned p : ivPositions)
    diff[p] = 1;
  int mergeIdx = -1;
  TransformAttr mergeOp;
  int64_t mergedDiffUnit = 0;
  for (size_t mapIdx = 0; mapIdx < transforms.size(); ++mapIdx) {
    TransformMapAttr map = transforms[mapIdx];
    auto upper = [&](unsigned d) -> int64_t {
      auto it = diff.find(d);
      return it == diff.end() ? 0 : it->second;
    };
    for (TransformAttr t : map.getOps()) {
      if (t.getType() != TransformType::Merge)
        continue;
      if (upper(t.getUpperDims()[0]) == 0)
        continue;
      if (mergeIdx != -1)
        return bail("more than one iv-traversed merge (not yet supported)");
      mergeIdx = static_cast<int>(mapIdx);
      mergeOp = t;
      mergedDiffUnit = upper(t.getUpperDims()[0]);
    }
    // We reconstruct the variant mask only from the sub-chain *below* the merge,
    // so no validity-impacting map may sit at or above it.
    if (mapImpactsValidity(map) &&
        (mergeIdx == -1 || static_cast<int>(mapIdx) <= mergeIdx))
      return bail("validity-impacting map at or above the iv-traversed merge");
    FailureOr<DenseMap<unsigned, int64_t>> lower = applyDiffOneMap(map, diff);
    if (failed(lower))
      return bail("non-constant diff while propagating to the merge");
    diff = std::move(*lower);
  }
  if (mergeIdx == -1)
    return bail("no iv-traversed merge found");

  // Per-loop-step carry digits from the merge's nested place values.
  ArrayRef<int64_t> e = mergeOp.getParams();
  int64_t D = mergedDiffUnit * stepConst;
  SmallVector<int64_t> P(e.size(), 1);
  for (int j = static_cast<int>(e.size()) - 2; j >= 0; --j)
    P[j] = P[j + 1] * e[j + 1];

  cand.kind = CandKind::Carry;
  cand.op = op;
  cand.rootBase = root;
  cand.transforms.assign(transforms.begin(), transforms.end());
  cand.mergeIdx = static_cast<unsigned>(mergeIdx);
  cand.variantPositions.assign(mergeOp.getLowerDims().begin(),
                               mergeOp.getLowerDims().end());
  cand.radices.assign(e.begin(), e.end());
  cand.digits.clear();
  for (size_t j = 0; j < e.size(); ++j)
    cand.digits.push_back(j == 0 ? (D / P[0]) : (D / P[j]) % e[j]);
  return true;
}

/// Decide whether the passed transforms_to_ptr op can be LICM'ed, and how
/// (affine accumulator vs. carried coordinates).
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

  // Affine fast path: the mask is loop-invariant (no validity-impacting map
  // constrains an iv-dependent coordinate) *and* the linearized offset has a
  // single constant per-iteration stride. A Pad/Embed on a dimension unrelated
  // to the iv (e.g. a padded gemmM while the iv lives in gemmK) keeps the mask
  // invariant. The stride propagation mirrors rocMLIR's index-diff rules and is
  // exact through Merge/Unmerge reconstructions (e.g. the conv gemmK packing).
  if (!maskDependsOnIv(transforms, ivPositions)) {
    DenseMap<unsigned, int64_t> ivDiff;
    for (unsigned p : ivPositions)
      ivDiff[p] = 1;
    FailureOr<int64_t> stride = linearizedDiffStride(transforms, ivDiff);
    if (succeeded(stride) && *stride != 0) {
      cand.kind = CandKind::Affine;
      cand.op = op;
      cand.unitStride = *stride;
      return true;
    }
  }

  // Carry fallback: the iv flows through a non-contiguous Merge and/or the mask
  // depends on the iv. Carry the merge's decomposed coordinates instead.
  if (analyzeCarryCandidate(op, loop, ivPositions, transforms, root, cand))
    return true;

  return bail("offset is not affine and no carry decomposition applies");
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

/// Create a constant i32 tensor of shape `tt`, every element equal to `v`.
static Value splatI32(OpBuilder &b, Location loc, RankedTensorType tt,
                      int64_t v) {
  Value s = arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(v));
  return triton::SplatOp::create(b, loc, tt, s);
}

/// Emit the per-iteration mixed-radix carry update of the decomposed merge
/// coordinates `carried` (highest place value first). Each coordinate advances
/// by its constant `digit`; overflow past `radix` carries 1 into the next-higher
/// place. Pure add/cmp/select - no div/mod. The highest place is never wrapped
/// (the loop bounds keep it in range).
static SmallVector<Value> emitCarryUpdate(OpBuilder &b, Location loc,
                                          ArrayRef<Value> carried,
                                          ArrayRef<int64_t> radices,
                                          ArrayRef<int64_t> digits) {
  auto tt = cast<RankedTensorType>(carried[0].getType());
  SmallVector<Value> out(carried.size());
  Value carryInt; // null == no incoming carry
  for (int j = static_cast<int>(carried.size()) - 1; j >= 0; --j) {
    Value v =
        arith::AddIOp::create(b, loc, carried[j], splatI32(b, loc, tt, digits[j]));
    if (carryInt)
      v = arith::AddIOp::create(b, loc, v, carryInt);
    if (j > 0) {
      Value radix = splatI32(b, loc, tt, radices[j]);
      Value ge = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, v,
                                       radix);
      Value vsub = arith::SubIOp::create(b, loc, v, radix);
      v = arith::SelectOp::create(b, loc, ge, vsub, v);
      carryInt = arith::SelectOp::create(b, loc, ge, splatI32(b, loc, tt, 1),
                                         splatI32(b, loc, tt, 0));
    }
    out[j] = v;
  }
  return out;
}

/// Per-candidate preheader values + iter_arg layout for the rewritten loop.
struct Reduced {
  CandKind kind;
  TransformsToPtrOp op;      // original in-loop op (to be removed)
  unsigned iterArgStart = 0; // first new iter_arg index for this candidate
  unsigned iterArgCount = 0; // number of new iter_args contributed

  // --- Affine ---
  Value basePtr;     // loop-invariant pointer tensor at iv == lb
  Value baseMask;    // loop-invariant mask
  Value strideSplat; // constant per-iteration offset increment
  Value accInit;     // zero offset accumulator initializer

  // --- Carry ---
  Value carryBasePtr;                      // pointer tensor at iv == lb
  Value offset0;                           // linearized offset at iv == lb
  SmallVector<Value> iter0Coords;          // merge-lower coord vector at iv==lb
  SmallVector<Value> carriedInits;         // init for the carried iter_args
  SmallVector<TransformMapAttr> belowMaps; // transform chain below the merge
  SmallVector<unsigned> variantPositions;  // carried positions in coord vector
  SmallVector<int64_t> radices;
  SmallVector<int64_t> digits;
  SmallVector<int64_t> outShape;
};

/// Try to LICM all eligible transforms_to_ptr ops in `loop`.
/// Returns true (and rewrites the loop) if at least one was reduced.
static bool tryHoistInvariantTransforms(scf::ForOp loop) {
  SmallVector<Candidate, 0> candidates;
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

  // Build per-candidate preheader values: a base pointer at iv == lb plus the
  // carried-state initializers (an affine accumulator, or the decomposed merge
  // coordinates).
  SmallVector<Reduced, 0> reduced;
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

    Reduced r;
    r.kind = cand.kind;
    r.op = cand.op;

    if (cand.kind == CandKind::Affine) {
      // The per-iteration offset increment is `step * unitStride`, expressed in
      // the pointer-offset element type. The offsets are integers independent
      // of the induction-variable type, so cast the step to that type before
      // multiplying by the constant stride.
      Type offsetTy = ptrType.getElementType();
      Value strideScalar = arith::MulIOp::create(
          b, loc, castScalar(b, loc, step, offsetTy),
          arith::ConstantOp::create(
              b, loc, b.getIntegerAttr(offsetTy, cand.unitStride)));
      r.basePtr = baseOp.getPointers();
      r.baseMask = baseOp.getMask();
      r.strideSplat = triton::SplatOp::create(b, loc, ptrType, strideScalar);
      r.accInit = arith::ConstantOp::create(
          b, loc, cast<TypedAttr>(b.getZeroAttr(ptrType)));
      r.iterArgCount = 1;
      reduced.push_back(std::move(r));
      continue;
    }

    // Carry path. Reconstruct the merge's lower coordinate vector at iv == lb by
    // expanding the chain above and including the merge over the (lb-pinned)
    // extra indices plus per-tile ranges - exactly the seed the lowering uses,
    // but stopping at the merge's lower space.
    ArrayRef<int64_t> outShape = ptrType.getShape();
    SmallVector<Value> initValues(baseIdx);
    for (size_t d = 0; d < outShape.size(); ++d)
      initValues.push_back(
          makeRange(b, loc, 0, outShape[d], outShape.size(), d));

    AffineMap aboveMap =
        composeTransforms(ArrayRef<TransformMapAttr>(cand.transforms)
                              .take_front(cand.mergeIdx + 1));
    FailureOr<SmallVector<Value>> iter0 =
        expandAffineMap(b, loc, aboveMap, initValues);
    assert(succeeded(iter0) && "merge-prefix expansion must succeed");

    SmallVector<TransformMapAttr> belowMaps(
        ArrayRef<TransformMapAttr>(cand.transforms)
            .drop_front(cand.mergeIdx + 1));
    FailureOr<OffsetAndMask> om0 =
        expandCoordsToOffsetAndMask(b, loc, belowMaps, *iter0, outShape);
    assert(succeeded(om0) && "sub-chain expansion must succeed");

    r.carryBasePtr = baseOp.getPointers();
    r.offset0 = om0->offset;
    r.iter0Coords = *iter0;
    r.belowMaps = std::move(belowMaps);
    r.variantPositions = cand.variantPositions;
    r.radices = cand.radices;
    r.digits = cand.digits;
    r.outShape.assign(outShape.begin(), outShape.end());
    for (unsigned pos : cand.variantPositions)
      r.carriedInits.push_back(
          broadcastToShape(b, loc, (*iter0)[pos], outShape));
    r.iterArgCount = r.carriedInits.size();
    reduced.push_back(std::move(r));
  }

  // New iter_args: the original ones followed by each candidate's carried state.
  unsigned numOrig = loop.getInitArgs().size();
  SmallVector<Value> newInits(loop.getInitArgs().begin(),
                              loop.getInitArgs().end());
  for (Reduced &r : reduced) {
    r.iterArgStart = newInits.size();
    if (r.kind == CandKind::Affine) {
      newInits.push_back(r.accInit);
    } else {
      for (Value v : r.carriedInits)
        newInits.push_back(v);
    }
  }

  auto newLoop =
      scf::ForOp::create(b, loc, lb, loop.getUpperBound(), step, newInits);

  // Map old body values into the new body.
  IRMapping bodyMap;
  bodyMap.map(iv, newLoop.getInductionVar());
  for (unsigned i = 0; i < numOrig; ++i)
    bodyMap.map(loop.getRegionIterArg(i), newLoop.getRegionIterArg(i));

  b.setInsertionPointToStart(newLoop.getBody());

  // Reconstruct each candidate's pointer/mask inside the body.
  //
  // Affine: pointer = basePtr + acc (a single add that RockToTTIR/
  // FuncToTritonFunc lower to a tt.addptr recurrence) with a loop-invariant
  // mask.
  //
  // Carry: rebuild the decomposed coordinate vector from the carried iter_args
  // (variant entries) and the hoisted iter-0 coordinates (invariant entries),
  // then re-expand the sub-chain below the merge to get this iteration's offset
  // and mask. The pointer is basePtr + (offset_k - offset_0), which keeps the
  // established "base pointer + integer offset" shape the downstream lowering
  // expects.
  llvm::SmallPtrSet<Operation *, 4> candidateOps;
  for (Reduced &r : reduced) {
    candidateOps.insert(r.op.getOperation());
    if (r.kind == CandKind::Affine) {
      Value acc = newLoop.getRegionIterArg(r.iterArgStart);
      Value ptr = arith::AddIOp::create(b, loc, r.basePtr, acc);
      bodyMap.map(r.op.getPointers(), ptr);
      bodyMap.map(r.op.getMask(), r.baseMask);
      continue;
    }

    SmallVector<Value> coords(r.iter0Coords);
    for (auto [k, pos] : llvm::enumerate(r.variantPositions))
      coords[pos] = newLoop.getRegionIterArg(r.iterArgStart + k);

    FailureOr<OffsetAndMask> om =
        expandCoordsToOffsetAndMask(b, loc, r.belowMaps, coords, r.outShape);
    assert(succeeded(om) && "sub-chain expansion must succeed");
    Value delta = arith::SubIOp::create(b, loc, om->offset, r.offset0);
    Value ptr = arith::AddIOp::create(b, loc, r.carryBasePtr, delta);
    bodyMap.map(r.op.getPointers(), ptr);
    bodyMap.map(r.op.getMask(), om->mask);
  }

  for (Operation &o : loop.getBody()->without_terminator()) {
    if (candidateOps.contains(&o))
      continue;
    b.clone(o, bodyMap);
  }

  // Build the new yield: original yields (remapped) plus the advanced carried
  // state for each candidate.
  auto oldYield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<Value> newYields;
  for (Value y : oldYield.getResults())
    newYields.push_back(bodyMap.lookupOrDefault(y));
  for (Reduced &r : reduced) {
    if (r.kind == CandKind::Affine) {
      Value acc = newLoop.getRegionIterArg(r.iterArgStart);
      newYields.push_back(arith::AddIOp::create(b, loc, acc, r.strideSplat));
      continue;
    }
    SmallVector<Value> carried;
    for (unsigned k = 0; k < r.iterArgCount; ++k)
      carried.push_back(newLoop.getRegionIterArg(r.iterArgStart + k));
    for (Value v : emitCarryUpdate(b, loc, carried, r.radices, r.digits))
      newYields.push_back(v);
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
