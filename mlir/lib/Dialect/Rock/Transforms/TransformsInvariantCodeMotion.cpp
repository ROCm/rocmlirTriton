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

    // Carry-neutrality guard: for any Merge the iv flows through (nonzero
    // merged diff), the lower dims must nest contiguously in the buffer
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

/// Returns true if a unit change in the merge-lower coordinate `pos` reaches a
/// validity-impacting bound check while propagating down `belowMaps` (the
/// sub-chain below the iv-traversed merge). Coordinates that impact validity
/// (the spatial taps of a conv: they feed a halo Pad / sliding-window Embed)
/// must be carried as full distributed tiles so the per-element mask can be
/// rebuilt without a cross-lane broadcast; coordinates that do not (the conv
/// channel, which only scales the buffer offset) need not be persisted at all -
/// their offset contribution is recovered from the carry-out of the coordinate
/// below them. Mirrors the bound checks emitted by `updateValidityAfter`.
static bool variantImpactsValidity(ArrayRef<TransformMapAttr> belowMaps,
                                   unsigned pos) {
  DenseMap<unsigned, int64_t> diff;
  diff[pos] = 1;
  for (TransformMapAttr map : belowMaps) {
    FailureOr<DenseMap<unsigned, int64_t>> lower = applyDiffOneMap(map, diff);
    if (failed(lower))
      return true; // cannot prove independence: conservatively impacting
    auto nonzero = [&](uint32_t d) {
      auto it = lower->find(d);
      return it != lower->end() && it->second != 0;
    };
    if (mapImpactsValidity(map)) {
      for (TransformAttr op : map.getOps()) {
        if (op.getType() == TransformType::Pad) {
          ArrayRef<int64_t> params = op.getParams();
          ArrayRef<uint32_t> ld = op.getLowerDims();
          for (size_t i = 0; i < ld.size(); ++i)
            if ((params[2 * i] != 0 || params[2 * i + 1] != 0) &&
                nonzero(ld[i]))
              return true;
        } else if (op.getType() == TransformType::Embed) {
          if (embedCanBeInvalid(map, op) && nonzero(op.getLowerDims()[0]))
            return true;
        }
      }
    }
    diff = std::move(*lower);
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
/// In theory, Affine candidates are always preferred over Carry candidates,
/// since the lowering is simpler for Affine candidates. However, the Affine
/// lowering is too simple to support all cases, so for the case where we bail
/// out, we fall back to Carry.
enum class CandKind {
  /// The linearized offset is affine in the IV and the mask is loop-invariant:
  /// carry a single integer accumulator (`base + IV*stride`).
  Affine,
  /// The IV flows through a non-contiguous `Merge` and/or the mask depends on
  /// the IV: carry the merge's decomposed lower coordinates (one
  /// `tensor<tile x i32>` each) and rebuild the offset + mask every iteration.
  Carry,
};

/// A simplification candidate: an `transforms_to_ptr` op together with the
/// information needed to incrementalize it.
struct Candidate {
  CandKind kind;        // Affine or Carry
  TransformsToPtrOp op; // The op to be simplified
  Value rootBase;       // The block argument of the op.

  // Used for Affine candidates.
  int64_t unitStride = 0;

  // Used for Carry candidates.
  //
  // The carry path decomposes the IV's merged dimension into several
  // coordinates. For example, consider a 2D coordinate case with values (3, 2).
  // Each coord has its own size (`coordSizes`). Every iteration, each
  // coordinate is incremented by a constant step (`coordSteps`); when a
  // coordinate reaches its size it wraps to 0 and carries +1 into the
  // next-higher coordinate — done with add/cmp/select, no div/mod.
  SmallVector<TransformMapAttr> transforms;
  unsigned mergeIdx = 0;
  SmallVector<unsigned> coordPositions; // which decomposed coords are carried
  SmallVector<int64_t> coordSizes;      // each coord's wrap limit (merge size)
  SmallVector<int64_t> coordSteps;      // per-iteration increment of each coord
};

/// Carry fallback for `analyzeCandidate`
///
/// If the transforms_to_ptr we are analyzing is supported by our current
/// implemetation, fill in the candidate struct and return true.
///
/// If we bail, it can be for 2 main reasons:
/// 1. Fundamentally we cannot handle this case: For example, if
/// the loop step is not constant, there is nothing we can do (not a limitation
/// of this transform)
/// 2. The implementation does not support this case. Here it's unclear if
/// supporting more cases would actually improve performance, since the carry
/// path on complex IRs can potentially make things worse due to register
/// pressure.
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
  std::optional<APInt> stepAP = loop.getConstantStep();
  if (!stepAP)
    return bail("loop step is not a compile time constant");
  int64_t stepConst = stepAP->getSExtValue();

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
    for (TransformAttr t : map.getOps()) {
      // We are looking for (only one) Merge that the IV
      // reaches with a nonzero diff. `diff.lookup` yields 0 for untouched dims.
      if (t.getType() != TransformType::Merge)
        continue;

      int64_t mergedDiff = diff.lookup(t.getUpperDims()[0]);
      if (mergedDiff == 0)
        continue;

      if (mergeIdx != -1)
        return bail("more than one iv-traversed merge (not yet supported)");

      mergeIdx = static_cast<int>(mapIdx);
      mergeOp = t;
      mergedDiffUnit = mergedDiff;
    }

    // We reconstruct the variant mask only from the sub-chain *below* the
    // merge, so no validity-impacting map may sit at or above it.
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

  // Per-iteration coord steps: the merged coordinate advances by `D` each loop
  // step, delinearized against the merge's nested place values (suffix
  // products). delinearize leaves the top coord un-wrapped, which is what we
  // want (the highest coord never wraps; the loop bounds keep it in range).
  ArrayRef<int64_t> mergeParams = mergeOp.getParams();
  // How far the merged coordinate moves per loop iteration
  int64_t D = mergedDiffUnit * stepConst;

  cand.kind = CandKind::Carry;
  cand.op = op;
  cand.rootBase = root;
  cand.transforms.assign(transforms.begin(), transforms.end());
  cand.mergeIdx = static_cast<unsigned>(mergeIdx);
  cand.coordPositions.assign(mergeOp.getLowerDims().begin(),
                             mergeOp.getLowerDims().end());
  cand.coordSizes.assign(mergeParams.begin(), mergeParams.end());
  cand.coordSteps = delinearize(D, computeSuffixProduct(mergeParams));
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

  // Classify each transforms_to_ptr index as the loop iv (which we
  // incrementalize), loop-invariant (referenced directly), or loop-variant but
  // not the iv. The last kind is only tracked here, not rejected: the affine
  // path handles it (it rewrites the base op in place and references the index
  // directly), while the carry path cannot and bails on it below. If no index
  // is the iv the pointer is already loop-invariant and there is nothing to do.
  ValueRange extra = op.getExtraIndices();
  SmallVector<unsigned> ivPositions;
  bool hasLoopVariantNonIvIdx = false;
  for (auto [pos, idx] : llvm::enumerate(extra)) {
    if (isInductionVar(idx, iv)) {
      ivPositions.push_back(pos);
      continue;
    }
    if (isDefinedInLoop(idx, loop))
      hasLoopVariantNonIvIdx = true;
  }
  if (ivPositions.empty())
    return bail("pointer does not depend on the iv (there is nothing to do)");

  // Walk the transform chain down to its root.
  SmallVector<TransformMapAttr> transforms;
  Value root;
  std::tie(root, std::ignore) = untransform(op.getSource(), transforms);
  if (!isa<BlockArgument>(root))
    return bail("transform chain root is not a block argument");
  cand.rootBase = root;

  // Affine path: the mask does not depend on the IV and the offset can be
  // incremented by a single constant per iteration.
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

  // Carry fallback: the IV flows through a non-contiguous Merge and/or the mask
  // depends on the IV. Carry the merge's decomposed coordinates instead.
  //
  // Unlike the affine path (which rewrites the base op in place and can
  // reference a loop-variant non-iv index directly), the carry path
  // cannot handle this case via cloneSliceBeforeLoop implementation, so we
  // bail.
  if (hasLoopVariantNonIvIdx)
    return bail("an extra index is loop-variant but is not the IV (carry path "
                "cannot rebuild it in the preheader)");
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

/// Result of advancing the full-tile distributed carry by one loop step: the
/// advanced validity (suffix) coordinates and the per-element offset increment.
struct FullTileCarry {
  SmallVector<Value> nextSuffix;
  Value offsetDelta;
};

/// Advance the full-tile distributed carry by one loop step. `suffix` holds the
/// validity-impacting merge coordinates as full `tensor<kIter x nPerBlock>`
/// tiles (highest place first); each advances by its constant `step`, wrapping
/// at its `size` with the carry rippling into the next-higher place. *Every*
/// suffix coordinate wraps when `hasPrefix` is set,
/// because the dropped high non-validity coordinate sits above them; its own
/// per-step offset contribution is `(prefixStep + carryOut) * prefixStride`,
/// recovered here without ever materializing that coordinate. The returned
/// `offsetDelta` is `sum_j (delta coord_j) * strides[j]` over the suffix plus
/// the prefix term - a full tile, so the offset recurrence stays vector-
/// resident.
static FullTileCarry
emitFullTileCarry(OpBuilder &b, Location loc, ArrayRef<Value> suffix,
                  ArrayRef<int64_t> sizes, ArrayRef<int64_t> steps,
                  ArrayRef<int64_t> strides, bool hasPrefix, int64_t prefixStep,
                  int64_t prefixStride) {
  auto tt = cast<RankedTensorType>(suffix[0].getType());
  FullTileCarry out;
  out.nextSuffix.resize(suffix.size());
  Value carry; // full-tile i32; null == no incoming carry
  Value delta; // full-tile i32; null == 0
  for (int j = static_cast<int>(suffix.size()) - 1; j >= 0; --j) {
    Value v = arith::AddIOp::create(b, loc, suffix[j],
                                    splatI32(b, loc, tt, steps[j]));
    if (carry)
      v = arith::AddIOp::create(b, loc, v, carry);
    // A suffix coordinate is the global top only when no prefix coordinate was
    // dropped above it; the loop bound keeps the global top from wrapping.
    if (j > 0 || hasPrefix) {
      Value size = splatI32(b, loc, tt, sizes[j]);
      Value ge =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, v, size);
      Value vsub = arith::SubIOp::create(b, loc, v, size);
      v = arith::SelectOp::create(b, loc, ge, vsub, v);
      carry = arith::SelectOp::create(b, loc, ge, splatI32(b, loc, tt, 1),
                                      splatI32(b, loc, tt, 0));
    } else {
      carry = Value();
    }
    out.nextSuffix[j] = v;
    Value dCoord = arith::SubIOp::create(b, loc, v, suffix[j]);
    Value term =
        arith::MulIOp::create(b, loc, dCoord, splatI32(b, loc, tt, strides[j]));
    delta = delta ? arith::AddIOp::create(b, loc, delta, term) : term;
  }
  if (hasPrefix) {
    Value dPrefix = splatI32(b, loc, tt, prefixStep);
    if (carry)
      dPrefix = arith::AddIOp::create(b, loc, dPrefix, carry);
    Value term = arith::MulIOp::create(b, loc, dPrefix,
                                       splatI32(b, loc, tt, prefixStride));
    delta = delta ? arith::AddIOp::create(b, loc, delta, term) : term;
  }
  out.offsetDelta = delta;
  return out;
}

/// Per-candidate carried state + iter_arg layout for the rewritten loop.
struct Reduced {
  TransformsToPtrOp op;      // original op (to be removed)
  unsigned iterArgStart = 0; // first new iter_arg index for this candidate

  RankedTensorType ptrType;
  Type maskType;

  // Coordinate vector at iv == lb. Invariant entries are used verbatim in the
  // loop body; the entries at `coordPositions` are replaced by the carried
  // iter_args and seed their initial values.
  SmallVector<Value> iter0Coords;
  SmallVector<Value> carriedInits;      // init for the carried coord iter_args
  SmallVector<unsigned> coordPositions; // carried positions in coord vector
  SmallVector<int64_t> coordSizes;      // each coord's wrap limit (merge size)
  SmallVector<int64_t> coordSteps;      // per-iteration increment of each coord

  // Coordinate decomposition: prefix vs. suffix.
  //
  // Example:
  //
  //   Merge{2, 3} -> [ci, tap]   means   gemmK = ci*3 + tap
  //
  // ci has positional weight 3, tap has weight 1, so [ci, tap] runs from the
  // most to the least significant coordinate. Therefore:
  //
  // - suffix: the least significant coordinates. They vary fastest:
  //   incrementing the merged index by 1 bumps the suffix.
  // - prefix: the most significant coordinate(s). They vary slowest
  //   and change only when a suffix coordinate wraps and carries in. At most
  //   one prefix coordinate exists here; it is dropped (never materialized) and
  //   its offset contribution is recovered from the suffix carry.

  // Information to compute the pointer recurrence carry.
  // This is the "heavy lifting" part of the carry. 3 parts:
  //
  // ---------------------------------
  // 1. Base tile reconstruction.
  // ---------------------------------
  // The base tile when iv == lb:
  // srcPre is the transforms_to_ptr source operand, cloned into the preheader
  // with every use of the iv replaced by lb baseIdx is the extra indices, again
  // with the iv slot pinned to lb
  Value srcPre;
  SmallVector<Value> baseIdx;

  // ---------------------------------
  // 2. Mask rebuild.
  // ---------------------------------
  // Transform chain below the merge: used to rebuild the mask on each
  // iteration.
  SmallVector<TransformMapAttr> belowMaps;

  // ---------------------------------
  // 3. Offset recurrence computation.
  // ---------------------------------
  // Buffer stride per coord: used to compute the offset delta on each
  // iteration.
  SmallVector<int64_t> offsetStrides;
  // The initial value used for the carry.
  Value offsetAccInit;
  // How many coords are carried as tiles.
  unsigned suffixCount = 0;
  // Was a high non-validity coord dropped?
  bool hasPrefix = false;
  // For the dropped coord, we store:
  // - its per-iteration increment (prefixStep)
  // - its offset stride (prefixStride)
  int64_t prefixStep = 0;
  int64_t prefixStride = 0;
};

/// Simplify an Affine candidate
///
/// The offset is affine in the IV with a single constant stride, so
/// offset(IV) = offset(LB) + (IV - LB) * unitStride. offset(LB) is the base
/// op's per-element tile; the tail is a uniform scalar splat onto it. The
/// resulting `addi(basePtr, splat)` is the "base pointer + integer offset"
/// shape RockToTTIR/TensorToTritonPtr lower to a tt.addptr.
static void simplifyAffineCandidate(Candidate &cand, Value iv, Value lb) {
  TransformsToPtrOp op = cand.op;
  OpBuilder b(op);
  Location loc = op.getLoc();

  // Pin the iv to lb: an index that is the iv (or a cast of it) is re-expressed
  // at lb, while loop-invariant indices are referenced directly.
  SmallVector<Value> baseIdx;
  for (Value idx : op.getExtraIndices())
    baseIdx.push_back(
        isInductionVar(idx, iv) ? castScalar(b, loc, lb, idx.getType()) : idx);

  auto ptrType = cast<RankedTensorType>(op.getPointers().getType());
  Type maskType = op.getMask().getType();
  auto base = TransformsToPtrOp::create(b, loc, ptrType, maskType,
                                        op.getSource(), baseIdx);

  // Emit the IR for the simplification offset computation.
  Type offsetTy = ptrType.getElementType();
  Value ivMinusLb = arith::SubIOp::create(b, loc, iv, lb);
  Value stride = arith::ConstantOp::create(
      b, loc, b.getIntegerAttr(offsetTy, cand.unitStride));
  Value delta = arith::MulIOp::create(
      b, loc, castScalar(b, loc, ivMinusLb, offsetTy), stride);
  Value deltaSplat = triton::SplatOp::create(b, loc, ptrType, delta);
  Value ptr = arith::AddIOp::create(b, loc, base.getPointers(), deltaSplat);

  op.getPointers().replaceAllUsesWith(ptr);
  op.getMask().replaceAllUsesWith(base.getMask());
  op.erase();
}

static void simplifyCarryCandidates(scf::ForOp loop,
                                    MutableArrayRef<Candidate> carryCands);

/// Simplify all eligible transforms_to_ptr ops in `loop`: Affine candidates are
/// rewritten in place, Carry candidates get loop-carried coordinate state.
/// Returns true if the IR was changed.
static bool tryHoistInvariantTransforms(scf::ForOp loop) {
  SmallVector<Candidate, 0> candidates;
  for (Operation &o : loop.getBody()->without_terminator()) {
    if (auto tp = dyn_cast<TransformsToPtrOp>(&o)) {
      Candidate cand;
      if (analyzeCandidate(tp, loop, cand))
        candidates.push_back(cand);
    }
  }
  if (candidates.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No candidates to simplify, skip " << loop.getLoc() << "\n");
    return false;
  }

  Value iv = loop.getInductionVar();
  Value lb = loop.getLowerBound();

  // Affine candidates are rewritten in place; Carry candidates need
  // loop-carried coordinate state, so they are collected here and handled by
  // simplifyCarryCandidates.
  SmallVector<Candidate, 0> carryCands;
  bool changed = false;
  for (Candidate &cand : candidates) {
    if (cand.kind == CandKind::Affine) {
      simplifyAffineCandidate(cand, iv, lb);
      changed = true;
    } else {
      carryCands.push_back(cand);
    }
  }

  if (!carryCands.empty())
    simplifyCarryCandidates(loop, carryCands);

  return changed;
}

/// Analyze each carry candidate and, for those eligible for the full-tile
/// pointer recurrence, emit their preheader init values (base coordinates,
/// carried suffix coordinates, zero offset accumulator) and return the
/// per-candidate `Reduced` state. Ineligible candidates are skipped (their op
/// is left in the loop, to be lowered in place by
/// rock-transforms-to-pointer-arith).
static SmallVector<Reduced, 0>
buildReducedCarries(OpBuilder &b, scf::ForOp loop,
                    MutableArrayRef<Candidate> carryCands) {
  Location loc = loop.getLoc();
  Value iv = loop.getInductionVar();
  Value lb = loop.getLowerBound();

  // Build each carry candidate's iter_arg initial values before the loop (as
  // SSA requires): the merge's decomposed coordinates at iv == lb.
  SmallVector<Reduced, 0> reduced;
  for (Candidate &cand : carryCands) {
    // Builds the transform chain below (after) the merge.
    // Note that cand.transforms is the full chain and cand.mergeIdx is the
    // index of the map containing the merge the IV flows through, so
    // drop_front(mergeIdx + 1) discards the merge and everything above it,
    // keeping only the sub-chain from the merge's decomposed coordinates down
    // to the raw buffer
    SmallVector<TransformMapAttr> belowMaps =
        llvm::to_vector(ArrayRef<TransformMapAttr>(cand.transforms)
                            .drop_front(cand.mergeIdx + 1));

    // Each carried coordinate contributes a constant
    // stride through the sub-chain below the merge.
    // When every coordinate linearizes, the offset can
    // be maintained by an add-only recurrence instead of re-expanded each
    // iteration.
    SmallVector<int64_t> strides;
    bool stridesOk = true;
    for (unsigned pos : cand.coordPositions) {
      DenseMap<unsigned, int64_t> unit;
      unit[pos] = 1;
      FailureOr<int64_t> s = linearizedDiffStride(belowMaps, unit);
      if (failed(s)) {
        stridesOk = false;
        break;
      }
      strides.push_back(*s);
    }
    // Non-linearizable offset: not eligible for the pointer-recurrence carry.
    if (!stridesOk)
      continue;

    assert(!cand.coordPositions.empty() &&
           "carry candidate must have a decomposed merge");

    // Full-tile distributed carry. Classify each carried coordinate as
    // validity-impacting or not. When it's not, we compute the value
    // of the decomposition and we carry the suffix and a full-tile offset
    // accumulator and *drop* the high non-validity coordinate (at most one; the
    // loop bound keeps the merge's top place from wrapping), recovering its
    // offset contribution from the suffix carry-out.
    bool fullTileOk = false;
    unsigned suffixCount = 0;
    unsigned prefixSize = 0;

    SmallVector<bool> impacts(cand.coordPositions.size());
    for (auto [i, pos] : llvm::enumerate(cand.coordPositions))
      impacts[i] = variantImpactsValidity(belowMaps, pos);

    // Eligible only when the validity-impacting coords are a contiguous suffix:
    // find the first impacting coord (everything before it is non-impacting by
    // construction) and require everything from there on to also impact. The
    // non-impacting prefix is dropped, so it may be at most one coord.
    auto *firstImpact = llvm::find(impacts, true);
    if (firstImpact != impacts.end()) {
      prefixSize = std::distance(impacts.begin(), firstImpact);
      suffixCount = impacts.size() - prefixSize;
      fullTileOk = prefixSize <= 1 &&
                   llvm::all_of(llvm::make_range(firstImpact, impacts.end()),
                                [](bool v) { return v; });
    }

    // Not eligible for the pointer-recurrence carry (the validity coordinates
    // are not a suffix, or more than one prefix coord would be dropped): bail
    // out.
    if (!fullTileOk)
      continue;

    // Pin the iv to lb: an index that is the iv (or a cast of it, e.g.
    // index_cast(iv) for an index-typed loop) is re-expressed at lb, while
    // loop-invariant indices are referenced directly. Loop-variant non-iv
    // indices were already rejected by analyzeCandidate, so this only clones
    // iv casts into the preheader.
    IRMapping cloneMap;
    cloneMap.map(iv, lb);
    Value srcPre = cloneSliceBeforeLoop(b, cand.op.getSource(), loop, cloneMap);
    SmallVector<Value> baseIdx;
    for (Value idx : cand.op.getExtraIndices())
      baseIdx.push_back(cloneSliceBeforeLoop(b, idx, loop, cloneMap));

    auto ptrType = cast<RankedTensorType>(cand.op.getPointers().getType());
    ArrayRef<int64_t> outShape = ptrType.getShape();

    // Reconstruct the merge's lower coordinate vector at iv == lb by expanding
    // the chain above and including the merge over the (lb-pinned) extra
    // indices plus per-tile ranges - the same seed the lowering uses, but
    // stopping at the merge's lower space.
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

    Reduced r;
    r.op = cand.op;
    r.ptrType = ptrType;
    r.maskType = cand.op.getMask().getType();
    r.iter0Coords = *iter0;
    r.coordPositions = cand.coordPositions;
    r.coordSizes = cand.coordSizes;
    r.coordSteps = cand.coordSteps;
    r.srcPre = srcPre;
    r.baseIdx = baseIdx;
    r.belowMaps = std::move(belowMaps);
    r.offsetStrides = strides;
    r.suffixCount = suffixCount;
    r.hasPrefix = (prefixSize == 1);
    if (r.hasPrefix) {
      r.prefixStep = cand.coordSteps[0];
      r.prefixStride = strides[0];
    }
    // Carry the validity (suffix) coordinates as full tiles plus a full-tile
    // zero offset accumulator (the base pointer already encodes offset0).
    for (unsigned pos :
         ArrayRef<unsigned>(cand.coordPositions).take_back(suffixCount))
      r.carriedInits.push_back(
          broadcastToShape(b, loc, (*iter0)[pos], outShape));
    auto offTy = RankedTensorType::get(outShape, b.getI32Type());
    r.offsetAccInit = splatI32(b, loc, offTy, 0);
    reduced.push_back(std::move(r));
  }
  return reduced;
}

/// Build the base pointer tile at iv == lb (all extra indices pinned to lb).
/// The op's mask output is unused (the mask is iv-dependent and rebuilt each
/// iteration by `buildCarryMask`).
static Value buildCarryBasePtr(OpBuilder &b, Location loc, const Reduced &r) {
  auto baseOp = TransformsToPtrOp::create(b, loc, r.ptrType, r.maskType,
                                          r.srcPre, r.baseIdx);
  return baseOp.getPointers();
}

/// Rebuild the validity mask for the current iteration: splice the carried
/// suffix coordinates (the new loop's iter_args) into the iv == lb coordinate
/// vector, then re-expand the sub-chain below the merge.
static Value buildCarryMask(OpBuilder &b, Location loc, const Reduced &r,
                            scf::ForOp newLoop) {
  SmallVector<Value> coords(r.iter0Coords);
  ArrayRef<unsigned> suffixPos =
      ArrayRef<unsigned>(r.coordPositions).take_back(r.suffixCount);
  for (auto [k, pos] : llvm::enumerate(suffixPos))
    coords[pos] = newLoop.getRegionIterArg(r.iterArgStart + k);
  FailureOr<OffsetAndMask> om = expandCoordsToOffsetAndMask(
      b, loc, r.belowMaps, coords, r.ptrType.getShape(), /*computeOffset=*/false);
  assert(succeeded(om) && "sub-chain mask expansion must succeed");
  return om->mask;
}

/// Form the current pointer as the base tile plus the carried full-tile offset
/// accumulator (the last iter_arg for this candidate). No offset re-expansion.
static Value buildCarryPtr(OpBuilder &b, Location loc, Value basePtr,
                           const Reduced &r, scf::ForOp newLoop) {
  Value offset = newLoop.getRegionIterArg(r.iterArgStart + r.suffixCount);
  return arith::AddIOp::create(b, loc, basePtr, offset);
}

/// Rewrite the eligible Carry candidates in `carryCands`: replace `loop` with a
/// new loop that carries the merge's decomposed coordinate state via the
/// full-tile pointer recurrence.
static void simplifyCarryCandidates(scf::ForOp loop,
                                    MutableArrayRef<Candidate> carryCands) {
  Location loc = loop.getLoc();
  Value iv = loop.getInductionVar();
  Value lb = loop.getLowerBound();
  Value step = loop.getStep();
  OpBuilder b(loop);

  SmallVector<Reduced, 0> reduced = buildReducedCarries(b, loop, carryCands);

  // No carry candidate qualified for the pointer-recurrence rewrite: bail out.
  if (reduced.empty())
    return;

  // New iter_args: the original ones followed by each candidate's carried
  // coordinates.
  unsigned numOrig = loop.getInitArgs().size();
  SmallVector<Value> newInits(loop.getInitArgs().begin(),
                              loop.getInitArgs().end());
  for (Reduced &r : reduced) {
    r.iterArgStart = newInits.size();
    for (Value v : r.carriedInits)
      newInits.push_back(v);
    // The full-tile offset accumulator is carried last.
    newInits.push_back(r.offsetAccInit);
  }

  auto newLoop =
      scf::ForOp::create(b, loc, lb, loop.getUpperBound(), step, newInits);

  // Map old body values into the new body.
  IRMapping bodyMap;
  bodyMap.map(iv, newLoop.getInductionVar());
  for (unsigned i = 0; i < numOrig; ++i)
    bodyMap.map(loop.getRegionIterArg(i), newLoop.getRegionIterArg(i));

  b.setInsertionPointToStart(newLoop.getBody());

  // Reconstruct each candidate's pointer/mask inside the body: rebuild the base
  // pointer/mask at iv == lb, splice the carried full-tile validity (suffix)
  // coordinates into the coord vector, rebuild only the mask, and form the
  // pointer as the base plus the carried full-tile offset accumulator (no
  // offset re-expansion).
  llvm::SmallPtrSet<Operation *, 4> candidateOps;
  for (Reduced &r : reduced) {
    candidateOps.insert(r.op.getOperation());

    Value basePtr = buildCarryBasePtr(b, loc, r);
    Value mask = buildCarryMask(b, loc, r, newLoop);
    Value ptr = buildCarryPtr(b, loc, basePtr, r, newLoop);
    bodyMap.map(r.op.getPointers(), ptr);
    bodyMap.map(r.op.getMask(), mask);
  }

  // Now just clone the other ops into the new body.
  for (Operation &o : loop.getBody()->without_terminator()) {
    if (candidateOps.contains(&o))
      continue;
    b.clone(o, bodyMap);
  }

  // Build the new yield
  auto oldYield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<Value> newYields;
  for (Value y : oldYield.getResults())
    newYields.push_back(bodyMap.lookupOrDefault(y));

  // Emit IR for the the advanced carried coordinates (coordinate carry) for
  // each candidate.
  for (Reduced &r : reduced) {
    // Advance the full-tile validity (suffix) coordinates and the
    // full-tile offset accumulator by the per-step delta.
    SmallVector<Value> suffix;
    for (unsigned k = 0; k < r.suffixCount; ++k)
      suffix.push_back(newLoop.getRegionIterArg(r.iterArgStart + k));

    FullTileCarry stepCarry = emitFullTileCarry(
        b, loc, suffix,
        ArrayRef<int64_t>(r.coordSizes).take_back(r.suffixCount),
        ArrayRef<int64_t>(r.coordSteps).take_back(r.suffixCount),
        ArrayRef<int64_t>(r.offsetStrides).take_back(r.suffixCount),
        r.hasPrefix, r.prefixStep, r.prefixStride);

    for (Value v : stepCarry.nextSuffix)
      newYields.push_back(v);

    Value offset = newLoop.getRegionIterArg(r.iterArgStart + r.suffixCount);
    newYields.push_back(
        arith::AddIOp::create(b, loc, offset, stepCarry.offsetDelta));
  }
  scf::YieldOp::create(b, loc, newYields);

  // Re-wire the original results and drop the old loop.
  for (unsigned i = 0; i < numOrig; ++i)
    loop.getResult(i).replaceAllUsesWith(newLoop.getResult(i));
  loop.erase();
}

} // end anonymous namespace

void RockTransformsInvariantCodeMotionPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // Re-walk after each rewrite: tryHoistInvariantTransforms may replace the
  // loop op (carry path), so collected handles would dangle. After a rewrite
  // the base ops are pinned to iv == lb and no longer depend on the iv, so they
  // are not re-selected as candidates and this terminates.
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
