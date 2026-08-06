//===- IncrementalPointerArith.cpp - Incrementalize in-loop ptr arith -----===//
//
// Copyright Advanced Micro Devices, Inc.
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

#include "PointerArithExpand.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKINCREMENTALPOINTERARITHPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-incremental-pointer-arith"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockIncrementalPointerArithPass
    : public rock::impl::RockIncrementalPointerArithPassBase<
          RockIncrementalPointerArithPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

namespace {

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

/// Propagate a constant coordinate *diff* through a transform chain and return
/// the resulting constant change in the single linearized buffer offset, or
/// failure if that change is not a compile-time constant.
///
/// This mirrors rocMLIR's index-diff rules (`IndexDiffUpdateRewritePattern` in
/// SugarToLoops.cpp): each transform maps an upper-space diff to a lower-space
/// diff. Because the input diff is a compile-time constant and every rule is
/// constant arithmetic, the whole propagation stays constant - including
/// through `Merge`/`Unmerge` reconstructions (e.g. the conv `gemmK` packing)
/// where a flattened affine map would keep opaque floordiv/mod.
///
/// Carry-neutrality guard: rocMLIR keeps the running coordinate valid via carry
/// propagation on `Merge`. We instead want a single loop-invariant stride,
/// which is only valid when those carries do not change the linearized offset,
/// i.e. when the merged dim's lower dims are laid out contiguously underneath
/// (their buffer strides nest as the merge factors). When that does not hold,
/// we bail.
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

/// Returns true if a unit change in the merge-lower coordinate `pos` reaches a
/// validity-impacting bound check while propagating down `belowMaps` (the
/// sub-chain below the iv-traversed merge). Mirrors the bound checks emitted by
/// `updateValidityAfter`.
///
/// Coordinates that impact validity must be carried as full distributed tiles,
/// so the per-element mask can be rebuilt without a cross-lane broadcast.
/// Coordinates that do not impact validity need not be persisted at all: their
/// offset contribution is recovered from the carry-out of the coordinate below
/// them.
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

/// The part of the analysis both the affine and carry paths share.
struct LoopPtrInfo {
  TransformsToPtrOp op; // The op to be simplified
  scf::ForOp loop;      // The loop `op` lives in
  /// The transform chain of `op`'s source, ordered from the upper view (index
  /// 0) down to the root buffer.
  SmallVector<TransformMapAttr> transforms;
  /// The positions in `op.getExtraIndices()` that hold the induction variable.
  SmallVector<unsigned> ivPositions;
  /// Set when some extra index is loop-variant but is not the induction
  /// variable. The affine path tolerates this (it rewrites the base op in place
  /// and references the index directly); the carry path cannot rebuild such an
  /// index in the preheader.
  bool hasLoopVariantNonIvIdx = false;
};

/// An op whose linearized offset is affine in the induction variable while its
/// mask is loop-invariant: a single integer accumulator (`base + IV*stride`)
/// is enough to advance it.
struct AffineCandidate {
  TransformsToPtrOp op; // The op to be simplified
  int64_t unitStride = 0;
};

/// An op whose induction variable flows through a `Merge` the affine path
/// cannot linearize, and/or whose mask moves with the induction variable.
struct CarryCandidate {
  TransformsToPtrOp op; // The op to be simplified
  /// The full transform chain, as in `LoopPtrInfo`.
  SmallVector<TransformMapAttr> transforms;
  /// Index into `transforms` of the map holding the iv-traversed merge.
  unsigned mergeIdx = 0;
  SmallVector<unsigned> coordPositions; // which decomposed coords are carried
  SmallVector<int64_t> coordSizes;      // each coord's wrap limit (merge size)
  SmallVector<int64_t> coordSteps;      // per-iteration increment of each coord

  /// The sub-chain above and including the iv-traversed merge.
  ArrayRef<TransformMapAttr> aboveMaps() const {
    return ArrayRef<TransformMapAttr>(transforms).take_front(mergeIdx + 1);
  }
  /// The sub-chain below the iv-traversed merge, i.e. from the merge's
  /// decomposed coordinates down to the raw buffer.
  ArrayRef<TransformMapAttr> belowMaps() const {
    return ArrayRef<TransformMapAttr>(transforms).drop_front(mergeIdx + 1);
  }
};

/// Carry fallback, tried when `analyzeAffineCandidate` bails.
///
/// If the transforms_to_ptr we are analyzing is supported by our current
/// implemetation, return the candidate.
///
/// If we bail, it can be for 2 main reasons:
/// 1. Fundamentally we cannot handle this case: For example, if
/// the loop step is not constant, there is nothing we can do (not a limitation
/// of this transform)
/// 2. The implementation does not support this case. Here it's unclear if
/// supporting more cases would actually improve performance, since the carry
/// path on complex IRs can potentially make things worse due to register
/// pressure.
static FailureOr<CarryCandidate>
analyzeCarryCandidate(const LoopPtrInfo &info) {
  TransformsToPtrOp op = info.op;
  ArrayRef<unsigned> ivPositions = info.ivPositions;
  ArrayRef<TransformMapAttr> transforms = info.transforms;
  auto bail = [&](const char *reason) {
    LLVM_DEBUG(llvm::dbgs() << "[analyzeCarryCandidate] skip " << op.getLoc()
                            << ": " << reason << "\n");
    return failure();
  };

  // Unlike the affine path, the carry path cannot reference a loop-variant
  // non-iv index directly: `cloneSliceBeforeLoop` would have to rebuild it in
  // the preheader, where it does not exist yet.
  if (info.hasLoopVariantNonIvIdx)
    return bail("an extra index is loop-variant but is not the IV (carry path "
                "cannot rebuild it in the preheader)");

  // analyzeLoopPointer allows multiple extraIndices positions to be classified
  // as the loop IV, whereas analyzeCarryCandidate only records one merge.
  // If the same IV affects another coordinate path outside that merge,
  // the carry rewrite pins all IV indices to lb and then only carries
  // the merge-decomposed coordinates, silently dropping the other IV-dependent
  // contribution. Therefore, we bail if there is more than one iv position.
  if (ivPositions.size() != 1)
    return bail("iv is used in more than one extra index; the carry path "
                "incrementalizes only one iv-driven coordinate, so the other "
                "iv contributions to the address/mask would be ignored");

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

  // No nested Merge may sit below the iv-traversed merge.
  //
  // Rejecting these chains costs nothing in practice: the mask rebuild has to
  // re-expand the sub-chain below the merge, so we would put the
  // floordiv/mod back into the loop body (so we would gain nothing).
  for (TransformMapAttr map : transforms.drop_front(mergeIdx + 1))
    for (TransformAttr t : map.getOps())
      if (t.getType() == TransformType::Merge)
        return bail("a nested merge sits below the iv-traversed merge; the "
                    "validity classification is not reliable through it");

  // Per-iteration coord steps: the merged coordinate advances by `D` each loop
  // step, delinearized against the merge's nested place values (suffix
  // products). delinearize leaves the top coord un-wrapped, which is what we
  // want (the highest coord never wraps; the loop bounds keep it in range).
  ArrayRef<int64_t> mergeParams = mergeOp.getParams();

  // The per-iteration merged diff must be a compile-time constant, so the loop
  // step must be constant.
  scf::ForOp loop = info.loop;
  std::optional<APInt> stepAP = loop.getConstantStep();
  if (!stepAP)
    return bail("loop step is not a compile time constant");
  int64_t stepConst = stepAP->getSExtValue();

  // How far the merged coordinate moves per loop iteration
  int64_t D = mergedDiffUnit * stepConst;

  // A non-positive D (caused by a negative loop step, or a negative coefficient
  // above the merge) would break the carry rewrite, so bail in such case.
  if (D <= 0)
    return bail("non-positive merged per-iteration diff");

  CarryCandidate cand;
  cand.op = op;
  cand.transforms.assign(transforms.begin(), transforms.end());
  cand.mergeIdx = static_cast<unsigned>(mergeIdx);
  cand.coordPositions.assign(mergeOp.getLowerDims().begin(),
                             mergeOp.getLowerDims().end());
  cand.coordSizes.assign(mergeParams.begin(), mergeParams.end());
  cand.coordSteps = delinearize(D, computeSuffixProduct(mergeParams));
  return cand;
}

/// Affine path: the mask does not depend on the IV and the offset can be
/// incremented by a single constant per iteration.
static FailureOr<AffineCandidate>
analyzeAffineCandidate(const LoopPtrInfo &info) {
  TransformsToPtrOp op = info.op;
  auto bail = [&](const char *reason) {
    LLVM_DEBUG(llvm::dbgs() << "[analyzeAffineCandidate] skip " << op.getLoc()
                            << ": " << reason << "\n");
    return failure();
  };

  if (validityDependsOnAnyDim(info.transforms, info.ivPositions))
    return bail("the validity mask depends on the iv");

  DenseMap<unsigned, int64_t> ivDiff;
  for (unsigned p : info.ivPositions)
    ivDiff[p] = 1;
  FailureOr<int64_t> stride = linearizedDiffStride(info.transforms, ivDiff);
  if (failed(stride))
    return bail("the offset has no single constant per-iteration stride");
  if (*stride == 0)
    return bail("the offset does not change with the iv");

  AffineCandidate cand;
  cand.op = op;
  cand.unitStride = *stride;
  return cand;
}

/// Collect what both incrementalization paths need from an in-loop
/// transforms_to_ptr op, or fail if the op is no candidate at all.
static FailureOr<LoopPtrInfo> analyzeLoopPointer(TransformsToPtrOp op,
                                                 scf::ForOp loop) {
  auto bail = [&](const char *reason) {
    LLVM_DEBUG(llvm::dbgs() << "[analyzeLoopPointer] skip " << op.getLoc()
                            << ": " << reason << "\n");
    return failure();
  };

  if (op->getBlock() != loop.getBody())
    return bail("op is not a direct child of the loop body");

  std::optional<Value> maybeIv =
      cast<LoopLikeOpInterface>(loop.getOperation()).getSingleInductionVar();
  if (!maybeIv)
    return bail("loop does not have exactly one induction variable");
  Value iv = *maybeIv;

  LoopPtrInfo info;
  info.op = op;
  info.loop = loop;

  // Classify each transforms_to_ptr index as the loop iv (which we
  // incrementalize), loop-invariant (referenced directly), or loop-variant but
  // not the iv. The last kind is only tracked here, not rejected: the affine
  // path handles it, while the carry path bails on it. If no index is the iv
  // the pointer is already loop-invariant and there is nothing to do.
  ValueRange extra = op.getExtraIndices();
  for (auto [pos, idx] : llvm::enumerate(extra)) {
    if (isInductionVar(idx, iv)) {
      info.ivPositions.push_back(pos);
      continue;
    }
    if (!loop.isDefinedOutsideOfLoop(idx))
      info.hasLoopVariantNonIvIdx = true;
  }
  if (info.ivPositions.empty())
    return bail("pointer does not depend on the iv (there is nothing to do)");

  // Walk the transform chain down to its root.
  Value root;
  std::tie(root, std::ignore) = untransform(op.getSource(), info.transforms);
  if (!isa<BlockArgument>(root))
    return bail("transform chain root is not a block argument");

  return info;
}

/// Clone, just before `loop`, the in-loop ops that define `v`, so that `v`
/// becomes available in the loop preheader. Ops defined outside the loop are
/// referenced directly. `map` caches and returns the rewired value.
static Value cloneSliceBeforeLoop(OpBuilder &b, Value v, scf::ForOp loop,
                                  IRMapping &map) {
  if (Value cached = map.lookupOrNull(v))
    return cached;
  if (loop.isDefinedOutsideOfLoop(v))
    return v;
  Operation *def = v.getDefiningOp();
  assert(def && "loop-defined value without a defining op should be the iv, "
                "which is excluded by analyzeLoopPointer");
  for (Value operand : def->getOperands())
    cloneSliceBeforeLoop(b, operand, loop, map);
  // clone() remaps operands via `map` and records result mappings into `map`.
  b.clone(*def, map);
  return map.lookup(v);
}

/// Create a constant tensor of shape/element-type `tt`, every element equal to
/// `v`.
static Value splatConst(OpBuilder &b, Location loc, RankedTensorType tt,
                        int64_t v) {
  Value s = arith::ConstantOp::create(b, loc,
                                      b.getIntegerAttr(tt.getElementType(), v));
  return triton::SplatOp::create(b, loc, tt, s);
}

/// Result of advancing the carry by one loop step: the advanced validity
/// (suffix) coordinates and the per-element offset increment.
struct CarryStep {
  SmallVector<Value> nextSuffix;
  Value offsetDelta;
};

/// Advance the carry by one loop step. `suffix` holds the
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
static CarryStep emitCarryStep(OpBuilder &b, Location loc,
                               ArrayRef<Value> suffix, ArrayRef<int64_t> sizes,
                               ArrayRef<int64_t> steps,
                               ArrayRef<int64_t> strides, bool hasPrefix,
                               int64_t prefixStep, int64_t prefixStride) {
  auto tt = cast<RankedTensorType>(suffix[0].getType());
  CarryStep out;
  out.nextSuffix.resize(suffix.size());
  Value carry; // tile-shaped; null == no incoming carry
  Value delta; // tile-shaped; null == 0
  for (int j = static_cast<int>(suffix.size()) - 1; j >= 0; --j) {
    Value v = arith::AddIOp::create(b, loc, suffix[j],
                                    splatConst(b, loc, tt, steps[j]));
    if (carry)
      v = arith::AddIOp::create(b, loc, v, carry);
    // A suffix coordinate is the global top only when no prefix coordinate was
    // dropped above it; the loop bound keeps the global top from wrapping.
    if (j > 0 || hasPrefix) {
      Value size = splatConst(b, loc, tt, sizes[j]);
      Value ge =
          arith::CmpIOp::create(b, loc, arith::CmpIPredicate::uge, v, size);
      Value vsub = arith::SubIOp::create(b, loc, v, size);
      v = arith::SelectOp::create(b, loc, ge, vsub, v);
      carry = arith::SelectOp::create(b, loc, ge, splatConst(b, loc, tt, 1),
                                      splatConst(b, loc, tt, 0));
    } else {
      carry = Value();
    }
    out.nextSuffix[j] = v;
    Value dCoord = arith::SubIOp::create(b, loc, v, suffix[j]);
    Value term = arith::MulIOp::create(b, loc, dCoord,
                                       splatConst(b, loc, tt, strides[j]));
    delta = delta ? arith::AddIOp::create(b, loc, delta, term) : term;
  }
  if (hasPrefix) {
    Value dPrefix = splatConst(b, loc, tt, prefixStep);
    if (carry)
      dPrefix = arith::AddIOp::create(b, loc, dPrefix, carry);
    Value term = arith::MulIOp::create(b, loc, dPrefix,
                                       splatConst(b, loc, tt, prefixStride));
    delta = delta ? arith::AddIOp::create(b, loc, delta, term) : term;
  }
  out.offsetDelta = delta;
  return out;
}

/// The plan to rewrite a carry candidate.
struct CarryPlan {
  // The analysis this state was built from. It owns the op to be replaced, the
  // transform chain (and thus `aboveMaps()` / `belowMaps()`) and the merge's
  // decomposed coordinates: their positions, wrap limits and per-step
  // increments.
  CarryCandidate cand;

  unsigned iterArgStart = 0; // first new iter_arg index for this candidate

  RankedTensorType ptrType;
  Type maskType;

  // Coordinate vector at iv == lb. Invariant entries are used verbatim in the
  // loop body; the entries at `cand.coordPositions` are replaced by the carried
  // iter_args and seed their initial values.
  SmallVector<Value> iter0Coords;
  SmallVector<Value> carriedInits; // init for the carried coord iter_args

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
  // Uses `cand.belowMaps()`, the transform chain below the merge, to rebuild
  // the mask on each iteration.

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
static void simplifyAffineCandidate(const AffineCandidate &cand, Value iv,
                                    Value lb) {
  TransformsToPtrOp op = cand.op;
  OpBuilder b(op);
  Location loc = op.getLoc();

  // Pin the iv to lb: an index that is the iv (or a cast of it) is re-expressed
  // at lb, while loop-invariant indices are referenced directly.
  SmallVector<Value> baseIdx;
  for (Value idx : op.getExtraIndices()) {
    if (isInductionVar(idx, iv))
      idx = getValueOrCreateCastToIndexLike(b, loc, idx.getType(), lb);
    baseIdx.push_back(idx);
  }

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
      b, loc, getValueOrCreateCastToIndexLike(b, loc, offsetTy, ivMinusLb),
      stride);
  Value deltaSplat = triton::SplatOp::create(b, loc, ptrType, delta);
  Value ptr = arith::AddIOp::create(b, loc, base.getPointers(), deltaSplat);

  op.getPointers().replaceAllUsesWith(ptr);
  op.getMask().replaceAllUsesWith(base.getMask());
  op.erase();
}

static bool simplifyCarryCandidates(scf::ForOp loop,
                                    ArrayRef<CarryCandidate> carryCands);

/// Simplify all eligible transforms_to_ptr ops in `loop`: Affine candidates are
/// rewritten in place, Carry candidates get loop-carried coordinate state.
/// Returns true if the IR was changed.
static bool trySimplifyTransformsCandidates(scf::ForOp loop) {
  // The affine path is always preferred over the carry path: its rewrite is
  // cheaper (one scalar accumulator rather than loop-carried coordinate tiles).
  // The carry path is therefore only tried for the ops whose address and mask
  // arithmetic the affine path cannot express.
  SmallVector<AffineCandidate, 0> affineCands;
  SmallVector<CarryCandidate, 0> carryCands;
  for (Operation &o : loop.getBody()->without_terminator()) {
    auto tp = dyn_cast<TransformsToPtrOp>(&o);
    if (!tp)
      continue;
    FailureOr<LoopPtrInfo> info = analyzeLoopPointer(tp, loop);
    if (failed(info))
      continue;
    if (FailureOr<AffineCandidate> affine = analyzeAffineCandidate(*info);
        succeeded(affine)) {
      affineCands.push_back(std::move(*affine));
      continue;
    }
    if (FailureOr<CarryCandidate> carry = analyzeCarryCandidate(*info);
        succeeded(carry))
      carryCands.push_back(std::move(*carry));
  }

  if (affineCands.empty() && carryCands.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "No candidates to simplify, skip " << loop.getLoc() << "\n");
    return false;
  }

  // Affine candidates are rewritten in place; Carry candidates need
  // loop-carried coordinate state, so they are handled together by
  // simplifyCarryCandidates (they share one rewrite of the loop).
  Value iv = loop.getInductionVar();
  Value lb = loop.getLowerBound();
  for (const AffineCandidate &cand : affineCands)
    simplifyAffineCandidate(cand, iv, lb);

  bool changed = !affineCands.empty();
  if (!carryCands.empty() && simplifyCarryCandidates(loop, carryCands))
    changed = true;

  return changed;
}

/// Analyze each carry candidate and, for those eligible for the pointer
/// recurrence, emit their preheader init values (base coordinates,
/// carried suffix coordinates, zero offset accumulator) and return the
/// per-candidate `CarryPlan` state. Ineligible candidates are skipped (their op
/// is left in the loop, to be lowered in place by
/// rock-transforms-to-pointer-arith).
static SmallVector<CarryPlan, 0>
buildCarryPlans(OpBuilder &b, scf::ForOp loop,
                ArrayRef<CarryCandidate> carryCands) {
  Location loc = loop.getLoc();
  Value iv = loop.getInductionVar();
  Value lb = loop.getLowerBound();

  // Build each carry candidate's iter_arg initial values before the loop (as
  // SSA requires): the merge's decomposed coordinates at iv == lb.
  SmallVector<CarryPlan, 0> plans;
  for (const CarryCandidate &cand : carryCands) {
    TransformsToPtrOp op = cand.op;
    ArrayRef<TransformMapAttr> belowMaps = cand.belowMaps();

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

    // Classify each carried coordinate as validity-impacting or not. When it's
    // not, we compute the value of the decomposition and we carry the suffix
    // and a full-tile offset accumulator and *drop* the high non-validity
    // coordinate (at most one; the loop bound keeps the merge's top place from
    // wrapping), recovering its offset contribution from the suffix carry-out.
    bool coordLayoutOk = false;
    unsigned suffixCount = 0;
    unsigned prefixSize = 0;

    SmallVector<bool> impacts(cand.coordPositions.size());
    for (auto [i, pos] : llvm::enumerate(cand.coordPositions))
      impacts[i] = variantImpactsValidity(belowMaps, pos);

    // Eligible only when the validity-impacting coords are a contiguous suffix:
    // find the first impacting coord (everything before it is non-impacting by
    // construction) and require everything from there on to also impact. The
    // non-impacting prefix is dropped, so it may be at most one coord.
    //
    // We support only one prefix mainly because the implementation currently
    // assumes so. For instance, `emitCarryStep` assumes this: it models the
    // dropped prefix as a single scalar step/stride pair fed by one carry.
    //
    // TODO: Support cases with more than one prefix coord.
    auto *firstImpact = llvm::find(impacts, true);
    if (firstImpact != impacts.end()) {
      prefixSize = std::distance(impacts.begin(), firstImpact);
      suffixCount = impacts.size() - prefixSize;
      coordLayoutOk = prefixSize <= 1 &&
                      llvm::all_of(llvm::make_range(firstImpact, impacts.end()),
                                   [](bool v) { return v; });
    }

    // Not eligible for the pointer-recurrence carry (the validity coordinates
    // are not a suffix, or more than one prefix coord would be dropped): bail
    // out.
    if (!coordLayoutOk)
      continue;

    // Pin the iv to lb: an index that is the iv (or a cast of it, e.g.
    // index_cast(iv) for an index-typed loop) is re-expressed at lb, while
    // loop-invariant indices are referenced directly. Loop-variant non-iv
    // indices were already rejected by analyzeCarryCandidate, so this only
    // clones iv casts into the preheader.
    IRMapping cloneMap;
    cloneMap.map(iv, lb);
    Value srcPre = cloneSliceBeforeLoop(b, op.getSource(), loop, cloneMap);
    SmallVector<Value> baseIdx;
    for (Value idx : op.getExtraIndices())
      baseIdx.push_back(cloneSliceBeforeLoop(b, idx, loop, cloneMap));

    auto ptrType = cast<RankedTensorType>(op.getPointers().getType());
    ArrayRef<int64_t> outShape = ptrType.getShape();
    // Offset element width (i32 or i64), matching the op's inferred pointer
    // result type. All coordinate arithmetic below is produced in this width.
    Type idxTy = ptrType.getElementType();

    // Reconstruct the merge's lower coordinate vector at iv == lb by expanding
    // the chain above and including the merge over the (lb-pinned) extra
    // indices plus per-tile ranges - the same seed the lowering uses, but
    // stopping at the merge's lower space. The extra indices arrive as i32 and
    // are widened to the index width so they compose with the make_range
    // coordinates (baseIdx itself stays i32: it still feeds a TransformsToPtrOp
    // whose extraIndices are i32).
    SmallVector<Value> initValues;
    for (Value idx : baseIdx)
      initValues.push_back(getValueOrCreateCastToIndexLike(b, loc, idxTy, idx));
    for (size_t d = 0; d < outShape.size(); ++d)
      initValues.push_back(
          makeRange(b, loc, 0, outShape[d], outShape.size(), d, idxTy));
    AffineMap aboveMap = composeTransforms(cand.aboveMaps());
    FailureOr<SmallVector<Value>> iter0 =
        expandAffineMap(b, loc, aboveMap, initValues, idxTy);
    if (failed(iter0))
      continue;

    CarryPlan plan;
    plan.cand = cand;
    plan.ptrType = ptrType;
    plan.maskType = op.getMask().getType();
    plan.iter0Coords = *iter0;
    plan.srcPre = srcPre;
    plan.baseIdx = baseIdx;
    plan.offsetStrides = strides;
    plan.suffixCount = suffixCount;
    plan.hasPrefix = (prefixSize == 1);
    if (plan.hasPrefix) {
      plan.prefixStep = cand.coordSteps[0];
      plan.prefixStride = strides[0];
    }
    // Carry the validity (suffix) coordinates as full tiles plus a full-tile
    // zero offset accumulator (the base pointer already encodes offset0).
    bool carriedInitsOk = true;
    for (unsigned pos :
         ArrayRef<unsigned>(cand.coordPositions).take_back(suffixCount)) {
      Value carried = broadcastToShape(b, loc, (*iter0)[pos], outShape);
      if (!carried) {
        carriedInitsOk = false;
        break;
      }
      plan.carriedInits.push_back(carried);
    }
    // A carried coordinate that can't be broadcast to the tile shape means this
    // candidate isn't eligible; skip it and let the op be lowered in place.
    if (!carriedInitsOk)
      continue;
    auto offTy = RankedTensorType::get(outShape, ptrType.getElementType());
    plan.offsetAccInit = splatConst(b, loc, offTy, 0);
    plans.push_back(std::move(plan));
  }
  return plans;
}

/// Build the base pointer tile at iv == lb (all extra indices pinned to lb).
/// The op's mask output is unused (the mask is iv-dependent and rebuilt each
/// iteration by `buildCarryMask`).
static Value buildCarryBasePtr(OpBuilder &b, Location loc,
                               const CarryPlan &plan) {
  auto baseOp = TransformsToPtrOp::create(b, loc, plan.ptrType, plan.maskType,
                                          plan.srcPre, plan.baseIdx);
  return baseOp.getPointers();
}

/// Rebuild the validity mask for the current iteration: splice the carried
/// suffix coordinates (the new loop's iter_args) into the iv == lb coordinate
/// vector, then re-expand the sub-chain below the merge. Returns failure if the
/// sub-chain expansion fails.
static FailureOr<Value> buildCarryMask(OpBuilder &b, Location loc,
                                       const CarryPlan &plan,
                                       scf::ForOp newLoop) {
  SmallVector<Value> coords(plan.iter0Coords);
  ArrayRef<unsigned> suffixPos =
      ArrayRef<unsigned>(plan.cand.coordPositions).take_back(plan.suffixCount);
  for (auto [k, pos] : llvm::enumerate(suffixPos))
    coords[pos] = newLoop.getRegionIterArg(plan.iterArgStart + k);
  FailureOr<OffsetAndMask> om = expandCoordsToOffsetAndMask(
      b, loc, plan.cand.belowMaps(), coords, plan.ptrType.getShape(),
      plan.ptrType.getElementType(),
      /*computeOffset=*/false);
  if (failed(om))
    return failure();
  return om->mask;
}

/// Form the current pointer as the base tile plus the carried full-tile offset
/// accumulator (the last iter_arg for this candidate). No offset re-expansion.
static Value buildCarryPtr(OpBuilder &b, Location loc, Value basePtr,
                           const CarryPlan &plan, scf::ForOp newLoop) {
  Value offset = newLoop.getRegionIterArg(plan.iterArgStart + plan.suffixCount);
  return arith::AddIOp::create(b, loc, basePtr, offset);
}

/// Rewrite the eligible Carry candidates in `carryCands`: replace `loop` with a
/// new loop that carries the merge's decomposed coordinate state via the
/// pointer recurrence. Returns true if the candidates were rewritten.
static bool simplifyCarryCandidates(scf::ForOp loop,
                                    ArrayRef<CarryCandidate> carryCands) {
  Location loc = loop.getLoc();
  OpBuilder b(loop);

  SmallVector<CarryPlan, 0> plans = buildCarryPlans(b, loop, carryCands);

  // No carry candidate qualified for the pointer-recurrence rewrite: bail out.
  if (plans.empty())
    return false;

  // New iter_args: the original ones followed by each candidate's carried
  // coordinates.
  unsigned numOrig = loop.getInitArgs().size();
  SmallVector<Value> extraInits;
  for (CarryPlan &plan : plans) {
    plan.iterArgStart = numOrig + extraInits.size();
    llvm::append_range(extraInits, plan.carriedInits);
    // The full-tile offset accumulator is carried last.
    extraInits.push_back(plan.offsetAccInit);
  }

  scf::ForOp newLoop = addIterArgsToLoop(b, loop, extraInits);

  // Reconstruct each candidate's pointer/mask inside the body: rebuild the base
  // pointer/mask at iv == lb, splice the carried full-tile validity (suffix)
  // coordinates into the coord vector, rebuild only the mask, and form the
  // pointer as the base plus the carried full-tile offset accumulator (no
  // offset re-expansion).
  SmallVector<std::pair<Value, Value>> ptrsAndMasks;
  for (CarryPlan &plan : plans) {
    b.setInsertionPoint(plan.cand.op);
    Value basePtr = buildCarryBasePtr(b, loc, plan);
    FailureOr<Value> mask = buildCarryMask(b, loc, plan, newLoop);
    if (failed(mask)) {
      SmallVector<Value> unchanged(
          newLoop.getRegionIterArgs().drop_front(numOrig));
      appendToForOpYield(newLoop, unchanged);
      return false;
    }
    ptrsAndMasks.emplace_back(buildCarryPtr(b, loc, basePtr, plan, newLoop),
                              *mask);
  }

  // Emit IR for the the advanced carried coordinates (coordinate carry) for
  // each candidate.
  b.setInsertionPoint(newLoop.getBody()->getTerminator());
  SmallVector<Value> carried;
  for (CarryPlan &plan : plans) {
    // Advance the full-tile validity (suffix) coordinates and the
    // full-tile offset accumulator by the per-step delta.
    SmallVector<Value> suffix;
    for (unsigned k = 0; k < plan.suffixCount; ++k)
      suffix.push_back(newLoop.getRegionIterArg(plan.iterArgStart + k));

    CarryStep carryStep = emitCarryStep(
        b, loc, suffix,
        ArrayRef<int64_t>(plan.cand.coordSizes).take_back(plan.suffixCount),
        ArrayRef<int64_t>(plan.cand.coordSteps).take_back(plan.suffixCount),
        ArrayRef<int64_t>(plan.offsetStrides).take_back(plan.suffixCount),
        plan.hasPrefix, plan.prefixStep, plan.prefixStride);

    llvm::append_range(carried, carryStep.nextSuffix);

    Value offset =
        newLoop.getRegionIterArg(plan.iterArgStart + plan.suffixCount);
    carried.push_back(
        arith::AddIOp::create(b, loc, offset, carryStep.offsetDelta));
  }
  appendToForOpYield(newLoop, carried);

  for (auto [plan, ptrAndMask] : llvm::zip_equal(plans, ptrsAndMasks)) {
    plan.cand.op.getPointers().replaceAllUsesWith(ptrAndMask.first);
    plan.cand.op.getMask().replaceAllUsesWith(ptrAndMask.second);
    plan.cand.op.erase();
  }
  return true;
}

} // end anonymous namespace

void RockIncrementalPointerArithPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // TODO(AIROCMLIR-1049): Investigate if this can be beneficial for
  // non-convolution kernels.
  if (!func->hasAttr(rock::ConvKernelAttr::getMnemonic()))
    return;

  // Re-walk after each rewrite: trySimplifyTransformsCandidates may replace the
  // loop op (carry path), so collected handles would dangle. After a rewrite
  // the base ops are pinned to iv == lb and no longer depend on the iv, so they
  // are not re-selected as candidates and this terminates.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp f) { loops.push_back(f); });
    for (scf::ForOp f : loops) {
      if (trySimplifyTransformsCandidates(f)) {
        changed = true;
        break;
      }
    }
  }
}
