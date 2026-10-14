//===- FoldClampToTriton.cpp - Fold min/max pairs into tt.clampf ----------===//
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
//
// Folds a nested arith.minnumf/arith.maxnumf pair into a single tt.clampf. For
// f16 and f32 the AMD backend emits one v_med3 for it; the other float types
// fall back to Triton's generic lowering, which is no worse than the pair we
// started from.
//
// This is an optimization and not part of lowering Rock to Triton, which is why
// it is not folded into RockToTTIR. It has to run after rock-allow-fast-math-
// flags, because it only matches a pair that already carries `nnan`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/tritonUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFOLDCLAMPTOTRITONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fold-clamp-to-triton"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockFoldClampToTritonPass
    : public rock::impl::RockFoldClampToTritonPassBase<
          RockFoldClampToTritonPass> {
  void runOnOperation() override;
};

/// Return the value behind a clamp bound, which is either a scalar float
/// constant or a splat of one.
static std::optional<APFloat> getConstantFloatBound(Value v) {
  Attribute attr;
  if (!matchPattern(v, m_Constant(&attr)))
    return std::nullopt;
  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return floatAttr.getValue();
  if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(attr))
    if (denseAttr.isSplat())
      return denseAttr.getSplatValue<APFloat>();
  return std::nullopt;
}

/// A nested minnumf/maxnumf pair that a single `tt.clampf` can replace.
struct ClampIdiom {
  Operation *inner;
  Value x;
  Value lo;
  Value hi;
};

/// Recognise a clamp written as a nested minnumf/maxnumf pair and check that
/// replacing it with a `tt.clampf` carrying `propagateNan = none` preserves
/// semantics. Both nestings reach here:
///
///   maxnumf(minnumf(x, hi), lo)   from tosa.clamp
///   minnumf(maxnumf(x, lo), hi)   from migraphx.clip
///
/// which the `nnan` required below makes equivalent. The NaN-propagating
/// `maximumf`/`minimumf` spelling is deliberately not matched, since a
/// non-propagating clamp is not a faithful replacement for it.
template <typename OuterOp, typename InnerOp>
static std::optional<ClampIdiom> matchClampIdiom(OuterOp outerOp) {
  // tt.clampf accepts FP8 operands, but Triton's type converter maps FP8 to i8,
  // so the lowering would compare integers. This is the same hazard that forces
  // FP8 casts through tt.fp_to_fp in RockToTTIR.
  if (hasFp8ElementType(outerOp.getType()))
    return std::nullopt;

  // Every producer of this idiom puts the bound on the right, which is also
  // where MLIR canonicalises the constant operand of a commutative op.
  auto innerOp = outerOp.getLhs().template getDefiningOp<InnerOp>();
  // An inner op read by anything else still has to be computed, so absorbing it
  // into the clamp would duplicate the work rather than remove it.
  if (!innerOp || !innerOp->hasOneUse())
    return std::nullopt;

  // Dropping NaN handling needs `nnan`, even though minnumf/maxnumf already
  // return the non-NaN operand. The max-outer pair still disagrees with v_med3
  // on which bound a NaN input collapses to: it yields `hi`, while v_med3
  // returns the smallest operand and so yields `lo`. Only `nnan` makes that
  // difference unobservable, and it is demanded of both nestings alike.
  constexpr arith::FastMathFlags nnan = arith::FastMathFlags::nnan;
  if (!bitEnumContainsAll(outerOp.getFastmath(), nnan) ||
      !bitEnumContainsAll(innerOp.getFastmath(), nnan))
    return std::nullopt;

  constexpr bool maxOuter = std::is_same_v<OuterOp, arith::MaxNumFOp>;
  Value loVal = maxOuter ? outerOp.getRhs() : innerOp.getRhs();
  Value hiVal = maxOuter ? innerOp.getRhs() : outerOp.getRhs();

  std::optional<APFloat> lo = getConstantFloatBound(loVal);
  std::optional<APFloat> hi = getConstantFloatBound(hiVal);
  if (!lo || !hi)
    return std::nullopt;
  // A median only equals the clamp when the bounds are ordered: for lo > hi the
  // two nestings saturate to opposite bounds and neither agrees with v_med3.
  // Unordered bounds (a NaN bound) compare as neither less-than nor equal and
  // are rejected too.
  APFloat::cmpResult order = lo->compare(*hi);
  if (order != APFloat::cmpLessThan && order != APFloat::cmpEqual)
    return std::nullopt;

  return ClampIdiom{innerOp, innerOp.getLhs(), loVal, hiVal};
}

//===----------------------------------------------------------------------===//
// ArithClampToClampFPattern - Fold a nested minnumf/maxnumf pair into
// tt.clampf.
//===----------------------------------------------------------------------===//
template <typename OuterOp, typename InnerOp>
struct ArithClampToClampFPattern : public OpRewritePattern<OuterOp> {
  using OpRewritePattern<OuterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(OuterOp op,
                                PatternRewriter &rewriter) const override {
    std::optional<ClampIdiom> clamp = matchClampIdiom<OuterOp, InnerOp>(op);
    if (!clamp)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "folding clamp into tt.clampf: " << op << "\n");
    auto propagateNan = triton::PropagateNanAttr::get(
        rewriter.getContext(), triton::PropagateNan::NONE);
    rewriter.replaceOpWithNewOp<triton::ClampFOp>(
        op, op.getType(), clamp->x, clamp->lo, clamp->hi, propagateNan);
    rewriter.eraseOp(clamp->inner);
    return success();
  }
};

using MaxOuterClampToClampFPattern =
    ArithClampToClampFPattern<arith::MaxNumFOp, arith::MinNumFOp>;
using MinOuterClampToClampFPattern =
    ArithClampToClampFPattern<arith::MinNumFOp, arith::MaxNumFOp>;
} // end anonymous namespace

void RockFoldClampToTritonPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  RewritePatternSet patterns(&getContext());
  patterns.add<MaxOuterClampToClampFPattern, MinOuterClampToClampFPattern>(
      &getContext());
  // Only the clamp patterns should fire. The default config would also fold
  // ops, CSE constants and simplify regions across the whole kernel, none of
  // which the clamp fold needs. The in-conversion pattern this pass replaces
  // did none of that either, so this keeps the rest of the kernel untouched.
  GreedyRewriteConfig config;
  config.enableFolding(false);
  config.enableConstantCSE(false);
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
  if (failed(applyPatternsGreedily(func, std::move(patterns), config)))
    signalPassFailure();
}
