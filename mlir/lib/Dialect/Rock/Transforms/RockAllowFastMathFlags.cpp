//===- RockAllowFastMathFlags.cpp ------------===//
//
// Copyright 2026 Advanced Micro Devices.
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
// ============================================================
//
// Tags float ops in kernels with the fast-math flags the AMDGPU backend can
// exploit, choosing per-op what's actually beneficial:
//   * `arcp`     on `arith.divf`  -> hardware reciprocal (v_rcp_f32).
//   * `contract` on `arith.{add,sub,mul}f` -> mul+add fused to v_fma_f32.
//                Also on round-tripping `arith.extf(arith.truncf %wide) ->
//                %wide` pairs whose intermediate is f16/bf16, so the redundant
//                materialization folds away.
//   * `nsz`      on `arith.{add,sub,mul,div,neg}f` -> permits ignoring the
//                sign of zero (enables a handful of LLVM peepholes such as
//                `x + 0 -> x`, `0 - x -> -x` via sign-bit XOR).
//   * `afn`      on `math.*` transcendentals -> hardware approximations
//                (v_exp_f32, v_log_f32, v_sqrt_f32, ...).

//===-----------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKALLOWFASTMATHFLAGSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-allow-fast-math-flags"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockAllowFastMathFlagsPass
    : public rock::impl::RockAllowFastMathFlagsPassBase<
          RockAllowFastMathFlagsPass> {
  void runOnOperation() override;
};

/// Generic rewrite pattern that ORs a fixed set of `FastMathFlags` into any op
/// implementing `arith::ArithFastMathInterface`. `OpTy` is the concrete op
/// type so the pattern is rooted on it (cheap matching, no extra dyn_cast).
template <typename OpTy>
struct AddFastMathFlagsPattern : public OpRewritePattern<OpTy> {
  AddFastMathFlagsPattern(MLIRContext *ctx, arith::FastMathFlags flagsToAdd,
                          PatternBenefit benefit = 1)
      : OpRewritePattern<OpTy>(ctx, benefit), flagsToAdd(flagsToAdd) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    arith::ArithFastMathInterface fmIface = op;
    arith::FastMathFlags current = fmIface.getFastMathFlagsAttr().getValue();
    arith::FastMathFlags updated = current | flagsToAdd;
    // Ensure greedy convergence: bail out once the desired bits are already on.
    if (current == updated)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Adding fast-math flags to " << *op << "\n");
    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr(fmIface.getFastMathAttrName(),
                  arith::FastMathFlagsAttr::get(op->getContext(), updated));
    });
    return success();
  }

  arith::FastMathFlags flagsToAdd;
};

/// Match `arith.extf(arith.truncf %wide) -> %wide` round-trips whose ends
/// meet at the same wide precision, and merge `fastmath<contract>` into both
/// casts' flag sets. Once both casts carry `contract`, upstream MLIR's
/// `arith.ExtFOp::fold` collapses the pair in the next greedy iteration;
/// pre-existing fast-math flags on either cast and the truncf rounding mode
/// are preserved verbatim, so this only relaxes precision in the contract
/// sense without dropping any IEEE guarantees the producer had set.
struct AnnotateExtTruncRoundTripPattern
    : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp extOp,
                                PatternRewriter &rewriter) const override {
    auto truncOp = extOp.getIn().getDefiningOp<arith::TruncFOp>();
    if (!truncOp)
      return failure();
    if (extOp.getOut().getType() != truncOp.getIn().getType())
      return failure();

    // Only collapse round-trips whose intermediate is a compute-precision
    // float the surrounding GEMM/elementwise ops already run in (f16/bf16).
    Type narrowTy = getElementTypeOrSelf(truncOp.getType());
    if (!isa<Float16Type, BFloat16Type>(narrowTy))
      return failure();

    arith::FastMathFlags extFlags =
        extOp.getFastmath().value_or(arith::FastMathFlags::none);
    arith::FastMathFlags truncFlags =
        truncOp.getFastmath().value_or(arith::FastMathFlags::none);
    arith::FastMathFlags mergedExt = extFlags | arith::FastMathFlags::contract;
    arith::FastMathFlags mergedTrunc =
        truncFlags | arith::FastMathFlags::contract;
    // Ensure greedy convergence: bail out once both casts already carry the
    // bit. The `arith.ExtFOp::fold` is then free to fire on the next visit.
    if (mergedExt == extFlags && mergedTrunc == truncFlags)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "annotating round-trip extf/truncf: " << extOp
                            << " / " << truncOp << "\n");
    rewriter.modifyOpInPlace(extOp, [&] { extOp.setFastmath(mergedExt); });
    rewriter.modifyOpInPlace(truncOp,
                             [&] { truncOp.setFastmath(mergedTrunc); });
    return success();
  }
};
} // end namespace

void RockAllowFastMathFlagsPass::runOnOperation() {
  auto func = getOperation();
  MLIRContext *ctx = &getContext();

  // x / y -> x * rcp(y) via hardware reciprocal.
  constexpr arith::FastMathFlags divFlags = arith::FastMathFlags::arcp |
                                            arith::FastMathFlags::nsz |
                                            arith::FastMathFlags::afn;
  // Allow mul+add to fuse into fma (v_fma_f32).
  constexpr arith::FastMathFlags fmaFlags =
      arith::FastMathFlags::contract | arith::FastMathFlags::nsz;
  // `0 - x` can lower to a sign-bit XOR; other ±0 peepholes too. This does not
  // imply `nnan`, so maximumf continues to propagate NaNs.
  constexpr arith::FastMathFlags nszOnly = arith::FastMathFlags::nsz;
  // Hardware approximate transcendentals (v_exp_f32, v_log_f32, ...).
  constexpr arith::FastMathFlags transcendentalFlags =
      arith::FastMathFlags::nsz | arith::FastMathFlags::contract |
      arith::FastMathFlags::afn;

  RewritePatternSet patterns(ctx);
  patterns.add<AddFastMathFlagsPattern<arith::DivFOp>>(ctx, divFlags);
  patterns.add<AddFastMathFlagsPattern<arith::AddFOp>,
               AddFastMathFlagsPattern<arith::SubFOp>,
               AddFastMathFlagsPattern<arith::MulFOp>>(ctx, fmaFlags);
  patterns.add<AddFastMathFlagsPattern<arith::NegFOp>,
               AddFastMathFlagsPattern<arith::RemFOp>,
               AddFastMathFlagsPattern<arith::MaximumFOp>,
               AddFastMathFlagsPattern<arith::MaxNumFOp>,
               AddFastMathFlagsPattern<arith::MinimumFOp>>(ctx, nszOnly);
  patterns.add<AddFastMathFlagsPattern<math::ExpOp>,
               AddFastMathFlagsPattern<math::Exp2Op>,
               AddFastMathFlagsPattern<math::ExpM1Op>,
               AddFastMathFlagsPattern<math::LogOp>,
               AddFastMathFlagsPattern<math::Log2Op>,
               AddFastMathFlagsPattern<math::Log10Op>,
               AddFastMathFlagsPattern<math::Log1pOp>,
               AddFastMathFlagsPattern<math::SinOp>,
               AddFastMathFlagsPattern<math::CosOp>,
               AddFastMathFlagsPattern<math::TanOp>,
               AddFastMathFlagsPattern<math::AsinOp>,
               AddFastMathFlagsPattern<math::AcosOp>,
               AddFastMathFlagsPattern<math::AtanOp>,
               AddFastMathFlagsPattern<math::Atan2Op>,
               AddFastMathFlagsPattern<math::SinhOp>,
               AddFastMathFlagsPattern<math::CoshOp>,
               AddFastMathFlagsPattern<math::TanhOp>,
               AddFastMathFlagsPattern<math::SqrtOp>,
               AddFastMathFlagsPattern<math::RsqrtOp>,
               AddFastMathFlagsPattern<math::CbrtOp>,
               AddFastMathFlagsPattern<math::PowFOp>,
               AddFastMathFlagsPattern<math::FPowIOp>,
               AddFastMathFlagsPattern<math::ErfOp>,
               AddFastMathFlagsPattern<math::ErfcOp>,
               AddFastMathFlagsPattern<math::AcoshOp>,
               AddFastMathFlagsPattern<math::AsinhOp>,
               AddFastMathFlagsPattern<math::AtanhOp>,
               AddFastMathFlagsPattern<math::SincosOp>>(ctx,
                                                        transcendentalFlags);
  patterns.add<AddFastMathFlagsPattern<math::ClampFOp>,
               AddFastMathFlagsPattern<math::AbsFOp>,
               AddFastMathFlagsPattern<math::CopySignOp>>(ctx, nszOnly);
  patterns.add<AddFastMathFlagsPattern<math::FmaOp>>(ctx, fmaFlags);
  patterns.add<AnnotateExtTruncRoundTripPattern>(ctx);

  if (failed(applyPatternsGreedily(func, std::move(patterns))))
    signalPassFailure();
}
