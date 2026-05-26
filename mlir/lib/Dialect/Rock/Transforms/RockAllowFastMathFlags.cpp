//===- RockAllowFastMathFlags.cpp ------------===//
//
// Copyright 2026 Advanced Micro Devices.
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
//   * `nsz`      on `arith.{add,sub,mul,div,neg}f` -> permits ignoring the
//                sign of zero (enables a handful of LLVM peepholes such as
//                `x + 0 -> x`, `0 - x -> -x` via sign-bit XOR).
//   * `afn`      on `math.*` transcendentals -> hardware approximations
//                (v_exp_f32, v_log_f32, v_sqrt_f32, ...).
//   * `contract` on round-tripping `arith.extf(arith.truncf %wide) -> %wide`
//                pairs inside `rock.kernel` functions, so the next
//                `-canonicalize` collapses them via upstream MLIR's
//                `arith.ExtFOp::fold` (which only fires when both casts
//                carry `contract`). The rock-side lowering pipeline injects
//                these pairs at splice points where the narrow type is only
//                a transport format (e.g. an f32 accumulator briefly
//                narrowed to f16 to match a downstream tile shape and then
//                widened back); marking them contractible removes a pair of
//                GPU conversion instructions that would otherwise run on
//                every element of the tile. The round-trip walk is gated on
//                the `rock.kernel` attribute so host functions are never
//                touched.

//===-----------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
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

/// Merge `fastmath<contract>` into `op`'s existing fast-math flags. The op's
/// other fast-math flags (`nnan`, `ninf`, ...) and its rounding mode are
/// preserved verbatim, so this only relaxes precision in the contract sense
/// without dropping any pre-existing IEEE guarantees the producer had set.
template <typename CastOp>
static void markContractible(CastOp op) {
  arith::FastMathFlags existing =
      op.getFastmath().value_or(arith::FastMathFlags::none);
  arith::FastMathFlags merged = existing | arith::FastMathFlags::contract;
  if (merged == existing)
    return;

  op.setFastmath(merged);
  LLVM_DEBUG(llvm::dbgs() << "marking contractible: " << op << "\n");
}
} // end namespace

void RockAllowFastMathFlagsPass::runOnOperation() {
  auto func = getOperation();
  if (!func->hasAttr("rock.kernel"))
    return;

  MLIRContext *ctx = &getContext();

  // x / y -> x * rcp(y) via hardware reciprocal.
  constexpr arith::FastMathFlags divFlags = arith::FastMathFlags::arcp |
                                            arith::FastMathFlags::nsz |
                                            arith::FastMathFlags::afn;
  // Allow mul+add to fuse into fma (v_fma_f32).
  constexpr arith::FastMathFlags fmaFlags =
      arith::FastMathFlags::contract | arith::FastMathFlags::nsz;
  // `0 - x` can lower to a sign-bit XOR; other ±0 peepholes too.
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

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();

  // Walk each `arith.extf` in kernel functions and check whether it sits on
  // top of an `arith.truncf` whose input type matches the extf's output type.
  // If so, tag both casts with `fastmath<contract>`. The next `-canonicalize`
  // in the kernel pipeline will then collapse the pair via upstream MLIR's
  // `arith.ExtFOp::fold`. The walk itself performs no IR rewrites and so is
  // safe to run after the greedy driver. Non-kernel (host) functions are
  // skipped: the precision-recovering direction is only known-safe at the
  // splice points the rock-side pipeline introduces inside `rock.kernel`.
  func::FuncOp funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  funcOp.walk([](arith::ExtFOp extOp) {
    auto truncOp = extOp.getIn().getDefiningOp<arith::TruncFOp>();
    if (!truncOp)
      return;
    if (extOp.getOut().getType() != truncOp.getIn().getType())
      return;
    markContractible(extOp);
    markContractible(truncOp);
  });
}
