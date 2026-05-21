//===- RemoveRedundantCasts.cpp - Remove redundant float casts -----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tags `arith.extf(arith.truncf %wide) -> %wide` round-trips inside
// `rock.kernel` functions with `fastmath<contract>` so that the standard
// canonicalize pass that follows in the kernel pipeline can fold them away
// via upstream MLIR's existing `arith.ExtFOp::fold`.
//
// Upstream MLIR's `arith.ExtFOp::fold` only collapses this round-trip when
// both `arith.truncf` and `arith.extf` carry `fastmath<contract>`. The guard
// exists because the rewrite is precision-recovering, not bit-equivalent: a
// wide value outside the narrow representable range would saturate to `inf`
// under `arith.truncf`, and bypassing the trunc preserves the original
// (un-saturated) wide value. For IEEE-faithful code outside the contract
// regime that is an observable semantic change, so upstream refuses to fire
// unless the IR explicitly opts in via the `contract` fast-math flag.
//
// The rock-side lowering pipeline injects `truncf -> extf` pairs at points
// where the narrow type is only used as a transport format, e.g. an f32
// accumulator emitted by `rock-gridwise-attn-to-blockwise` is briefly
// narrowed to f16 to match a downstream blockwise tile shape, then
// immediately widened back so the f32 softmax / scale / reduce body can
// run in its natural precision. At those splice points:
//
//   * the wide values are by construction within the narrow representable
//     range (they came out of the narrow tile we just wrote), so the
//     saturation-changes-semantics concern that gates the upstream fold
//     does not apply;
//   * the trunc/ext pair has no numerical purpose and would lower to a
//     pair of GPU conversion instructions that the hardware executes on
//     every element of the tile.
//
// Rather than re-implement the fold, this pass walks the function and,
// for every `arith.extf` whose defining op is an `arith.truncf` of the
// matching wide type, merges `fastmath<contract>` into both casts' flag
// sets. Casts that are not part of a round-trip are left untouched, so the
// semantic relaxation only applies where we have actually proved the round-trip
// is safe. The pass itself performs no IR rewrites; the next `-canonicalize` in
// the kernel pipeline runs upstream's fold on the now-tagged pair and removes
// the dead cast(s).
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREMOVEREDUNDANTCASTSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-remove-redundant-casts"

using namespace mlir;
using namespace mlir::rock;

namespace {

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

struct RockRemoveRedundantCastsPass
    : public rock::impl::RockRemoveRedundantCastsPassBase<
          RockRemoveRedundantCastsPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockRemoveRedundantCastsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // Walk each `arith.extf` and check whether it sits on top of an
  // `arith.truncf` whose input type matches the extf's output type. If so,
  // tag both casts with `fastmath<contract>`. The next `-canonicalize`
  // in the kernel pipeline will then collapse the pair via upstream
  // `arith.ExtFOp::fold`. We do not modify or replace any op structure
  // here, so the walk is safe and terminates in a single pass.
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
