//===- VerifyPackedUpcastLayout.cpp - Check packed upcast layouts --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AMD lowerings of the packed FP4/FP8 upcasts consume a thread's values in
// fixed-size groups, because one group is exactly the register the conversion
// sequence operates on: four packed bytes fill the 32-bit operand of the
// `v_cvt_scalef32_*` sequences, and eight fill the wider `v_cvt_pk_scale_pk8_*`
// one. `ttg.fp4_to_fp`, `amdg.scaled_upcast_fp4` and `amdg.scaled_upcast_fp8`
// all assert that precondition and then index the group unconditionally, so a
// layout that leaves a thread holding a partial group is an out-of-bounds read
// once assertions are compiled out.
//
// Rock cannot avoid such a layout when it builds the kernel: Triton derives the
// distribution from the block tile and `numWarps` in
// `tritonamdgpu-accelerate-matmul`, so the per-thread count is only known once
// the encodings exist. This pass therefore runs at the end of the TTGIR
// pipeline, immediately before the conversion to LLVM, where the layouts are
// final. A violating (kernel x perf-config) combination is reported through
// `markAsNotApplicable` so the tuning driver records it as inapplicable and
// keeps searching, instead of dying on a compiler crash.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKVERIFYPACKEDUPCASTLAYOUTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-verify-packed-upcast-layout"

using namespace mlir;
using namespace mlir::rock;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace {
struct RockVerifyPackedUpcastLayoutPass
    : public rock::impl::RockVerifyPackedUpcastLayoutPassBase<
          RockVerifyPackedUpcastLayoutPass> {
  using rock::impl::RockVerifyPackedUpcastLayoutPassBase<
      RockVerifyPackedUpcastLayoutPass>::RockVerifyPackedUpcastLayoutPassBase;
  void runOnOperation() override;
};
} // end anonymous namespace

/// Number of packed values one lowered upcast group consumes. Four packed
/// bytes fill the 32-bit register that `upcast8xMxfp4_SW` and the
/// `v_cvt_scalef32_*` sequences read.
static constexpr unsigned kPackedUpcastGroup = 4;

/// The `v_cvt_pk_scale_pk8_*` form of the scaled upcast takes twice as wide an
/// operand, so a thread's packed values are consumed eight at a time.
static constexpr unsigned kWidePackedUpcastGroup = 8;

/// Report `op` as inapplicable when its source layout hands a thread a partial
/// upcast group. Returns failure when the op must be rejected.
static LogicalResult checkUpcastSource(Operation *op, Value src,
                                       unsigned group) {
  auto srcTy = dyn_cast<RankedTensorType>(src.getType());
  // Before the layouts are assigned there is nothing to check; the pass is
  // scheduled where they exist, but stay defensive so it is also usable on
  // hand-written IR.
  if (!srcTy || !srcTy.getEncoding())
    return success();

  unsigned elemsPerThread = ttg::getTotalElemsPerThread(srcTy);
  LLVM_DEBUG(llvm::dbgs() << "packed values per thread: " << elemsPerThread
                          << " for " << *op << "\n");
  if (elemsPerThread % group == 0)
    return success();

  rock::markAsNotApplicable(op);
  return op->emitOpError() << "upcast consumes " << group
                           << " packed values per thread, but this layout "
                              "provides "
                           << elemsPerThread;
}

void RockVerifyPackedUpcastLayoutPass::runOnOperation() {
  // An unset arch means hand-written IR rather than the pipeline; assume the
  // narrow conversion, which every target has.
  unsigned fp8Group = !arch.empty() && rock::supportsScaledUpcastPk8(arch)
                          ? kWidePackedUpcastGroup
                          : kPackedUpcastGroup;

  WalkResult res = getOperation()->walk([&](Operation *op) -> WalkResult {
    LogicalResult checked = success();
    if (auto fp4ToFp = dyn_cast<ttg::Fp4ToFpOp>(op))
      checked = checkUpcastSource(op, fp4ToFp.getSrc(), kPackedUpcastGroup);
    else if (auto scaledFp4 = dyn_cast<ttag::ScaledUpcastFp4Op>(op))
      checked = checkUpcastSource(op, scaledFp4.getInput(), kPackedUpcastGroup);
    else if (auto scaledFp8 = dyn_cast<ttag::ScaledUpcastFp8Op>(op))
      checked = checkUpcastSource(op, scaledFp8.getInput(), fp8Group);
    return failed(checked) ? WalkResult::interrupt() : WalkResult::advance();
  });

  if (res.wasInterrupted())
    signalPassFailure();
}
