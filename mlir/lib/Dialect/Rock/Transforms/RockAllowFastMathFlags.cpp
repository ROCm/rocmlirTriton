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

//===-----------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/TypeSwitch.h"
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
} // end namespace

template <typename OpTy>
static void addFastMathFlags(OpTy op, arith::FastMathFlags extra) {
  LLVM_DEBUG(llvm::dbgs() << "Adding fast-math flags to " << *op.getOperation()
                          << "\n");
  op.setFastmath(op.getFastmath() | extra);
}

void RockAllowFastMathFlagsPass::runOnOperation() {

  getOperation().walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        // x / y -> x * rcp(y) via hardware reciprocal.
        .Case<arith::DivFOp>([&](auto operation) {
          addFastMathFlags(operation, arith::FastMathFlags::arcp |
                                          arith::FastMathFlags::nsz);
        })
        // Allow mul+add to fuse into fma (v_fma_f32).
        .Case<arith::AddFOp, arith::SubFOp, arith::MulFOp>([&](auto operation) {
          addFastMathFlags(operation, arith::FastMathFlags::contract |
                                          arith::FastMathFlags::nsz);
        })
        // `0 - x` can lower to a sign-bit XOR; other ±0 peepholes too.
        .Case<arith::NegFOp>([&](auto operation) {
          addFastMathFlags(operation, arith::FastMathFlags::nsz);
        })
        // Hardware approximate transcendentals (v_exp_f32, v_log_f32, ...).
        .Case<math::ExpOp, math::Exp2Op, math::ExpM1Op, math::LogOp,
              math::Log2Op, math::Log10Op, math::Log1pOp, math::SinOp,
              math::CosOp, math::TanOp, math::AsinOp, math::AcosOp,
              math::AtanOp, math::Atan2Op, math::SinhOp, math::CoshOp,
              math::TanhOp, math::SqrtOp, math::RsqrtOp, math::CbrtOp,
              math::PowFOp, math::FPowIOp, math::ErfOp, math::ErfcOp>(
            [&](auto operation) {
              addFastMathFlags(operation, arith::FastMathFlags::afn);
            });
  });
}
