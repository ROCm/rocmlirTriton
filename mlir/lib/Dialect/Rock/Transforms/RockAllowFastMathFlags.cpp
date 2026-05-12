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
// Tags floating-point divisions with fastmath `arcp` so backends may lower
// them as multiply-by-reciprocal (same idea as attention output scaling).
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

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

void RockAllowFastMathFlagsPass::runOnOperation() {
  func::FuncOp func = getOperation();

  func.walk([](arith::DivFOp divOp) {
    LLVM_DEBUG(llvm::dbgs() << "Tagging arcp on " << divOp << "\n");
    divOp.setFastmath(divOp.getFastmath() | arith::FastMathFlags::arcp);
  });
}
