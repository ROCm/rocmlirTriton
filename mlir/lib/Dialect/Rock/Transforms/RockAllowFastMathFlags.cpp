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
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

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

static LogicalResult allowFastMathFlags(func::FuncOp func) {
  IRRewriter rewriter(func->getContext());

  // Collect first: walk + replace in place can invalidate the walk iterator.
  SmallVector<arith::DivFOp, 8> divOps;
  func.walk([&](arith::DivFOp divOp) { divOps.push_back(divOp); });

  for (arith::DivFOp divOp : divOps) {
    assert(divOp.getNumOperands() == 2);
    LLVM_DEBUG(llvm::dbgs() << "Op to modify: " << divOp << "\n");
    arith::FastMathFlags combinedFlags =
        divOp.getFastmath() | arith::FastMathFlags::arcp;
    rewriter.setInsertionPoint(divOp);
    rewriter.replaceOpWithNewOp<arith::DivFOp>(divOp, divOp.getLhs(),
                                                divOp.getRhs(), combinedFlags);
  }
  return success();
}

void RockAllowFastMathFlagsPass::runOnOperation() {
  func::FuncOp func = getOperation();

  if (failed(allowFastMathFlags(func))) {
    return signalPassFailure();
  }
}
