//===- FusionSplitkRegularizationPass.cpp ------------===//
//
// Copyright 2025 Advanced Micro Devices.
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
// This pass modifies fusion ops for split-k fusions. It converts any
// arith.addf/arith.subf gemmOut, other to arith.addf gemmOut,
// other/splitkFactor.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFUSIONSPLITKREGULARIZATIONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fusion-splitk-regularization"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockFusionSplitkRegularizationPass
    : public rock::impl::RockFusionSplitkRegularizationPassBase<
          RockFusionSplitkRegularizationPass> {
  void runOnOperation() override;
};
} // end namespace

static LogicalResult
divideAddBySplitkFactor(Value gemmResult, int64_t splitKFactor, IRRewriter &b) {
  SmallVector<std::tuple<Operation *, int>> adds;
  if (failed(checkValidOutputFusion(gemmResult, adds)))
    return gemmResult.getDefiningOp()->emitOpError(
        "has invalid output fusion for split-k");

  for (auto [arithOp, gemmOutIndex] : adds) {
    assert(arithOp->getNumOperands() == 2);
    assert(gemmOutIndex == 0 || gemmOutIndex == 1);
    LLVM_DEBUG(llvm::dbgs() << "Op to modify: " << arithOp << "\n");
    b.setInsertionPoint(arithOp);
    Value gemmOut = arithOp->getOperand(gemmOutIndex);
    Value otherValue =
        (gemmOutIndex == 0) ? arithOp->getOperand(1) : arithOp->getOperand(0);
    Type otherElmType = cast<ShapedType>(otherValue.getType()).getElementType();
    auto splitKFactorValue =
        createConstantFloatOp(b, arithOp->getLoc(), otherValue.getType(),
                              otherElmType, static_cast<float>(splitKFactor));
    Value otherBySplitk = b.createOrFold<arith::DivFOp>(
        arithOp->getLoc(), otherValue, splitKFactorValue);
    if (isa<arith::AddFOp>(arithOp)) {
      b.replaceOpWithNewOp<arith::AddFOp>(arithOp, gemmOut, otherBySplitk);
    } else if (isa<arith::SubFOp>(arithOp)) {
      if (gemmOutIndex == 0)
        b.replaceOpWithNewOp<arith::SubFOp>(arithOp, gemmOut, otherBySplitk);
      else
        b.replaceOpWithNewOp<arith::SubFOp>(arithOp, otherBySplitk, gemmOut);
    } else {
      return failure();
    }
  }
  return success();
}

static LogicalResult rewriteFusionForSplitK(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());
  // TODO: extend this for gemm+gemm
  SmallVector<GemmOp> gemmOps;
  bool foundGemmWithoutParams = false;

  func.walk([&](GemmOp gemmOp) {
    auto params = gemmOp.getParams();
    if (!params) {
      foundGemmWithoutParams = true;
      return;
    }
    int64_t splitKFactor = params->getSplitKFactor();
    if (splitKFactor > 1) {
      gemmOps.push_back(gemmOp);
    }
  });

  if (foundGemmWithoutParams) {
    func->emitError("rewriteFusionForSplitK: found gemm op without params");
    return failure();
  }

  // This is relevant for backward convs (where we have multiple gemms in the
  // same kernel)
  // TODO: fix this when we allow fusions for backward convs
  if (gemmOps.size() > 1) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "More than once GEMM found, skipping rewriteFusionForSplitK\n");
    return success();
  }

  if (gemmOps.size() == 1) {
    GemmOp gemmOp = gemmOps[0];
    int64_t splitKFactor = gemmOp.getParams()->getSplitKFactor();

    if (failed(divideAddBySplitkFactor(gemmOp.getResult(), splitKFactor,
                                       rewriter)))
      return failure();
  }

  return success();
}

void RockFusionSplitkRegularizationPass::runOnOperation() {
  func::FuncOp func = getOperation();

  if (failed(rewriteFusionForSplitK(func))) {
    return signalPassFailure();
  }
} // namespace
