//===- GemmLinalgSplitkNormalizationPass.cpp ------------===//
//
// Copyright 2025 Advanced Micro Devices.
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
// This pass modifies linalg.generic for split-k fusions. It converts any
// arith.addf/arith.subf gemmOut, other to arith.addf gemmOut,
// other/splitkFactor.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGEMMLINALGSPLITKNORMALIZATIONPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gemm-linalg-splitk-normalization"

using namespace mlir;
using namespace mlir::rock;

namespace {
class RockGemmLinalgSplitkNormalizationPass
    : public rock::impl::RockGemmLinalgSplitkNormalizationPassBase<
          RockGemmLinalgSplitkNormalizationPass> {
  void runOnOperation() override;
};
} // end namespace

static LogicalResult divideAddBySplitkFactor(linalg::GenericOp genericOp,
                                             Value gemmResult,
                                             int64_t splitKFactor,
                                             IRRewriter &b) {
  SmallVector<std::tuple<Operation *, int>> adds;
  if (failed(checkValidOutputFusion(genericOp, gemmResult, adds)))
    return failure();

  for (auto [arithOp, gemmOutIndex] : adds) {
    assert(gemmOutIndex == 0 || gemmOutIndex == 1);
    LLVM_DEBUG(llvm::dbgs() << "Op to modify: " << arithOp << "\n");
    b.setInsertionPoint(arithOp);
    Value gemmOut = arithOp->getOperand(gemmOutIndex);
    Value otherValue =
        (gemmOutIndex == 0) ? arithOp->getOperand(1) : arithOp->getOperand(0);
    auto splitKFactorValue = createConstantFloatOp(
        b, arithOp->getLoc(), otherValue.getType(), otherValue.getType(),
        static_cast<float>(splitKFactor));
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

static LogicalResult rewriteLinalgForSplitK(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());
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
    func->emitError("rewriteLinalgForSplitK: found gemm op without params");
    return failure();
  }

  if (gemmOps.size() > 1)
    return failure();

  if (gemmOps.size() == 1) {
    GemmOp gemmOp = gemmOps[0];
    // GemmOp no longer has an output argument - use the result directly
    auto gemmResult = gemmOp.getResult();
    int64_t splitKFactor = gemmOp.getParams()->getSplitKFactor();

    // save all `linalg::GenericOp` that read from a gemm output
    auto genericOpOperands =
        traceGemmOutputToGenericOps(gemmResult, func);

    // GEMM result could come from a block argument, so if it fails, we return
    // success()
    if (failed(genericOpOperands))
      return success();

    // check if generic ops are valid fusions
    for (OpOperand *genericOpOperand : genericOpOperands.value()) {
      auto genericOp = cast<linalg::GenericOp>(genericOpOperand->getOwner());
      LLVM_DEBUG(llvm::dbgs()
                 << "Found linalg::GenericOp that reads GEMM output, let's "
                    "modify it if it has addf and/or subf. Op="
                 << genericOp << "\n");
      auto inputAlloc = findMemrefAlloc(genericOpOperand->get());
      if (failed(inputAlloc))
        return failure();

      if (failed(divideAddBySplitkFactor(genericOp, inputAlloc.value(),
                                         splitKFactor, rewriter)))
        return failure();
    }
  }

  return success();
}

void RockGemmLinalgSplitkNormalizationPass::runOnOperation() {
  func::FuncOp func = getOperation();

  if (failed(rewriteLinalgForSplitK(func))) {
    return signalPassFailure();
  }
} // namespace
