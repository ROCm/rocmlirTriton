//===- FusionSplitkRegularizationPass.cpp ------------===//
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

// Append to `splitKOps`, for every op of type `OpTy` that runs with split-k,
// the result its output fusion consumes and the factor that fusion has to be
// divided by. The tuning parameters carrying the factor are read through
// `getSplitKParams`; an op that has none is an error reported as
// `missingParamsMsg`, since this pass runs after they have been affixed.
// Attention is skipped -- it implements the gemm+gemm interface but partitions
// its reduction with splitKV rather than with the perf-config splitKFactor;
// the check never fires for a plain GEMM.
template <typename OpTy, typename GetSplitKParamsFn>
static LogicalResult
collectSplitKOps(func::FuncOp &func, GetSplitKParamsFn getSplitKParams,
                 StringRef missingParamsMsg,
                 SmallVectorImpl<std::pair<Value, int64_t>> &splitKOps) {
  WalkResult res = func.walk([&](OpTy op) -> WalkResult {
    if (isa<AttentionOp>(op.getOperation()))
      return WalkResult::advance();
    auto params = getSplitKParams(op);
    if (!params.has_value()) {
      op->emitError(missingParamsMsg);
      return WalkResult::interrupt();
    }
    int64_t splitKFactor = params->getSplitKFactor();
    if (splitKFactor > 1)
      splitKOps.emplace_back(op->getResult(0), splitKFactor);
    return WalkResult::advance();
  });
  return success(!res.wasInterrupted());
}

static LogicalResult rewriteFusionForSplitK(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());
  SmallVector<std::pair<Value, int64_t>> splitKOps;
  if (failed(collectSplitKOps<GemmOp>(
          func, [](GemmOp op) { return op.getParams(); },
          "rewriteFusionForSplitK: found gemm op without params", splitKOps)))
    return failure();

  // Split-k on a gemm+gemm chain partitions the dimension shared by the two
  // GEMMs, so only gemm1's params carry the factor; gemm0 is always unsplit.
  if (failed(collectSplitKOps<RockGemmGemmWrapperInterface>(
          func,
          [](RockGemmGemmWrapperInterface op) { return op.getGemm1Params(); },
          "rewriteFusionForSplitK: found gemm+gemm op without gemm1 params",
          splitKOps)))
    return failure();

  // This is relevant for backward convs (where we have multiple gemms in the
  // same kernel)
  // TODO: fix this when we allow fusions for backward convs
  if (splitKOps.size() > 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "More than one split-k op (gemm or gemm+gemm) found, "
                  "skipping rewriteFusionForSplitK\n");
    return success();
  }

  if (splitKOps.empty())
    return success();

  auto [gemmResult, splitKFactor] = splitKOps.front();
  return divideAddBySplitkFactor(gemmResult, splitKFactor, rewriter);
}

void RockFusionSplitkRegularizationPass::runOnOperation() {
  func::FuncOp func = getOperation();

  if (failed(rewriteFusionForSplitK(func))) {
    return signalPassFailure();
  }
} // namespace
