//===- fusionUtils.cpp - Rock utility for fusion -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::rock;
using namespace arith;

static bool validOperationGemmOut(Operation *op) {
  return isa<MulFOp, DivFOp, AddFOp, SubFOp, SIToFPOp, UIToFPOp, NegFOp,
             ExtUIOp, ExtSIOp, ExtFOp, TruncFOp, TruncIOp>(op);
}

LogicalResult mlir::rock::checkValidOutputFusion(
    Value gemmResult, SmallVector<std::tuple<Operation *, int>> &adds) {
  /* We can only fuse:
  - add/sub gemmResult, otherTensor (which will be converted to add gemmResult,
  otherTensor/splitKFactor)
  - add/sub gemmResult, gemmResult
  - mul/div gemmResult, otherTensor
  - neg
  - type conversion functions
  Where gemmResult != otherTensor for all cases.
  */
  auto fusionInfo = rock::collectFusionInfo(gemmResult);
  for (Operation *fusionOp : fusionInfo.fusionOps) {
    // check if any operand is derived from the GEMM result
    int numGemmResults = 0;
    for (Value operand : fusionOp->getOperands()) {
      if (fusionInfo.chainValues.contains(operand))
        numGemmResults++;
    }
    if (numGemmResults > 0) {
      // check it's a valid operation
      if (!validOperationGemmOut(fusionOp)) {
        return failure();
      }

      if (isa<MulFOp, DivFOp>(fusionOp) && numGemmResults > 1) {
        // gemmOut^2 is not allowed
        return failure();
      }

      // save add and sub ops to modify them: divide by splitKFactor
      // if both operands come from gemmOut, no need to modify anything
      if (isa<AddFOp, SubFOp>(fusionOp) && numGemmResults == 1) {
        int index =
            fusionInfo.chainValues.contains(fusionOp->getOperand(0)) ? 0 : 1;
        adds.push_back(std::make_tuple(fusionOp, index));
      }
    }
  }
  return success();
}

bool mlir::rock::gemmGemmHasPreSecondGemmFusion(
    RockGemmGemmWrapperInterface gemmGemmOp) {
  Region &region = gemmGemmOp.getPreSecondGemmRegion();
  if (region.empty())
    return false;
  return !region.front().without_terminator().empty();
}

static LogicalResult checkValidSplitKOutputTypes(Value output,
                                                 func::FuncOp func) {
  FailureOr<SmallVector<BlockArgument>> outputArgs =
      traceRootOutputToArgs(output, func);
  if (failed(outputArgs))
    return failure();

  for (BlockArgument outputArg : *outputArgs) {
    Type elementType = cast<ShapedType>(outputArg.getType()).getElementType();
    if (!isAtomicRMWTypeSupported(elementType))
      return failure();
  }
  return success();
}

LogicalResult mlir::rock::testFusionLegalitySplitK(func::FuncOp func) {
  // can't fuse reduce_max with split-k
  WalkResult reduceMaxRes = func.walk([](ReduceOp reduceOp) -> WalkResult {
    if (reduceOp.getReduceMethod() == ReduceMethod::Max)
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  WalkResult gemmWalkResult =
      func.walk([&](rock::RockGemmWrapperInterface gemmOp) -> WalkResult {
        // Use the result directly if there's no output argument (e.g., GemmOp)
        Value gemmResult = gemmOp->getResult(0);

        if (failed(checkValidSplitKOutputTypes(gemmResult, func)))
          return WalkResult::interrupt();

        SmallVector<std::tuple<Operation *, int>> adds;
        if (failed(checkValidOutputFusion(gemmResult, adds)))
          return WalkResult::interrupt();

        return WalkResult::advance();
      });

  WalkResult gemmGemmWalkResult = func.walk(
      [&](rock::RockGemmGemmWrapperInterface gemmGemmOp) -> WalkResult {
        // Attention uses splitKV rather than the perf-config splitKFactor.
        if (isa<AttentionOp>(gemmGemmOp))
          return WalkResult::interrupt();

        // Only gemm+gemm reaches here, so there is a single result.
        auto gemmGemmResult = gemmGemmOp->getResult(0);

        if (failed(checkValidSplitKOutputTypes(gemmGemmResult, func)))
          return WalkResult::interrupt();

        // The output fusion has to survive being applied once per split and
        // then summed by the atomic_add, same requirement as a plain GEMM.
        SmallVector<std::tuple<Operation *, int>> adds;
        if (failed(checkValidOutputFusion(gemmGemmResult, adds)))
          return WalkResult::interrupt();

        return WalkResult::advance();
      });

  return success(!gemmWalkResult.wasInterrupted() &&
                 !gemmGemmWalkResult.wasInterrupted() &&
                 !reduceMaxRes.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalitySplitK(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  assert(std::distance(funcs.begin(), funcs.end()) &&
         "expected ModuleOp containing a single func::FuncOp");
  func::FuncOp func = *(funcs.begin());
  return testFusionLegalitySplitK(func);
}

LogicalResult mlir::rock::testFusionLegalityReduce(func::FuncOp func) {
  WalkResult walkResult = func.walk([](ReduceOp reduceOp) -> WalkResult {
    Type elementType = reduceOp.getResult().getType().getElementType();
    return isAtomicRMWTypeSupported(elementType) ? WalkResult::advance()
                                                 : WalkResult::interrupt();
  });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalityReduce(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  assert(std::distance(funcs.begin(), funcs.end()) &&
         "expected ModuleOp containing a single func::FuncOp");
  func::FuncOp func = *(funcs.begin());
  return testFusionLegalityReduce(func);
}

// Matches a pure element-wise widening of the attention result: a lone
// `arith.extf` fed directly by the result. Widening is lossless, so it
// commutes with the split-kv LSE combine; anything chained onto it does not.
static bool isPureElementwiseExtF(const FusionInfo &fusionInfo) {
  if (fusionInfo.fusionOps.size() != 1 || !fusionInfo.reduceOps.empty())
    return false;
  auto extFOp = dyn_cast<ExtFOp>(fusionInfo.fusionOps.front());
  return extFOp && fusionInfo.chainValues.contains(extFOp.getIn());
}

// Whether the fusion chain hanging off `partial` survives the split-kv
// combine. Only a missing epilogue or a lone widening does.
static bool splitKVOutputFusionIsLegal(Value partial) {
  FusionInfo fusionInfo = rock::collectFusionInfo(partial);
  bool hasOutputFusion =
      !fusionInfo.fusionOps.empty() || !fusionInfo.reduceOps.empty();
  return !hasOutputFusion || isPureElementwiseExtF(fusionInfo);
}

LogicalResult
mlir::rock::testFusionLegalityAttentionSplitKV(func::FuncOp func) {
  // Input fusions and fusions between the two gemms stay legal under
  // splitKV > 1; only output fusions are rejected, because each split produces
  // a partial result that an LSE-based combine has yet to rescale.
  WalkResult walkResult = func.walk([](rock::AttentionOp attnOp) -> WalkResult {
    if (attnOp.getSplitKV() <= 1)
      return WalkResult::advance();

    if (!splitKVOutputFusionIsLegal(attnOp.getResult()))
      return WalkResult::interrupt();

    // The LSE is a per-split partial log-sum-exp that the same combine has to
    // reconcile, so an epilogue on it is no more legal than one on the result.
    // The verifier guarantees it is present once splitKV > 1.
    if (Value lse = attnOp.getLse())
      if (!splitKVOutputFusionIsLegal(lse))
        return WalkResult::interrupt();

    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalityAttentionSplitKV(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  bool isFusible = true;
  for (auto f : funcs) {
    isFusible &= succeeded(testFusionLegalityAttentionSplitKV(f));
  }

  return success(isFusible);
}

LogicalResult mlir::rock::testFusionLegalityBwdDataConv(func::FuncOp func) {
  // For right now, no BwdDataConv ops are fusible
  WalkResult walkResult = func.walk([&](Operation *op) -> WalkResult {
    if (auto bwdData = dyn_cast<rock::ConvBwdDataOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return success(!walkResult.wasInterrupted());
}

LogicalResult mlir::rock::testFusionLegalityBwdDataConv(ModuleOp mod) {
  auto funcs = mod.getOps<func::FuncOp>();
  bool isFusible = true;
  for (auto f : funcs) {
    isFusible &= succeeded(testFusionLegalityBwdDataConv(f));
  }

  return success(isFusible);
}
