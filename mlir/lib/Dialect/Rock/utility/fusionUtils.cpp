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
    Value gemmResult,
    SmallVector<std::tuple<Operation *, int>> &adds) {
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

        if (failed(traceRootOutputToArgs(gemmResult, func)))
          return WalkResult::interrupt();

        SmallVector<std::tuple<Operation *, int>> adds;
        if (failed(checkValidOutputFusion(gemmResult, adds)))
          return WalkResult::interrupt();

        return WalkResult::advance();
      });

  WalkResult gemmGemmWalkResult = func.walk(
      [&](rock::RockGemmGemmWrapperInterface gemmGemmOp) -> WalkResult {
        // We have two results for attention (output + LSE)
        // But we only support split-k for gemm+gemm, so there's a single result
        // here
        auto gemmGemmResult = gemmGemmOp->getResult(0);

        if (failed(traceRootOutputToArgs(gemmGemmResult, func)))
          return WalkResult::interrupt();

        // no fusions allowed for now
        auto fusionInfo = rock::collectFusionInfo(gemmGemmResult);
        if (!fusionInfo.fusionOps.empty())
          return WalkResult::interrupt();

        // fusions between gemm0 and gemm1 are not allowed
        bool fusionsFound = gemmGemmHasPreSecondGemmFusion(gemmGemmOp);
        if (fusionsFound)
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
  // Only reduce-max needs an arch gate. Atomic add is always safe to emit:
  // where the hardware lacks a native instruction the backend expands it to a
  // compare-and-swap loop. Float atomic max gets no such expansion, because it
  // lowers to a buffer-atomic intrinsic that AtomicExpandPass never sees, so
  // fusing it on an unsupported arch is a hard codegen failure rather than a
  // slow path.
  WalkResult walkResult = func.walk([&](rock::ReduceOp reduceOp) -> WalkResult {
    if (reduceOp.getReduceMethod() != ReduceMethod::Max)
      return WalkResult::advance();
    auto outElemType = reduceOp.getResult().getType().getElementType();
    if (!isFastAtomicMaxSupported(rock::getArchValue(reduceOp), outElemType))
      return WalkResult::interrupt();
    return WalkResult::advance();
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
