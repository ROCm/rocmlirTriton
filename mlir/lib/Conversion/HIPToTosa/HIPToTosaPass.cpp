//===- HIPToTosaPass.cpp - Lowering HIP to Tosa Dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass legalizes HIP operations to the Tosa dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/HIPToTosa/HIPToTosa.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "hip/Dialect/IR/HipDialect.h"

#define DEBUG_TYPE "hip-to-tosa"

namespace mlir {
#define GEN_PASS_DEF_HIPTOTOSAPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct HIPToTosa : public impl::HIPToTosaPassBase<HIPToTosa> {
public:
  using HIPToTosaPassBase::HIPToTosaPassBase;
  void runOnOperation() override;
};
} // end namespace

void HIPToTosa::runOnOperation() {
  func::FuncOp func = getOperation();

  RewritePatternSet patterns(&getContext());
  hip::populateHIPToTosaConversionPatterns(patterns);
  if (failed(applyPatternsGreedily(func, std::move(patterns))))
    return signalPassFailure();
}

void mlir::hip::addHIPToTosaPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createHIPToTosaPass());
}
