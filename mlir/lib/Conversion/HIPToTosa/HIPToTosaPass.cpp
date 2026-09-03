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
#include "mlir/Transforms/Passes.h"

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

  // TODO: see the matching TODO in HIPToTosa.cpp -- this does not belong here.
  hip::annotateAsRockKernel(func, arch);
  hip::eraseUnusedContextArgs(func);
  hip::eraseOnnxAttrs(func);
}

void mlir::hip::addHIPToTosaPasses(OpPassManager &pm, StringRef arch) {
  auto &funcPm = pm.nest<func::FuncOp>();
  HIPToTosaPassOptions opts;
  opts.arch = arch.str();
  funcPm.addPass(createHIPToTosaPass(opts));
  funcPm.addPass(createCanonicalizerPass());
  funcPm.addPass(createCSEPass());
}
