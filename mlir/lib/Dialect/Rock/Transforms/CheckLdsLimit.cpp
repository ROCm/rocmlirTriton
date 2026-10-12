//===- CheckLdsLimit.cpp - Reject kernels that overflow LDS ---------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rejects a kernel as soon as Triton's shared-memory allocation shows that it
// needs more LDS than the target has, instead of after the lowering to LLVM
// dialect, which for such kernels is often the most expensive step of the
// whole compile.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCHECKLDSLIMITPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {

struct RockCheckLdsLimitPass
    : public rock::impl::RockCheckLdsLimitPassBase<RockCheckLdsLimitPass> {
  using RockCheckLdsLimitPassBase::RockCheckLdsLimitPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (arch.empty()) {
      moduleOp.emitError("rock-check-lds-limit requires the arch option");
      return signalPassFailure();
    }

    auto sharedAttr = moduleOp->getAttrOfType<IntegerAttr>("ttg.shared");
    if (!sharedAttr) {
      moduleOp.emitError("ttg.shared attribute not found on module");
      return signalPassFailure();
    }

    if (failed(rock::checkSharedMemoryFitsLDS(moduleOp, sharedAttr.getInt(),
                                              arch)))
      signalPassFailure();
  }
};

} // namespace
