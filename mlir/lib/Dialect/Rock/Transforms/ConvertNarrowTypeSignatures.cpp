//===- ConvertNarrowTypeSignatures.cpp - Func signature conversion --------===//
//
// Copyright Advanced Micro Devices, Inc.
// Copyright 2026 The MLIR Authors.
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
//===----------------------------------------------------------------------===//
//
// Converts function signatures containing sub-byte memref types
// (e.g. memref<Nxi4>, memref<Nxf4E2M1FN>) to packed wider-type storage
// (e.g. memref<ceil(N/2)xi8>).
//
// Runs before RockEmulateNarrowTypesPass so that block arguments are already
// the converted type when the upstream memref narrow-type emulation patterns
// rewrite loads and stores.  This avoids a crash in the upstream
// ConvertMemRefLoad/ConvertMemrefStore patterns that call
// extract_strided_metadata on the original (pre-conversion) block argument.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKCONVERTNARROWTYPESIGNATURESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {

static bool has4BitTypes(func::FuncOp funcOp) {
  auto is4Bit = [](Type t) {
    Type elem = getElementTypeOrSelf(t);
    return elem.isIntOrFloat() && elem.getIntOrFloatBitWidth() == 4;
  };
  for (Type inputTy : funcOp.getFunctionType().getInputs())
    if (is4Bit(inputTy))
      return true;
  for (Type resultTy : funcOp.getFunctionType().getResults())
    if (is4Bit(resultTy))
      return true;
  // Also check the body: a caller may have call ops with 4-bit operands
  // even when the caller's own signature has no 4-bit types.
  auto result = funcOp.walk([&](Operation *op) {
    for (Type t : op->getOperandTypes())
      if (is4Bit(t))
        return WalkResult::interrupt();
    for (Type t : op->getResultTypes())
      if (is4Bit(t))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

struct RockConvertNarrowTypeSignaturesPass
    : public rock::impl::RockConvertNarrowTypeSignaturesPassBase<
          RockConvertNarrowTypeSignaturesPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    if (!has4BitTypes(funcOp))
      return;

    auto typeConverter = rock::create4BitTypeConverter();

    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&typeConverter](func::FuncOp op) {
          return typeConverter.isLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&typeConverter](func::CallOp op) {
          return typeConverter.isLegal(op);
        });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&typeConverter](func::ReturnOp op) {
          return typeConverter.isLegal(op);
        });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // end anonymous namespace
