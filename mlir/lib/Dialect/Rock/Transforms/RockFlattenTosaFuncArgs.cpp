//===- RockFlattenTosaFuncArgs.cpp - Flatten N-D func boundaries to 1-D --===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices
//
//===----------------------------------------------------------------------===//
//
// Downstream passes require all function boundary tensors to be 1-D.  The
// MIGraphX-to-TOSA conversion already produces 1-D boundaries, but
// hand-written TOSA test kernels may have N-D tensor arguments and results.
//
// This pass rewrites each function so that every N-D tensor argument becomes a
// 1-D tensor with a tosa.reshape at the top of the entry block to recover the
// original shape, and every N-D return operand is flattened with tosa.reshape
// before the func.return.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFLATTENTOSAFUNCARGSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;

namespace {

struct RockFlattenTosaFuncArgs
    : public rock::impl::RockFlattenTosaFuncArgsPassBase<
          RockFlattenTosaFuncArgs> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    Block &entryBlock = func.front();
    bool changed = false;

    // --- Flatten arguments ---
    ImplicitLocOpBuilder builder(func.getLoc(), &entryBlock,
                                 entryBlock.begin());
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      BlockArgument arg = entryBlock.getArgument(i);
      auto tensorTy = dyn_cast<RankedTensorType>(arg.getType());
      if (!tensorTy || tensorTy.getRank() <= 1)
        continue;

      int64_t numElements = tensorTy.getNumElements();
      auto flatTy =
          RankedTensorType::get({numElements}, tensorTy.getElementType());
      arg.setType(flatTy);

      Value shapeValue = tosa::getTosaConstShape(builder, tensorTy.getShape());
      auto reshapeOp =
          tosa::ReshapeOp::create(builder, tensorTy, arg, shapeValue);
      arg.replaceAllUsesExcept(reshapeOp.getResult(), reshapeOp.getOperation());
      changed = true;
    }

    // --- Flatten return values ---
    func.walk([&](func::ReturnOp ret) {
      ImplicitLocOpBuilder retBuilder(ret.getLoc(), ret);
      for (unsigned i = 0; i < ret.getNumOperands(); ++i) {
        Value val = ret.getOperand(i);
        auto tensorTy = dyn_cast<RankedTensorType>(val.getType());
        if (!tensorTy || tensorTy.getRank() <= 1)
          continue;

        int64_t numElements = tensorTy.getNumElements();
        auto flatTy =
            RankedTensorType::get({numElements}, tensorTy.getElementType());
        Value shapeValue =
            tosa::getTosaConstShape(retBuilder, ArrayRef<int64_t>{numElements});
        auto flattened =
            tosa::ReshapeOp::create(retBuilder, flatTy, val, shapeValue);
        ret.setOperand(i, flattened);
        changed = true;
      }
    });

    // --- Update function signature ---
    if (changed) {
      SmallVector<Type> argTypes;
      for (BlockArgument arg : entryBlock.getArguments())
        argTypes.push_back(arg.getType());

      SmallVector<Type> resultTypes;
      func.walk([&](func::ReturnOp ret) {
        if (resultTypes.empty())
          resultTypes.append(ret.getOperandTypes().begin(),
                             ret.getOperandTypes().end());
      });

      func.setType(FunctionType::get(func.getContext(), argTypes, resultTypes));
    }
  }
};

} // namespace
