//===- SerializeHostFuncs.cpp - Serialize and erase host functions -------===//
//
// Copyright 2026 The MLIR Authors.
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
// Serializes non-kernel (host) functions into a module attribute and erases
// them. Must run before func-level passes that change kernel signatures
// (e.g. RockToTTIRPass sets kernel return to void).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSERIALIZEHOSTFUNCSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockSerializeHostFuncsPass
    : public rock::impl::RockSerializeHostFuncsPassBase<
          RockSerializeHostFuncsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    SmallVector<func::FuncOp> nonKernelFuncs;
    moduleOp.walk([&](func::FuncOp funcOp) {
      if (funcOp->getParentOfType<ModuleOp>() != moduleOp)
        return;
      if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
        nonKernelFuncs.push_back(funcOp);
    });

    if (nonKernelFuncs.empty())
      return;

    OpBuilder builder(ctx);
    SmallVector<Attribute> funcStrings;
    for (func::FuncOp funcOp : nonKernelFuncs) {
      std::string funcStr;
      llvm::raw_string_ostream os(funcStr);
      funcOp.print(os, OpPrintingFlags().useLocalScope());
      funcStrings.push_back(StringAttr::get(ctx, funcStr));
    }
    moduleOp->setAttr("rock.host_functions", ArrayAttr::get(ctx, funcStrings));

    // Erase order doesn't matter: func.call holds a FlatSymbolRefAttr (a
    // string), not an SSA use-def edge, so erasing a callee before its
    // caller won't fail.
    for (func::FuncOp funcOp : nonKernelFuncs)
      funcOp.erase();
  }
};

} // namespace
