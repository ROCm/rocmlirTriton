//===- ScheduleUtils.cpp - Common utilities for transform schedules -------===//
//
// Copyright 2026 Advanced Micro Devices.
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
// =============================================================================
//
// This file implements common utility functions used by transform schedule
// builders.
//
//===----------------------------------------------------------------------===//

#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

OwningOpRef<ModuleOp> cpu::createTransformModule(MLIRContext *ctx) {
  auto loc = UnknownLoc::get(ctx);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  (*module)->setAttr(transform::TransformDialect::kWithNamedSequenceAttrName,
                     UnitAttr::get(ctx));
  return module;
}

transform::AnyOpType cpu::getAnyOpType(MLIRContext *ctx) {
  return transform::AnyOpType::get(ctx);
}

OwningOpRef<ModuleOp> cpu::buildTransformModule(MLIRContext *ctx,
                                                TransformBodyBuilder bodyBuilder) {
  OwningOpRef<ModuleOp> module = createTransformModule(ctx);
  auto loc = UnknownLoc::get(ctx);
  ImplicitLocOpBuilder builder(loc, ctx);
  builder.setInsertionPointToStart(module->getBody());

  auto anyOpType = getAnyOpType(ctx);

  builder.create<transform::NamedSequenceOp>(
      "__transform_main",
      /*rootType=*/anyOpType,
      /*resultTypes=*/TypeRange{},
      /*bodyBuilder=*/
      [&](OpBuilder &b, Location loc, BlockArgument arg) {
        ImplicitLocOpBuilder ib(loc, b);
        bodyBuilder(ib, arg);
        ib.create<transform::YieldOp>();
      });

  return module;
}

llvm::ArrayRef<utils::IteratorType> cpu::getMatmulIteratorTypes() {
  // Canonical 3-D matmul iter-type signature: two parallel dims (M, N)
  // followed by one reduction dim (K). This is the shape we rely on
  // *after* `fold_unit_extent_dims_via_slices` collapses the size-1
  // batch dim present in both the rocmlir-gen GEMM (G=1) and the
  // fused-conv-derived matmul (N=1, G=1, Ho=1, Fh=1, Fw=1 -> all
  // folded). Keeping the signature 3-D unifies the conv-and-pure-gemm
  // code paths.
  //
  // NOTE: This describes the *iter-type* signature, not the iter-space
  // *order*: the position of M / N / K within the iter-space depends on
  // how the op was generated. Callers that need the per-dim role should
  // recover it from the operand maps (see
  // `LowerCpuVerifier::classifyMatmulDims`).
  static constexpr utils::IteratorType kIters[] = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  return llvm::ArrayRef<utils::IteratorType>(kIters);
}

DictionaryAttr cpu::getMatmulIteratorTypesAttr(MLIRContext *ctx) {
  SmallVector<Attribute> iteratorTypeAttrs;
  iteratorTypeAttrs.reserve(getMatmulIteratorTypes().size());
  for (utils::IteratorType iter : getMatmulIteratorTypes())
    iteratorTypeAttrs.push_back(linalg::IteratorTypeAttr::get(ctx, iter));
  return DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, "iterator_types"),
                           ArrayAttr::get(ctx, iteratorTypeAttrs))});
}

transform::MatchOp cpu::createMatchMatmulOp(ImplicitLocOpBuilder &ib,
                                            MLIRContext *ctx, Value target) {
  auto anyOpType = getAnyOpType(ctx);
  auto opAttrs = getMatmulIteratorTypesAttr(ctx);
  return ib.create<transform::MatchOp>(
      /*resultTypes=*/anyOpType,
      /*target=*/target,
      /*ops=*/ArrayAttr::get(ctx, {StringAttr::get(ctx, "linalg.generic")}),
      /*interface=*/transform::MatchInterfaceEnumAttr{},
      /*opAttrs=*/opAttrs,
      /*filterResultType=*/TypeAttr{},
      /*filterOperandTypes=*/ArrayAttr{});
}

transform::MatchOp cpu::createMatchCpuVerifierFuncOp(ImplicitLocOpBuilder &ib,
                                                     MLIRContext *ctx,
                                                     Value target) {
  auto anyOpType = getAnyOpType(ctx);
  auto cpuVerifierAttr = DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, "rock.cpu_verifier"),
                           UnitAttr::get(ctx))});
  return ib.create<transform::MatchOp>(
      /*resultTypes=*/anyOpType,
      /*target=*/target,
      /*ops=*/ArrayAttr::get(ctx, {StringAttr::get(ctx, "func.func")}),
      /*interface=*/transform::MatchInterfaceEnumAttr{},
      /*opAttrs=*/cpuVerifierAttr,
      /*filterResultType=*/TypeAttr{},
      /*filterOperandTypes=*/ArrayAttr{});
}
