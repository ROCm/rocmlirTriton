//===- PrePostSchedules.cpp - Pre/Post transform schedules ----------------===//
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
// This file implements the pre and post transform schedules using the MLIR
// C++ API. These schedules are applied before and after each main transform
// phase (tiling, vectorization, lowering).
//
//===----------------------------------------------------------------------===//

#include "PrePostSchedules.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::cpu;

//===----------------------------------------------------------------------===//
// Pre-Schedule Builder
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> cpu::buildPreSchedule(MLIRContext *ctx) {
  // Create module with transform.with_named_sequence attribute
  OwningOpRef<ModuleOp> module = createTransformModule(ctx);
  auto loc = UnknownLoc::get(ctx);
  ImplicitLocOpBuilder builder(loc, ctx);
  builder.setInsertionPointToStart(module->getBody());

  auto anyOpType = getAnyOpType(ctx);

  // Build: transform.named_sequence @__transform_main(%arg0: !transform.any_op)
  builder.create<transform::NamedSequenceOp>(
      "__transform_main",
      /*rootType=*/anyOpType,
      /*resultTypes=*/TypeRange{},
      /*bodyBuilder=*/
      [&](OpBuilder &b, Location loc, BlockArgument arg) {
        ImplicitLocOpBuilder ib(loc, b);

        // %func = transform.structured.match ops{["func.func"]} in %arg0
        auto matchFunc = ib.create<transform::MatchOp>(
            anyOpType, arg, ArrayRef<StringRef>{"func.func"});

        // %1 = transform.apply_registered_pass "canonicalize" to %func
        auto canonicalize = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, matchFunc.getResults(),
            /*passName=*/"canonicalize",
            /*options=*/DictionaryAttr::get(ctx),
            /*dynamicOptions=*/ValueRange{});

        // transform.apply_cse to %1
        ib.create<transform::ApplyCommonSubexpressionEliminationOp>(
            canonicalize.getResult());

        // transform.yield
        ib.create<transform::YieldOp>();
      });

  return module;
}

//===----------------------------------------------------------------------===//
// Post-Schedule Builder
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> cpu::buildPostSchedule(MLIRContext *ctx) {
  // Create module with transform.with_named_sequence attribute
  OwningOpRef<ModuleOp> module = createTransformModule(ctx);
  auto loc = UnknownLoc::get(ctx);
  ImplicitLocOpBuilder builder(loc, ctx);
  builder.setInsertionPointToStart(module->getBody());

  auto anyOpType = getAnyOpType(ctx);

  // Build: transform.named_sequence @__transform_main(%arg0: !transform.any_op)
  builder.create<transform::NamedSequenceOp>(
      "__transform_main",
      /*rootType=*/anyOpType,
      /*resultTypes=*/TypeRange{},
      /*bodyBuilder=*/
      [&](OpBuilder &b, Location loc, BlockArgument arg) {
        ImplicitLocOpBuilder ib(loc, b);

        // %func = transform.structured.match ops{["func.func"]} in %arg0
        auto matchFunc = ib.create<transform::MatchOp>(
            anyOpType, arg, ArrayRef<StringRef>{"func.func"});

        // %loops = transform.structured.match interface{LoopLikeInterface} in %func
        auto matchLoops = ib.create<transform::MatchOp>(
            /*resultTypes=*/anyOpType,
            /*target=*/matchFunc.getResults(),
            /*ops=*/ArrayAttr{},
            /*interface=*/transform::MatchInterfaceEnumAttr::get(
                ctx, transform::MatchInterfaceEnum::LoopLikeInterface),
            /*opAttrs=*/DictionaryAttr{},
            /*filterResultType=*/TypeAttr{},
            /*filterOperandTypes=*/ArrayAttr{});

        // transform.apply_licm to %loops
        ib.create<transform::ApplyLoopInvariantCodeMotionOp>(
            matchLoops.getResults());

        // %func2 = transform.structured.match ops{["func.func"]} in %func
        auto matchFunc2 = ib.create<transform::MatchOp>(
            anyOpType, matchFunc.getResults(), ArrayRef<StringRef>{"func.func"});

        // %1 = transform.structured.hoist_redundant_vector_transfers %func2
        auto hoistTransfers =
            ib.create<transform::HoistRedundantVectorTransfersOp>(
                anyOpType, matchFunc2.getResults(),
                /*verifyNonZeroTrip=*/false);

        // %2 = transform.structured.hoist_redundant_vector_broadcasts %1
        auto hoistBroadcasts =
            ib.create<transform::HoistRedundantVectorBroadcastsOp>(
                anyOpType, hoistTransfers.getTransformed());

        // %3 = transform.apply_registered_pass "canonicalize" to %2
        ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, hoistBroadcasts.getTransformed(),
            /*passName=*/"canonicalize",
            /*options=*/DictionaryAttr::get(ctx),
            /*dynamicOptions=*/ValueRange{});

        // transform.yield
        ib.create<transform::YieldOp>();
      });

  return module;
}
