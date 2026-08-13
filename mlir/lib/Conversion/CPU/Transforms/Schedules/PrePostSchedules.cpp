//===- PrePostSchedules.cpp - Pre/Post transform schedules ----------------===//
//
// Copyright 2026 Advanced Micro Devices.
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
// =============================================================================

#include "PrePostSchedules.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

OwningOpRef<ModuleOp> cpu::buildPreSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib, BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);

    auto matchFunc = createMatchCpuVerifierFuncOp(ib, ctx, arg);

    auto canonicalize = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, matchFunc.getResults(),
        /*passName=*/"canonicalize",
        /*options=*/DictionaryAttr::get(ctx),
        /*dynamicOptions=*/ValueRange{});

    ib.create<transform::ApplyCommonSubexpressionEliminationOp>(
        canonicalize.getResult());
  });
}

OwningOpRef<ModuleOp> cpu::buildPostSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib, BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);

    auto matchFunc = createMatchCpuVerifierFuncOp(ib, ctx, arg);

    auto matchLoops = ib.create<transform::MatchOp>(
        /*resultTypes=*/anyOpType,
        /*target=*/matchFunc.getResults(),
        /*ops=*/ArrayAttr{},
        /*interface=*/transform::MatchInterfaceEnumAttr::get(
            ctx, transform::MatchInterfaceEnum::LoopLikeInterface),
        /*opAttrs=*/DictionaryAttr{},
        /*filterResultType=*/TypeAttr{},
        /*filterOperandTypes=*/ArrayAttr{});

    ib.create<transform::ApplyLoopInvariantCodeMotionOp>(
        matchLoops.getResults());

    auto matchFunc2 = createMatchCpuVerifierFuncOp(ib, ctx, matchFunc.getResults());

    auto hoistTransfers =
        ib.create<transform::HoistRedundantVectorTransfersOp>(
            anyOpType, matchFunc2.getResults(),
            /*verifyNonZeroTrip=*/false);

    auto hoistBroadcasts =
        ib.create<transform::HoistRedundantVectorBroadcastsOp>(
            anyOpType, hoistTransfers.getTransformed());

    ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, hoistBroadcasts.getTransformed(),
        /*passName=*/"canonicalize",
        /*options=*/DictionaryAttr::get(ctx),
        /*dynamicOptions=*/ValueRange{});
  });
}
