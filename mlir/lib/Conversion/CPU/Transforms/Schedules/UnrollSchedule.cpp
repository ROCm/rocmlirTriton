//===- UnrollSchedule.cpp - Unroll transform schedule ---------------------===//
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

#include "UnrollSchedule.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

OwningOpRef<ModuleOp> cpu::buildUnrollSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib, BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);

    auto matchContract = ib.create<transform::MatchOp>(
        anyOpType, arg, ArrayRef<StringRef>{"vector.contract"});

    auto getParent = ib.create<transform::GetParentOp>(
        /*resultType=*/anyOpType,
        /*target=*/matchContract.getResults(),
        /*isolated_from_above=*/false,
        /*allow_empty_results=*/true,
        /*op_name=*/StringAttr::get(ctx, "scf.for"),
        /*deduplicate=*/false,
        /*nth_parent=*/1);

    auto foreachOp = ib.create<transform::ForeachOp>(
        /*resultTypes=*/TypeRange{},
        /*targets=*/ValueRange{getParent.getParent()},
        /*with_zip_shortest=*/false);
    {
      OpBuilder::InsertionGuard guard(ib);
      Region &bodyRegion = foreachOp.getBody();
      Block *body = ib.createBlock(&bodyRegion, bodyRegion.end(), {anyOpType},
                                   {ib.getLoc()});
      ib.create<transform::LoopUnrollOp>(
          /*target=*/body->getArgument(0),
          /*factor=*/8);
      ib.create<transform::YieldOp>(/*operands=*/ValueRange{});
    }
  });
}
