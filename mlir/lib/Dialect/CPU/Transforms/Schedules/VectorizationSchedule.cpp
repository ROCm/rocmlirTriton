//===- VectorizationSchedule.cpp - Vectorization transform schedule -------===//
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
// This file implements the vectorization transform schedule using the MLIR
// C++ API. This schedule vectorizes linalg ops marked with rock.matmul.
//
//===----------------------------------------------------------------------===//

#include "VectorizationSchedule.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::cpu;

//===----------------------------------------------------------------------===//
// Vectorization Schedule Builder
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> cpu::buildVectorizationSchedule(MLIRContext *ctx) {
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

        // %0 = transform.structured.match attributes {rock.matmul} in %arg0
        // Create a DictionaryAttr with rock.matmul = unit
        auto matmulAttr = DictionaryAttr::get(
            ctx, {NamedAttribute(StringAttr::get(ctx, "rock.matmul"),
                                 UnitAttr::get(ctx))});
        auto matchMatmul = ib.create<transform::MatchOp>(
            /*resultTypes=*/anyOpType,
            /*target=*/arg,
            /*ops=*/ArrayAttr{},
            /*interface=*/transform::MatchInterfaceEnumAttr{},
            /*opAttrs=*/matmulAttr,
            /*filterResultType=*/TypeAttr{},
            /*filterOperandTypes=*/ArrayAttr{});

        // %1 = transform.get_parent_op %0 {isolated_from_above}
        auto getParent = ib.create<transform::GetParentOp>(
            /*resultType=*/anyOpType,
            /*target=*/matchMatmul.getResults(),
            /*isolated_from_above=*/true,
            /*allow_empty_results=*/false,
            /*op_name=*/StringAttr{},
            /*deduplicate=*/false,
            /*nth_parent=*/1);

        // %2 = transform.structured.vectorize_children_and_apply_patterns %1
        //      {vectorize_nd_extract, vectorize_padding}
        ib.create<transform::VectorizeChildrenAndApplyPatternsOp>(
            /*target=*/getParent.getParent(),
            /*foldTypeExtensionsIntoContract=*/false,
            /*vectorizePadding=*/true,
            /*vectorizeNDExtract=*/true,
            /*flatten1DDepthwise=*/false);

        // transform.yield
        ib.create<transform::YieldOp>();
      });

  return module;
}
