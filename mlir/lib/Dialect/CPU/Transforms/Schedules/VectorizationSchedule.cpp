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

#include "VectorizationSchedule.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

OwningOpRef<ModuleOp> cpu::buildVectorizationSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib, BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);

    auto matchMatmul = createMatchMatmulOp(ib, ctx, arg);

    // Step 1: Vectorize the matched op
    transform::VectorizeOp::create(ib,
        /*target=*/matchMatmul.getResult(),
        /*vectorSizes=*/ValueRange{},
        /*staticVectorSizes=*/ArrayRef<int64_t>{},
        /*vectorizeNDExtract=*/ib.getUnitAttr(),
        /*assumeDynamicDimsMatchVecSizes=*/UnitAttr{},
        /*createNamedContraction=*/UnitAttr{},
        /*scalableSizes=*/ArrayRef<bool>{});

    // Step 2: Match the func.func to apply patterns to
    auto matchFunc = transform::MatchOp::create(ib,
        /*resultTypes=*/anyOpType,
        /*target=*/arg,
        /*ops=*/ib.getStrArrayAttr({"func.func"}),
        /*interface=*/transform::MatchInterfaceEnumAttr{},
        /*opAttrs=*/DictionaryAttr{},
        /*filterResultType=*/TypeAttr{},
        /*filterOperandTypes=*/ArrayAttr{});

    // Step 3: Apply the same patterns as vectorize_children_and_apply_patterns
    transform::ApplyPatternsOp::create(ib,
        matchFunc.getResult(),
        [](OpBuilder &b, Location loc) {
          transform::ApplyTransferPermutationPatternsOp::create(b, loc);
          transform::ApplyVectorReductionToContractPatternsOp::create(b, loc);
          transform::ApplySinkVectorPatternsOp::create(b, loc);
          transform::ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp::create(b, loc);
          transform::ApplyFoldArithExtensionPatternsOp::create(b, loc);
          transform::ApplyCanonicalizationPatternsOp::create(b, loc);
          transform::ApplyPadVectorizationPatternsOp::create(b, loc);
          transform::ApplyDecomposeTensorPadPatternsOp::create(b, loc);
        });
  });
}
