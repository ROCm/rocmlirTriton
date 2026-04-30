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

namespace {

/// Populate the body of a `transform.apply_patterns` op with the same set of
/// ops that `transform.structured.vectorize_children_and_
/// apply_patterns` would apply. This mirrors the upstream populate
/// calls in `VectorizeChildrenAndApplyPatternsOp::applyToOne`
/// (mlir/lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp).
static void populateVectorizationCleanup(ImplicitLocOpBuilder &ib,
                                         bool vectorizePadding,
                                         bool foldTypeExtensionsIntoContract) {
  ib.create<transform::ApplyTransferPermutationPatternsOp>();
  ib.create<transform::ApplyVectorReductionToContractPatternsOp>();
  ib.create<transform::ApplySinkVectorPatternsOp>();
  ib.create<transform::ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp>();
  ib.create<transform::ApplyCanonicalizationPatternsOp>();
  if (foldTypeExtensionsIntoContract)
    ib.create<transform::ApplyFoldArithExtensionPatternsOp>();
  if (vectorizePadding) {
    ib.create<transform::ApplyPadVectorizationPatternsOp>();
    ib.create<transform::ApplyDecomposeTensorPadPatternsOp>();
  }
}

} // namespace

OwningOpRef<ModuleOp> cpu::buildVectorizationSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib,
                                         BlockArgument arg) {
    // Match parameters that VectorizeChildrenAndApplyPatternsOp used to be
    // invoked with.
    constexpr bool kVectorizePadding = true;
    constexpr bool kVectorizeNDExtract = true;
    constexpr bool kFoldTypeExtensionsIntoContract = false;

    auto matchFunc = createMatchCpuVerifierFuncOp(ib, ctx, arg);

    // Vectorize the matmul only
    auto matchMatmul = createMatchMatmulOp(ib, ctx, arg);
    ib.create<transform::VectorizeOp>(
        /*target=*/matchMatmul.getResults(),
        /*vector_sizes=*/ValueRange{},
        /*static_vector_sizes=*/DenseI64ArrayAttr{},
        /*vectorize_nd_extract=*/
        kVectorizeNDExtract ? UnitAttr::get(ctx) : UnitAttr{},
        /*assume_dynamic_dims_match_vec_sizes=*/UnitAttr{},
        /*create_named_contraction=*/UnitAttr{},
        /*scalable_sizes=*/DenseBoolArrayAttr{});

    // Apply the same cleanup pattern set that
    // VectorizeChildrenAndApplyPatternsOp would apply.
    ib.create<transform::ApplyPatternsOp>(
        /*target=*/matchFunc.getResults(),
        /*bodyBuilder=*/
        [&](OpBuilder &b, Location loc) {
          ImplicitLocOpBuilder nested(loc, b);
          populateVectorizationCleanup(nested, kVectorizePadding,
                                       kFoldTypeExtensionsIntoContract);
        });
  });
}
