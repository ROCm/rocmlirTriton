//===- LowerToLLVMSchedule.cpp - Lower to LLVM transform schedule ---------===//
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

#include "LowerToLLVMSchedule.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

OwningOpRef<ModuleOp> cpu::buildLowerToLLVMSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib, BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);
    auto emptyOptions = DictionaryAttr::get(ctx);
    auto emptyDynamic = ValueRange{};

    // Step 1: One-shot bufferization
    auto bufferize = transform::OneShotBufferizeOp::create(
        ib, anyOpType, arg,
        /*function_boundary_type_conversion=*/
        bufferization::LayoutMapOptionAttr::get(
            ctx, bufferization::LayoutMapOption::IdentityLayoutMap),
        /*allow_return_allocs_from_loops=*/false,
        /*allow_unknown_ops=*/false,
        /*bufferize_function_boundaries=*/true,
        /*dump_alias_sets=*/false,
        /*test_analysis_only=*/false,
        /*print_conflicts=*/false,
        /*check_parallel_regions=*/true,
        /*memcpy_op=*/"memref.copy");

    // Step 2: Initial cleanup passes
    auto matchFunc = createMatchCpuVerifierFuncOp(ib, ctx, bufferize.getTransformed());

    // Get the target module (parent of the func) for later module-level operations
    auto targetModule = ib.create<transform::GetParentOp>(
        /*resultType=*/anyOpType,
        /*target=*/matchFunc.getResults(),
        /*isolated_from_above=*/false,
        /*allow_empty_results=*/false,
        /*op_name=*/StringAttr::get(ctx, "builtin.module"),
        /*deduplicate=*/false,
        /*nth_parent=*/1);

    auto cse = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, matchFunc.getResults(),
        /*passName=*/"cse", emptyOptions, emptyDynamic);

    auto canonicalize = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, cse.getResult(),
        /*passName=*/"canonicalize", emptyOptions, emptyDynamic);

    // Step 3: Buffer hoisting passes
    auto bufferHoisting = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, canonicalize.getResult(),
        /*passName=*/"buffer-hoisting", emptyOptions, emptyDynamic);

    auto bufferLoopHoisting = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, bufferHoisting.getResult(),
        /*passName=*/"buffer-loop-hoisting", emptyOptions, emptyDynamic);

    // Step 4: Vector lowering patterns
    auto matchFuncForVector =
        createMatchCpuVerifierFuncOp(ib, ctx, bufferLoopHoisting.getResult());

    transform::ApplyPatternsOp::create(
        ib, matchFuncForVector.getResults(),
        /*bodyBuilder=*/
        [&](OpBuilder &pib, Location ploc) {
          ImplicitLocOpBuilder pbuilder(ploc, pib);

          transform::ApplyLowerContractionPatternsOp::create(
              pbuilder, vector::VectorContractLowering::OuterProduct);

          pbuilder.create<transform::ApplyLowerOuterProductPatternsOp>();

          pbuilder.create<transform::ApplyTransferPermutationPatternsOp>();

          transform::ApplySplitTransferFullPartialPatternsOp::create(
              pbuilder, vector::VectorTransferSplitAttr::get(
                            ctx, vector::VectorTransferSplit::LinalgCopy));

          transform::ApplyTransferToScfPatternsOp::create(
              pbuilder,
              /*max_transfer_rank=*/pbuilder.getI64IntegerAttr(4),
              /*full_unroll=*/pbuilder.getBoolAttr(true));

          transform::ApplyLowerTransferPatternsOp::create(
              pbuilder, /*max_transfer_rank=*/1);

          pbuilder.create<transform::ApplyLowerShapeCastPatternsOp>();

          transform::ApplyLowerTransposePatternsOp::create(
              pbuilder, vector::VectorTransposeLoweringAttr::get(
                            ctx, vector::VectorTransposeLowering::Shuffle1D));
        });

    // Step 5: Further lowering passes
    auto convertVectorToScf = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, matchFuncForVector.getResults(),
        /*passName=*/"convert-vector-to-scf", emptyOptions, emptyDynamic);

    auto convertLinalgToLoops = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, convertVectorToScf.getResult(),
        /*passName=*/"convert-linalg-to-loops", emptyOptions, emptyDynamic);

    auto bufferLoopHoisting2 = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, convertLinalgToLoops.getResult(),
        /*passName=*/"buffer-loop-hoisting", emptyOptions, emptyDynamic);

    auto promoteBuffersToStack = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, bufferLoopHoisting2.getResult(),
        /*passName=*/"promote-buffers-to-stack", emptyOptions, emptyDynamic);

    auto convertScfToCf = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, promoteBuffersToStack.getResult(),
        /*passName=*/"convert-scf-to-cf", emptyOptions, emptyDynamic);

    auto expandStridedMetadata = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, convertScfToCf.getResult(),
        /*passName=*/"expand-strided-metadata", emptyOptions, emptyDynamic);

    auto lowerAffine = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, expandStridedMetadata.getResult(),
        /*passName=*/"lower-affine", emptyOptions, emptyDynamic);

    auto convertVectorToLLVMOptions = DictionaryAttr::get(
        ctx, {NamedAttribute(StringAttr::get(ctx, "reassociate-fp-reductions"),
                             BoolAttr::get(ctx, false))});
    ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, lowerAffine.getResult(),
        /*passName=*/"convert-vector-to-llvm", convertVectorToLLVMOptions,
        emptyDynamic);

    // Step 6: Convert memref.global at module level
    // This must happen before conversion patterns so llvm.mlir.addressof can reference
    // the converted llvm.mlir.global
    auto updatedModule = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, targetModule.getResult(),
        /*passName=*/"finalize-memref-to-llvm", emptyOptions, emptyDynamic);

    // Step 7: Apply conversion patterns to LLVM
    // Re-match func from updated module (previous handles invalidated by module transform)
    auto funcForConversion =
        createMatchCpuVerifierFuncOp(ib, ctx, updatedModule.getResult());

    auto applyConversion = transform::ApplyConversionPatternsOp::create(
        ib, funcForConversion.getResults(),
        /*patternsBodyBuilder=*/
        [&](OpBuilder &pib, Location ploc) {
          ImplicitLocOpBuilder pbuilder(ploc, pib);

          transform::ApplyToLLVMConversionPatternsOp::create(pbuilder, "math");

          pbuilder.create<transform::ApplyVectorToLLVMConversionPatternsOp>(
              /*reassociate_fp_reductions=*/false,
              /*force_32bit_vector_indices=*/true,
              /*use_vector_alignment=*/false);

          // memref conversion handled by finalize-memref-to-llvm above
          transform::ApplyToLLVMConversionPatternsOp::create(pbuilder, "index");
          transform::ApplyToLLVMConversionPatternsOp::create(pbuilder, "arith");
          transform::ApplyToLLVMConversionPatternsOp::create(pbuilder, "cf");
        },
        /*typeConverterBodyBuilder=*/
        [&](OpBuilder &tcb, Location tcloc) {
          ImplicitLocOpBuilder tcbuilder(tcloc, tcb);

          transform::MemrefToLLVMTypeConverterOp::create(
              tcbuilder,
              /*use_aligned_alloc=*/false,
              /*index_bitwidth=*/64,
              /*use_generic_functions=*/false,
              /*use_bare_ptr_call_conv=*/false,
              /*data_layout=*/StringAttr{});
        });

    applyConversion.setLegalDialectsAttr(
        ArrayAttr::get(ctx, {StringAttr::get(ctx, "llvm")}));
    applyConversion.setPartialConversionAttr(UnitAttr::get(ctx));

    // Step 8: Final reconciliation
    auto matchFunc2 =
        createMatchCpuVerifierFuncOp(ib, ctx, updatedModule.getResult());

    ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, matchFunc2.getResults(),
        /*passName=*/"reconcile-unrealized-casts", emptyOptions, emptyDynamic);
  });
}
