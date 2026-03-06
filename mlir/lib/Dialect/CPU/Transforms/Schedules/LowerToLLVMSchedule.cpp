//===- LowerToLLVMSchedule.cpp - Lower to LLVM transform schedule ---------===//
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
// This file implements the lowering transform schedule using the MLIR C++ API.
//
//===----------------------------------------------------------------------===//

#include "LowerToLLVMSchedule.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::cpu;

//===----------------------------------------------------------------------===//
// LowerToLLVMSchedule Builder
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> cpu::buildLowerToLLVMSchedule(MLIRContext *ctx) {
  // Create module with transform.with_named_sequence attribute
  OwningOpRef<ModuleOp> module = createTransformModule(ctx);
  auto loc = UnknownLoc::get(ctx);
  ImplicitLocOpBuilder builder(loc, ctx);
  builder.setInsertionPointToStart(module->getBody());

  auto anyOpType = getAnyOpType(ctx);
  auto emptyOptions = DictionaryAttr::get(ctx);
  auto emptyDynamic = ValueRange{};

  // Build: transform.named_sequence @__transform_main(%arg0: !transform.any_op)
  builder.create<transform::NamedSequenceOp>(
      "__transform_main",
      /*rootType=*/anyOpType,
      /*resultTypes=*/TypeRange{},
      /*bodyBuilder=*/
      [&](OpBuilder &b, Location loc, BlockArgument arg) {
        ImplicitLocOpBuilder ib(loc, b);

        // Step 1: One-shot bufferization
        // %10 = transform.bufferization.one_shot_bufferize
        //         layout{IdentityLayoutMap} %arg0
        //         {bufferize_function_boundaries = true}
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
        // %func = transform.structured.match ops{["func.func"]} in %10
        auto matchFunc = ib.create<transform::MatchOp>(
            anyOpType, bufferize.getTransformed(),
            ArrayRef<StringRef>{"func.func"});

        // %3 = transform.apply_registered_pass "cse" to %func
        auto cse = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, matchFunc.getResults(),
            /*passName=*/"cse", emptyOptions, emptyDynamic);

        // %4 = transform.apply_registered_pass "canonicalize" to %3
        auto canonicalize = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, cse.getResult(),
            /*passName=*/"canonicalize", emptyOptions, emptyDynamic);

        // Step 3: Buffer hoisting passes
        // %b10 = transform.apply_registered_pass "buffer-hoisting" to %4
        auto bufferHoisting = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, canonicalize.getResult(),
            /*passName=*/"buffer-hoisting", emptyOptions, emptyDynamic);

        // %c10 = transform.apply_registered_pass "buffer-loop-hoisting" to %b10
        auto bufferLoopHoisting = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, bufferHoisting.getResult(),
            /*passName=*/"buffer-loop-hoisting", emptyOptions, emptyDynamic);

        // Step 4: Vector lowering patterns
        // %fff = transform.structured.match ops{["func.func"]} in %c10
        auto matchFuncForVector = ib.create<transform::MatchOp>(
            anyOpType, bufferLoopHoisting.getResult(),
            ArrayRef<StringRef>{"func.func"});

        // transform.apply_patterns to %fff { ... vector lowering patterns ... }
        transform::ApplyPatternsOp::create(
            ib, matchFuncForVector.getResults(),
            /*bodyBuilder=*/
            [&](OpBuilder &pib, Location ploc) {
              ImplicitLocOpBuilder pbuilder(ploc, pib);

              // vector.lower_contraction lowering_strategy = "outerproduct"
              transform::ApplyLowerContractionPatternsOp::create(
                  pbuilder, vector::VectorContractLowering::OuterProduct);

              // vector.lower_outerproduct
              pbuilder.create<transform::ApplyLowerOuterProductPatternsOp>();

              // vector.transfer_permutation_patterns
              pbuilder.create<transform::ApplyTransferPermutationPatternsOp>();

              // vector.split_transfer_full_partial split_transfer_strategy =
              // "linalg-copy"
              transform::ApplySplitTransferFullPartialPatternsOp::create(
                  pbuilder, vector::VectorTransferSplitAttr::get(
                                ctx, vector::VectorTransferSplit::LinalgCopy));

              // vector.transfer_to_scf max_transfer_rank = 4 full_unroll = true
              transform::ApplyTransferToScfPatternsOp::create(
                  pbuilder,
                  /*max_transfer_rank=*/pbuilder.getI64IntegerAttr(4),
                  /*full_unroll=*/pbuilder.getBoolAttr(true));

              // vector.lower_transfer max_transfer_rank = 1
              transform::ApplyLowerTransferPatternsOp::create(
                  pbuilder, /*max_transfer_rank=*/1);

              // vector.lower_shape_cast
              pbuilder.create<transform::ApplyLowerShapeCastPatternsOp>();

              // vector.lower_transpose lowering_strategy = "shuffle_1d"
              transform::ApplyLowerTransposePatternsOp::create(
                  pbuilder, vector::VectorTransposeLoweringAttr::get(
                                ctx, vector::VectorTransposeLowering::Shuffle1D));
            });

        // Step 5: Further lowering passes
        // %f = transform.apply_registered_pass "convert-vector-to-scf" to %fff
        auto convertVectorToScf = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, matchFuncForVector.getResults(),
            /*passName=*/"convert-vector-to-scf", emptyOptions, emptyDynamic);

        // %f2 = transform.apply_registered_pass "convert-linalg-to-loops" to %f
        auto convertLinalgToLoops = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, convertVectorToScf.getResult(),
            /*passName=*/"convert-linalg-to-loops", emptyOptions, emptyDynamic);

        // IMPORTANT: convert-vector-to-scf creates new allocations.
        // We must run buffer-loop-hoisting again.
        // %fh1 = transform.apply_registered_pass "buffer-loop-hoisting" to %f2
        auto bufferLoopHoisting2 = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, convertLinalgToLoops.getResult(),
            /*passName=*/"buffer-loop-hoisting", emptyOptions, emptyDynamic);

        // %fh2 = transform.apply_registered_pass "promote-buffers-to-stack" to
        // %fh1
        auto promoteBuffersToStack = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, bufferLoopHoisting2.getResult(),
            /*passName=*/"promote-buffers-to-stack", emptyOptions, emptyDynamic);

        // %f3 = transform.apply_registered_pass "convert-scf-to-cf" to %fh2
        auto convertScfToCf = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, promoteBuffersToStack.getResult(),
            /*passName=*/"convert-scf-to-cf", emptyOptions, emptyDynamic);

        // %f4 = transform.apply_registered_pass "expand-strided-metadata" to %f3
        auto expandStridedMetadata =
            ib.create<transform::ApplyRegisteredPassOp>(
                anyOpType, convertScfToCf.getResult(),
                /*passName=*/"expand-strided-metadata", emptyOptions,
                emptyDynamic);

        // %f5 = transform.apply_registered_pass "lower-affine" to %f4
        auto lowerAffine = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, expandStridedMetadata.getResult(),
            /*passName=*/"lower-affine", emptyOptions, emptyDynamic);

        // %f5b = transform.apply_registered_pass "convert-vector-to-llvm"
        //        with options = {"reassociate-fp-reductions" = false} to %f5
        auto convertVectorToLLVMOptions = DictionaryAttr::get(
            ctx, {NamedAttribute(
                     StringAttr::get(ctx, "reassociate-fp-reductions"),
                     BoolAttr::get(ctx, false))});
        auto convertVectorToLLVM = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, lowerAffine.getResult(),
            /*passName=*/"convert-vector-to-llvm", convertVectorToLLVMOptions,
            emptyDynamic);

        // Step 6: Apply conversion patterns to LLVM
        // transform.apply_conversion_patterns to %f5b { ... }
        auto applyConversion = transform::ApplyConversionPatternsOp::create(
            ib, convertVectorToLLVM.getResult(),
            /*patternsBodyBuilder=*/
            [&](OpBuilder &pib, Location ploc) {
              ImplicitLocOpBuilder pbuilder(ploc, pib);

              // transform.apply_conversion_patterns.dialect_to_llvm "math"
              transform::ApplyToLLVMConversionPatternsOp::create(pbuilder,
                                                                 "math");

              // transform.apply_conversion_patterns.vector.vector_to_llvm
              pbuilder.create<transform::ApplyVectorToLLVMConversionPatternsOp>(
                  /*reassociate_fp_reductions=*/false,
                  /*force_32bit_vector_indices=*/true,
                  /*use_vector_alignment=*/false);

              // transform.apply_conversion_patterns.dialect_to_llvm "memref"
              transform::ApplyToLLVMConversionPatternsOp::create(pbuilder,
                                                                 "memref");

              // transform.apply_conversion_patterns.dialect_to_llvm "index"
              transform::ApplyToLLVMConversionPatternsOp::create(pbuilder,
                                                                 "index");

              // transform.apply_conversion_patterns.dialect_to_llvm "arith"
              transform::ApplyToLLVMConversionPatternsOp::create(pbuilder,
                                                                 "arith");

              // transform.apply_conversion_patterns.dialect_to_llvm "cf"
              transform::ApplyToLLVMConversionPatternsOp::create(pbuilder, "cf");
            },
            /*typeConverterBodyBuilder=*/
            [&](OpBuilder &tcb, Location tcloc) {
              ImplicitLocOpBuilder tcbuilder(tcloc, tcb);

              // transform.apply_conversion_patterns.memref.memref_to_llvm_type_converter
              //   {index_bitwidth = 64, ...}
              transform::MemrefToLLVMTypeConverterOp::create(
                  tcbuilder,
                  /*use_aligned_alloc=*/false,
                  /*index_bitwidth=*/64,
                  /*use_generic_functions=*/false,
                  /*use_bare_ptr_call_conv=*/false,
                  /*data_layout=*/StringAttr{});
            });

        // Set attributes on the conversion op
        applyConversion.setLegalDialectsAttr(
            ArrayAttr::get(ctx, {StringAttr::get(ctx, "llvm")}));
        applyConversion.setPartialConversionAttr(UnitAttr::get(ctx));

        // Step 7: Final reconciliation
        // %f6 = transform.structured.match ops{["llvm.func"]} in %10
        auto matchLLVMFunc = ib.create<transform::MatchOp>(
            anyOpType, bufferize.getTransformed(),
            ArrayRef<StringRef>{"llvm.func"});

        // %f7 = transform.apply_registered_pass "reconcile-unrealized-casts" to
        // %f6
        ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, matchLLVMFunc.getResults(),
            /*passName=*/"reconcile-unrealized-casts", emptyOptions,
            emptyDynamic);

        // transform.yield
        ib.create<transform::YieldOp>();
      });

  return module;
}
