//===- VectorizationSchedule.cpp - Vectorization transform schedule -------===//
//
// Copyright Advanced Micro Devices, Inc.
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

#include "VectorizationSchedule.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

namespace {

/// Name of the named matcher sequence used by `transform.collect_matching` to
/// pick out only the static-shaped matmul-like `linalg.generic` op.
constexpr llvm::StringLiteral staticMatmulMatcherName =
    "match_static_matmul_generic";

/// Populate the body of a `transform.apply_patterns` op with the same set of
/// ops that `transform.structured.vectorize_children_and_apply_patterns` would
/// apply. Mirrors `VectorizeChildrenAndApplyPatternsOp::applyToOne` upstream.
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

/// Build the matcher named sequence:
///   transform.named_sequence @match_static_matmul_generic(
///       %candidate: !transform.any_op {transform.readonly}) ->
///       !transform.any_op
static void buildStaticMatmulMatcher(OpBuilder &builder, Location loc,
                                     ModuleOp module,
                                     ArrayRef<int64_t> expectedDims) {
  MLIRContext *ctx = builder.getContext();
  auto anyOpType = transform::AnyOpType::get(ctx);
  auto i64ParamType = transform::ParamType::get(ctx, builder.getI64Type());

  auto readonlyAttr = builder.getDictionaryAttr(
      {builder.getNamedAttr("transform.readonly", builder.getUnitAttr())});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());

  auto matcher = transform::NamedSequenceOp::create(
      builder, loc, staticMatmulMatcherName,
      /*rootType=*/anyOpType,
      /*resultTypes=*/TypeRange{anyOpType},
      /*bodyBuilder=*/
      [&](OpBuilder &b, Location nestedLoc, BlockArgument candidate) {
        ImplicitLocOpBuilder ib(nestedLoc, b);
        ib.create<transform::MatchOperationNameOp>(
            candidate, b.getStrArrayAttr({"linalg.generic"}));

        auto matchStructured = ib.create<transform::MatchStructuredOp>(
            /*outputs=*/TypeRange{anyOpType},
            /*current=*/candidate,
            /*failure_propagation_mode=*/
            transform::FailurePropagationModeAttr{});

        Region &body = matchStructured.getBodyRegion();
        Block *block =
            b.createBlock(&body, body.begin(), {anyOpType}, {nestedLoc});
        BlockArgument structHandle = block->getArgument(0);

        ImplicitLocOpBuilder bodyIb(nestedLoc, b);
        bodyIb.setInsertionPointToStart(block);

        // Reject candidates whose loop rank is not 3.
        auto rankParam = bodyIb.create<transform::MatchStructuredRankOp>(
            /*rank=*/i64ParamType,
            /*operand_handle=*/structHandle);
        auto expectedRank = bodyIb.create<transform::ParamConstantOp>(
            /*param=*/i64ParamType,
            /*value=*/b.getI64IntegerAttr(3));
        bodyIb.create<transform::MatchParamCmpIOp>(
            /*param=*/rankParam.getResult(),
            /*reference=*/expectedRank.getResult(),
            /*predicate=*/transform::MatchCmpIPredicate::eq);

        // Iterator-type predicates: dims [0, 1] parallel, dim [2] reduction.
        bodyIb.create<transform::MatchStructuredDimOp>(
            /*result=*/Type{},
            /*operand_handle=*/structHandle,
            /*raw_dim_list=*/b.getDenseI64ArrayAttr({0, 1}),
            /*is_inverted=*/UnitAttr{},
            /*is_all=*/UnitAttr{},
            /*parallel=*/b.getUnitAttr(),
            /*reduction=*/UnitAttr{});
        bodyIb.create<transform::MatchStructuredDimOp>(
            /*result=*/Type{},
            /*operand_handle=*/structHandle,
            /*raw_dim_list=*/b.getDenseI64ArrayAttr({2}),
            /*is_inverted=*/UnitAttr{},
            /*is_all=*/UnitAttr{},
            /*parallel=*/UnitAttr{},
            /*reduction=*/b.getUnitAttr());

        // `param.constant` only accepts a scalar attribute, so compare each
        // loop range individually against its expected static size.
        SmallVector<Value, 4> expectedConsts;
        expectedConsts.reserve(expectedDims.size());
        for (int64_t expected : expectedDims) {
          auto constOp = bodyIb.create<transform::ParamConstantOp>(
              /*param=*/i64ParamType,
              /*value=*/b.getI64IntegerAttr(expected));
          expectedConsts.push_back(constOp.getResult());
        }

        for (auto [i, expectedConst] : llvm::enumerate(expectedConsts)) {
          auto capturedOp = bodyIb.create<transform::MatchStructuredDimOp>(
              /*result=*/i64ParamType,
              /*operand_handle=*/structHandle,
              /*raw_dim_list=*/
              b.getDenseI64ArrayAttr({static_cast<int64_t>(i)}),
              /*is_inverted=*/UnitAttr{},
              /*is_all=*/UnitAttr{},
              /*parallel=*/UnitAttr{},
              /*reduction=*/UnitAttr{});
          bodyIb.create<transform::MatchParamCmpIOp>(
              /*param=*/capturedOp.getResult(),
              /*reference=*/expectedConst,
              /*predicate=*/transform::MatchCmpIPredicate::eq);
        }

        bodyIb.create<transform::MatchStructuredYieldOp>(
            ValueRange{structHandle});

        // Yield the matched op as the result of the named sequence.
        ib.setInsertionPointAfter(matchStructured);
        ib.create<transform::YieldOp>(matchStructured.getResults());
      });

  // NamedSequenceOp::build does not propagate the `argAttrs` parameter, so
  // set the arg attributes after construction. `transform.readonly` is
  // required by `transform.collect_matching` on the matcher's argument.
  matcher.setArgAttrs(0, readonlyAttr);
}

} // namespace

OwningOpRef<ModuleOp> cpu::buildVectorizationSchedule(MLIRContext *ctx) {
  // Match parameters that VectorizeChildrenAndApplyPatternsOp used to be
  // invoked with.
  constexpr bool vectorizePadding = true;
  constexpr bool vectorizeNDExtract = true;
  constexpr bool foldTypeExtensionsIntoContract = false;

  // If peeling was applied, we will have multiple matmuls with different
  // shapes: The original matmul plus the peeled one. But we want to vectorize
  // only the original one, which we know that has the static shape {8, 8, 8}
  // after the 3-D matmul micro-tile. So match that only.
  static constexpr int64_t expectedDims[] = {8, 8, 8};

  OwningOpRef<ModuleOp> module = createTransformModule(ctx);
  auto loc = UnknownLoc::get(ctx);
  ImplicitLocOpBuilder builder(loc, ctx);

  // Emit the matcher named sequence at module scope first so it can be
  // referenced by symbol from `__transform_main`.
  buildStaticMatmulMatcher(builder, loc, *module, expectedDims);

  // Emit `__transform_main` as a sibling named sequence.
  builder.setInsertionPointToEnd(module->getBody());
  auto anyOpType = getAnyOpType(ctx);
  builder.create<transform::NamedSequenceOp>(
      "__transform_main",
      /*rootType=*/anyOpType,
      /*resultTypes=*/TypeRange{},
      /*bodyBuilder=*/
      [&](OpBuilder &b, Location nestedLoc, BlockArgument arg) {
        ImplicitLocOpBuilder ib(nestedLoc, b);

        auto matchFunc = createMatchCpuVerifierFuncOp(ib, ctx, arg);

        // Collect only matmul `linalg.generic` ops whose static loop ranges
        // match `expectedDims` -- i.e. the post-peeling main-loop matmuls.
        auto staticMatmuls = ib.create<transform::CollectMatchingOp>(
            /*results=*/TypeRange{anyOpType},
            /*root=*/arg,
            /*matcher=*/
            SymbolRefAttr::get(StringAttr::get(ctx, staticMatmulMatcherName)));

        ib.create<transform::VectorizeOp>(
            /*target=*/staticMatmuls.getResults().front(),
            /*vector_sizes=*/ValueRange{},
            /*static_vector_sizes=*/DenseI64ArrayAttr{},
            /*vectorize_nd_extract=*/
            vectorizeNDExtract ? UnitAttr::get(ctx) : UnitAttr{},
            /*assume_dynamic_dims_match_vec_sizes=*/UnitAttr{},
            /*create_named_contraction=*/UnitAttr{},
            /*scalable_sizes=*/DenseBoolArrayAttr{});

        // Apply the same cleanup pattern set that
        // VectorizeChildrenAndApplyPatternsOp would apply.
        ib.create<transform::ApplyPatternsOp>(
            /*target=*/matchFunc.getResults(),
            /*bodyBuilder=*/
            [&](OpBuilder &nb, Location loc) {
              ImplicitLocOpBuilder nested(loc, nb);
              populateVectorizationCleanup(nested, vectorizePadding,
                                           foldTypeExtensionsIntoContract);
            });

        ib.create<transform::YieldOp>();
      });

  return module;
}
