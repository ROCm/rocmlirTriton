//===- FusedConvToMatmulSchedule.cpp - Fused conv -> matmul shape ---------===//
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

#include "FusedConvToMatmulSchedule.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
using namespace mlir::cpu;

/// Build the iterator-types `DictionaryAttr` corresponding to the
/// rocmlir-gen batched-GEMM iter-space `(G, M, N, K)` -> `[par, par, par,
/// red]`. Used by the transform-dialect matcher below.
static DictionaryAttr getBatchedGemmIteratorTypesAttr(MLIRContext *ctx) {
  static constexpr utils::IteratorType kIters[] = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::parallel, utils::IteratorType::reduction};
  SmallVector<Attribute> iteratorTypeAttrs;
  iteratorTypeAttrs.reserve(std::size(kIters));
  for (utils::IteratorType iter : kIters)
    iteratorTypeAttrs.push_back(linalg::IteratorTypeAttr::get(ctx, iter));
  return DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, "iterator_types"),
                           ArrayAttr::get(ctx, iteratorTypeAttrs))});
}

OwningOpRef<ModuleOp> cpu::buildFusedConvToMatmulSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib,
                                         BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);

    // Match the fused 8-D conv emitted by --cpu-conv-to-gemm.
    auto fusedConvAttrs = DictionaryAttr::get(
        ctx, {NamedAttribute(
                 StringAttr::get(ctx, rock::CpuFusedConvAttr::getMnemonic()),
                 UnitAttr::get(ctx))});
    auto matchFused = ib.create<transform::MatchOp>(
        /*resultTypes=*/anyOpType,
        /*target=*/arg,
        /*ops=*/ArrayAttr::get(ctx, {StringAttr::get(ctx, "linalg.generic")}),
        /*interface=*/transform::MatchInterfaceEnumAttr{},
        /*opAttrs=*/fusedConvAttrs,
        /*filterResultType=*/TypeAttr{},
        /*filterOperandTypes=*/ArrayAttr{});

    // Iteration-space order of the fused op (set by `createFusedConvOp` in
    // ConvToGemm.cpp; do not reorder without updating both sides):
    //
    //   d0 = N   parallel
    //   d1 = G   parallel
    //   d2 = C   parallel
    //   d3 = Ho  parallel
    //   d4 = Wo  parallel
    //   d5 = KC  reduction
    //   d6 = Fh  reduction
    //   d7 = Fw  reduction
    //
    // We tile by 1 along (N, Ho, Fh, Fw) to peel them off as outer scalar
    // loops. The remaining iter dims (G, C, Wo, KC) stay inside the inner
    // `linalg.generic` at full extent. After unit-extent folding below,
    // those collapse to a 3-D `(C, Wo, KC)` matmul shape with iter types
    // `[par, par, red]` (G is unit and folds away too).
    SmallVector<int64_t> convTileSizes = {1, 0, 0, 1, 0, 0, 1, 1};

    // Number of returned `scf.for` loops equals the count of non-zero tile
    // sizes. Build the loop-result type list to match.
    SmallVector<Type> convLoopTypes;
    for (int64_t s : convTileSizes)
      if (s != 0)
        convLoopTypes.push_back(anyOpType);

    ib.create<transform::TileUsingForOp>(
        /*loopTypes=*/convLoopTypes,
        /*target=*/matchFused.getResult(),
        /*staticTileSizes=*/convTileSizes,
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalableSizes=*/std::nullopt);

    // Now tile the G dimension, to get a 3D matmul.
    // Tile sizes index by iter-space dim; G is at d0:
    //                                d0 d1 d2 d3
    //                                 G  M  N  K
    SmallVector<int64_t> gemmTileSizes = {1, 0, 0, 0};
    SmallVector<Type> gemmLoopTypes(/*Size=*/1, anyOpType);

    auto gemmAttrs = getBatchedGemmIteratorTypesAttr(ctx);
    auto matchBatchedGemm = ib.create<transform::MatchOp>(
        /*resultTypes=*/anyOpType,
        /*target=*/arg,
        /*ops=*/ArrayAttr::get(ctx, {StringAttr::get(ctx, "linalg.generic")}),
        /*interface=*/transform::MatchInterfaceEnumAttr{},
        /*opAttrs=*/gemmAttrs,
        /*filterResultType=*/TypeAttr{},
        /*filterOperandTypes=*/ArrayAttr{});

    ib.create<transform::TileUsingForOp>(
        /*loopTypes=*/gemmLoopTypes,
        /*target=*/matchBatchedGemm.getResult(),
        /*staticTileSizes=*/gemmTileSizes,
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalableSizes=*/std::nullopt);

    // Drop the unit-extent iteration dims that tiling introduced on the
    // inner ops (plus any pre-existing unit dims like G=1) so each inner
    // kernel collapses to the matmul shape that downstream schedules
    // recognise.
    auto matchFunc = createMatchCpuVerifierFuncOp(ib, ctx, arg);
    ib.create<transform::ApplyPatternsOp>(
        /*target=*/matchFunc.getResult(),
        /*bodyBuilder=*/
        [&](OpBuilder &nb, Location loc) {
          ImplicitLocOpBuilder nested(loc, nb);
          nested
              .create<transform::ApplyFoldUnitExtentDimsViaSlicesPatternsOp>();
          nested.create<transform::ApplyCanonicalizationPatternsOp>();
        });
  });
}
