//===- TilingSchedule.cpp - Tiling transform schedule ---------------------===//
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

#include "TilingSchedule.h"
#include "ScheduleUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

OwningOpRef<ModuleOp> cpu::buildTilingSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib, BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);

    // Helper to create parallel iterator type attribute
    auto parallelIterType =
        linalg::IteratorTypeAttr::get(ctx, utils::IteratorType::parallel);
    auto linalgGenericOps =
        ArrayAttr::get(ctx, {StringAttr::get(ctx, "linalg.generic")});

    // Match 1D elementwise ops: iterator_types = [parallel]
    auto elemwise1DOpAttrs = DictionaryAttr::get(
        ctx, {NamedAttribute(StringAttr::get(ctx, "iterator_types"),
                             ArrayAttr::get(ctx, {parallelIterType}))});
    auto matchElemwise1D = ib.create<transform::MatchOp>(
        /*resultTypes=*/anyOpType,
        /*target=*/arg,
        /*ops=*/linalgGenericOps,
        /*interface=*/transform::MatchInterfaceEnumAttr{},
        /*opAttrs=*/elemwise1DOpAttrs,
        /*filterResultType=*/TypeAttr{},
        /*filterOperandTypes=*/ArrayAttr{});

    // Tile 1D elementwise ops: tile_sizes [8]
    SmallVector<Type> tile1DLoopTypes(1, anyOpType);
    ib.create<transform::TileUsingForOp>(
        /*loopTypes=*/tile1DLoopTypes,
        /*target=*/matchElemwise1D.getResult(),
        /*staticTileSizes=*/ArrayRef<int64_t>{8},
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalableSizes=*/std::nullopt);

    // Match 2D elementwise ops: iterator_types = [parallel, parallel]
    auto elemwise2DOpAttrs = DictionaryAttr::get(
        ctx,
        {NamedAttribute(
            StringAttr::get(ctx, "iterator_types"),
            ArrayAttr::get(ctx, {parallelIterType, parallelIterType}))});
    auto matchElemwise2D = ib.create<transform::MatchOp>(
        /*resultTypes=*/anyOpType,
        /*target=*/arg,
        /*ops=*/linalgGenericOps,
        /*interface=*/transform::MatchInterfaceEnumAttr{},
        /*opAttrs=*/elemwise2DOpAttrs,
        /*filterResultType=*/TypeAttr{},
        /*filterOperandTypes=*/ArrayAttr{});

    // Tile 2D elementwise ops: tile_sizes [8, 8]
    SmallVector<Type> tile2DLoopTypes(2, anyOpType);
    ib.create<transform::TileUsingForOp>(
        /*loopTypes=*/tile2DLoopTypes,
        /*target=*/matchElemwise2D.getResult(),
        /*staticTileSizes=*/ArrayRef<int64_t>{8, 8},
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalableSizes=*/std::nullopt);

    // Match 3D elementwise ops: iterator_types = [parallel, parallel, parallel]
    auto elemwise3DOpAttrs = DictionaryAttr::get(
        ctx, {NamedAttribute(StringAttr::get(ctx, "iterator_types"),
                             ArrayAttr::get(ctx, {parallelIterType,
                                                  parallelIterType,
                                                  parallelIterType}))});
    auto matchElemwise3D = ib.create<transform::MatchOp>(
        /*resultTypes=*/anyOpType,
        /*target=*/arg,
        /*ops=*/linalgGenericOps,
        /*interface=*/transform::MatchInterfaceEnumAttr{},
        /*opAttrs=*/elemwise3DOpAttrs,
        /*filterResultType=*/TypeAttr{},
        /*filterOperandTypes=*/ArrayAttr{});

    // Tile 3D elementwise ops: tile_sizes [8, 8, 8]
    SmallVector<Type> tile3DLoopTypes(3, anyOpType);
    ib.create<transform::TileUsingForOp>(
        /*loopTypes=*/tile3DLoopTypes,
        /*target=*/matchElemwise3D.getResult(),
        /*staticTileSizes=*/ArrayRef<int64_t>{8, 8, 8},
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalableSizes=*/std::nullopt);

    auto matchMatmul = createMatchMatmulOp(ib, ctx, arg);

    SmallVector<Type> fuseLoopTypes(3, anyOpType);
    auto fuse = ib.create<transform::FuseOp>(
        /*loopTypes=*/fuseLoopTypes,
        /*target=*/matchMatmul.getResults(),
        /*staticTileSizes=*/ArrayRef<int64_t>{1, 256, 64},
        /*staticTileInterchange=*/ArrayRef<int64_t>{},
        /*applyCleanup=*/false,
        /*useForall=*/false);

    SmallVector<Type> tile1LoopTypes(1, anyOpType);
    auto tile1 = ib.create<transform::TileUsingForOp>(
        /*loopTypes=*/tile1LoopTypes,
        /*target=*/fuse.getTransformed(),
        /*staticTileSizes=*/ArrayRef<int64_t>{0, 0, 0, 64},
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalableSizes=*/std::nullopt);

    SmallVector<Type> tile2LoopTypes(3, anyOpType);
    ib.create<transform::TileUsingForOp>(
        /*loopTypes=*/tile2LoopTypes,
        /*target=*/tile1.getTiledLinalgOp(),
        /*staticTileSizes=*/ArrayRef<int64_t>{0, 8, 8, 1},
        /*interchange=*/ArrayRef<int64_t>{},
        /*scalableSizes=*/std::nullopt);

    auto matchFunc = ib.create<transform::MatchOp>(
        anyOpType, arg, ArrayRef<StringRef>{"func.func"});

    auto canonicalize = ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, matchFunc.getResults(),
        /*passName=*/"canonicalize",
        /*options=*/DictionaryAttr::get(ctx),
        /*dynamicOptions=*/ValueRange{});

    ib.create<transform::ApplyRegisteredPassOp>(
        anyOpType, canonicalize.getResult(),
        /*passName=*/"lower-affine",
        /*options=*/DictionaryAttr::get(ctx),
        /*dynamicOptions=*/ValueRange{});
  });
}
