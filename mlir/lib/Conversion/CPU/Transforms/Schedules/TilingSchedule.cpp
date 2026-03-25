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

/// Match and tile N-dimensional elementwise ops (linalg.generic with N parallel
/// iterator types).
static void tileElementwiseOps(ImplicitLocOpBuilder &ib, MLIRContext *ctx,
                               Value target, int vectorSize, unsigned numDims) {
  auto anyOpType = getAnyOpType(ctx);
  auto parallelIterType =
      linalg::IteratorTypeAttr::get(ctx, utils::IteratorType::parallel);

  // Create iterator_types = [parallel, parallel, ...] with numDims elements
  SmallVector<Attribute> iteratorTypes(numDims, parallelIterType);
  auto opAttrs = DictionaryAttr::get(
      ctx, {NamedAttribute(StringAttr::get(ctx, "iterator_types"),
                           ArrayAttr::get(ctx, iteratorTypes))});

  // Match linalg.generic ops with the specified iterator types
  auto matchOp = ib.create<transform::MatchOp>(
      /*resultTypes=*/anyOpType,
      /*target=*/target,
      /*ops=*/ArrayAttr::get(ctx, {StringAttr::get(ctx, "linalg.generic")}),
      /*interface=*/transform::MatchInterfaceEnumAttr{},
      /*opAttrs=*/opAttrs,
      /*filterResultType=*/TypeAttr{},
      /*filterOperandTypes=*/ArrayAttr{});

  // Tile with the given vector size for each dimension
  SmallVector<Type> loopTypes(numDims, anyOpType);
  SmallVector<int64_t> tileSizes(numDims, vectorSize);
  ib.create<transform::TileUsingForOp>(
      /*loopTypes=*/loopTypes,
      /*target=*/matchOp.getResult(),
      /*staticTileSizes=*/tileSizes,
      /*interchange=*/ArrayRef<int64_t>{},
      /*scalableSizes=*/std::nullopt);
}

OwningOpRef<ModuleOp> cpu::buildTilingSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib, BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);

    // AVX vector width is 256 bits. 
    // For fp32, AVX takes 8 elements.
    int vectorSize = 8;

    // Tile elementwise ops of different dimensions to prevent huge vectors
    tileElementwiseOps(ib, ctx, arg, vectorSize, /*numDims=*/1);
    tileElementwiseOps(ib, ctx, arg, vectorSize, /*numDims=*/2);
    tileElementwiseOps(ib, ctx, arg, vectorSize, /*numDims=*/3);

    // Now tile (and optionally fuse) the matmul
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
        /*staticTileSizes=*/ArrayRef<int64_t>{0, vectorSize, vectorSize, 1},
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
