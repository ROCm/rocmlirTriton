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

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

OwningOpRef<ModuleOp> cpu::buildTilingSchedule(MLIRContext *ctx) {
  return buildTransformModule(ctx, [ctx](ImplicitLocOpBuilder &ib, BlockArgument arg) {
    auto anyOpType = getAnyOpType(ctx);

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
