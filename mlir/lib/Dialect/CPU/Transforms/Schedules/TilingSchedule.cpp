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
//
// This file implements the tiling transform schedule using the MLIR C++ API.
// This schedule tiles matmul operations for CPU cache efficiency.
//
//===----------------------------------------------------------------------===//

#include "TilingSchedule.h"
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
// Tiling Schedule Builder
//===----------------------------------------------------------------------===//

OwningOpRef<ModuleOp> cpu::buildTilingSchedule(MLIRContext *ctx) {
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

        // %transformed, %loops:3 = transform.structured.fuse %0
        //     tile_sizes [1, 256, 64]
        // Returns: transformed op + 3 loop handles
        SmallVector<Type> fuseLoopTypes(3, anyOpType);
        auto fuse = ib.create<transform::FuseOp>(
            /*loopTypes=*/fuseLoopTypes,
            /*target=*/matchMatmul.getResults(),
            /*staticTileSizes=*/ArrayRef<int64_t>{1, 256, 64},
            /*staticTileInterchange=*/ArrayRef<int64_t>{},
            /*applyCleanup=*/false,
            /*useForall=*/false);

        // %xxx, %loop = transform.structured.tile_using_for %transformed
        //     tile_sizes [0, 0, 0, 64]
        // Returns: tiled op + 1 loop handle (only 1 non-zero tile size)
        SmallVector<Type> tile1LoopTypes(1, anyOpType);
        auto tile1 = ib.create<transform::TileUsingForOp>(
            /*loopTypes=*/tile1LoopTypes,
            /*target=*/fuse.getTransformed(),
            /*staticTileSizes=*/ArrayRef<int64_t>{0, 0, 0, 64},
            /*interchange=*/ArrayRef<int64_t>{},
            /*scalableSizes=*/std::nullopt);

        // %microkernel, %microkernel_loops:3 = transform.structured.tile_using_for
        //     %xxx tile_sizes [0, 8, 8, 1]
        // Returns: tiled op + 3 loop handles (3 non-zero tile sizes)
        SmallVector<Type> tile2LoopTypes(3, anyOpType);
        ib.create<transform::TileUsingForOp>(
            /*loopTypes=*/tile2LoopTypes,
            /*target=*/tile1.getTiledLinalgOp(),
            /*staticTileSizes=*/ArrayRef<int64_t>{0, 8, 8, 1},
            /*interchange=*/ArrayRef<int64_t>{},
            /*scalableSizes=*/std::nullopt);

        // TODO: We will probably need padding and/or packing here.

        // %1 = transform.structured.match ops{["func.func"]} in %arg0
        auto matchFunc = ib.create<transform::MatchOp>(
            anyOpType, arg, ArrayRef<StringRef>{"func.func"});

        // %2 = transform.apply_registered_pass "canonicalize" to %1
        auto canonicalize = ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, matchFunc.getResults(),
            /*passName=*/"canonicalize",
            /*options=*/DictionaryAttr::get(ctx),
            /*dynamicOptions=*/ValueRange{});

        // %3 = transform.apply_registered_pass "lower-affine" to %2
        ib.create<transform::ApplyRegisteredPassOp>(
            anyOpType, canonicalize.getResult(),
            /*passName=*/"lower-affine",
            /*options=*/DictionaryAttr::get(ctx),
            /*dynamicOptions=*/ValueRange{});

        // transform.yield
        ib.create<transform::YieldOp>();
      });

  return module;
}
