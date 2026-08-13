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
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::cpu;

/// Match and tile N-dimensional elementwise ops (linalg.generic with N parallel
/// iterator types).
static void tileElementwiseOps(ImplicitLocOpBuilder &ib, MLIRContext *ctx,
                               Value target, int64_t vectorSize,
                               unsigned numDims) {
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

OwningOpRef<ModuleOp>
cpu::buildTilingSchedule(MLIRContext *ctx, const MatmulTileSizes &tileSizes) {
  return buildTransformModule(
      ctx, [ctx, tileSizes](ImplicitLocOpBuilder &ib, BlockArgument arg) {
        auto anyOpType = getAnyOpType(ctx);
        auto scfForType = transform::OperationType::get(ctx, "scf.for");
        constexpr int64_t vectorSize = MatmulTileSizes::kVectorSize;

        // Tile elementwise ops of different dimensions. In general, this
        // improves performance.
        for (int numDims = 1; numDims <= 8; numDims++) {
          tileElementwiseOps(ib, ctx, arg, vectorSize, numDims);
        }

        // Now tile (and optionally fuse) the matmul.
        auto matchMatmul = createMatchMatmulOp(ib, ctx, arg);

        const unsigned mDim = tileSizes.mDim;
        const unsigned nDim = tileSizes.nDim;
        const unsigned kDim = tileSizes.kDim;

        // Interchange puts N on the outside (i.e. iterate over N first)
        // by listing parallel dims in [N, M, K] order.
        SmallVector<int64_t, 3> fuseTileSizes(3, 0);
        fuseTileSizes[mDim] = tileSizes.mFuse;
        fuseTileSizes[nDim] = tileSizes.nFuse;
        SmallVector<int64_t, 3> fuseInterchange = {nDim, mDim, kDim};
        SmallVector<Type> fuseLoopTypes(2, scfForType);
        auto fuse = ib.create<transform::FuseOp>(
            /*loopTypes=*/fuseLoopTypes,
            /*target=*/matchMatmul.getResults(),
            /*staticTileSizes=*/fuseTileSizes,
            /*staticTileInterchange=*/fuseInterchange,
            /*applyCleanup=*/false,
            /*useForall=*/false);

        auto peelLoop = [&](Value loop) {
          ib.create<transform::LoopPeelOp>(
              /*peeled_loop=*/scfForType,
              /*remainder_loop=*/scfForType,
              /*target=*/loop,
              /*peel_front=*/false,
              /*fail_if_already_divisible=*/false);
        };

        // Peel if tiling does not perfectly divide the loop.
        // We do this to make sure we can vectorize the loop.
        bool peeledFuseLoop = false;
        if (!tileSizes.mDivisible) {
          peelLoop(fuse.getLoops()[1]);
          peeledFuseLoop = true;
        }
        if (!tileSizes.nDivisible) {
          peelLoop(fuse.getLoops()[0]);
          peeledFuseLoop = true;
        }

        // Re-match the matmul after peeling.
        Value kTileTarget = peeledFuseLoop
                                ? createMatchMatmulOp(ib, ctx, arg).getResults()
                                : fuse.getTransformed();

        // Tile only the K (reduction) dim: tile sizes index by iter-space
        // dim, so we put `kTile` at the K position and 0 elsewhere.
        SmallVector<int64_t, 3> kTileSizes(3, 0);
        kTileSizes[kDim] = tileSizes.kTile;
        SmallVector<Type> tile1LoopTypes(1, scfForType);
        auto tile1 = ib.create<transform::TileUsingForOp>(
            /*loopTypes=*/tile1LoopTypes,
            /*target=*/kTileTarget,
            /*staticTileSizes=*/kTileSizes,
            /*interchange=*/ArrayRef<int64_t>{},
            /*scalableSizes=*/std::nullopt);

        // Same for K: peel if the K dim isn't a multiple of kTile,
        // then re-match the matmul before the inner micro-tile runs.
        Value microTileTarget = tile1.getTiledLinalgOp();
        if (!tileSizes.kDivisible) {
          peelLoop(tile1.getLoops()[0]);
          microTileTarget = createMatchMatmulOp(ib, ctx, arg).getResults();
        }

        // Innermost register-blocking micro-tile over (M, N, K). Place
        // each per-dim micro-tile at its iter-space position.
        SmallVector<int64_t, 3> microTileSizes(3, 0);
        microTileSizes[mDim] = tileSizes.microTileM;
        microTileSizes[nDim] = tileSizes.microTileN;
        microTileSizes[kDim] = tileSizes.microTileK;
        SmallVector<Type> tile2LoopTypes(3, anyOpType);
        ib.create<transform::TileUsingForOp>(
            /*loopTypes=*/tile2LoopTypes,
            /*target=*/microTileTarget,
            /*staticTileSizes=*/microTileSizes,
            /*interchange=*/ArrayRef<int64_t>{},
            /*scalableSizes=*/std::nullopt);

        auto matchFunc = createMatchCpuVerifierFuncOp(ib, ctx, arg);

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

OwningOpRef<ModuleOp> cpu::buildTilingSchedule(MLIRContext *ctx) {
  return buildTilingSchedule(ctx, MatmulTileSizes{});
}
