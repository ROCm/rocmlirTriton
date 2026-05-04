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

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "cpu-tiling-schedule"

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

/// Flatten elementwise linalg.generic ops containing arith.extf or
/// arith.truncf. This prevents IR explosion when lowering multi-dimensional
/// vectors to LLVM, since LLVM only supports 1D vectors and must decompose each
/// leaf vector individually for operations on nested arrays.
static void flattenExtfTruncfOps(ImplicitLocOpBuilder &ib, MLIRContext *ctx,
                                 Value target) {
  auto anyOpType = getAnyOpType(ctx);

  // Match arith.extf ops and get their parent linalg.generic
  auto matchExtf = ib.create<transform::MatchOp>(
      anyOpType, target, ArrayRef<StringRef>{"arith.extf"});
  auto extfLinalg = ib.create<transform::GetParentOp>(
      /*parent=*/anyOpType,
      /*target=*/matchExtf.getResult(),
      /*isolated_from_above=*/UnitAttr{},
      /*allow_empty_results=*/UnitAttr{},
      /*op_name=*/StringAttr::get(ctx, "linalg.generic"),
      /*deduplicate=*/UnitAttr{},
      /*nth_parent=*/IntegerAttr{});

  // Match arith.truncf ops and get their parent linalg.generic
  auto matchTruncf = ib.create<transform::MatchOp>(
      anyOpType, target, ArrayRef<StringRef>{"arith.truncf"});
  auto truncfLinalg = ib.create<transform::GetParentOp>(
      /*parent=*/anyOpType,
      /*target=*/matchTruncf.getResult(),
      /*isolated_from_above=*/UnitAttr{},
      /*allow_empty_results=*/UnitAttr{},
      /*op_name=*/StringAttr::get(ctx, "linalg.generic"),
      /*deduplicate=*/UnitAttr{},
      /*nth_parent=*/IntegerAttr{});

  // Merge handles and flatten the elementwise ops
  auto merged = ib.create<transform::MergeHandlesOp>(
      anyOpType, ValueRange{extfLinalg.getResult(), truncfLinalg.getResult()},
      /*deduplicate=*/UnitAttr{});
  ib.create<transform::FlattenElementwiseLinalgOp>(anyOpType,
                                                   merged.getResult());
}

OwningOpRef<ModuleOp>
cpu::buildTilingSchedule(MLIRContext *ctx, const MatmulTileSizes &tileSizes) {
  LLVM_DEBUG(llvm::dbgs()
             << "buildTilingSchedule: matmul tile sizes G=" << tileSizes.gFuse
             << " M=" << tileSizes.mFuse << "(div=" << tileSizes.mDivisible
             << ") N=" << tileSizes.nFuse << "(div=" << tileSizes.nDivisible
             << ") K=" << tileSizes.kTile << "(div=" << tileSizes.kDivisible
             << ") microTile=(" << tileSizes.microTileM << ","
             << tileSizes.microTileN << "," << tileSizes.microTileK << ")\n");

  return buildTransformModule(
      ctx, [ctx, tileSizes](ImplicitLocOpBuilder &ib, BlockArgument arg) {
        auto anyOpType = getAnyOpType(ctx);
        // Typed handle for `scf.for` payloads -- required as the operand
        // type of `transform.loop.peel`, so we ask `fuse` and the K-tile
        // to return their loops as `!transform.op<"scf.for">` instead of
        // the generic `!transform.any_op`.
        auto scfForType = transform::OperationType::get(ctx, "scf.for");

        // AVX vector width is 256 bits.
        // For fp32, AVX takes 8 elements.
        int vectorSize = 8;

        // Flatten extf/truncf linalg.generic ops to 1D to avoid IR explosion
        // when lowering multi-dimensional vectors to LLVM
        flattenExtfTruncfOps(ib, ctx, arg);

        // Tile elementwise ops of different dimensions to prevent huge vectors
        tileElementwiseOps(ib, ctx, arg, vectorSize, /*numDims=*/1);
        tileElementwiseOps(ib, ctx, arg, vectorSize, /*numDims=*/2);
        tileElementwiseOps(ib, ctx, arg, vectorSize, /*numDims=*/3);
        tileElementwiseOps(ib, ctx, arg, vectorSize, /*numDims=*/4);
        tileElementwiseOps(ib, ctx, arg, vectorSize, /*numDims=*/5);

        // Now tile (and optionally fuse) the matmul.
        //
        // Loop layout produced by `fuse` with interchange [0, 2, 1] and
        // tile sizes [gFuse, mFuse, nFuse] (G, M, N parallel, K reduction):
        //   loops[0] -> G  (tile = gFuse)
        //   loops[1] -> N  (tile = nFuse)
        //   loops[2] -> M  (tile = mFuse)
        // K is tiled separately by `tile1` below.
        //
        // For each parallel dim that the caller marked as not divisible by
        // its tile, peel the corresponding loop so the main body has a
        // clean static trip count and only the (small) remainder body
        // carries a partial iteration. Vectorization downstream targets
        // only the static main body. Peel the M loop first (innermost of
        // the fused triple) so the N peel that follows operates on a
        // simpler structure; functionally either order works.
        auto matchMatmul = createMatchMatmulOp(ib, ctx, arg);

        SmallVector<Type> fuseLoopTypes(3, scfForType);
        auto fuse = ib.create<transform::FuseOp>(
            /*loopTypes=*/fuseLoopTypes,
            /*target=*/matchMatmul.getResults(),
            /*staticTileSizes=*/
            ArrayRef<int64_t>{tileSizes.gFuse, tileSizes.mFuse,
                              tileSizes.nFuse},
            /*staticTileInterchange=*/ArrayRef<int64_t>{0, 2, 1},
            /*applyCleanup=*/false,
            /*useForall=*/false);

        // Helper: peel a loop's last (partial) iteration.
        auto peelLoop = [&](Value loop) {
          ib.create<transform::LoopPeelOp>(
              /*peeled_loop=*/scfForType,
              /*remainder_loop=*/scfForType,
              /*target=*/loop,
              /*peel_front=*/false,
              /*fail_if_already_divisible=*/false);
        };

        bool peeledFuseLoop = false;
        if (!tileSizes.mDivisible) {
          peelLoop(fuse.getLoops()[2]);
          peeledFuseLoop = true;
        }
        if (!tileSizes.nDivisible) {
          peelLoop(fuse.getLoops()[1]);
          peeledFuseLoop = true;
        }

        // Peeling clones the matmul into both main and remainder bodies and
        // invalidates the `fuse.getTransformed()` handle. Re-match so the
        // K-tile applies to the new payload.
        Value kTileTarget =
            peeledFuseLoop
                ? createMatchMatmulOp(ib, ctx, arg).getResults()
                : fuse.getTransformed();

        SmallVector<Type> tile1LoopTypes(1, scfForType);
        auto tile1 = ib.create<transform::TileUsingForOp>(
            /*loopTypes=*/tile1LoopTypes,
            /*target=*/kTileTarget,
            /*staticTileSizes=*/ArrayRef<int64_t>{0, 0, 0, tileSizes.kTile},
            /*interchange=*/ArrayRef<int64_t>{},
            /*scalableSizes=*/std::nullopt);

        // Same dance for K: peel if the K dim isn't a multiple of kTile,
        // then re-match the matmul before the inner micro-tile runs.
        Value microTileTarget = tile1.getTiledLinalgOp();
        if (!tileSizes.kDivisible) {
          peelLoop(tile1.getLoops()[0]);
          microTileTarget = createMatchMatmulOp(ib, ctx, arg).getResults();
        }

        SmallVector<Type> tile2LoopTypes(3, anyOpType);
        ib.create<transform::TileUsingForOp>(
            /*loopTypes=*/tile2LoopTypes,
            /*target=*/microTileTarget,
            /*staticTileSizes=*/
            ArrayRef<int64_t>{0, tileSizes.microTileM, tileSizes.microTileN,
                              tileSizes.microTileK},
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
