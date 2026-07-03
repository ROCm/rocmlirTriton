//===- GridwiseElementwiseToBlockwise.cpp - lower gridwise_elementwise ---===//
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
// ============================================================
//
// This pass is the pure-elementwise analogue of
// rock-gridwise-gemm-to-blockwise. It lowers a rock.gridwise_elementwise root
// into the same StoreMarkerOp-anchored output tile that a gemm produces, but
// the tile is loaded from the primary input instead of being computed:
//
//   %tile = rock.load_marker %input views [<output_tiling>] [%g, %m, %n]
//   %full = rock.store_marker %tile views [<output_tiling>] [%g, %m, %n]
//
// After this, rock-insert-output-fusion-loads turns the remaining elementwise
// inputs into output-fusion loads and rock-lower-stores emits the per-block
// blockwise_store, exactly as for a fused gemm + elementwise tail.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "GridLayoutEmitter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGRIDWISEELEMENTWISETOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gridwise-elementwise-to-blockwise"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockGridwiseElementwiseToBlockwisePass
    : public rock::impl::RockGridwiseElementwiseToBlockwisePassBase<
          RockGridwiseElementwiseToBlockwisePass> {
  void runOnOperation() override;
};

struct GridwiseElementwiseRewritePattern
    : public OpConversionPattern<GridwiseElementwiseOp> {
  using OpConversionPattern<GridwiseElementwiseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GridwiseElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto inTy = cast<RankedTensorType>(input.getType());
    ArrayRef<int64_t> shape = inTy.getShape();
    int64_t G = shape[0], M = shape[1], N = shape[2];
    int64_t mPerBlock = op.getMPerBlock();
    int64_t nPerBlock = op.getNPerBlock();
    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;
    Type elemType = inTy.getElementType();

    StringRef arch = getArchValue(op).getValue();
    int64_t numCU = getNumCUValue(op);
    int64_t numChiplets = getNumChipletsValue(op);

    // Current workgroup id and its (g, m, n) block coordinates.
    Value bid =
        triton::GetProgramIdOp::create(b, loc, triton::ProgramIDDim::X);
    layout::GridLayoutInfo info{G,        mBlocks,  nBlocks,
                                numCU,    numChiplets, elemType,
                                elemType, /*gridGroupSize=*/0};
    layout::GridCoordinates gridCoords =
        layout::makeGroupedGridLayout(b, loc, bid, info, arch);

    FailureOr<ArrayAttr> maybeViews = computeOutputTransforms(
        b, loc, mPerBlock, nPerBlock, {G, mBlocks, nBlocks});
    if (failed(maybeViews))
      return failure();
    ArrayAttr outputViews = maybeViews.value();

    SmallVector<Value> indices{gridCoords.g_block, gridCoords.m_block,
                               gridCoords.n_block};

    // Load the primary input tile, then mark it as the full-tensor output tile
    // so the existing fusion/store path can consume it.
    auto tileType = RankedTensorType::get({mPerBlock, nPerBlock}, elemType);
    auto loadMarker = LoadMarkerOp::create(b, loc, tileType, input, outputViews,
                                           indices, rock::CacheModifier::NONE);
    auto storeMarker =
        StoreMarkerOp::create(b, loc, op.getOutput().getType(),
                              loadMarker.getResult(), outputViews, indices);

    b.replaceOp(op, storeMarker.getResult());
    return success();
  }
};
} // end anonymous namespace

void RockGridwiseElementwiseToBlockwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::GridwiseElementwiseOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         triton::TritonDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseElementwiseRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
