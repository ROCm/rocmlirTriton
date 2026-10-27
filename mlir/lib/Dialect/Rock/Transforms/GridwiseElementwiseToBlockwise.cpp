//===- GridwiseElementwiseToBlockwise.cpp -------------------------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGRIDWISEELEMENTWISETOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

namespace {

static bool isStandaloneElementwiseKernel(func::FuncOp funcOp) {
  return funcOp->hasAttr(rock::KernelAttr::getMnemonic()) &&
         funcOp->hasAttr(rock::ElementwiseAttr::getMnemonic());
}

static int64_t getTileSize(func::FuncOp func) {
  if (auto blockSize =
          func->getAttrOfType<IntegerAttr>(rock::BlockSizeAttr::getMnemonic()))
    return blockSize.getInt();
  return 256;
}

static llvm::SetVector<Value> getElementwiseKernelInputs(func::FuncOp funcOp) {
  llvm::SetVector<Value> leafInputs;
  funcOp.walk([&](Operation *op) {
    if (!isFusionOp(op))
      return;
    for (Value operand : op->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      SmallVector<TransformMapAttr> transforms;
      auto [underlying, _] = rock::untransform(operand, transforms);
      if (!isa<BlockArgument>(underlying) &&
          !underlying.getDefiningOp<arith::ConstantOp>())
        continue;
      leafInputs.insert(operand);
    }
  });
  return leafInputs;
}

static ArrayAttr buildTilingViews(OpBuilder &b, Location loc, int64_t gridSize,
                                  int64_t tileSize) {
  TopDownTMBuilder builder(b, {"bid", "tile_elem"}, {gridSize, tileSize}, loc);
  builder.unmerge("elem", 0, {"bid", "tile_elem"}, {gridSize, tileSize});
  return b.getArrayAttr({builder.get()});
}

struct RockGridwiseElementwiseToBlockwisePass
    : public rock::impl::RockGridwiseElementwiseToBlockwisePassBase<
          RockGridwiseElementwiseToBlockwisePass> {
  void runOnOperation() override;
};

} // namespace

void RockGridwiseElementwiseToBlockwisePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!isStandaloneElementwiseKernel(func))
    return;

  auto gridSizeAttr =
      func->getAttrOfType<IntegerAttr>(rock::GridSizeAttr::getMnemonic());
  if (!gridSizeAttr) {
    func.emitError("missing rock.grid_size for elementwise kernel");
    return signalPassFailure();
  }

  int64_t tileSize = getTileSize(func);
  int64_t gridSize = gridSizeAttr.getInt();
  if (tileSize <= 0 || gridSize <= 0) {
    func.emitError("invalid block size/grid size for elementwise kernel");
    return signalPassFailure();
  }

  OpBuilder b(func.getContext());
  Location loc = func.getLoc();

  b.setInsertionPointToStart(&func.getBody().front());
  Value bid = triton::GetProgramIdOp::create(b, loc, triton::ProgramIDDim::X);

  ArrayAttr tilingViews = buildTilingViews(b, loc, gridSize, tileSize);

  llvm::SetVector<Value> elementwiseInputs = getElementwiseKernelInputs(func);
  for (Value input : elementwiseInputs) {
    auto vType = cast<RankedTensorType>(input.getType());
    auto tileType = RankedTensorType::get({tileSize}, vType.getElementType());
    OpBuilder::InsertionGuard guard(b);
    if (auto *defOp = input.getDefiningOp())
      b.setInsertionPointAfter(defOp);
    else
      b.setInsertionPointAfter(bid.getDefiningOp());

    auto loadMarker = LoadMarkerOp::create(b, loc, tileType, input, tilingViews,
                                           ValueRange{bid},
                                           rock::CacheModifier::CS);
    propagateOutputType(input, loadMarker.getResult());
  }

  int64_t paddedSize = gridSize * tileSize;
  func.walk([&](StoreOp storeOp) {
    Value source = storeOp.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto paddedType =
        RankedTensorType::get({paddedSize}, sourceType.getElementType());
    b.setInsertionPoint(storeOp);
    auto storeMarker = StoreMarkerOp::create(b, loc, paddedType, source,
                                             tilingViews, ValueRange{bid});
    storeOp.getSourceMutable().assign(storeMarker.getResult());
  });
}
