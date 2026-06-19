//===------- GridwiseElementwiseToBlockwise.cpp
//----------------------------===//
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
//===----------------------------------------------------------------------===//
//
// This pass tiles standalone elementwise fusion kernels by inserting
// load_marker and store_marker ops. Only inputs tracing to kernel block
// arguments get load_markers; only outputs flowing to rock.store get
// store_markers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

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

private:
  LogicalResult processKernel(func::FuncOp func);
};

} // namespace

/// Build tiling views: a single Unmerge that maps from (bid, tile_elem)
/// down to the flat padded element index. Padding is already applied by
/// ElementwiseToGridwise, so the source tensor size equals gridSize * tileSize.
static ArrayAttr buildTilingViews(OpBuilder &b, Location loc, int64_t gridSize,
                                  int64_t tileSize) {
  TopDownTMBuilder builder(b, {"bid", "tile_elem"}, {gridSize, tileSize}, loc);
  builder.unmerge("elem", 0, {"bid", "tile_elem"}, {gridSize, tileSize});
  return b.getArrayAttr({builder.get()});
}

LogicalResult
RockGridwiseElementwiseToBlockwisePass::processKernel(func::FuncOp func) {
  if (!isElementwiseKernel(func))
    return success();

  auto elemParamsAttr =
      func->getAttrOfType<ElementwiseParamsAttr>("perf_config");
  if (!elemParamsAttr)
    return func.emitError("missing perf_config for elementwise kernel");

  auto gridSizeAttr =
      func->getAttrOfType<IntegerAttr>(rock::GridSizeAttr::getMnemonic());
  if (!gridSizeAttr)
    return func.emitError("missing rock.grid_size for elementwise kernel");

  int64_t tileSize = elemParamsAttr.getTileSize();
  int64_t gridSize = gridSizeAttr.getInt();

  OpBuilder b(func.getContext());
  Location loc = func.getLoc();

  // Insert GetProgramIdOp at the start of the function.
  b.setInsertionPointToStart(&func.getBody().front());
  Value bid = triton::GetProgramIdOp::create(b, loc, triton::ProgramIDDim::X);

  ArrayAttr tilingViews = buildTilingViews(b, loc, gridSize, tileSize);

  // Insert load_markers on inputs and propagate tile types through
  // the fusion chain.
  SetVector<Value> elementwiseInputs = getElementwiseKernelInputs(func);
  for (Value input : elementwiseInputs) {
    auto vType = cast<RankedTensorType>(input.getType());
    auto tileType = RankedTensorType::get({tileSize}, vType.getElementType());
    OpBuilder::InsertionGuard guard(b);
    if (auto *defOp = input.getDefiningOp())
      b.setInsertionPointAfter(defOp);
    else
      b.setInsertionPointAfter(bid.getDefiningOp());
    auto loadMarker = LoadMarkerOp::create(b, loc, tileType, input, tilingViews,
                                           ValueRange{bid});
    propagateOutputType(input, loadMarker.getResult());
  }

  // Insert store_markers on store sources.
  // The store_marker maps tile-sized data back to the full padded space.
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

  return success();
}

void RockGridwiseElementwiseToBlockwisePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (failed(processKernel(func)))
    return signalPassFailure();
}
