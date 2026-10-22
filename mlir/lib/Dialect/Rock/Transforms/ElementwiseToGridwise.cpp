//===-------------- ElementwiseToGridwise.cpp
//------------------------------===//
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
// This pass computes the grid_size for standalone elementwise fusion kernels
// and flattens + pads each fusion operand and store destination to 1D padded
// tensors of size gridSize * tileSize.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKELEMENTWISETOGRIDWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-elementwise-to-gridwise"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct RockElementwiseToGridwisePass
    : public rock::impl::RockElementwiseToGridwisePassBase<
          RockElementwiseToGridwisePass> {
  void runOnOperation() override;
};

} // namespace

/// Merge a value to 1D, then pad to paddedSize.
static Value flattenAndPad(OpBuilder &b, Location loc, Value v,
                           int64_t paddedSize) {
  auto vType = cast<RankedTensorType>(v.getType());

  OpBuilder::InsertionGuard guard(b);
  if (auto *defOp = v.getDefiningOp())
    b.setInsertionPointAfter(defOp);
  else
    b.setInsertionPointToStart(&v.getParentRegion()->getBlocks().front());

  // Step 1: Merge to 1D if multi-dimensional.
  Value flat = v;
  if (vType.getRank() > 1) {
    SmallVector<SmallString<4>> nameStorage(vType.getRank());
    SmallVector<StringRef> dimNames;
    for (int64_t i = 0; i < vType.getRank(); i++) {
      nameStorage[i] = ("d" + Twine(i)).str();
      dimNames.push_back(nameStorage[i]);
    }
    BottomUpTMBuilder merger(b, dimNames, vType.getShape(), loc);
    merger.merge("flat", 0, dimNames);
    flat = TransformOp::create(b, loc, v, merger.get());
  }

  // Step 2: Pad to paddedSize if needed.
  int64_t numElems = cast<RankedTensorType>(flat.getType()).getDimSize(0);
  if (numElems != paddedSize) {
    int64_t padAmount = paddedSize - numElems;
    BottomUpTMBuilder padder(b, {"flat"}, {numElems}, loc);
    padder.pad("padded", "flat", 0, padAmount);
    flat = TransformOp::create(b, loc, flat, padder.get());
  }

  return flat;
}

void RockElementwiseToGridwisePass::runOnOperation() {
  func::FuncOp func = getOperation();

  if (!isElementwiseKernel(func))
    return;

  auto elemParamsAttr =
      func->getAttrOfType<ElementwiseParamsAttr>("perf_config");
  if (!elemParamsAttr) {
    func.emitError("missing perf_config for elementwise kernel");
    signalPassFailure();
  }

  int64_t tileSize = elemParamsAttr.getTileSize();

  // Find the biggest tensor among fusion op operands and results.
  int64_t maxElements = 0;
  func.walk([&](Operation *op) {
    if (!isFusionOp(op))
      return;
    for (Value operand : op->getOperands()) {
      if (auto shaped = dyn_cast<ShapedType>(operand.getType()))
        if (shaped.hasStaticShape())
          maxElements = std::max(maxElements, shaped.getNumElements());
    }
    for (Value result : op->getResults()) {
      if (auto shaped = dyn_cast<ShapedType>(result.getType()))
        if (shaped.hasStaticShape())
          maxElements = std::max(maxElements, shaped.getNumElements());
    }
  });

  if (maxElements == 0) {
    func.emitError("elementwise kernel has no shaped arguments");
    return signalPassFailure();
  }

  int64_t gridSize = llvm::divideCeil(maxElements, tileSize);

  int64_t paddedSize = gridSize * tileSize;

  LLVM_DEBUG(llvm::dbgs() << "Elementwise: maxElements=" << maxElements
                          << " tileSize=" << tileSize
                          << " gridSize=" << gridSize << "\n");

  OpBuilder b(func.getContext());
  Location loc = func.getLoc();
  func->setAttr(rock::GridSizeAttr::getMnemonic(),
                b.getI32IntegerAttr(gridSize));

  SetVector<Value> elementwiseInputs = getElementwiseKernelInputs(func);

  // Step 2: Flatten + pad each input, then propagateOutputType to update
  // downstream fusion ops' operands and result types through the chain.
  for (Value input : elementwiseInputs) {
    Value padded = flattenAndPad(b, loc, input, paddedSize);
    propagateOutputType(input, padded);
  }

  // Flatten + pad each store destination; source is already padded.
  // The store result type is NOT changed — it preserves the original
  // buffer type, so the function return type stays correct.
  func.walk([&](StoreOp storeOp) {
    storeOp.getDestMutable().assign(
        flattenAndPad(b, loc, storeOp.getDest(), paddedSize));
  });
}
