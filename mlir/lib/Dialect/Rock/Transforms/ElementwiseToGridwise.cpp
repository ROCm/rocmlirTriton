//===- ElementwiseToGridwise.cpp -----------------------------------------===//
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

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKELEMENTWISETOGRIDWISEPASS
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

/// Merge `v` to 1D then pad to `paddedSize`.
static Value flattenAndPad(OpBuilder &b, Location loc, Value v,
                           int64_t paddedSize) {
  auto vType = cast<RankedTensorType>(v.getType());

  OpBuilder::InsertionGuard guard(b);
  if (auto *defOp = v.getDefiningOp())
    b.setInsertionPointAfter(defOp);
  else
    b.setInsertionPointToStart(&v.getParentRegion()->getBlocks().front());

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

  int64_t numElems = cast<RankedTensorType>(flat.getType()).getDimSize(0);
  if (numElems != paddedSize) {
    int64_t padAmount = paddedSize - numElems;
    BottomUpTMBuilder padder(b, {"flat"}, {numElems}, loc);
    padder.pad("padded", "flat", 0, padAmount);
    flat = TransformOp::create(b, loc, flat, padder.get());
  }

  return flat;
}

struct RockElementwiseToGridwisePass
    : public rock::impl::RockElementwiseToGridwisePassBase<
          RockElementwiseToGridwisePass> {
  void runOnOperation() override;
};

} // namespace

void RockElementwiseToGridwisePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!isStandaloneElementwiseKernel(func))
    return;

  int64_t tileSize = getTileSize(func);
  if (tileSize <= 0) {
    func.emitError("rock.block_size must be positive for elementwise kernels");
    return signalPassFailure();
  }

  int64_t maxElements = 0;
  func.walk([&](Operation *op) {
    if (!isFusionOp(op))
      return;
    for (Value operand : op->getOperands()) {
      if (auto shaped = dyn_cast<ShapedType>(operand.getType());
          shaped && shaped.hasStaticShape())
        maxElements = std::max(maxElements, shaped.getNumElements());
    }
    for (Value result : op->getResults()) {
      if (auto shaped = dyn_cast<ShapedType>(result.getType());
          shaped && shaped.hasStaticShape())
        maxElements = std::max(maxElements, shaped.getNumElements());
    }
  });

  if (maxElements == 0)
    return;

  int64_t gridSize = llvm::divideCeil(maxElements, tileSize);
  int64_t paddedSize = gridSize * tileSize;

  OpBuilder b(func.getContext());
  Location loc = func.getLoc();
  func->setAttr(rock::GridSizeAttr::getMnemonic(), b.getI32IntegerAttr(gridSize));

  llvm::SetVector<Value> elementwiseInputs = getElementwiseKernelInputs(func);
  for (Value input : elementwiseInputs) {
    Value padded = flattenAndPad(b, loc, input, paddedSize);
    propagateOutputType(input, padded);
  }

  func.walk([&](StoreOp storeOp) {
    storeOp.getDestMutable().assign(
        flattenAndPad(b, loc, storeOp.getDest(), paddedSize));
  });
}
