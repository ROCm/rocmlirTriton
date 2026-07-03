//===- ElementwiseToGridwise.cpp - Rock pure-elementwise kernel root -----===//
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
// This pass is the pure-elementwise analogue of rock-gemm-to-gridwise. A
// kernel that has no FusionRoot (no gemm/conv/attention) but whose return value
// is produced by an arith/math fusion chain cannot be lowered today: there is
// no root to drive the output tiling, so InsertOutputStores leaves it without a
// rock.store and the ToBlockwise passes find no StoreMarkerOp.
//
// For such a kernel this pass:
//   1. picks the "primary" elementwise input (operand 0 of the fusion op that
//      feeds the return) and wraps it in a rock.gridwise_elementwise root,
//   2. sets the func-level rock.block_size / rock.grid_size, and
//   3. inserts the output argument + rock.store for the return value.
//
// rock-gridwise-elementwise-to-blockwise then lowers the gridwise_elementwise
// into a load_marker + store_marker output tile, after which the existing
// insert-output-fusion-loads / lower-loads / lower-stores path takes over,
// treating every remaining elementwise input as a normal output-fusion load.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
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
} // end anonymous namespace

// Trace backwards from `v` through rock.transform views and return the first
// elementwise (arith/math) fusion op encountered, or nullptr.
static Operation *findFusionOp(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (isFusionOp(def))
      return def;
    if (auto tOp = dyn_cast<TransformOp>(def)) {
      v = tOp.getInput();
      continue;
    }
    break;
  }
  return nullptr;
}

void RockElementwiseToGridwisePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  // Only handle kernels that have no other FusionRoot.
  bool hasRoot = false;
  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::rock::FusionRoot>())
      hasRoot = true;
  });
  if (hasRoot)
    return;

  // This pass must run before any rock.store ops are inserted.
  bool hasStore =
      funcOp.walk([](StoreOp) { return WalkResult::interrupt(); })
          .wasInterrupted();
  if (hasStore)
    return;

  auto returnOp =
      cast<func::ReturnOp>(funcOp.getBody().back().getTerminator());
  // For now we support a single returned value (the common slice+combine case).
  if (returnOp.getNumOperands() != 1)
    return;

  Value retVal = returnOp.getOperand(0);
  Operation *fusionOp = findFusionOp(retVal);
  if (!fusionOp)
    return; // pure passthrough or unsupported chain: leave untouched

  // Pick the primary input: the first operand that is a ranked tensor matching
  // the gemm-style (G x M x N) output space.
  Value primary;
  for (Value operand : fusionOp->getOperands()) {
    auto ty = dyn_cast<RankedTensorType>(operand.getType());
    if (ty && ty.getRank() == 3) {
      primary = operand;
      break;
    }
  }
  if (!primary)
    return;

  auto primTy = cast<RankedTensorType>(primary.getType());
  ArrayRef<int64_t> shape = primTy.getShape();
  int64_t G = shape[0], M = shape[1], N = shape[2];

  // Linear output tiling: one (M x N) tile per group => grid = G.
  int64_t mPerBlock = M;
  int64_t nPerBlock = N;
  int64_t gridSize = G;

  StringAttr archAttr = getArchValue(funcOp);
  if (!archAttr) {
    funcOp.emitError(
        "rock-elementwise-to-gridwise: kernel is missing rock.arch");
    return signalPassFailure();
  }
  int64_t blockSize = rock::getWaveSize(archAttr.getValue());

  OpBuilder b(fusionOp);
  Location loc = fusionOp->getLoc();

  // 1. Wrap the primary input in a gridwise_elementwise root.
  auto gpOp = GridwiseElementwiseOp::create(b, loc, primTy, primary,
                                            b.getI64IntegerAttr(mPerBlock),
                                            b.getI64IntegerAttr(nPerBlock));
  fusionOp->replaceUsesOfWith(primary, gpOp.getResult());

  // 2. Set func-level launch parameters.
  funcOp->setAttr(rock::GridSizeAttr::getMnemonic(),
                  b.getI32IntegerAttr(gridSize));
  funcOp->setAttr(rock::BlockSizeAttr::getMnemonic(),
                  b.getI32IntegerAttr(blockSize));

  // 3. Insert the output argument + rock.store for the return value.
  Type retType = retVal.getType();
  unsigned newArgIdx = funcOp.getNumArguments();
  if (failed(funcOp.insertArgument(newArgIdx, retType, DictionaryAttr(),
                                   funcOp.getLoc())))
    return signalPassFailure();
  BlockArgument storeArg = funcOp.getArgument(newArgIdx);

  OpBuilder sb(returnOp);
  auto storeMethodAttr = sb.getAttr<StoreMethodAttr>(StoreMethod::Set);
  auto storeOp = StoreOp::create(sb, returnOp.getLoc(), /*result=*/retType,
                                 /*source=*/retVal, /*dest=*/storeArg,
                                 storeMethodAttr);
  returnOp.setOperand(0, storeOp.getResult());
}
