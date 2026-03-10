//===---------------------- InsertOutputStores.cpp ------------------------===//
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
// This pass inserts rock.store ops for kernel functions that return values
// but lack explicit stores. For each return value from a FusionRoot chain,
// it adds a new output argument, inserts a rock.store, and updates the
// func.return to use the store results.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKINSERTOUTPUTSTORESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-insert-output-stores"

using namespace mlir;
using namespace mlir::rock;

namespace {

// Forward flood-fill from a FusionRoot result.
// Collects all reachable values into `chainSet`. For any rock.store found
// whose source is in the chain, the store result is added to `storedValues`
// (and to `chainSet`) but the flood-fill does not follow past the store.
// This lets the caller distinguish return operands that already have stores
// from those that still need one.
static LogicalResult floodFillFromRoot(Value root, DenseSet<Value> &chainSet,
                                       DenseSet<Value> &storedValues) {
  SmallVector<Value> worklist;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!chainSet.insert(v).second)
      continue;

    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();

      // For an existing rock.store whose source is in the chain, record its
      // result as already-stored and add it to the chain (so it can match a
      // return operand), but don't flood-fill past the store.
      if (auto storeOp = dyn_cast<StoreOp>(owner)) {
        if (use.getOperandNumber() == 0) { // source operand
          Value storeResult = storeOp.getResult();
          chainSet.insert(storeResult);
          storedValues.insert(storeResult);
        }
        continue;
      }

      // Follow through fusion-like ops
      if (isForwardTraceOp(owner)) {
        for (Value res : owner->getResults())
          worklist.push_back(res);
        continue;
      }

      // func.return is an expected terminal use.
      if (isa<func::ReturnOp>(owner))
        continue;

      // Any other use of a FusionRoot chain value is unexpected.
      return owner->emitError(
          "unexpected use of FusionRoot chain value by ")
          << owner->getName();
    }
  }
  return success();
}

// Information about a return value that needs a rock.store.
struct ReturnStoreInfo {
  unsigned returnIndex;    // Index in the func.return operand list
  Value returnOperand;     // The value being returned
  BlockArgument storeArg;  // New output arg (set by addOutputArguments)
};

// For each FusionRoot, flood-fill forward and identify which return operands
// need stores. A return operand that is already the result of a rock.store is
// left alone; all other return operands reachable from the root get a new
// store. If a root result has neither an existing store nor a path to a
// func.return operand, an error is emitted. After processing all roots, any
// return operand not covered by a root chain must be a block argument
// passthrough; otherwise it is an error.
static FailureOr<SmallVector<ReturnStoreInfo>>
identifyReturnStores(ArrayRef<Operation *> fusionRoots,
                     func::ReturnOp returnOp) {
  SmallVector<ReturnStoreInfo> storeInfos;

  // Track which return operand indices are covered by some FusionRoot chain.
  DenseSet<unsigned> coveredReturnIndices;

  for (Operation *rootOp : fusionRoots) {
    for (Value rootResult : rootOp->getResults()) {
      DenseSet<Value> chainSet;
      DenseSet<Value> storedValues;
      if (failed(floodFillFromRoot(rootResult, chainSet, storedValues)))
        return failure();

      // Check which return operands are in this chain
      bool foundReturnOperand = false;
      for (unsigned i = 0, e = returnOp.getNumOperands(); i < e; ++i) {
        Value retVal = returnOp.getOperand(i);
        if (!chainSet.contains(retVal))
          continue;

        foundReturnOperand = true;
        coveredReturnIndices.insert(i);

        // Skip return operands that are already the result of a rock.store.
        if (storedValues.contains(retVal)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Return " << i
                     << " already has store, skipping\n");
          continue;
        }

        ReturnStoreInfo info;
        info.returnIndex = i;
        info.returnOperand = retVal;
        storeInfos.push_back(info);
      }

      // A FusionRoot result with no store and no path to a return is an error.
      if (!foundReturnOperand) {
        return rootOp->emitError(
            "FusionRoot result has no rock.store and does not "
            "reach a function return");
      }
    }
  }

  // Verify every return operand is either covered by a FusionRoot chain or is
  // a block argument passthrough. An uncovered non-argument return value would
  // mean a computed result is silently dropped without a store.
  for (unsigned i = 0, e = returnOp.getNumOperands(); i < e; ++i) {
    if (coveredReturnIndices.contains(i))
      continue;
    Value retVal = returnOp.getOperand(i);
    if (isa<BlockArgument>(retVal))
      continue;
    return returnOp.emitError("return operand ")
           << i << " is not a block argument and is not reachable from any "
           << "FusionRoot";
  }

  return storeInfos;
}

// Add a new output argument to the kernel for each return value that needs a
// store. Transfers all result attributes (e.g. rock.prefill) to new arguments.
static LogicalResult
addOutputArguments(func::FuncOp funcOp,
                   SmallVectorImpl<ReturnStoreInfo> &storeInfos) {
  // Sort by returnIndex so output arguments are added in the same order as the
  // original return operands: return %a, %b -> func(inputs..., %a, %b).
  llvm::sort(storeInfos,
             [](const ReturnStoreInfo &a, const ReturnStoreInfo &b) {
               return a.returnIndex < b.returnIndex;
             });

  for (auto &info : storeInfos) {
    Type retType = info.returnOperand.getType();
    unsigned newArgIdx = funcOp.getNumArguments();

    // Transfer all result attributes (e.g. rock.prefill) to the new argument.
    DictionaryAttr resultAttrs = funcOp.getResultAttrDict(info.returnIndex);

    if (failed(funcOp.insertArgument(newArgIdx, retType,
                                     resultAttrs ? resultAttrs
                                                 : DictionaryAttr(),
                                     funcOp.getLoc())))
      return failure();
    info.storeArg = funcOp.getArgument(newArgIdx);

    LLVM_DEBUG(llvm::dbgs()
               << "Return " << info.returnIndex
               << ": created output arg " << newArgIdx
               << " with type " << retType << "\n");
  }
  return success();
}

// Insert rock.store ops just before func.return for each identified return
// value, storing the result into the corresponding output argument.
static void insertStoreOps(func::FuncOp funcOp, func::ReturnOp returnOp,
                            ArrayRef<ReturnStoreInfo> storeInfos) {
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPoint(returnOp);

  for (const auto &info : storeInfos) {
    assert(info.storeArg && "store destination must be set by now");

    auto storeMethodAttr =
        builder.getAttr<StoreMethodAttr>(StoreMethod::Set);
    auto storeOp = StoreOp::create(builder, returnOp.getLoc(),
                                   /*result=*/info.returnOperand.getType(),
                                   /*source=*/info.returnOperand,
                                   /*dest=*/info.storeArg,
                                   /*storeMethod=*/storeMethodAttr);
    returnOp.setOperand(info.returnIndex, storeOp.getResult());
  }
}


struct RockInsertOutputStoresPass
    : public rock::impl::RockInsertOutputStoresPassBase<
          RockInsertOutputStoresPass> {
  void runOnOperation() override;

private:
  LogicalResult processKernel(func::FuncOp funcOp, ModuleOp moduleOp);
};

} // namespace

LogicalResult
RockInsertOutputStoresPass::processKernel(func::FuncOp funcOp,
                                          ModuleOp moduleOp) {
  SmallVector<Operation *> fusionRoots;
  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::rock::FusionRoot>())
      fusionRoots.push_back(op);
  });

  if (fusionRoots.empty())
    return success();

  auto returnOp = cast<func::ReturnOp>(funcOp.getBody().back().getTerminator());

  auto maybeStoreInfos = identifyReturnStores(fusionRoots, returnOp);
  if (failed(maybeStoreInfos))
    return failure();
  auto &storeInfos = *maybeStoreInfos;

  if (storeInfos.empty())
    return success();

  // This pass runs before wrapper generation, so the kernel must not have
  // any call sites yet.
  if (!funcOp.symbolKnownUseEmpty(moduleOp))
    return funcOp.emitError(
        "kernel has callers; InsertOutputStores expects no call sites");

  if (failed(addOutputArguments(funcOp, storeInfos)))
    return failure();

  insertStoreOps(funcOp, returnOp, storeInfos);

  return success();
}

void RockInsertOutputStoresPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Collect kernel functions
  SmallVector<func::FuncOp> kernelFuncs;
  moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp->hasAttr(KernelAttr::getMnemonic()))
      kernelFuncs.push_back(funcOp);
  });

  for (func::FuncOp funcOp : kernelFuncs) {
    LLVM_DEBUG(llvm::dbgs()
               << "Processing kernel: " << funcOp.getName() << "\n");
    if (failed(processKernel(funcOp, moduleOp)))
      return signalPassFailure();
  }
}
