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
// it adds a new output argument, inserts a rock.store, removes the return
// value from the function signature, and updates all callers accordingly.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/BitVector.h"
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

// Returns true if the operation should be followed during flood-fill.
// This includes fusion ops (arith/math elementwise), linalg.generic,
// rock.transform, and rock.reduce.
static bool isFloodFillOp(Operation *op) {
  return isFusionOp(op) || isa<TransformOp>(op) || isa<ReduceOp>(op) ||
         isa<linalg::GenericOp>(op);
}

// Forward flood-fill from a FusionRoot result.
// Collects all reachable values into `chainSet`. If a rock.store is found
// whose source is in the chain, sets `hasExistingStore` to true.
static void floodFillFromRoot(Value root, DenseSet<Value> &chainSet,
                              bool &hasExistingStore) {
  SmallVector<Value> worklist;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!chainSet.insert(v).second)
      continue;

    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();

      // Check for existing rock.store
      if (auto storeOp = dyn_cast<StoreOp>(owner)) {
        if (use.getOperandNumber() == 0) { // source operand
          hasExistingStore = true;
          return;
        }
        continue;
      }

      // Follow through fusion-like ops
      if (isFloodFillOp(owner)) {
        for (Value res : owner->getResults())
          worklist.push_back(res);
      }
    }
  }
}

// Information about a return value that needs a rock.store.
struct ReturnStoreInfo {
  unsigned returnIndex;    // Index in the func.return operand list
  Value returnOperand;     // The value being returned
  BlockArgument storeArg;  // New output arg (set by addOutputArguments)
};

// For each FusionRoot, flood-fill forward and identify which return operands
// need stores. Each FusionRoot result must either already have a rock.store or
// trace forward to a func.return operand. If neither is true, emits an error.
static FailureOr<SmallVector<ReturnStoreInfo>>
identifyReturnStores(ArrayRef<Operation *> fusionRoots,
                     func::ReturnOp returnOp) {
  SmallVector<ReturnStoreInfo> storeInfos;
  for (Operation *rootOp : fusionRoots) {
    for (Value rootResult : rootOp->getResults()) {
      DenseSet<Value> chainSet;
      bool hasExistingStore = false;
      floodFillFromRoot(rootResult, chainSet, hasExistingStore);

      if (hasExistingStore) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Root already has store, skipping: " << *rootOp << "\n");
        continue;
      }

      // Check which return operands are in this chain
      bool foundReturnOperand = false;
      for (unsigned i = 0, e = returnOp.getNumOperands(); i < e; ++i) {
        Value retVal = returnOp.getOperand(i);
        if (!chainSet.contains(retVal))
          continue;

        foundReturnOperand = true;
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
  return storeInfos;
}

// Add a new output argument to the kernel for each return value that needs a
// store. Transfers rock.prefill attributes from results to new arguments.
// Returns the types of the newly added arguments.
static FailureOr<SmallVector<Type>>
addOutputArguments(func::FuncOp funcOp,
                   SmallVectorImpl<ReturnStoreInfo> &storeInfos) {
  SmallVector<Type> newArgTypes;
  for (auto &info : storeInfos) {
    Type retType = info.returnOperand.getType();
    unsigned newArgIdx = funcOp.getNumArguments();

    if (failed(funcOp.insertArgument(newArgIdx, retType,
                                     /*argAttrs=*/DictionaryAttr(),
                                     funcOp.getLoc())))
      return failure();
    info.storeArg = funcOp.getArgument(newArgIdx);
    newArgTypes.push_back(retType);

    LLVM_DEBUG(llvm::dbgs()
               << "Return " << info.returnIndex
               << ": created output arg " << newArgIdx
               << " with type " << retType << "\n");

    // Transfer rock.prefill from result attr to the new argument attr
    if (Attribute prefillAttr = funcOp.getResultAttr(
            info.returnIndex, PrefillAttr::getMnemonic())) {
      funcOp.setArgAttr(newArgIdx, PrefillAttr::getMnemonic(), prefillAttr);
    }
  }
  return newArgTypes;
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
    StoreOp::create(builder, returnOp.getLoc(),
                    /*resultType=*/info.returnOperand.getType(),
                    /*source=*/info.returnOperand,
                    /*dest=*/info.storeArg,
                    /*storeMethod=*/storeMethodAttr);
  }
}

// Remove stored values from the func.return operands and erase the
// corresponding result types from the function's type signature.
static LogicalResult
removeStoredReturns(func::FuncOp funcOp, func::ReturnOp returnOp,
                    ArrayRef<ReturnStoreInfo> storeInfos,
                    unsigned origNumResults) {
  DenseSet<unsigned> storedIndices;
  for (const auto &info : storeInfos)
    storedIndices.insert(info.returnIndex);

  // Remove return operands in reverse order for index stability
  SmallVector<unsigned> sortedStoredIndices(storedIndices.begin(),
                                            storedIndices.end());
  llvm::sort(sortedStoredIndices);
  for (int i = sortedStoredIndices.size() - 1; i >= 0; --i)
    returnOp.getOperation()->eraseOperand(sortedStoredIndices[i]);

  // Remove function result types (handles type and result attributes)
  llvm::BitVector resultsToRemove(origNumResults);
  for (unsigned idx : storedIndices)
    resultsToRemove.set(idx);
  return funcOp.eraseResults(resultsToRemove);
}

// Update all callers of a kernel function after adding output arguments and
// removing stored return values. New output operands are filled with
// tensor.empty, and uses of removed call results are replaced with the
// corresponding tensor.empty operand.
static LogicalResult
updateCallers(func::FuncOp funcOp, ModuleOp moduleOp, unsigned origNumResults,
              ArrayRef<ReturnStoreInfo> storeInfos,
              ArrayRef<Type> newArgTypes) {
  StringRef kernelName = funcOp.getName();

  // Build a map from removed return indices to their store arg indices.
  DenseSet<unsigned> removedReturnIndices;
  DenseMap<unsigned, unsigned> returnToArgIdx;
  for (const auto &info : storeInfos) {
    removedReturnIndices.insert(info.returnIndex);
    returnToArgIdx[info.returnIndex] = info.storeArg.getArgNumber();
  }

  // Walk the module for all func.call ops that call this kernel
  SmallVector<func::CallOp> callsToUpdate;
  moduleOp.walk([&](func::CallOp callOp) {
    if (callOp.getCallee() == kernelName)
      callsToUpdate.push_back(callOp);
  });

  for (func::CallOp callOp : callsToUpdate) {
    OpBuilder builder(callOp);

    // Build new operand list: original operands + tensor.empty for new args
    SmallVector<Value> newOperands(callOp.getOperands());
    for (Type argType : newArgTypes) {
      auto tensorType = cast<RankedTensorType>(argType);
      Value empty = tensor::EmptyOp::create(builder, callOp.getLoc(),
                                            tensorType.getShape(),
                                            tensorType.getElementType());
      newOperands.push_back(empty);
    }

    // Compute new result types
    SmallVector<Type> newResultTypes;
    for (unsigned i = 0; i < origNumResults; ++i) {
      if (!removedReturnIndices.contains(i))
        newResultTypes.push_back(callOp.getResultTypes()[i]);
    }

    // Create a new call with updated operands and result types
    auto newCallOp = func::CallOp::create(
        builder, callOp.getLoc(), callOp.getCallee(),
        newResultTypes, newOperands);

    // Replace uses of old call results:
    // - Removed results: replace with the tensor.empty output operand
    // - Kept results: replace with the new call's result at adjusted index
    unsigned newResIdx = 0;
    for (unsigned i = 0; i < origNumResults; ++i) {
      if (removedReturnIndices.contains(i)) {
        // This result was stored, replace uses with the output operand
        unsigned argIdx = returnToArgIdx[i];
        callOp.getResult(i).replaceAllUsesWith(newOperands[argIdx]);
      } else {
        // This result is kept, replace with the new call's result
        callOp.getResult(i).replaceAllUsesWith(newCallOp.getResult(newResIdx));
        newResIdx++;
      }
    }

    callOp.erase();
  }

  return success();
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

  auto storeInfosOr = identifyReturnStores(fusionRoots, returnOp);
  if (failed(storeInfosOr))
    return failure();
  auto &storeInfos = *storeInfosOr;

  if (storeInfos.empty())
    return success();

  unsigned origNumResults = funcOp.getNumResults();

  auto newArgTypesOr = addOutputArguments(funcOp, storeInfos);
  if (failed(newArgTypesOr))
    return failure();

  insertStoreOps(funcOp, returnOp, storeInfos);

  if (failed(removeStoredReturns(funcOp, returnOp, storeInfos, origNumResults)))
    return failure();

  return updateCallers(funcOp, moduleOp, origNumResults, storeInfos,
                       *newArgTypesOr);
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
