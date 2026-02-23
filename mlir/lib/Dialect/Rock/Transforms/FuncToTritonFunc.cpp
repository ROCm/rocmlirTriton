//===- FuncToTritonFunc.cpp - Convert func.func to tt.func for Triton -----===//
//
// Copyright 2026 The MLIR Authors.
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
// This pass transforms Rock kernel functions from func.func to Triton's tt.func.
// It converts tensor arguments to Triton pointer types (!tt.ptr), eliminates the
// pointer extraction chain (rock.extract_ptr), converts arith.addi on pointer
// tensors to tt.addptr, and sets up pointer attributes for optimization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFUNCTOTRITONFUNCPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-func-to-triton-func"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::triton;
using namespace mlir::arith;

namespace {

/// Helper to determine if a type is a tensor with pointer element type
static bool isTensorOfPointers(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return false;
}

struct RockFuncToTritonFuncPass
    : public rock::impl::RockFuncToTritonFuncPassBase<RockFuncToTritonFuncPass> {
  void runOnOperation() override;

private:
  /// Map from original i32 values to converted tt.ptr values
  IRMapping valueMapping;

  /// Process a single kernel function (convert to tt.func)
  void processFunction(func::FuncOp funcOp);
};

} // end anonymous namespace

void RockFuncToTritonFuncPass::processFunction(func::FuncOp funcOp) {
  valueMapping.clear();
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);

  // Step 1: Find all rock.extract_ptr patterns and collect info
  // Pattern: block_arg (tensor) -> rock.extract_ptr -> tt.splat
  struct ArgConversionInfo {
    unsigned argIndex;
    Type elementType;
    RankedTensorType tensorType; // Original tensor type for size calculation
    SmallVector<Value> valuesToReplace; // extract_ptr results to replace with block arg
    // Ops in the chain that need to be erased (in order: splats, extract_ptr)
    SmallVector<triton::SplatOp> oldSplatOps;
    rock::ExtractPtrOp extractPtrOp;
  };
  SmallVector<ArgConversionInfo> argsToConvert;

  funcOp.walk([&](rock::ExtractPtrOp extractPtrOp) {
    Value tensorOperand = extractPtrOp.getSource();

    // Check if the source is a block argument (tensor)
    auto blockArg = dyn_cast<BlockArgument>(tensorOperand);
    if (!blockArg)
      return;

    auto tensorType = dyn_cast<RankedTensorType>(tensorOperand.getType());
    if (!tensorType)
      return;

    // The result of extract_ptr is i32, which is what we need to replace
    if (extractPtrOp.getResult().getType().isInteger(32)) {
      // Found the pattern - record it
      ArgConversionInfo info;
      info.argIndex = blockArg.getArgNumber();
      info.elementType = tensorType.getElementType();
      info.tensorType = tensorType;
      info.valuesToReplace.push_back(extractPtrOp.getResult());
      info.extractPtrOp = extractPtrOp;
      argsToConvert.push_back(info);
    }
  });

  if (argsToConvert.empty())
    return;

  // Step 2: Build new function type with tt.ptr arguments
  FunctionType funcType = funcOp.getFunctionType();
  SmallVector<Type> newInputTypes;
  DenseMap<unsigned, Type> argElementTypes;

  for (const auto &info : argsToConvert) {
    argElementTypes[info.argIndex] = info.elementType;
  }

  for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
    auto it = argElementTypes.find(i);
    if (it != argElementTypes.end()) {
      newInputTypes.push_back(triton::PointerType::get(it->second, 1));
    } else {
      newInputTypes.push_back(funcType.getInput(i));
    }
  }

  auto newFuncType = FunctionType::get(ctx, newInputTypes, funcType.getResults());

  // Collect attributes to copy
  SmallVector<NamedAttribute> attrsToKeep;
  for (NamedAttribute attr : funcOp->getAttrs()) {
    StringRef name = attr.getName();
    if (name == "function_type" || name == "sym_name" || name == "sym_visibility")
      continue;
    attrsToKeep.push_back(attr);
  }

  // Step 3: For each extract_ptr result, replace its users with the block argument
  // We do this BEFORE changing types so the old ops become dead
  Block &entryBlock = funcOp.front();
  for (auto &info : argsToConvert) {
    BlockArgument blockArg = entryBlock.getArgument(info.argIndex);
    auto ptrType = triton::PointerType::get(info.elementType, 1);

    for (Value oldValue : info.valuesToReplace) {
      // For each user of the extract_ptr result (like tt.splat), create a replacement
      for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses())) {
        Operation *user = use.getOwner();

        if (auto splatOp = dyn_cast<triton::SplatOp>(user)) {
          // Create new splat with pointer type
          builder.setInsertionPoint(splatOp);
          auto resultType = cast<RankedTensorType>(splatOp.getResult().getType());
          auto newResultType = RankedTensorType::get(
              resultType.getShape(), ptrType, resultType.getEncoding());
          Value newSplat = triton::SplatOp::create(
              builder, splatOp.getLoc(), newResultType, blockArg);

          // Map old splat result to new for downstream propagation
          valueMapping.map(splatOp.getResult(), newSplat);

          // Replace all uses of old splat with new splat
          splatOp.getResult().replaceAllUsesWith(newSplat);

          // Track old splat for later erasure
          info.oldSplatOps.push_back(splatOp);
        }
      }
    }

    // Update the block argument type
    blockArg.setType(ptrType);
  }

  // Erase the ops in the chain (users first, producers last)
  // Order: old splats -> extract_ptr
  for (auto &info : argsToConvert) {
    // First erase the old splat ops (they use extract_ptr result)
    for (auto splatOp : info.oldSplatOps) {
      splatOp.erase();
    }
    // Then erase extract_ptr (uses block arg)
    if (info.extractPtrOp)
      info.extractPtrOp.erase();
  }

  // Step 4: Create tt.func and move body
  builder.setInsertionPoint(funcOp);
  auto ttFuncOp = triton::FuncOp::create(
      builder, funcOp.getLoc(), funcOp.getName(), newFuncType, attrsToKeep);
  ttFuncOp->setAttr("noinline", builder.getBoolAttr(true));

  // Set tt.divisibility = 16 on pointer arguments to enable better vectorization
  // Set tt.pointer_range = 32 if tensor is statically known to be < 2GB
  constexpr int64_t k2GBLimit = (1LL << 31); // 2GB
  for (const auto &info : argsToConvert) {
    // as we use gpu malloc, this is always the case
    // we use 16 because 128-bit is the maximum vector load/store width on
    // modern GPUs
    ttFuncOp.setArgAttr(info.argIndex, "tt.divisibility",
                        builder.getI32IntegerAttr(16));

    // Check if tensor size is statically known and < 2GB
    if (info.tensorType.hasStaticShape()) {
      int64_t numElements = info.tensorType.getNumElements();
      assert(info.elementType.isIntOrFloat());
      unsigned elementBitWidth = info.elementType.getIntOrFloatBitWidth();
      assert(elementBitWidth > 0);
        int64_t tensorSizeBytes = llvm::divideCeil(numElements * elementBitWidth, 8);
      if (tensorSizeBytes < k2GBLimit) {
        // Tensor fits in 32-bit address range, enable buffer ops optimization
        ttFuncOp.setArgAttr(info.argIndex, "tt.pointer_range",
                            builder.getI32IntegerAttr(32));
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Tensor (idx=" << info.argIndex
                                << ") is too big to add tt.pointer_range=32\n");
      }
    }
  }

  Region &oldRegion = funcOp.getBody();
  Region &newRegion = ttFuncOp.getBody();
  newRegion.takeBody(oldRegion);

  // Convert func.return to tt.return
  ttFuncOp.walk([&](func::ReturnOp returnOp) {
    builder.setInsertionPoint(returnOp);
    triton::ReturnOp::create(builder, returnOp.getLoc(), returnOp.getOperands());
    returnOp.erase();
  });

  funcOp.erase();

  // Continue with remaining transformations
  SmallVector<Operation *, 8> opsToErase;

  // Step 5: Convert arith.addi on pointer tensors to tt.addptr
  bool changed = true;
  while (changed) {
    changed = false;
    ttFuncOp.walk([&](arith::AddIOp addOp) {
      // Skip if already scheduled for erasure
      if (llvm::is_contained(opsToErase, addOp.getOperation()))
        return;

      Value lhs = addOp.getLhs();
      Value rhs = addOp.getRhs();

      Value mappedLhs = valueMapping.lookupOrNull(lhs);
      Value mappedRhs = valueMapping.lookupOrNull(rhs);

      // Check if either operand is a pointer tensor (either mapped or directly)
      Value ptrOperand = nullptr;
      Value offsetOperand = nullptr;

      if (mappedLhs && isTensorOfPointers(mappedLhs.getType())) {
        ptrOperand = mappedLhs;
        offsetOperand = mappedRhs ? mappedRhs : rhs;
      } else if (mappedRhs && isTensorOfPointers(mappedRhs.getType())) {
        ptrOperand = mappedRhs;
        offsetOperand = mappedLhs ? mappedLhs : lhs;
      } else if (isTensorOfPointers(lhs.getType())) {
        // Direct pointer tensor (already converted)
        ptrOperand = lhs;
        offsetOperand = mappedRhs ? mappedRhs : rhs;
      } else if (isTensorOfPointers(rhs.getType())) {
        ptrOperand = rhs;
        offsetOperand = mappedLhs ? mappedLhs : lhs;
      }

      if (!ptrOperand)
        return;

      // Don't convert if offset is also a pointer (shouldn't happen)
      if (isTensorOfPointers(offsetOperand.getType()))
        return;

      Location loc = addOp.getLoc();
      builder.setInsertionPoint(addOp);

      // Create tt.addptr
      Value newAddPtr = triton::AddPtrOp::create(
          builder, loc, ptrOperand.getType(), ptrOperand, offsetOperand);

      valueMapping.map(addOp.getResult(), newAddPtr);
      opsToErase.push_back(addOp);
      changed = true;
    });
  }

  // Step 6: Propagate pointer tensors through rock.cast_to_ptr ops
  changed = true;
  while (changed) {
    changed = false;

    // Handle rock.cast_to_ptr - if input maps to pointer tensor, replace it
    ttFuncOp.walk([&](rock::CastToPtrOp castOp) {
      if (llvm::is_contained(opsToErase, castOp.getOperation()))
        return;

      Value src = castOp.getSrc();
      Value mappedSrc = valueMapping.lookupOrNull(src);

      // in some cases, there's no tt.add_ptr, so mappedSrc = nullptr
      if (!mappedSrc)
        mappedSrc = src;

      if (!isTensorOfPointers(mappedSrc.getType()))
        return;

      // Check if already mapped
      if (valueMapping.contains(castOp.getResult()))
        return;

      // The cast_to_ptr produces a pointer tensor, but we already have one
      valueMapping.map(castOp.getResult(), mappedSrc);
      opsToErase.push_back(castOp);
      changed = true;
    });
  }

  // Step 7: Update all remaining uses
  ttFuncOp.walk([&](Operation *op) {
    // Skip ops we're about to erase
    if (llvm::is_contained(opsToErase, op))
      return;

    bool needsUpdate = false;
    for (Value operand : op->getOperands()) {
      if (valueMapping.contains(operand)) {
        needsUpdate = true;
        break;
      }
    }

    if (!needsUpdate)
      return;

    // Update operands
    for (OpOperand &operand : op->getOpOperands()) {
      if (Value mapped = valueMapping.lookupOrNull(operand.get())) {
        operand.set(mapped);
      }
    }
  });

  // Erase the converted operations in reverse order
  for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
    (*it)->erase();
  }
}

void RockFuncToTritonFuncPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Collect kernel functions (host functions were already serialized and
  // erased by RockSerializeHostFuncsPass earlier in the pipeline).
  SmallVector<func::FuncOp> funcsToProcess;
  moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp->getParentOfType<ModuleOp>() != moduleOp)
      return;
    if (funcOp->hasAttr(rock::KernelAttr::getMnemonic()) ||
        funcOp->hasAttr(rock::KernelLegacyAttr::getMnemonic()))
      funcsToProcess.push_back(funcOp);
  });

  // Store kernel grid/block sizes as module attributes BEFORE converting to
  // tt.func (which erases the func::FuncOp). These will be used later for
  // gpu.launch_func.
  for (func::FuncOp funcOp : funcsToProcess) {
    std::string kernelName = funcOp.getName().str();
    auto gridAttr = funcOp->getAttrOfType<IntegerAttr>(
        rock::GridSizeAttr::getMnemonic());
    if (!gridAttr)
      gridAttr = funcOp->getAttrOfType<IntegerAttr>(
          rock::GridSizeLegacyAttr::getMnemonic());
    if (gridAttr) {
      moduleOp->setAttr("rock.grid_size." + kernelName, gridAttr);
    }
  }

  // Process kernel functions (convert to tt.func)
  for (func::FuncOp funcOp : funcsToProcess) {
    processFunction(funcOp);
  }

  // Verify no Rock dialect ops remain after conversion.
  WalkResult result = moduleOp->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() ==
                                rock::RockDialect::getDialectNamespace()) {
      op->emitError("unexpected Rock op remaining after FuncToTritonFunc");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
