//===- TensorToTritonPtr.cpp - Convert tensor semantic kernels (rock) to pointer
// semantic kernels (triton) --===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// This pass transforms Rock kernel functions from func.func to Triton's
// tt.func. It converts tensor arguments to Triton pointer types (!tt.ptr),
// eliminates the pointer extraction chain (rock.extract_ptr), converts
// arith.addi on pointer tensors to tt.addptr, and sets up pointer attributes
// for optimization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTENSORTOTRITONPTRPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-tensor-to-triton-ptr"

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

using ConstantGlobalMap = DenseMap<Attribute, LLVM::GlobalOp>;

struct ConstantGlobalState {
  ConstantGlobalMap globals;
  unsigned nextSymbolSuffix = 0;
};

static bool isReusableConstantGlobal(LLVM::GlobalOp global) {
  return global.getConstant() &&
         global.getLinkage() == LLVM::Linkage::Internal &&
         global.getAddrSpace() == 1 &&
         global.getAlignment().value_or(0) >= 16 && !global.getThreadLocal_() &&
         !global.getExternallyInitialized();
}

/// Normalize a reusable global's initializer to the same flat dense form used
/// for newly created globals.
static DenseElementsAttr normalizeGlobalInitializer(LLVM::GlobalOp global) {
  if (!isReusableConstantGlobal(global))
    return {};
  auto arrayType = dyn_cast<LLVM::LLVMArrayType>(global.getGlobalType());
  if (!arrayType || !TensorType::isValidElementType(arrayType.getElementType()))
    return {};

  int64_t numElements = arrayType.getNumElements();
  auto flatType =
      RankedTensorType::get({numElements}, arrayType.getElementType());
  Attribute initializer = global.getValueAttr();
  if (auto dense = dyn_cast_or_null<DenseElementsAttr>(initializer)) {
    if (dense.getElementType() != arrayType.getElementType() ||
        dense.getNumElements() != numElements)
      return {};
    return dense.reshape(flatType);
  }

  auto array = dyn_cast_or_null<ArrayAttr>(initializer);
  if (!array || array.size() != static_cast<size_t>(numElements) ||
      !llvm::all_of(array, [&](Attribute element) {
        auto typed = dyn_cast<TypedAttr>(element);
        return typed && typed.getType() == arrayType.getElementType();
      }))
    return {};
  return DenseElementsAttr::get(flatType, array.getValue());
}

/// Index reusable globals once so each constant lookup is constant-time.
static ConstantGlobalState indexConstantGlobals(ModuleOp module) {
  ConstantGlobalState state;
  for (LLVM::GlobalOp global : module.getOps<LLVM::GlobalOp>())
    if (DenseElementsAttr initializer = normalizeGlobalInitializer(global))
      state.globals.try_emplace(initializer, global);
  return state;
}

/// Return an internal GPU global containing `constant`'s flattened elements.
/// Identical constants share one global.
static LLVM::GlobalOp getOrCreateConstantGlobal(OpBuilder &builder,
                                                ModuleOp module,
                                                arith::ConstantOp constant,
                                                ConstantGlobalState &state) {
  auto values = cast<DenseElementsAttr>(constant.getValue());
  auto tensorType = cast<RankedTensorType>(constant.getType());
  int64_t numElements = tensorType.getNumElements();
  auto globalType =
      LLVM::LLVMArrayType::get(tensorType.getElementType(), numElements);
  auto flatType =
      RankedTensorType::get({numElements}, tensorType.getElementType());
  DenseElementsAttr initializer = values.reshape(flatType);

  if (auto it = state.globals.find(initializer); it != state.globals.end())
    return it->second;

  SmallString<32> name = SymbolTable::generateSymbolName<32>(
      "__rock_constant",
      [&](StringRef candidate) {
        return module.lookupSymbol(candidate) != nullptr;
      },
      state.nextSymbolSuffix);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  LLVM::GlobalOp global = LLVM::GlobalOp::create(
      builder, constant.getLoc(), globalType,
      /*isConstant=*/true, LLVM::Linkage::Internal, name, initializer,
      /*alignment=*/16,
      /*addrSpace=*/1);
  state.globals.try_emplace(initializer, global);
  return global;
}

/// Reject extraction forms that this pass cannot convert without leaving a
/// dangling integer placeholder or changing its scalar semantics.
static LogicalResult validateExtractPtr(rock::ExtractPtrOp extractPtrOp) {
  auto tensorType = cast<RankedTensorType>(extractPtrOp.getSource().getType());
  if (tensorType.getNumElements() == 0)
    return extractPtrOp.emitOpError(
        "zero-sized tensors cannot provide a storage pointer");

  for (Operation *user : extractPtrOp.getResult().getUsers())
    if (!isa<triton::SplatOp>(user))
      return extractPtrOp.emitOpError(
                 "expected every result user to be tt.splat")
             << ", but found " << user->getName();
  return success();
}

/// Replace pointer-placeholder splats fed by `extractPtrOp` with splats of the
/// real Triton pointer, then remove the placeholder extraction.
static void replaceExtractPtrWithPointer(IRRewriter &rewriter,
                                         rock::ExtractPtrOp extractPtrOp,
                                         Value pointer) {
  auto pointerType = cast<triton::PointerType>(pointer.getType());
  for (Operation *user :
       llvm::make_early_inc_range(extractPtrOp.getResult().getUsers())) {
    auto splatOp = cast<triton::SplatOp>(user);
    auto resultType = cast<RankedTensorType>(splatOp.getResult().getType());
    auto newResultType = RankedTensorType::get(
        resultType.getShape(), pointerType, resultType.getEncoding());
    rewriter.setInsertionPoint(splatOp);
    rewriter.replaceOpWithNewOp<triton::SplatOp>(splatOp, newResultType,
                                                 pointer);
  }
  rewriter.eraseOp(extractPtrOp);
}

struct RockTensorToTritonPtrPass
    : public rock::impl::RockTensorToTritonPtrPassBase<
          RockTensorToTritonPtrPass> {
  void runOnOperation() override;

private:
  /// Process a single kernel function (convert to tt.func)
  LogicalResult processFunction(func::FuncOp funcOp,
                                ConstantGlobalState &constantGlobals);
};

} // end anonymous namespace

LogicalResult RockTensorToTritonPtrPass::processFunction(
    func::FuncOp funcOp, ConstantGlobalState &constantGlobals) {
  MLIRContext *ctx = &getContext();
  OpBuilder builder(ctx);
  IRRewriter rewriter(ctx);

  // Step 1: Find all rock.extract_ptr patterns and collect info
  // Pattern: block_arg (tensor) -> rock.extract_ptr -> tt.splat
  struct ArgConversionInfo {
    unsigned argIndex;
    rock::ExtractPtrOp extractPtrOp;
  };
  SmallVector<ArgConversionInfo> argsToConvert;
  struct ConstantConversionInfo {
    arith::ConstantOp constant;
    rock::ExtractPtrOp extractPtrOp;
  };
  SmallVector<ConstantConversionInfo> constantsToConvert;

  funcOp.walk([&](rock::ExtractPtrOp extractPtrOp) {
    Value tensorOperand = extractPtrOp.getSource();

    if (auto constant = tensorOperand.getDefiningOp<arith::ConstantOp>()) {
      constantsToConvert.push_back({constant, extractPtrOp});
      return;
    }

    auto blockArg = cast<BlockArgument>(tensorOperand);
    argsToConvert.push_back({blockArg.getArgNumber(), extractPtrOp});
  });

  for (const ArgConversionInfo &info : argsToConvert)
    if (failed(validateExtractPtr(info.extractPtrOp)))
      return failure();
  for (const ConstantConversionInfo &info : constantsToConvert)
    if (failed(validateExtractPtr(info.extractPtrOp)))
      return failure();

  // Step 2: Build new function type with tt.ptr arguments
  FunctionType funcType = funcOp.getFunctionType();

  // Some tensor inputs may be dead (no rock.extract_ptr use) but still need
  // conversion to pointers since Triton kernels cannot have bare tensor args.
  bool hasTensorArgs = llvm::any_of(
      funcType.getInputs(), [](Type t) { return isa<RankedTensorType>(t); });

  // A kernel with only compiler-owned dense constants still needs conversion:
  // its extract_ptr operations become addresses of generated GPU globals.
  if (!hasTensorArgs && constantsToConvert.empty())
    return success();

  // Build the new pointer-based signature. Every tensor argument becomes a
  // !tt.ptr.
  SmallVector<Type> newInputTypes;
  for (Type inputType : funcType.getInputs()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(inputType))
      newInputTypes.push_back(
          triton::PointerType::get(tensorType.getElementType(), 1));
    else
      newInputTypes.push_back(inputType);
  }

  auto newFuncType =
      FunctionType::get(ctx, newInputTypes, funcType.getResults());

  // Step 3: Rewrite each pointer block argument to have the type !tt.ptr.
  // Also rebuild the tt.splat
  // that broadcasts it as a ptr tensor, and drop the dead extract_ptr.
  Block &entryBlock = funcOp.front();
  for (unsigned i = 0, e = newInputTypes.size(); i < e; ++i)
    entryBlock.getArgument(i).setType(newInputTypes[i]);

  // For each used tensor argument, rebuild the tt.splat that broadcast
  // its extract_ptr result so it now broadcasts the (already retyped) !tt.ptr
  // block argument, then drop the dead extract_ptr.
  for (auto &info : argsToConvert) {
    BlockArgument blockArg = entryBlock.getArgument(info.argIndex);
    replaceExtractPtrWithPointer(rewriter, info.extractPtrOp, blockArg);
  }

  // Convert LLVM pointers to integers so tt.int_to_ptr can bridge into Triton's
  // pointer type without mixed-dialect pointer-cast conversion support.
  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  for (auto &info : constantsToConvert) {
    LLVM::GlobalOp global = getOrCreateConstantGlobal(
        builder, module, info.constant, constantGlobals);
    auto tensorType = cast<RankedTensorType>(info.constant.getType());
    auto llvmPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
    auto tritonPtrType =
        triton::PointerType::get(tensorType.getElementType(), 1);

    rewriter.setInsertionPoint(info.extractPtrOp);
    Value address = LLVM::AddressOfOp::create(
        rewriter, info.extractPtrOp.getLoc(), llvmPtrType, global.getSymName());
    Value integerAddress = LLVM::PtrToIntOp::create(
        rewriter, info.extractPtrOp.getLoc(), rewriter.getI64Type(), address);
    Value tritonPtr = triton::IntToPtrOp::create(
        rewriter, info.extractPtrOp.getLoc(), tritonPtrType, integerAddress);

    replaceExtractPtrWithPointer(rewriter, info.extractPtrOp, tritonPtr);
    if (info.constant->use_empty())
      rewriter.eraseOp(info.constant);
  }

  // Step 4: Create tt.func and move body
  builder.setInsertionPoint(funcOp);
  SmallVector<NamedAttribute> attrsToKeep(funcOp->getDiscardableAttrs());
  auto ttFuncOp = triton::FuncOp::create(
      builder, funcOp.getLoc(), funcOp.getName(), newFuncType, attrsToKeep);
  ttFuncOp->setAttr("noinline", builder.getBoolAttr(true));

  // Propagate arg attributes (tt.divisibility, tt.pointer_range, LLVM attrs)
  // set by RockAnalyzeMemoryUsePass from func.func to tt.func.
  if (auto allArgAttrs = funcOp.getAllArgAttrs())
    ttFuncOp.setAllArgAttrs(allArgAttrs);

  Region &oldRegion = funcOp.getBody();
  Region &newRegion = ttFuncOp.getBody();
  newRegion.takeBody(oldRegion);

  // Convert func.return to tt.return
  ttFuncOp.walk([&](func::ReturnOp returnOp) {
    builder.setInsertionPoint(returnOp);
    triton::ReturnOp::create(builder, returnOp.getLoc(),
                             returnOp.getOperands());
    returnOp.erase();
  });

  funcOp.erase();

  // Step 5: Convert arith.addi on pointer tensors to tt.addptr.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<arith::AddIOp> toConvert;
    ttFuncOp.walk([&](arith::AddIOp addOp) {
      bool lhsPtr = isTensorOfPointers(addOp.getLhs().getType());
      bool rhsPtr = isTensorOfPointers(addOp.getRhs().getType());
      // Adding two pointer tensors is not valid pointer arithmetic and should
      // never be produced upstream (TransformsToPointerArith only ever adds an
      // integer offset to a base pointer).
      assert(!(lhsPtr && rhsPtr) && "arith.addi on two pointer tensors is not "
                                    "valid pointer arithmetic");
      // Convert only when exactly one operand is a pointer tensor; neither is
      // an ordinary integer add.
      if (lhsPtr != rhsPtr)
        toConvert.push_back(addOp);
    });
    for (arith::AddIOp addOp : toConvert) {
      Value lhs = addOp.getLhs(), rhs = addOp.getRhs();
      bool lhsPtr = isTensorOfPointers(lhs.getType());
      Value ptr = lhsPtr ? lhs : rhs;
      Value offset = lhsPtr ? rhs : lhs;
      rewriter.setInsertionPoint(addOp);
      rewriter.replaceOpWithNewOp<triton::AddPtrOp>(addOp, ptr.getType(), ptr,
                                                    offset);
      changed = true;
    }
  }

  // Step 6: A rock.cast_to_ptr whose source already became a pointer tensor
  // (through the addptr chain or a ptr splat) is now an identity and folds
  // away.
  changed = true;
  while (changed) {
    changed = false;
    SmallVector<rock::CastToPtrOp> toFold;
    ttFuncOp.walk([&](rock::CastToPtrOp castOp) {
      if (isTensorOfPointers(castOp.getSrc().getType()))
        toFold.push_back(castOp);
    });
    for (rock::CastToPtrOp castOp : toFold) {
      rewriter.replaceOp(castOp, castOp.getSrc());
      changed = true;
    }
  }
  return success();
}

void RockTensorToTritonPtrPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Collect kernel functions (host functions were already serialized and
  // erased by RockSerializeHostFuncsPass earlier in the pipeline).
  SmallVector<func::FuncOp> funcsToProcess;
  moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp->getParentOfType<ModuleOp>() != moduleOp)
      return;
    if (funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
      funcsToProcess.push_back(funcOp);
  });

  // Store kernel metadata as module attributes before converting to
  // tt.func (which erases the func::FuncOp). These will be used later by
  // collectKernelInfo / RockEmitGpuBinaryPass.
  OpBuilder builder(&getContext());
  for (func::FuncOp funcOp : funcsToProcess) {
    std::string kernelName = funcOp.getName().str();
    if (auto gridAttr = funcOp->getAttrOfType<IntegerAttr>(
            rock::GridSizeAttr::getMnemonic())) {
      moduleOp->setAttr(rock::GridSizeAttr::getModuleAttrName(kernelName),
                        gridAttr);
    }

    // Collect rock.prefill arg attributes.
    SmallVector<Attribute> prefillEntries;
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      if (auto initVal =
              funcOp.getArgAttr(i, rock::PrefillAttr::getMnemonic())) {
        SmallVector<NamedAttribute> entry;
        entry.push_back(
            builder.getNamedAttr("index", builder.getI64IntegerAttr(i)));
        entry.push_back(builder.getNamedAttr("value", initVal));
        prefillEntries.push_back(builder.getDictionaryAttr(entry));
      }
    }
    if (!prefillEntries.empty()) {
      moduleOp->setAttr("rock.prefill_args." + kernelName,
                        builder.getArrayAttr(prefillEntries));
    }
  }

  // Process kernel functions (convert to tt.func). Index existing globals only
  // once; newly created globals are added to the same lookup.
  ConstantGlobalState constantGlobals = indexConstantGlobals(moduleOp);
  for (func::FuncOp funcOp : funcsToProcess) {
    if (failed(processFunction(funcOp, constantGlobals)))
      return signalPassFailure();
  }

  // Verify no Rock dialect ops remain after conversion.
  WalkResult result = moduleOp->walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() ==
                                rock::RockDialect::getDialectNamespace()) {
      op->emitError("unexpected Rock op remaining after RockTensorToTritonPtr");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();
}
