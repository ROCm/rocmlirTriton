//===- RockToTTIR.cpp - Convert Rock dialect to Triton IR -----------------===//
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
// This pass converts Rock dialect operations (blockwise loads, stores, gemm,
// etc.) to Triton IR counterparts (tt.load, tt.store, tt.dot, etc.).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/tritonUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKTOTTIRPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-to-ttir"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::triton;
using namespace mlir::arith;

namespace {
struct RockToTTIRPass : public rock::impl::RockToTTIRPassBase<RockToTTIRPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// RockBlockwiseReduceOpRewritePattern - Convert rock.blockwise_reduce to tt.reduce
//===----------------------------------------------------------------------===//
struct RockBlockwiseReduceOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseReduceOp> {
  using OpRewritePattern<rock::BlockwiseReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseReduceOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    auto inputType = cast<RankedTensorType>(input.getType());
    Type elemType = inputType.getElementType();
    int axis = static_cast<int>(op.getAxisAttr().getInt());
    
    // Create tt.reduce operation
    auto reduceOp = triton::ReduceOp::create(
        rewriter, loc, ValueRange{input}, axis);
    
    // Build the combiner region
    Block *block = rewriter.createBlock(&reduceOp.getCombineOp());
    
    // Add block arguments - tt.reduce expects scalar arguments for the combiner
    block->addArgument(elemType, loc);
    block->addArgument(elemType, loc);
    
    // Create the combiner operation based on reduce method
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);
    
    Value lhs = block->getArgument(0);
    Value rhs = block->getArgument(1);
    Value result;
    
    ReduceMethod method = op.getReduceMethod();
    bool isFloat = isa<FloatType>(elemType);
    
    if (method == ReduceMethod::Max) {
      if (isFloat) {
        result = arith::MaximumFOp::create(rewriter, loc, lhs, rhs);
      } else {
        // Use signed max for integers (MaxSIOp)
        result = arith::MaxSIOp::create(rewriter, loc, lhs, rhs);
      }
    } else if (method == ReduceMethod::Sum) {
      if (isFloat) {
        result = arith::AddFOp::create(rewriter, loc, lhs, rhs);
      } else {
        result = arith::AddIOp::create(rewriter, loc, lhs, rhs);
      }
    } else {
      return failure();
    }
    
    // Create reduce.return
    triton::ReduceReturnOp::create(rewriter, loc, ValueRange{result});
    
    rewriter.replaceOp(op, reduceOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockLoadPtrOpRewritePattern - Convert rock.blockwise_load_ptr to
// tt.load
//===----------------------------------------------------------------------===//
struct RockLoadPtrOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseLoadPtrOp> {
  using OpRewritePattern<rock::BlockwiseLoadPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseLoadPtrOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get operands (all tensors now)
    Value pointerTensor = op.getPointerTensor();
    Value maskTensor = op.getMaskTensor();

    // Get the element type and shape from the result type
    auto resultTensorType = cast<RankedTensorType>(op.getResult().getType());
    Type elementType = resultTensorType.getElementType();

    // Verify pointerTensor is a tensor of i32
    auto ptrTensorType = dyn_cast<RankedTensorType>(pointerTensor.getType());
    if (!ptrTensorType || !ptrTensorType.getElementType().isInteger(32)) {
      LLVM_DEBUG(llvm::dbgs() << "Pointer tensor is not a tensor of i32\n");
      return failure();
    }

    // Verify maskTensor is a tensor of i1
    auto maskTensorType = dyn_cast<RankedTensorType>(maskTensor.getType());
    if (!maskTensorType || !maskTensorType.getElementType().isInteger(1)) {
      LLVM_DEBUG(llvm::dbgs() << "Mask tensor is not a tensor of i1\n");
      return failure();
    }

    // Create pointer type: !tt.ptr<elementType>
    // Use address space 1 (global) as default for Triton
    triton::PointerType ptrType = triton::PointerType::get(elementType, 1);

    // Create tensor of pointers: tensor<...x!tt.ptr<elementType>>
    RankedTensorType ptrTensorOfPtrsType =
        RankedTensorType::get(ptrTensorType.getShape(), ptrType,
                              ptrTensorType.getEncoding());

    // Convert tensor of i32 to tensor of pointers
    Value ptrTensorOfPtrs =
        rock::CastToPtrOp::create(rewriter, loc, ptrTensorOfPtrsType, pointerTensor);

    // Create tt.load operation
    // LoadOp takes: ptr, mask (optional), other (optional), boundaryCheck,
    // padding, cache, evict, isVolatile Create attributes with default values
    auto boundaryCheckAttr = rewriter.getDenseI32ArrayAttr({});
    auto cacheAttr = triton::CacheModifierAttr::get(
        rewriter.getContext(), triton::CacheModifier::NONE);
    auto evictAttr = triton::EvictionPolicyAttr::get(
        rewriter.getContext(), triton::EvictionPolicy::NORMAL);
    auto isVolatileAttr = rewriter.getBoolAttr(false);

    Value result = triton::LoadOp::create(
        rewriter, loc, resultTensorType, ptrTensorOfPtrs, maskTensor,
        /*other=*/Value(), boundaryCheckAttr,
        /*padding=*/nullptr, cacheAttr, evictAttr, isVolatileAttr);

    // Replace the op with the loaded tensor result
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RockBlockwiseGemmOpRewritePattern - Convert rock.blockwise_gemm to
// tt.dot
//===----------------------------------------------------------------------===//
struct RockBlockwiseGemmOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseGemmOp> {
  using OpRewritePattern<rock::BlockwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseGemmOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get operands (already tensors)
    Value a = op.getMatrixA();
    Value b = op.getMatrixB();
    Value c = op.getMatrixC();
    Value scaledA = op.getMatrixScaleA();
    Value scaledB = op.getMatrixScaleB();

    // Get the tensor types
    auto aTensorType = cast<RankedTensorType>(a.getType());
    auto bTensorType = cast<RankedTensorType>(b.getType());
    auto cTensorType = cast<RankedTensorType>(c.getType());

    // Use tt.dot_scaled when either:
    //   (1) original element types were recorded, or
    //   (2) scale operands are present.
    // This ensures f8 uses float MFMA (via dot_scaled) instead of
    // integer MFMA (via dot with i8 operands).
    bool hasOrigElemTypes = op.getMatrixAOrigElemType().has_value() &&
                            op.getMatrixBOrigElemType().has_value();
    bool hasScales = scaledA && scaledB;
    bool useDotScaled = hasOrigElemTypes || hasScales;

    Value result;
    if (useDotScaled) {
      // Use original element types saved by LegalizeFloatTypes (if any).
      Type aOrigType =
          op.getMatrixAOrigElemType().value_or(aTensorType.getElementType());
      Type bOrigType =
          op.getMatrixBOrigElemType().value_or(bTensorType.getElementType());
      auto aElemTy = rock::mlirTypeToScaleDotElemType(aOrigType);
      auto bElemTy = rock::mlirTypeToScaleDotElemType(bOrigType);
      if (failed(aElemTy) || failed(bElemTy))
        return op.emitError("unsupported element type for tt.dot_scaled");

      bool matrixAKPack = op.getMatrixAKPack().value_or(true);
      bool matrixBKPack = op.getMatrixBKPack().value_or(true);

      result = triton::DotScaledOp::create(rewriter, loc, cTensorType, a, b, c,
                                           scaledA, scaledB, aElemTy.value(),
                                           bElemTy.value(), /*fastMath=*/false,
                                           matrixAKPack, matrixBKPack);
    } else {
      // Create tt.dot operation
      result =
          triton::DotOp::create(rewriter, loc, cTensorType, a, b, c,
                                /*inputPrecision=*/triton::InputPrecision::IEEE,
                                /*maxNumImpreciseAcc=*/0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct RockStorePtrOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseStorePtrOp> {
  using OpRewritePattern<rock::BlockwiseStorePtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseStorePtrOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get operands (all tensors now)
    Value pointerTensor = op.getPointerTensor();
    Value maskTensor = op.getMaskTensor();

    // Get the value to store from the op's source operand.
    // This correctly handles output fusion where the source is the result
    // of fusion ops (e.g., arith.addf) rather than the raw GEMM result.
    Value valueToStore = op.getSource();

    // 2. Verify pointer tensor is a tensor of i32
    auto ptrTensorType = dyn_cast<RankedTensorType>(pointerTensor.getType());
    if (!ptrTensorType || !ptrTensorType.getElementType().isInteger(32)) {
      LLVM_DEBUG(llvm::dbgs() << "Pointer tensor is not a tensor of i32\n");
      return failure();
    }

    // 3. Verify mask tensor is a tensor of i1
    auto maskTensorType = dyn_cast<RankedTensorType>(maskTensor.getType());
    if (!maskTensorType || !maskTensorType.getElementType().isInteger(1)) {
      LLVM_DEBUG(llvm::dbgs() << "Mask tensor is not a tensor of i1\n");
      return failure();
    }

    // 4. Convert the pointer tensor (tensor of i32) to tensor of triton pointers
    // Get element type from the value to store
    auto valueType = cast<RankedTensorType>(valueToStore.getType());
    Type elementType = valueType.getElementType();

    // Create triton pointer type: !tt.ptr<elementType>
    triton::PointerType ptrType = triton::PointerType::get(elementType, 1);

    // Create tensor of pointers type
    RankedTensorType ptrTensorOfPtrsType = RankedTensorType::get(
        ptrTensorType.getShape(), ptrType, ptrTensorType.getEncoding());

    // Cast the i32 tensor to tensor of pointers
    Value ptrTensorOfPtrs = rock::CastToPtrOp::create(
        rewriter, loc, ptrTensorOfPtrsType, pointerTensor);

    // 5. Create triton::StoreOp or triton::AtomicRMWOp depending on storeMethod
    auto storeMethod = op.getStoreMethod();
    if (storeMethod == rock::StoreMethod::AtomicAdd) {
      // Use FADD for floating point, ADD for integer
      triton::RMWOp rmwOp =
          elementType.isIntOrIndex() ? triton::RMWOp::ADD : triton::RMWOp::FADD;
      // AtomicRMWOp returns the old value, but we don't need it
      triton::AtomicRMWOp::create(
          rewriter, loc, valueType, rmwOp, ptrTensorOfPtrs, valueToStore, maskTensor,
          triton::MemSemantic::RELAXED, triton::MemSyncScope::GPU);
    } else if (storeMethod == rock::StoreMethod::AtomicMax) {
      // Use MAX for signed int, UMAX for unsigned int
      // For floating point, Triton doesn't have a direct FMAX atomic,
      // so we use MAX (which may need special handling downstream)
      triton::RMWOp rmwOp;
      if (elementType.isUnsignedInteger()) {
        rmwOp = triton::RMWOp::UMAX;
      } else {
        // Signed integer or floating point - use MAX
        rmwOp = triton::RMWOp::MAX;
      }
      triton::AtomicRMWOp::create(
          rewriter, loc, valueType, rmwOp, ptrTensorOfPtrs, valueToStore, maskTensor,
          triton::MemSemantic::RELAXED, triton::MemSyncScope::GPU);
    } else {
      // Default: StoreMethod::Set - regular store
      // Signature: (ptr, value, mask, boundaryCheck, cache, evict)
      triton::StoreOp::create(
          rewriter, loc, ptrTensorOfPtrs, valueToStore, maskTensor,
          /*boundaryCheck=*/ArrayRef<int32_t>{},
          /*cache=*/triton::CacheModifier::NONE,
          /*evict=*/triton::EvictionPolicy::NORMAL);
    }

    // Replace the op with the stored value (the result represents the stored tensor)
    rewriter.replaceOp(op, valueToStore);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReturnOpRewritePattern - Update return ops to return nothing and update
// the parent function signature to return void
//===----------------------------------------------------------------------===//
struct ReturnOpRewritePattern : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp returnOp,
                                PatternRewriter &rewriter) const override {
    // Only convert return ops that have operands
    if (returnOp.getOperands().empty())
      return failure();

    // Update the parent function's signature to return void
    auto funcOp = returnOp->getParentOfType<func::FuncOp>();
    if (funcOp && funcOp.getFunctionType().getNumResults() > 0) {
      FunctionType newFuncType = FunctionType::get(
          rewriter.getContext(), funcOp.getFunctionType().getInputs(),
          /*results=*/{});
      rewriter.modifyOpInPlace(funcOp, [&]() {
        funcOp.setFunctionType(newFuncType);
        funcOp.setAllResultAttrs(ArrayRef<DictionaryAttr>{});
      });
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);
    return success();
  }
};
} // end anonymous namespace

void RockToTTIRPass::runOnOperation() {
  MLIRContext *ctx = &getContext();

  auto funcOp = getOperation();
  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    return;
  }

  ConversionTarget target(*ctx);

  // Mark Rock ops as illegal - they should be converted
  target.addIllegalOp<rock::BlockwiseReduceOp>();
  target.addIllegalOp<rock::BlockwiseLoadPtrOp>();
  target.addIllegalOp<rock::BlockwiseGemmOp>();
  target.addIllegalOp<rock::BlockwiseStorePtrOp>();

  // Triton and Rock dialects are legal (Rock for now, will be converted later)
  target.addLegalDialect<triton::TritonDialect>();
  target.addLegalDialect<rock::RockDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<math::MathDialect>();
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [](func::ReturnOp op) { return op.getOperands().empty(); });

  RewritePatternSet patterns(ctx);
  patterns.add<RockBlockwiseReduceOpRewritePattern>(ctx);
  patterns.add<RockLoadPtrOpRewritePattern>(ctx);
  patterns.add<RockBlockwiseGemmOpRewritePattern>(ctx);
  patterns.add<RockStorePtrOpRewritePattern>(ctx);
  patterns.add<ReturnOpRewritePattern>(ctx);

  // Apply partial conversion - convert tensor.splat and Rock ops to Triton ops
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}
