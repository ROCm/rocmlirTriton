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

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

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
// RockLoadTilePtrOpRewritePattern - Convert rock.blockwise_load_tile_ptr to
// tt.load
//===----------------------------------------------------------------------===//
struct RockLoadTilePtrOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseLoadTilePtrOp> {
  using OpRewritePattern<rock::BlockwiseLoadTilePtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseLoadTilePtrOp op,
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
// RockBlockwiseGemmOpRewritePattern - Convert rock.blockwise_gemm_accel to
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

    // Get the tensor types
    auto aTensorType = dyn_cast<RankedTensorType>(a.getType());
    auto bTensorType = dyn_cast<RankedTensorType>(b.getType());
    auto cTensorType = dyn_cast<RankedTensorType>(c.getType());
    if (!aTensorType || !bTensorType || !cTensorType)
      return failure();

    // Create tt.dot operation
    Value result = triton::DotOp::create(
        rewriter, loc, cTensorType, a, b, c,
        /*inputPrecision=*/triton::InputPrecision::IEEE,
        /*maxNumImpreciseAcc=*/0);

    // We dont use replaceOp because result has one result whereas op has none.
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct RockStoreTilePtrOpRewritePattern
    : public OpRewritePattern<rock::BlockwiseStoreTilePtrOp> {
  using OpRewritePattern<rock::BlockwiseStoreTilePtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::BlockwiseStoreTilePtrOp op,
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
      rewriter.modifyOpInPlace(funcOp,
                               [&]() { funcOp.setFunctionType(newFuncType); });
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
  target.addIllegalOp<rock::BlockwiseLoadTilePtrOp>();
  target.addIllegalOp<rock::BlockwiseGemmOp>();

  // Triton and Rock dialects are legal (Rock for now, will be converted later)
  target.addLegalDialect<triton::TritonDialect>();
  target.addLegalDialect<rock::RockDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<arith::ArithDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<RockLoadTilePtrOpRewritePattern>(ctx);
  patterns.add<RockBlockwiseGemmOpRewritePattern>(ctx);

  // Apply partial conversion - convert tensor.splat and Rock ops to Triton ops
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }

  // Second conversion step: convert the micro kernel loop
  // by converting the scf.for op to a scf.for op with iter_args and
  // yield and rewrite the store tile ptr op to triton::store op.
  ConversionTarget target2(*ctx);
  target2.addLegalDialect<scf::SCFDialect>();
  target2.addLegalDialect<func::FuncDialect>();
  target2.addLegalDialect<arith::ArithDialect>();
  target2.addLegalDialect<rock::RockDialect>();
  target2.addLegalDialect<triton::TritonDialect>();
  target2.addDynamicallyLegalOp<scf::ForOp>(
      [](scf::ForOp op) { return op.getNumResults() > 0; });
  target2.addIllegalOp<rock::BlockwiseStoreTilePtrOp>();
  target2.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
    return op.getOperands().empty();
  });

  RewritePatternSet patterns2(ctx);
  patterns2.add<RockStoreTilePtrOpRewritePattern>(ctx);
  patterns2.add<ReturnOpRewritePattern>(ctx);
  if (failed(applyFullConversion(getOperation(), target2,
                                    std::move(patterns2)))) {
    return signalPassFailure();
  }
}
