/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEEPILOGUE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kLoopedGeluAttr = "amdg.looped_gelu";

bool isOneOperandElementwiseOp(Operation *op) {
  if (llvm::isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::FPToSIOp,
                arith::FPToUIOp, arith::NegFOp, arith::SIToFPOp,
                arith::TruncFOp, arith::TruncIOp, arith::UIToFPOp>(op))
    return true;
  if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                math::CeilOp, math::CosOp, math::SinOp,
                math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                math::ExpM1Op, math::FloorOp, math::LogOp, math::Log10Op,
                math::Log1pOp, math::Log2Op, math::SqrtOp, math::RsqrtOp,
                math::TanhOp>(op))
    return true;
  if (llvm::isa<triton::IntToPtrOp, triton::PtrToIntOp, triton::BitcastOp,
                triton::FpToFpOp>(op))
    return true;
  if (auto externElementwiseOp = dyn_cast<triton::ExternElementwiseOp>(op))
    return op->getNumOperands() == 1 && op->getNumResults() == 1 &&
           externElementwiseOp.getPure();
  return false;
}

FloatAttr getSplatF32Constant(Value value) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return getSplatF32Constant(splat.getSrc());

  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return {};
  if (auto scalar = dyn_cast<FloatAttr>(constant.getValueAttr()))
    return scalar.getType().isF32() ? scalar : FloatAttr{};
  auto elements = dyn_cast<DenseFPElementsAttr>(constant.getValueAttr());
  if (!elements || !elements.isSplat() || !elements.getElementType().isF32())
    return {};
  return FloatAttr::get(elements.getElementType(),
                        elements.getSplatValue<APFloat>());
}

struct ValueAndConstant {
  Value value;
  FloatAttr constant;
  bool valueIsLhs;
};

std::optional<ValueAndConstant> matchValueAndConstant(arith::MulFOp mul) {
  if (FloatAttr rhs = getSplatF32Constant(mul.getRhs()))
    return ValueAndConstant{mul.getLhs(), rhs, /*valueIsLhs=*/true};
  if (FloatAttr lhs = getSplatF32Constant(mul.getLhs()))
    return ValueAndConstant{mul.getRhs(), lhs, /*valueIsLhs=*/false};
  return std::nullopt;
}

std::optional<ValueAndConstant> matchValueAndConstant(arith::AddFOp add) {
  if (FloatAttr rhs = getSplatF32Constant(add.getRhs()))
    return ValueAndConstant{add.getLhs(), rhs, /*valueIsLhs=*/true};
  if (FloatAttr lhs = getSplatF32Constant(add.getLhs()))
    return ValueAndConstant{add.getRhs(), lhs, /*valueIsLhs=*/false};
  return std::nullopt;
}

bool isFedByBlockedF32Dot(Value value, llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (auto dot = dyn_cast<triton::DotOp>(def)) {
    auto resultTy = dot.getD().getType();
    return resultTy.getElementType().isF32() &&
           isa<triton::gpu::BlockedEncodingAttr>(resultTy.getEncoding());
  }
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return false;
    unsigned resultNumber = result.getResultNumber();
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    return isFedByBlockedF32Dot(forOp.getInitArgs()[resultNumber], visited) ||
           isFedByBlockedF32Dot(yield.getOperand(resultNumber), visited);
  }
  if (!isa<arith::AddFOp, arith::MulFOp, triton::gpu::ConvertLayoutOp>(def))
    return false;
  return llvm::any_of(def->getOperands(), [&](Value operand) {
    auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
    return tensorTy && tensorTy.getElementType().isF32() &&
           isFedByBlockedF32Dot(operand, visited);
  });
}

bool isFedByBlockedF32Dot(Value value) {
  llvm::DenseSet<Value> visited;
  return isFedByBlockedF32Dot(value, visited);
}

void addFastMathMetadata(NamedAttrList &metadata, StringRef name,
                         Operation *op) {
  auto fastMath = dyn_cast<arith::ArithFastMathInterface>(op);
  if (!fastMath)
    return;
  arith::FastMathFlagsAttr attr = fastMath.getFastMathFlagsAttr();
  if (attr && attr.getValue() != arith::FastMathFlags::none)
    metadata.append(name, attr);
}

// Marks an exact x * (erf(x * scale) + bias) [* outputScale] store for a
// compact store-side lowering. Keeping only x as the store value prevents the
// ordinary elementwise conversion from cloning the GELU once per accumulator;
// the dictionary records everything needed to reconstruct the exact scalar
// expression inside a bounded LLVM loop.
class MarkLoopedGeluStore : public OpRewritePattern<triton::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (store->hasAttr(kLoopedGeluAttr))
      return failure();
    auto storedTy = dyn_cast<RankedTensorType>(store.getValue().getType());
    if (!storedTy || !storedTy.getElementType().isF32() ||
        !isa<triton::gpu::BlockedEncodingAttr>(storedTy.getEncoding()) ||
        triton::gpu::getTotalElemsPerThread(storedTy) < 32)
      return failure();

    Value coreValue = store.getValue();
    arith::SelectOp maskedValueSelect;
    if (auto select = coreValue.getDefiningOp<arith::SelectOp>()) {
      // MIGraphX materializes masked stores as
      //   store(ptr, select(mask, value, 0), mask).
      // The false arm is unobservable because the same lanes are suppressed by
      // the store. Peel that wrapper so the production GELU chain can be
      // recognized, but only when the predicates are the exact same SSA value.
      if (!select->hasOneUse() || !store.getMask() ||
          select.getCondition() != store.getMask())
        return failure();
      maskedValueSelect = select;
      coreValue = select.getTrueValue();
    }

    arith::MulFOp outputMul;
    std::optional<ValueAndConstant> outputScale;
    if (auto maybeOutputMul = coreValue.getDefiningOp<arith::MulFOp>()) {
      if (auto matched = matchValueAndConstant(maybeOutputMul)) {
        outputMul = maybeOutputMul;
        outputScale = matched;
        coreValue = matched->value;
      }
    }

    auto coreMul = coreValue.getDefiningOp<arith::MulFOp>();
    if (!coreMul || !coreMul->hasOneUse())
      return failure();
    arith::AddFOp biasAdd = coreMul.getRhs().getDefiningOp<arith::AddFOp>();
    Value x = coreMul.getLhs();
    bool xIsLhs = true;
    if (!biasAdd) {
      biasAdd = coreMul.getLhs().getDefiningOp<arith::AddFOp>();
      x = coreMul.getRhs();
      xIsLhs = false;
    }
    if (!biasAdd || !biasAdd->hasOneUse())
      return failure();

    std::optional<ValueAndConstant> bias = matchValueAndConstant(biasAdd);
    if (!bias)
      return failure();
    auto erf = bias->value.getDefiningOp<math::ErfOp>();
    if (!erf || !erf->hasOneUse())
      return failure();
    auto scaleMul = erf.getOperand().getDefiningOp<arith::MulFOp>();
    if (!scaleMul || !scaleMul->hasOneUse())
      return failure();
    std::optional<ValueAndConstant> scale = matchValueAndConstant(scaleMul);
    if (!scale || scale->value != x || !isFedByBlockedF32Dot(x))
      return failure();
    if (outputMul &&
        (!outputMul->hasOneUse() || outputScale->value != coreMul.getResult()))
      return failure();

    NamedAttrList metadata;
    metadata.append("scale", scale->constant);
    metadata.append("bias", bias->constant);
    metadata.append("x_is_lhs", rewriter.getBoolAttr(xIsLhs));
    metadata.append("scaled_x_is_lhs", rewriter.getBoolAttr(scale->valueIsLhs));
    metadata.append("erf_is_lhs", rewriter.getBoolAttr(bias->valueIsLhs));
    if (outputScale) {
      metadata.append("output_scale", outputScale->constant);
      metadata.append("core_is_lhs",
                      rewriter.getBoolAttr(outputScale->valueIsLhs));
    }
    addFastMathMetadata(metadata, "scale_fastmath", scaleMul);
    addFastMathMetadata(metadata, "erf_fastmath", erf);
    addFastMathMetadata(metadata, "bias_fastmath", biasAdd);
    addFastMathMetadata(metadata, "core_fastmath", coreMul);
    if (outputMul)
      addFastMathMetadata(metadata, "output_fastmath", outputMul);

    rewriter.modifyOpInPlace(store, [&] {
      store.getValueMutable().assign(x);
      store->setAttr(kLoopedGeluAttr,
                     DictionaryAttr::get(store.getContext(), metadata));
    });
    if (maskedValueSelect)
      rewriter.eraseOp(maskedValueSelect);
    if (outputMul)
      rewriter.eraseOp(outputMul);
    rewriter.eraseOp(coreMul);
    rewriter.eraseOp(biasAdd);
    rewriter.eraseOp(erf);
    rewriter.eraseOp(scaleMul);
    return success();
  }
};

// Tries to optimize oldStoreOp with v_permlane*_swap instruction when possible.
// Returns null store op if not suitable.
static triton::StoreOp
usePermlaneSwapToOptimizeStore(PatternRewriter &rewriter, Value ptr, Value val,
                               Value mask, triton::StoreOp oldStoreOp) {
  auto ptrType = cast<RankedTensorType>(ptr.getType());
  auto valType = cast<RankedTensorType>(val.getType());

  // Create a new layout where each thread holds 8 consecutive elements, in
  // order to enable wide 128-bit global stores.
  std::optional<triton::LinearLayout> storeLL =
      triton::gpu::chooseMfmaLikeStoreLayout(valType);
  if (!storeLL)
    return nullptr;

  Attribute newEncoding = triton::gpu::LinearEncodingAttr::get(
      oldStoreOp.getContext(), std::move(storeLL.value()));
  auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
  Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                      newPtrType, ptr);

  auto newValType = valType.cloneWithEncoding(newEncoding);
  Value newVal = triton::gpu::ConvertLayoutOp::create(rewriter, val.getLoc(),
                                                      newValType, val);

  Value newMask = mask;
  if (mask) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    auto newMaskType = maskType.cloneWithEncoding(newEncoding);
    newMask = triton::gpu::ConvertLayoutOp::create(rewriter, mask.getLoc(),
                                                   newMaskType, mask);
  }

  return triton::StoreOp::create(rewriter, oldStoreOp.getLoc(), newPtr, newVal,
                                 newMask, oldStoreOp.getCache(),
                                 oldStoreOp.getEvict());
}

// Whether issuing the store directly in the layout of `srcType` is worth
// skipping the relayout into the coalesced layout of `dstType`, the type the
// original store used.
//
// For an MMA source it always is: the relayout is a pure LDS round trip and
// the MFMA/WMMA layouts store acceptably.
//
// For a blocked source it is profitable when the store stays as wide and as
// coalesced as it would be in the layout of `dstType`. Both layouts hold the
// same number of elements per thread, so a shorter contiguous run means that
// each store would cover fewer elements, thus needing more store instructions,
// which would harm performance.
static bool isProfitableBypassSource(RankedTensorType srcType,
                                     RankedTensorType dstType) {
  Attribute srcEncoding = srcType.getEncoding();
  Attribute dstEncoding = dstType.getEncoding();
  if (!srcEncoding || !dstEncoding)
    return false;

  if (isa<triton::gpu::MmaEncodingTrait>(srcEncoding))
    return true;

  auto srcBlocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(srcEncoding);
  auto dstBlocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(dstEncoding);
  if (!srcBlocked || !dstBlocked)
    return false;

  // A differing fastest-varying dimension would make the store run in a
  // different axis, so the contiguous runs are not comparable.
  const unsigned contigDim = srcBlocked.getOrder()[0];
  if (contigDim != dstBlocked.getOrder()[0])
    return false;

  // Since we are considering using the src layout, make sure that
  // lanes in that layout write to consecutive addresses.
  if (triton::gpu::getThreadOrder(srcType)[0] != contigDim)
    return false;

  // Make sure that using the src layout would not increase the store count.
  // A store is at most 128 bits wide, so we cap both runs at the number of
  // elements that fit in it (4 for f32, 8 for f16): a src run of 8 f16 against
  // a dst run of 16 needs no more stores, since neither can write more than 8
  // elements at a time, and rejecting it would keep the LDS round trip for a
  // penalty the hardware does not have.
  Type elemType = dstType.getElementType();
  const unsigned elemBitWidth =
      elemType.isIntOrFloat() ? elemType.getIntOrFloatBitWidth() : 0;
  const unsigned maxElems =
      elemBitWidth ? std::max(128u / elemBitWidth, 1u) : ~0u;
  return std::min(triton::gpu::getContigPerThread(srcType)[contigDim],
                  maxElems) >=
         std::min(triton::gpu::getContigPerThread(dstType)[contigDim],
                  maxElems);
}

// convert(val) : xmma -> blocked
// elementWiseOp(val) : blocked
// ...
// elementWiseOp(val) : blocked
// tt.store(ptr, val, mask, ...) : blocked
// ==>
// convert(ptr) : blocked -> xmma
// convert(mask) : blocked -> xmma
// elementWiseOp(val) : xmma
// ...
// elementWiseOp(val) : xmma
// tt.store(ptr, val, mask, ...) : xmma
//
// Store with xmma layout directly
//
// xmma layout is either MFMA or WMMA, or the blocked layout of an FMA dot when
// that keeps the store as wide as the coalesced layout would.
class BypassEpilogueSMEM : public mlir::OpRewritePattern<triton::StoreOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp stOp,
                  mlir::PatternRewriter &rewriter) const override {

    Value ptr = stOp.getPtr();
    Value val = stOp.getValue();
    Value mask = stOp.getMask();
    auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
    auto valType = dyn_cast<RankedTensorType>(val.getType());
    if (!ptrType || !valType ||
        !isa<triton::gpu::BlockedEncodingAttr>(ptrType.getEncoding()) ||
        !isa<triton::gpu::BlockedEncodingAttr>(valType.getEncoding()))
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *> chainedOps;
    while (true) {
      auto chainedOp = val.getDefiningOp();
      if (!chainedOp)
        return mlir::failure();
      if (llvm::isa<triton::gpu::ConvertLayoutOp>(chainedOp))
        break;
      if (!chainedOp->hasOneUse())
        return mlir::failure();
      if (!isOneOperandElementwiseOp(chainedOp))
        return mlir::failure();
      val = chainedOp->getOperand(0);
      chainedOps.push_back(chainedOp);
    }

    auto cvtOp = val.getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (!cvtOp)
      return mlir::failure();

    if (!isProfitableBypassSource(cvtOp.getSrc().getType(), valType))
      return mlir::failure();

    if (!cvtOp.getResult().hasOneUse())
      return mlir::failure();

    triton::FuncOp funcOp = stOp->getParentOfType<triton::FuncOp>();
    if (funcOp && funcOp->getDiscardableAttr("rock.prefer_lds_epilogue"))
      return mlir::failure();

    auto newEncoding =
        cast<RankedTensorType>(cvtOp.getSrc().getType()).getEncoding();

    auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
    Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                        newPtrType, ptr);

    auto newVal = cvtOp.getSrc();

    for (auto chainedOp : llvm::reverse(chainedOps)) {
      auto oldType =
          cast<mlir::RankedTensorType>(chainedOp->getResult(0).getType());
      chainedOp->setOperand(0, newVal);
      newVal = llvm::cast<mlir::TypedValue<RankedTensorType>>(
          chainedOp->getResult(0));

      auto newType = oldType.cloneWithEncoding(newEncoding);
      newVal.setType(newType);
    }

    Value newMask = mask;
    if (mask) {
      auto maskType = dyn_cast<RankedTensorType>(mask.getType());
      auto newMaskType = maskType.cloneWithEncoding(newEncoding);
      newMask = triton::gpu::ConvertLayoutOp::create(rewriter, mask.getLoc(),
                                                     newMaskType, mask);
    }
    triton::StoreOp newStoreOp =
        usePermlaneSwapToOptimizeStore(rewriter, newPtr, newVal, newMask, stOp);
    if (!newStoreOp) {
      newStoreOp =
          triton::StoreOp::create(rewriter, stOp.getLoc(), newPtr, newVal,
                                  newMask, stOp.getCache(), stOp.getEvict());
    }
    if (Attribute loopedGelu = stOp->getAttr(kLoopedGeluAttr))
      newStoreOp->setAttr(kLoopedGeluAttr, loopedGelu);

    rewriter.replaceOp(stOp, newStoreOp);
    return mlir::success();
  }
};

} // anonymous namespace

class TritonAMDGPUOptimizeEpiloguePass
    : public impl::TritonAMDGPUOptimizeEpilogueBase<
          TritonAMDGPUOptimizeEpiloguePass> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonAMDGPUOptimizeEpiloguePass)
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<MarkLoopedGeluStore, BypassEpilogueSMEM>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
