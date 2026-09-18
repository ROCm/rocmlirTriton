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
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AMDGPUAllocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUOPTIMIZEEPILOGUE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Whether the bypass can retype `op` in place, given its operands retyped to
// the same layout. Elementwise means every lane's result depends only on that
// lane's operands, so the layout is a free choice, and memory-effect-free
// means the retype cannot be observed through memory. This replaces a
// hand-maintained opcode list that omitted the entire binary arith set,
// `arith::AddFOp` and `arith::MulFOp` among them, and so rejected most real
// epilogues. The same predicate is used to select ops for the backward slice
// and to halt the traversal through anything else.
//
// `tt.broadcast` is not elementwise, since it changes the shape, but it
// carries SameOperandsAndResultEncoding and stretching a unit dimension is a
// per-lane read of the source element, so it relayouts exactly like one. It
// has to be included: a bias or per-channel scale reaches the epilogue as a
// broadcast of an Mx1 load, so halting on it puts an unclassifiable value on
// the cone's boundary and refuses the whole epilogue.
static bool isRelayoutableInPlace(Operation *op) {
  if (isa<triton::BroadcastOp>(op))
    return true;
  return op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op);
}

// Whether `op`'s result can be built directly in any layout, so that a bypass
// needs no conversion for it at all. This is the set `canUseResultEncoding`
// names, narrowed to the ops this pattern knows how to rebuild. It is what
// makes ReLU (`maxnumf(acc, cst)`), clamps, and alpha scaling free.
static bool isLayoutAgnostic(Operation *op) {
  if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
    // A non-splat constant would need its per-element values permuted into the
    // new layout. Those are rare in epilogues, so bail rather than build that.
    auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue());
    return dense && dense.isSplat();
  }
  return isa<triton::SplatOp, triton::MakeRangeOp>(op);
}

// Rebuild the result of a layout-agnostic `op` in `encoding`. A splat holds the
// same scalar in every thread under any layout, and `make_range` is lane-index
// arithmetic, so neither needs cross-lane movement to change layout.
static Value rematerializeInEncoding(PatternRewriter &rewriter, Operation *op,
                                     Attribute encoding) {
  auto oldType = cast<RankedTensorType>(op->getResult(0).getType());
  auto newType = oldType.cloneWithEncoding(encoding);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
    // The DenseElementsAttr is typed, so it has to be rebuilt against the new
    // tensor type rather than carried over.
    auto dense = cast<DenseElementsAttr>(cstOp.getValue());
    return arith::ConstantOp::create(rewriter, op->getLoc(), newType,
                                     dense.resizeSplat(newType));
  }
  if (auto splatOp = dyn_cast<triton::SplatOp>(op))
    return triton::SplatOp::create(rewriter, op->getLoc(), newType,
                                   splatOp.getSrc());
  auto rangeOp = cast<triton::MakeRangeOp>(op);
  return triton::MakeRangeOp::create(rewriter, op->getLoc(), newType,
                                     rangeOp.getStart(), rangeOp.getEnd());
}

// Rebuild a side load, a bias or residual operand, so that it delivers its
// data directly in `encoding` instead of being converted into it. Output
// fusion extras are loads from func block arguments, so which layout they
// arrive in is a free choice, and the pointer and mask are index arithmetic of
// exactly the kind this pattern already relayouts for the store itself.
//
// Removing the load's conversion is the point, rather than making it cheap:
// gating side operands on !cvtNeedsSharedMemory would reject the common bias
// case, since `blocked -> mma` usually does need shared memory.
static Value rematerializeLoadInEncoding(PatternRewriter &rewriter,
                                         triton::LoadOp loadOp,
                                         Attribute encoding) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loadOp);

  auto relayout = [&](Value v) -> Value {
    auto tensorType = v ? dyn_cast<RankedTensorType>(v.getType()) : nullptr;
    if (!tensorType)
      return v;
    return triton::gpu::ConvertLayoutOp::create(
        rewriter, v.getLoc(), tensorType.cloneWithEncoding(encoding), v);
  };

  // Bound in order, since the order of evaluation of call arguments would not
  // be, and it decides the order the conversions are emitted in.
  Value newPtr = relayout(loadOp.getPtr());
  Value newMask = relayout(loadOp.getMask());
  Value newOther = relayout(loadOp.getOther());

  auto newType =
      cast<RankedTensorType>(loadOp.getType()).cloneWithEncoding(encoding);
  return triton::LoadOp::create(rewriter, loadOp.getLoc(), newType, newPtr,
                                newMask, newOther, loadOp.getCache(),
                                loadOp.getEvict(), loadOp.getIsVolatile());
}

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

// Whether the conversion this pattern would remove needs more LDS than the
// caller allowed the kernel as a whole. Epilogue scratch reuses the space the
// pipelined operand buffers free once the loop is done, so what it has to fit
// in is the whole budget rather than a remainder.
//
// `rock.max_lds` is a caller-supplied ceiling in bytes, carried onto the
// tt.func as a plain IntegerAttr by rock-tensor-to-triton-ptr. Absent means
// the caller names no ceiling, leaving the decision to the performance
// heuristics. A non-positive value is rejected downstream by
// rock-resolve-kernel-launch-params, so it is no reason to force a bypass
// here.
static bool exceedsLdsBudget(triton::FuncOp funcOp,
                             triton::gpu::ConvertLayoutOp cvtOp,
                             triton::AMD::TargetInfo &targetInfo) {
  if (!funcOp)
    return false;
  auto maxLds =
      dyn_cast_or_null<IntegerAttr>(funcOp->getDiscardableAttr("rock.max_lds"));
  if (!maxLds || maxLds.getInt() <= 0)
    return false;

  unsigned roundTripBytes = triton::AMD::getConvertLayoutScratchInBytes(
      cvtOp.getSrc().getType(), cvtOp.getType(), targetInfo);
  return static_cast<int64_t>(roundTripBytes) > maxLds.getInt();
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
// elementWiseOp(val, ...) : blocked
// ...
// elementWiseOp(val, ...) : blocked
// tt.store(ptr, val, mask, ...) : blocked
// ==>
// convert(ptr) : blocked -> xmma
// convert(mask) : blocked -> xmma
// elementWiseOp(val, ...) : xmma
// ...
// elementWiseOp(val, ...) : xmma
// tt.store(ptr, val, mask, ...) : xmma
//
// Store with xmma layout directly
//
// xmma layout is either MFMA or WMMA, or the blocked layout of an FMA dot when
// that keeps the store as wide as the coalesced layout would.
//
// The ops between the conversion and the store form a DAG, not a chain: an
// epilogue can read the accumulator more than once (SiLU is `acc * sigmoid
// (acc)`) and can mix in operands that are free in any layout (ReLU's zero
// splat). Both are bypassed at no cost. A side load, a bias or residual, is
// bypassed by rematerializing the load in the accumulator layout, which trades
// the load's conversion for a conversion of its pointer and mask. What is not
// handled is anything needing cross-lane movement, such as a row reduction or
// a softmax tail; those keep their round trip.
class BypassEpilogueSMEM : public mlir::OpRewritePattern<triton::StoreOp> {

public:
  BypassEpilogueSMEM(mlir::MLIRContext *context,
                     triton::AMD::TargetInfo &targetInfo)
      : OpRewritePattern(context), targetInfo(targetInfo) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp stOp,
                  mlir::PatternRewriter &rewriter) const override {

    Value ptr = stOp.getPtr();
    Value val = stOp.getValue();
    auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
    auto valType = dyn_cast<RankedTensorType>(val.getType());
    if (!ptrType || !valType ||
        !isa<triton::gpu::BlockedEncodingAttr>(ptrType.getEncoding()) ||
        !isa<triton::gpu::BlockedEncodingAttr>(valType.getEncoding()))
      return mlir::failure();

    Operation *valDef = val.getDefiningOp();
    if (!valDef)
      return mlir::failure();

    // Collect the elementwise cone feeding the store. The filter both selects
    // ops and halts the traversal through them, so the cone terminates by
    // construction at the layout conversion, at loads, and at constants. The
    // result is topologically sorted, which is the order the retype below has
    // to run in. An empty cone is the unfused case, where the store consumes
    // the conversion directly.
    SetVector<Operation *> cone;
    if (!isa<triton::gpu::ConvertLayoutOp>(valDef)) {
      if (!isRelayoutableInPlace(valDef))
        return mlir::failure();
      BackwardSliceOptions options;
      options.omitBlockArguments = true;
      options.filter = isRelayoutableInPlace;
      if (failed(getBackwardSlice(valDef, &cone, options)))
        return mlir::failure();
      // getBackwardSlice omits its own root, which is topologically last.
      cone.insert(valDef);
    }

    // Classify the values entering the cone. Exactly one has to be the
    // accumulator's layout conversion, the one this pattern removes; every
    // other one has to be buildable in the accumulator layout for free.
    triton::gpu::ConvertLayoutOp cvtOp;
    llvm::SmallVector<Operation *> agnosticDefs;
    llvm::SmallVector<triton::LoadOp> sideLoads;
    auto classifyBoundary = [&](Value operand) -> LogicalResult {
      Operation *def = operand.getDefiningOp();
      if (!def)
        return mlir::failure();
      if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(def)) {
        // Two different accumulator sources would each want their own layout.
        if (cvtOp && cvtOp != cvt)
          return mlir::failure();
        cvtOp = cvt;
        return mlir::success();
      }
      if (auto load = dyn_cast<triton::LoadOp>(def)) {
        if (!isa<RankedTensorType>(load.getType()))
          return mlir::failure();
        if (!llvm::is_contained(sideLoads, load))
          sideLoads.push_back(load);
        return mlir::success();
      }
      if (!isLayoutAgnostic(def))
        return mlir::failure();
      if (!llvm::is_contained(agnosticDefs, def))
        agnosticDefs.push_back(def);
      return mlir::success();
    };

    if (cone.empty()) {
      if (failed(classifyBoundary(val)))
        return mlir::failure();
    } else {
      for (Operation *op : cone) {
        if (op->getNumResults() != 1 ||
            !isa<RankedTensorType>(op->getResult(0).getType()))
          return mlir::failure();
        for (Value operand : op->getOperands()) {
          // Scalar operands carry no layout, so the bypass leaves them be.
          if (!isa<RankedTensorType>(operand.getType()))
            continue;
          if (cone.contains(operand.getDefiningOp()))
            continue;
          if (failed(classifyBoundary(operand)))
            return mlir::failure();
        }
      }
    }
    if (!cvtOp)
      return mlir::failure();

    // A kernel with several results stores more than one value off the same
    // accumulator, so a value in the cone, or the conversion itself, can be
    // read by a store other than the one matched here:
    //
    //   %c   = ttg.convert_layout %acc : #mma -> #blocked
    //   tt.store %p0, %c                              // the raw result
    //   %r   = arith.maxnumf %c, %zero                // the activated one
    //   tt.store %p1, %r
    //
    // Rewriting one store in isolation is impossible: whichever is matched,
    // the other still reads the old layout. They have to move together, so
    // collect them and treat a store as an acceptable user of the cone.
    // The retype below mutates result types in place, so nothing outside the
    // cone may still be observing the old type. Reconvergence *inside* the
    // cone is fine, which is what makes a SiLU- or GELU-shaped epilogue
    // bypassable: the accumulator is one value read more than once. The
    // conversion has to become dead too, or the LDS round trip stays and the
    // bypass buys an uncoalesced store for nothing.
    //
    // The one outside user that does not disqualify the bypass is another
    // store, which is how a kernel with several results reads one epilogue:
    //
    //   %c = ttg.convert_layout %acc : #mma -> #blocked
    //   tt.store %p0, %c                              // the raw result
    //   %r = arith.maxnumf %c, %zero                  // the activated one
    //   tt.store %p1, %r
    //
    // Neither store can move on its own, since whichever is rewritten first
    // leaves the other reading a value whose layout just changed. Collect
    // them and move them together.
    llvm::SmallVector<triton::StoreOp> stores{stOp};
    auto collectStoreUsers = [&](Operation *op) -> LogicalResult {
      for (OpOperand &use : op->getResult(0).getUses()) {
        Operation *user = use.getOwner();
        if (user == stOp || cone.contains(user))
          continue;
        // Only the stored value may come from the epilogue. An epilogue value
        // used as a store's pointer or mask means addressing computed from
        // the data, which is not a shape this pattern rewrites.
        auto otherSt = dyn_cast<triton::StoreOp>(user);
        if (!otherSt || use.get() != otherSt.getValue())
          return mlir::failure();
        auto otherPtrType =
            dyn_cast<RankedTensorType>(otherSt.getPtr().getType());
        if (!otherPtrType ||
            !isa<triton::gpu::BlockedEncodingAttr>(otherPtrType.getEncoding()))
          return mlir::failure();
        if (!llvm::is_contained(stores, otherSt))
          stores.push_back(otherSt);
      }
      return mlir::success();
    };

    for (Operation *op : cone)
      if (failed(collectStoreUsers(op)))
        return mlir::failure();
    if (failed(collectStoreUsers(cvtOp)))
      return mlir::failure();

    // A side load is replaced, not duplicated, so it too has to become dead.
    // Leaving a user behind would mean loading the same data twice, once per
    // layout, which costs more than the conversion being removed.
    for (triton::LoadOp load : sideLoads)
      for (Operation *user : load->getUsers())
        if (user != stOp && !cone.contains(user))
          return mlir::failure();

    // Everything above is legality: epilogue shapes this pattern cannot
    // express, which it has to refuse whatever the budget says. What follows
    // is policy, and the caller's LDS ceiling outranks all of it.
    //
    // MIGraphX picks a perf config against an unfused problem key and reuses
    // it once the kernel is fused, so the epilogue round trip can be the
    // difference between the tile fitting in LDS and not. Keeping it for a
    // wider store would only move the failure to
    // rock-resolve-kernel-launch-params, which has no tile left to shrink.
    triton::FuncOp funcOp = stOp->getParentOfType<triton::FuncOp>();
    const bool mustBypass = exceedsLdsBudget(funcOp, cvtOp, targetInfo);

    if (!mustBypass) {
      if (!isProfitableBypassSource(cvtOp.getSrc().getType(), valType))
        return mlir::failure();

      if (funcOp && funcOp->getDiscardableAttr("rock.prefer_lds_epilogue"))
        return mlir::failure();
    }

    auto newEncoding =
        cast<RankedTensorType>(cvtOp.getSrc().getType()).getEncoding();

    auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
    Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                        newPtrType, ptr);

    // Map the cone's boundary values into the accumulator layout: the
    // conversion collapses to its own source, the layout-agnostic operands get
    // rebuilt there, and the side loads get reissued there.
    llvm::DenseMap<Value, Value> remapped;
    remapped[cvtOp.getResult()] = cvtOp.getSrc();
    for (Operation *def : agnosticDefs)
      remapped[def->getResult(0)] =
          rematerializeInEncoding(rewriter, def, newEncoding);
    for (triton::LoadOp load : sideLoads)
      remapped[load.getResult()] =
          rematerializeLoadInEncoding(rewriter, load, newEncoding);

    // Retype the cone in topological order, rewiring every operand rather than
    // only operand 0. Values defined inside the cone keep their identity, so
    // they need no remapping: their type has already been updated by the time
    // a later op reads them.
    for (Operation *op : cone) {
      rewriter.modifyOpInPlace(op, [&]() {
        for (OpOperand &operand : op->getOpOperands())
          if (Value mapped = remapped.lookup(operand.get()))
            operand.set(mapped);
        Value result = op->getResult(0);
        result.setType(cast<RankedTensorType>(result.getType())
                           .cloneWithEncoding(newEncoding));
      });
    }

    // Each store keeps its own destination and its own value; only the layout
    // they agree on has changed. The matched store's pointer was converted
    // above, the rest are converted here.
    for (triton::StoreOp curSt : stores) {
      // Build each replacement where its original stood, so the conversions
      // it needs are dominated by their operands and the stores keep their
      // order relative to the rest of the block.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(curSt);

      Value curPtr = curSt == stOp ? newPtr : nullptr;
      if (!curPtr) {
        auto curPtrType = cast<RankedTensorType>(curSt.getPtr().getType());
        curPtr = triton::gpu::ConvertLayoutOp::create(
            rewriter, curSt.getPtr().getLoc(),
            curPtrType.cloneWithEncoding(newEncoding), curSt.getPtr());
      }

      // The stored value has already been retyped in place, unless it is the
      // conversion itself, which collapses to its own source.
      Value curVal = curSt.getValue();
      if (Value mapped = remapped.lookup(curVal))
        curVal = mapped;

      Value curMask = curSt.getMask();
      if (curMask) {
        auto maskType = cast<RankedTensorType>(curMask.getType());
        curMask = triton::gpu::ConvertLayoutOp::create(
            rewriter, curMask.getLoc(), maskType.cloneWithEncoding(newEncoding),
            curMask);
      }

      triton::StoreOp newStoreOp = usePermlaneSwapToOptimizeStore(
          rewriter, curPtr, curVal, curMask, curSt);
      if (!newStoreOp) {
        newStoreOp = triton::StoreOp::create(rewriter, curSt.getLoc(), curPtr,
                                             curVal, curMask, curSt.getCache(),
                                             curSt.getEvict());
      }
      rewriter.replaceOp(curSt, newStoreOp);
    }
    return mlir::success();
  }

private:
  // Not const: getConvertLayoutScratchInBytes takes a mutable TargetInfoBase.
  triton::AMD::TargetInfo &targetInfo;
};

// Deliver a loaded value in the layout its consumer wants, instead of loading
// it in one layout and converting the data afterwards.
//
// BypassEpilogueSMEM handles the operands it finds inside the store's backward
// slice, but a bias add is already written in the accumulator's layout by the
// time this pass runs:
//
//   %bias = tt.load %ptr            : tensor<MxNxf16, #blocked>
//   %c    = ttg.convert_layout %bias : #blocked -> #mma
//   %sum  = arith.addf %acc, %c      : tensor<MxNxf16, #mma>
//
// so the accumulator never needs a conversion and the store bypass has nothing
// left to fix, while `%c` still costs M*N*sizeof(elem) of scratch. That is the
// whole epilogue cost of the most common fusion there is, and it is invisible
// unless the tile is large enough to outgrow what the main loop already
// allocated for its operand buffers.
//
// A load's layout is a free choice, so the conversion can be removed outright
// by relayouting the pointer instead of the data.
struct BypassLoadConversion
    : public mlir::OpRewritePattern<triton::gpu::ConvertLayoutOp> {
  BypassLoadConversion(mlir::MLIRContext *context,
                       triton::AMD::TargetInfo &targetInfo)
      : mlir::OpRewritePattern<triton::gpu::ConvertLayoutOp>(context),
        targetInfo(targetInfo) {}

  mlir::LogicalResult
  matchAndRewrite(triton::gpu::ConvertLayoutOp cvtOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loadOp = cvtOp.getSrc().getDefiningOp<triton::LoadOp>();
    if (!loadOp || !isa<RankedTensorType>(loadOp.getType()))
      return mlir::failure();

    // The load is moved, not duplicated. A second reader would mean fetching
    // the same data once per layout, which costs more than the conversion.
    if (!loadOp->hasOneUse())
      return mlir::failure();

    auto newType = dyn_cast<RankedTensorType>(cvtOp.getType());
    if (!newType)
      return mlir::failure();
    Attribute newEncoding = newType.getEncoding();
    if (!isa_and_nonnull<triton::gpu::DistributedEncodingTrait>(newEncoding))
      return mlir::failure();

    // A `blocked -> dot_op` conversion is the GEMM's own operand staging, not
    // a fusion artifact: that trip through LDS is how the tile reaches the
    // matrix cores, and the loop is built around it. Only conversions that
    // exist to reconcile a fused value with the accumulator are fair game.
    if (isa<triton::gpu::DotOperandEncodingAttr>(newEncoding))
      return mlir::failure();

    // Reissuing the load in the consumer's layout can cost global-load
    // coalescing, so it is worth it only to stay inside a ceiling the caller
    // actually named. Without `rock.max_lds` there is no reason to prefer one
    // over the other and the conversion stays.
    triton::FuncOp funcOp = cvtOp->getParentOfType<triton::FuncOp>();
    if (!exceedsLdsBudget(funcOp, cvtOp, targetInfo))
      return mlir::failure();

    Value newLoad = rematerializeLoadInEncoding(rewriter, loadOp, newEncoding);
    rewriter.replaceOp(cvtOp, newLoad);
    rewriter.eraseOp(loadOp);
    return mlir::success();
  }

private:
  // Not const: getConvertLayoutScratchInBytes takes a mutable TargetInfoBase.
  triton::AMD::TargetInfo &targetInfo;
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

    triton::AMD::TargetInfo targetInfo(getAMDArch(m));

    mlir::RewritePatternSet patterns(context);

    patterns.add<BypassEpilogueSMEM, BypassLoadConversion>(context, targetInfo);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
