//===- SetReductionLayout.cpp - Redistribute reduction-operand load ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "amd/include/Analysis/AMDGPUAllocation.h"
#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "lib/TritonAMDGPUToLLVM/TargetInfo.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <utility>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSETREDUCTIONLAYOUTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-set-reduction-layout"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockSetReductionLayoutPass
    : public rock::impl::RockSetReductionLayoutPassBase<
          RockSetReductionLayoutPass> {
  using rock::impl::RockSetReductionLayoutPassBase<
      RockSetReductionLayoutPass>::RockSetReductionLayoutPassBase;
  void runOnOperation() override;
};

// Walk from a dot operand back to the value the rewrite should redistribute,
// stepping through the layout-only convert_layout / in_thread_transpose and
// the local_alloc / local_load pair that stages an operand through shared
// memory.
//
// On an unfused kernel the walk ends on the global load itself. On a fused one
// it ends on the tail of the fusion prologue, which is anchored on only under
// the conditions below. Returns the result the walk arrived through, or null
// when neither is reached. The fusion path requires a blocked encoding; on the
// load path that is rewriteGatherLoad's to check.
//
// Please note that this will not rewrite direct-to-LDS loads or TDM.
Value findGatherAnchor(Value operand) {
  Value gathered = operand;
  while (Operation *def = gathered.getDefiningOp()) {
    if (isa<triton::LoadOp, triton::amdgpu::BufferLoadOp>(def))
      return gathered;
    if (isa<triton::gpu::ConvertLayoutOp, triton::amdgpu::InThreadTransposeOp>(
            def)) {
      gathered = def->getOperand(0);
      continue;
    }

    if (auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(def)) {
      auto alloc =
          localLoad.getSrc().getDefiningOp<triton::gpu::LocalAllocOp>();
      if (!alloc || !alloc.getSrc())
        return nullptr;
      gathered = alloc.getSrc();
      continue;
    }

    // Search for anchors on fusions

    // Bail out on loops/conditionals.
    if (def->getNumRegions() != 0)
      return nullptr;

    // The anchor must be a compute op.
    if (!isMemoryEffectFree(def))
      return nullptr;

    // The checks below prove the anchor's operands and result agree on one
    // layout; these traits prove the op does not care which layout that is.
    // tt.gather is why both are needed: its source, indices and result can all
    // carry the same shape and encoding, yet its lowering reads the layout.
    //
    // This is also what Triton's LayoutConversion checks before rewriting (ie
    // in RemoveLayoutConversions.cpp in LayoutPropagation class)
    if (!(def->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
          def->hasTrait<OpTrait::Elementwise>()))
      return nullptr;

    // The walk ends on this op. Everything below either takes `gathered` as the
    // anchor or gives up on this dot operand.
    auto ty = dyn_cast<RankedTensorType>(gathered.getType());
    if (!ty || !isa<triton::gpu::BlockedEncodingAttr>(ty.getEncoding()))
      return nullptr;

    // Make sure the encodings and the shapes of the operands agree with the
    // anchor. The checks above only cover one of the two.
    for (Value anchorOperand : def->getOperands()) {
      auto operandTy = dyn_cast<RankedTensorType>(anchorOperand.getType());
      if (!operandTy)
        continue; // A scalar operand carries no encoding to keep in step.
      if (operandTy.getEncoding() != ty.getEncoding() ||
          operandTy.getShape() != ty.getShape())
        return nullptr;
    }
    return gathered;
  }
  return nullptr;
}

// Redistribute a single blocked-encoded gather's warps onto its reduction (K)
// dim, given by kDim. The caller supplies kDim from the dot operand this
// gather feeds, and `anchor` from findGatherAnchor.
// The rewrite is scoped to the anchor's own use-def slice
// (the anchor, the backward slice feeding it, and any in_thread_transpose
// consuming it) rather than applied module-wide: TTG
// encodings are uniqued by content, so a module-wide substitution keyed on the
// encoding would also rewrite unrelated values that merely happen to share it.
// Returns true if the layout was rewritten.
bool rewriteGatherLoad(Value anchor, unsigned kDim) {
  Operation *anchorOp = anchor.getDefiningOp();
  MLIRContext *ctx = anchor.getContext();
  auto ty = dyn_cast<RankedTensorType>(anchor.getType());
  if (!ty)
    return false;
  auto oldBlocked =
      dyn_cast<triton::gpu::BlockedEncodingAttr>(ty.getEncoding());
  if (!oldBlocked)
    return false;
  ArrayRef<int64_t> shape = ty.getShape();

  // Only act on the "gather" operand (i.e., the reduction operand whose K is
  // the strided/slow axis)
  SmallVector<unsigned> order(oldBlocked.getOrder());
  if (order.empty() || order.back() != kDim)
    return false;

  SmallVector<unsigned> sizePerThread(oldBlocked.getSizePerThread());
  SmallVector<unsigned> threadsPerWarp(oldBlocked.getThreadsPerWarp());
  SmallVector<unsigned> warpsPerCTA(oldBlocked.getWarpsPerCTA());

  // Put every warp on the reduction dim, leaving the contiguous dim to lanes.
  // sizePerThread and threadsPerWarp are unchanged, so the total lane/warp
  // counts are preserved.
  unsigned totalWarps = 1;
  for (unsigned w : warpsPerCTA)
    totalWarps *= w;
  for (unsigned i = 0; i < warpsPerCTA.size(); ++i)
    warpsPerCTA[i] = (i == kDim) ? totalWarps : 1u;

  // Bail if the redistributed layout no longer tiles the load shape (e.g. the
  // reduction dim is too small to hold every warp), leaving it unchanged.
  bool tiles = true;
  for (unsigned d = 0; d < shape.size(); ++d) {
    unsigned cover = sizePerThread[d] * threadsPerWarp[d] * warpsPerCTA[d];
    assert(cover != 0 && "blocked encoding tile factors must be >= 1");
    if (shape[d] % cover != 0) {
      tiles = false;
      break;
    }
  }
  if (!tiles) {
    anchorOp->emitWarning("rock-set-reduction-layout: warps do not tile the "
                          "reduction dim; skipping");
    return false;
  }

  auto newBlocked = triton::gpu::BlockedEncodingAttr::get(
      ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order,
      oldBlocked.getCGALayout());
  if (newBlocked == oldBlocked) {
    LLVM_DEBUG(llvm::dbgs() << "rock-set-reduction-layout: load already in the "
                               "desired layout; skipping\n");
    return false;
  }

  // in_thread_transpose pairs this blocked encoding with a #linear derived via
  // deduceOutputLayout. Remap that pair too; when no in_thread_transpose
  // consumes the load, no value carries this #linear and the remap is inert.
  triton::LinearLayout oldLL =
      triton::amdgpu::InThreadTransposeOp::deduceOutputLayout(shape,
                                                              oldBlocked);
  triton::LinearLayout newLL =
      triton::amdgpu::InThreadTransposeOp::deduceOutputLayout(shape,
                                                              newBlocked);
  Attribute oldLinear =
      triton::gpu::LinearEncodingAttr::get(ctx, std::move(oldLL));
  Attribute newLinear =
      triton::gpu::LinearEncodingAttr::get(ctx, std::move(newLL));

  // Collect exactly the ops whose types must move to the redistributed layout:
  // the anchor itself, its backward slice (the loads it gathers from, any
  // fusion prologue between them, and the
  // splat/addptr/make_range/broadcast/offset constants that share the anchor's
  // encoding), and any in_thread_transpose consuming it.
  llvm::SetVector<Operation *> scope;
  BackwardSliceOptions sliceOpts;
  sliceOpts.omitBlockArguments = true;
  (void)getBackwardSlice(anchorOp, &scope, sliceOpts);
  scope.insert(anchorOp);
  for (Operation *user : anchor.getUsers())
    if (isa<triton::amdgpu::InThreadTransposeOp>(user))
      scope.insert(user);

  // Close the scope over scf.for loop-carried edges. The gather's pointer/mask
  // operands may be computed inside an scf.for from a value carried across
  // iterations, i.e. read from an iter_arg block argument and advanced through
  // the loop's yield. The backward slice stops at that block argument, so it
  // captures only the in-loop uses; but an iter_arg's type is one and the same
  // as its init operand, its yielded value, and the loop result. Retyping just
  // the in-loop uses would therefore leave the init/yield/iter_arg/result at
  // the old layout and produce a type mismatch on the loop signature. Pull the
  // init and yielded-value producers into the scope, and record which
  // iter_args/results must be retyped alongside (done after the rewrite).
  llvm::MapVector<scf::ForOp, llvm::SmallDenseSet<unsigned>> forFixups;
  auto addOpAndSlice = [&](Operation *op) {
    if (!op)
      return;
    // getBackwardSlice (inclusive=false) removes the root op from the set on
    // exit, so slice first and insert the op itself afterwards.
    (void)getBackwardSlice(op, &scope, sliceOpts);
    scope.insert(op);
  };
  for (bool grew = true; grew;) {
    grew = false;
    for (Operation *op : SmallVector<Operation *>(scope.begin(), scope.end())) {
      for (Value operand : op->getOperands()) {
        auto ba = dyn_cast<BlockArgument>(operand);
        if (!ba)
          continue;
        auto forOp = dyn_cast_or_null<scf::ForOp>(ba.getOwner()->getParentOp());
        if (!forOp || ba.getArgNumber() == 0) // 0 == induction variable
          continue;
        unsigned iterIdx = ba.getArgNumber() - 1;
        if (!forFixups[forOp].insert(iterIdx).second)
          continue;
        grew = true;
        addOpAndSlice(forOp.getInitArgs()[iterIdx].getDefiningOp());
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        addOpAndSlice(yieldOp.getOperand(iterIdx).getDefiningOp());
      }
    }
  }

  AttrTypeReplacer replacer;
  replacer.addReplacement([oldBlocked, newBlocked, oldLinear, newLinear](
                              Attribute attr) -> std::optional<Attribute> {
    if (attr == oldBlocked)
      return Attribute(newBlocked);
    if (attr == oldLinear)
      return newLinear;
    return std::nullopt;
  });

  // Correctness guard against shared producers. The in-place rewrite changes
  // the type seen by every consumer of a scoped op's result, so bail unless
  // each such consumer is one we handle. convert_layout and local_alloc are
  // safe outside consumers: they sink the distributed tensor into a
  // differently-typed result (a dot-operand layout or a shared-memory memdesc)
  // that is decoupled from the source's distributed encoding, so only its
  // element type and shape -- unchanged by this rewrite -- must still match.
  for (Operation *op : scope) {
    for (Value result : op->getResults()) {
      if (replacer.replace(result.getType()) == result.getType())
        continue;
      for (Operation *user : result.getUsers())
        if (!scope.contains(user) &&
            !isa<triton::gpu::ConvertLayoutOp, triton::gpu::LocalAllocOp,
                 scf::ForOp, scf::YieldOp>(user)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "rock-set-reduction-layout: rewrite would escape its "
                        "scope (value shared with an outside consumer); "
                        "skipping\n");
          return false;
        }
    }
  }

  // Guard for the case where we would retype a value that is read after the
  // loop. Its readers sit outside the scope and are not rewritten, so they
  // would be left expecting the old layout; bail out to avoid corrupting them.
  for (auto &[forOp, indices] : forFixups) {
    for (unsigned i : indices) {
      Value res = forOp.getResult(i);
      // Does `res` carry a layout that will be rewritten?
      // If not, it's safe, we don't need to bail out.
      if (replacer.replace(res.getType()) == res.getType())
        continue;
      if (!res.use_empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "rock-set-reduction-layout: loop-carried slot has a "
                      "post-loop use that would keep the old layout; "
                      "skipping\n");
        return false;
      }
    }
  }

  // The replacer recurses into nested encodings, so slice<{parent = #blocked}>
  // and the like are rewritten too. Applying it per scoped op rewrites
  // attribute dictionaries and result types locally.
  for (Operation *op : scope)
    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

  // arith.constant keeps its value as an inherent attribute (a property), which
  // the dictionary rewrite above does not reach. Reshape any scoped constant so
  // its dense/splat value type stays consistent with the freshly rewritten
  // result type.
  for (Operation *op : scope) {
    auto constOp = dyn_cast<arith::ConstantOp>(op);
    if (!constOp)
      continue;
    auto dense = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!dense)
      continue;
    Type nt = replacer.replace(dense.getType());
    if (nt != dense.getType())
      constOp.setValueAttr(dense.reshape(cast<ShapedType>(nt)));
  }

  // Retype the loop-carried slots discovered above. The init and yield
  // producers were rewritten as scoped ops (so the ForOp operand and yield
  // operand types already moved); the block-argument and result types are held
  // on the ForOp itself and are updated here to keep the loop signature
  // consistent.
  for (auto &[forOp, indices] : forFixups) {
    for (unsigned i : indices) {
      BlockArgument arg = forOp.getRegionIterArg(i);
      arg.setType(replacer.replace(arg.getType()));
      Value res = forOp.getResult(i);
      Type newResTy = replacer.replace(res.getType());
      // Re-check what the guard above established, so that a later change to
      // it fails here rather than somewhere downstream of this pass.
      assert((newResTy == res.getType() || res.use_empty()) &&
             "retyping a loop result that is still read would corrupt its "
             "readers");
      res.setType(newResTy);
    }
  }
  return true;
}

// Redistribute every gather in `mod` that feeds a dot unambiguously. With
// `forceAll`, every kernel is considered; otherwise only convolution ones.
// Returns true if any gather was rewritten.
bool redistributeGathers(ModuleOp mod, bool forceAll) {
  // Associate each dot operand with the gather anchor that feeds it and the
  // reduction (K) dim implied by its operand position.
  llvm::MapVector<Value, unsigned> anchorKDim;
  llvm::DenseSet<Value> conflicting;
  // TODO: Support the case where the same gather is assigned to multiple
  // tt.dots. This can be beneficial specially if we used DecomposeNonPow2 pass.
  auto record = [&](Value operand, unsigned kDim) {
    Value anchor = findGatherAnchor(operand);
    if (!anchor)
      return;
    auto [it, inserted] = anchorKDim.try_emplace(anchor, kDim);
    if (!inserted && it->second != kDim)
      conflicting.insert(anchor);
  };
  mod.walk([&](triton::FuncOp func) {
    if (!forceAll && !func->hasAttr(rock::ConvKernelAttr::getMnemonic()))
      return;
    func.walk([&](triton::DotOpInterface dot) {
      record(dot.getA(), /*kDim=*/1u);
      record(dot.getB(), /*kDim=*/0u);
    });
  });
  if (anchorKDim.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-set-reduction-layout: no dot operand is fed "
                  "by a global load; nothing to redistribute\n");
    return false;
  }

  // A single gather that feeds two dots as different operands (conflicting
  // reduction dims) is ambiguous; leave it untouched rather than guess.
  //
  // TODO: Duplicating the gather (one clone per reduction dim) would let each
  // dot keep its ideal layout, but whether it's beneficial is not clear.
  bool rewrote = false;
  for (auto [anchor, kDim] : anchorKDim) {
    if (conflicting.contains(anchor)) {
      anchor.getDefiningOp()->emitWarning(
          "rock-set-reduction-layout: load feeds dot operands "
          "with conflicting reduction dims; skipping");
      continue;
    }
    rewrote |= rewriteGatherLoad(anchor, kDim);
  }
  return rewrote;
}

// The LDS that `mod` will allocate, computed with the very
// analysis `AllocateAMDGPUSharedMemory` (which is the one that allocates LDS)
// runs a few passes later. Nothing
// between the two passes changes tensor encodings, so the value returned here
// is the same that will actually be allocated later. Returns nullopt when the
// module carries no AMD target to derive it from.
//
// Also, we dont want to grow the LDS size because MIGraphX assumes one
// perfConfig would not change its LDS size if fusions are involved.
// So making sure the LDS size does not grow also enforces that assumption.
std::optional<size_t> sharedMemoryFootprint(ModuleOp mod) {
  std::optional<StringRef> arch = getAMDArch(mod);
  if (!arch)
    return std::nullopt;
  triton::AMD::TargetInfo targetInfo(arch->str());
  auto scratchSizeGetter = [&targetInfo](Operation *op) {
    return triton::AMD::AMDAllocationAnalysisScratchSizeFn(op, targetInfo);
  };
  return ModuleAllocation(mod, scratchSizeGetter,
                          targetInfo.getSharedMemoryPartitionSize())
      .getSharedMemorySize();
}
} // end anonymous namespace

void RockSetReductionLayoutPass::runOnOperation() {
  ModuleOp mod = getOperation();

  // The `useReductionLayout` perfConfig knob is a tri-state gate:
  //   -1 (heuristic default): rewrite only convolution kernels (those carrying
  //      the `rock.conv_kernel` attribute).
  //    0 (off): disable the rewrite entirely; no kernel is touched.
  //    1 (on): force the rewrite on every kernel.
  // TODO(AIROCMLIR-1049): Investigate if this can be beneficial for
  // non-convolution kernels.
  if (useReductionLayout == 0)
    return;
  bool forceAll = useReductionLayout == 1;

  // Redistributing the layout may hurt performance if LDS size is increased.
  // So we decide if we should rewrite the layout or not based on the LDS size.
  // The rewrite is performed on a cloned module, and that clone replaces the
  // original only if its shared-memory footprint did not grow.
  OwningOpRef<ModuleOp> probe(mod.clone());
  if (!redistributeGathers(*probe, forceAll))
    return;

  // With no target attribute to size shared memory from, keep the rewrite
  // rather than silently dropping it.
  std::optional<size_t> before = sharedMemoryFootprint(mod);
  std::optional<size_t> after = sharedMemoryFootprint(*probe);
  if (before && after && *after > *before) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-set-reduction-layout: rewrite would grow shared memory "
                  "from "
               << *before << " to " << *after << " bytes; skipping\n");
    return;
  }

  mod.getBodyRegion().takeBody(probe->getBodyRegion());
}
