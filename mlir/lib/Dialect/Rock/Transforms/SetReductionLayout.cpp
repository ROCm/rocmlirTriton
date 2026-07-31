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
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

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

// Walk from a dot operand back to the global load that produces it. This pass
// runs before the operand is staged through shared memory, so the only thing
// standing between the two is the layout-only convert_layout that feeds the
// accelerator dot. Returns null if the def chain does not find a
// tt.load / amdgpu.buffer_load.
Operation *findFeedingLoad(Value operand) {
  Operation *def = operand.getDefiningOp();
  while (def) {
    if (isa<triton::LoadOp, triton::amdgpu::BufferLoadOp>(def))
      return def;
    if (!isa<triton::gpu::ConvertLayoutOp>(def))
      return nullptr;
    def = def->getOperand(0).getDefiningOp();
  }
  return nullptr;
}

// The blocked encoding of `load`'s result, or null when that result is not a
// blocked-encoded tensor and so holds nothing this pass can redistribute.
triton::gpu::BlockedEncodingAttr getLoadBlockedEncoding(Operation *load) {
  auto ty = dyn_cast<RankedTensorType>(load->getResult(0).getType());
  if (!ty)
    return nullptr;
  return dyn_cast<triton::gpu::BlockedEncodingAttr>(ty.getEncoding());
}

// Which side of a dot a value feeds, which is what fixes where its reduction
// dim sits.
enum class OperandRole { A, B };

// The reduction (K) dim of a dot operand. Per the `DotOpInterface` contract
// (see its `verifyDims`), K is the last dim of A and the second-to-last dim of
// B. Operands may be 2D or 3D (batched), so the dim follows from the rank
// rather than being fixed at 1 / 0. Returns none for operands whose type this
// pass cannot reason about.
std::optional<unsigned> getReductionDim(Value operand, OperandRole role) {
  auto ty = dyn_cast<RankedTensorType>(operand.getType());
  if (!ty || ty.getRank() < 2)
    return std::nullopt;
  int64_t rank = ty.getRank();
  return static_cast<unsigned>(role == OperandRole::A ? rank - 1 : rank - 2);
}

// True when `load` reads its operand "gather"-style for a reduction along
// `kDim`: K is the slowest-varying axis of the blocked encoding, so lanes run
// along the contiguous axis and each K index a thread touches costs its own
// address computation. Only such a load has anything to gain from moving warps
// onto K.
//
// At most one `kDim` can satisfy this for a given load, since `order.back()` is
// a single dim. Two dot operands that disagree about K therefore cannot both
// see the same load as a gather, which is what keeps the load -> kDim mapping
// unambiguous without any conflict tracking.
//
// TODO: This gate is narrower than the register-pressure argument requires.
// What makes the collapse pay off is that the slowest-varying dim is
// warp-uniform (`threadsPerWarp[dim] == 1`), so the offsets for its repetitions
// are scalar; that the dim is also K is incidental. An A operand with
// `threadsPerWarp = [1, 32]` and `order = [1, 0]` has a warp-uniform free dim
// (M) and a lane-varying K, and is declined here even though collapsing warps
// onto M would shrink the same scalar address chain. A sweep of the tier1 conv
// configs found such a load in 24 of 1363 kernels. Worth supporting, but gated
// and measured on its own rather than folded into the K case.
bool isGatherLoad(Operation *load, unsigned kDim) {
  auto blocked = getLoadBlockedEncoding(load);
  if (!blocked)
    return false;
  auto order = blocked.getOrder();
  return !order.empty() && order.back() == kDim;
}

// True when `user` keeps working if an operand it reads changes distributed
// layout, so a scoped value it consumes needs no old-layout copy.
// convert_layout sinks the tensor into a differently-typed result (a
// dot-operand layout) that is decoupled from the source's distributed encoding,
// so only its element type and shape -- which this rewrite leaves alone -- must
// still match. scf.for and scf.yield carry the value across a loop edge whose
// type is fixed up explicitly.
bool toleratesRelayout(Operation *user) {
  return isa<triton::gpu::ConvertLayoutOp, scf::ForOp, scf::YieldOp>(user);
}

// Redistribute a gather-style blocked global load's warps onto its reduction
// (K) dim, given by kDim. The caller supplies kDim from the dot operand this
// load feeds, and must have established `isGatherLoad(load, kDim)`.
// The rewrite is scoped to the load's own use-def slice (the load, the backward
// slice feeding its pointer/mask/other operands, and the loop-carried edges that
// slice crosses) rather than applied module-wide: TTG encodings are uniqued by
// content, so a module-wide substitution keyed on the encoding would also
// rewrite unrelated values that merely happen to share it.
// Where that slice overlaps a chain another op still needs at the old layout,
// the overlap is duplicated rather than the rewrite declined.
// Returns true when the load's layout was changed.
bool rewriteGatherLoad(Operation *load, unsigned kDim) {
  auto oldBlocked = getLoadBlockedEncoding(load);
  assert(oldBlocked && isGatherLoad(load, kDim) &&
         "rewriteGatherLoad expects a gather-style blocked-encoded load");
  auto ty = cast<RankedTensorType>(load->getResult(0).getType());
  ArrayRef<int64_t> shape = ty.getShape();

  SmallVector<unsigned> order(oldBlocked.getOrder());
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
    LLVM_DEBUG(llvm::dbgs() << "rock-set-reduction-layout: warps do not tile "
                               "the reduction dim; skipping\n");
    return false;
  }

  MLIRContext *ctx = load->getContext();
  auto newBlocked = triton::gpu::BlockedEncodingAttr::get(
      ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order,
      oldBlocked.getCGALayout());
  if (newBlocked == oldBlocked) {
    LLVM_DEBUG(llvm::dbgs() << "rock-set-reduction-layout: load already in the "
                               "desired layout; skipping\n");
    return false;
  }

  // Collect exactly the ops whose types must move to the redistributed layout:
  // the load itself and the backward slice feeding its pointer/mask/other
  // operands (splat/addptr/make_range/broadcast/offset constants that share the
  // load encoding).
  llvm::SetVector<Operation *> scope;
  BackwardSliceOptions sliceOpts;
  sliceOpts.omitBlockArguments = true;
  (void)getBackwardSlice(load, &scope, sliceOpts);
  scope.insert(load);

  // Close the scope over scf.for loop-carried edges. The load's pointer/mask
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
  replacer.addReplacement(
      [oldBlocked, newBlocked](Attribute attr) -> std::optional<Attribute> {
        if (attr == oldBlocked)
          return Attribute(newBlocked);
        return std::nullopt;
      });

  auto changesType = [&replacer](Value v) {
    return replacer.replace(v.getType()) != v.getType();
  };

  // Retyping a scoped op in place changes the type seen by every consumer of
  // its result, including consumers outside the scope. Collect the scoped ops
  // that some outside consumer still needs at the old layout, so they must keep
  // an old-layout version. This is routine rather than exceptional: CSE merges
  // the gather load's address arithmetic with that of an unrelated load or of
  // the epilogue store whenever the two span the same extent in the same
  // layout, leaving one make_range / addi / constant chain feeding both.
  llvm::SetVector<Operation *> keepOldCopy;
  for (Operation *op : scope)
    for (Value result : op->getResults()) {
      if (!changesType(result))
        continue;
      if (llvm::any_of(result.getUsers(), [&scope](Operation *user) {
            return !scope.contains(user) && !toleratesRelayout(user);
          }))
        keepOldCopy.insert(op);
    }

  // Close backwards over the scope: an op held at the old layout needs its
  // operands at the old layout too. Indexing a SetVector while inserting into
  // it is the usual worklist idiom, since insertions append.
  for (unsigned i = 0; i < keepOldCopy.size(); ++i)
    for (Value operand : keepOldCopy[i]->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (scope.contains(def) && changesType(operand))
          keepOldCopy.insert(def);

  // The load is the one op that cannot be duplicated: its result is what has to
  // reach the dot at the new layout, so an outside consumer holding it at the
  // old layout leaves nothing to gain.
  if (keepOldCopy.contains(load)) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-set-reduction-layout: the load's own result is read at "
                  "the old layout outside the rewrite scope; skipping\n");
    return false;
  }
  // Retyping a loop-carried slot moves the layout of the whole slot at once:
  // its init operand, its iter_arg, its yielded value and its loop result. The
  // cases below are the ones where some part of the slot cannot follow, which
  // would leave the loop signature inconsistent and fail the verifier. Since
  // this pass is an optimization behind a knob, it declines instead: it must
  // never turn a kernel that compiles into one that does not. All of these are
  // therefore settled before the first mutation, so that declining leaves the
  // IR exactly as it was found.
  for (auto &[forOp, indices] : forFixups)
    for (unsigned i : indices) {
      // A reader of the iter_arg that stays at the old layout -- an outside
      // consumer that does not tolerate a relayout, or a scoped op kept at the
      // old layout to serve one -- would need the slot to hold both layouts at
      // once, that is, a second iter_arg.
      if (llvm::any_of(forOp.getRegionIterArg(i).getUsers(),
                       [&scope, &keepOldCopy](Operation *user) {
                         return keepOldCopy.contains(user) ||
                                (!scope.contains(user) &&
                                 !toleratesRelayout(user));
                       })) {
        LLVM_DEBUG(llvm::dbgs()
                   << "rock-set-reduction-layout: a loop-carried iter_arg is "
                      "also read at the old layout; skipping\n");
        return false;
      }
      // The loop result is retyped along with the slot, so a post-loop reader
      // would need a convert_layout on the way out.
      if (!forOp.getResult(i).use_empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "rock-set-reduction-layout: a retyped loop-carried slot "
                      "has post-loop uses; skipping\n");
        return false;
      }
      // The init and yielded values move with the slot only if some op the
      // rewrite reaches produces them. One arriving as a block argument -- an
      // enclosing loop's iter_arg, say -- has no producer to retype here.
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (Value edge : {forOp.getInitArgs()[i], yieldOp.getOperand(i)})
        if (!edge.getDefiningOp() && changesType(edge)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "rock-set-reduction-layout: a loop-carried edge is "
                        "itself a block argument; skipping\n");
          return false;
        }
    }

  // An scf.for init operand and the matching scf.yield operand sit on ops that
  // are not in the scope themselves -- the loop encloses the load rather than
  // feeding it -- yet they carry the slots retyped above. Name them here so
  // that a duplicated producer rewires them along with the in-scope uses,
  // instead of leaving the loop reading its init at the old layout.
  auto isRetypedLoopOperand = [&forFixups](OpOperand &use) {
    Operation *owner = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
      auto it = forFixups.find(forOp);
      if (it == forFixups.end())
        return false;
      unsigned firstInit = forOp.getNumControlOperands();
      unsigned idx = use.getOperandNumber();
      return idx >= firstInit && it->second.contains(idx - firstInit);
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
      auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
      if (!forOp)
        return false;
      auto it = forFixups.find(forOp);
      return it != forFixups.end() &&
             it->second.contains(use.getOperandNumber());
    }
    return false;
  };

  // Duplicate the shared part of the chain instead of declining the rewrite.
  // The original keeps the old layout for its outside consumers; the copy takes
  // over the uses inside the scope and is retyped in its place. What gets
  // duplicated is address arithmetic on coordinates that are uniform anyway, so
  // the cost is a handful of scalar instructions.
  llvm::SetVector<Operation *> rewrite(scope.begin(), scope.end());
  IRMapping mapping;
  for (Operation *op : topologicalSort(keepOldCopy)) {
    OpBuilder builder(op);
    builder.setInsertionPointAfter(op);
    // Cloning in topological order means the copy's operands are remapped to
    // the copies of any producers duplicated alongside it, while operands from
    // ops retyped in place stay shared.
    Operation *copy = builder.clone(*op, mapping);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op->getResults(), copy->getResults()))
      oldResult.replaceUsesWithIf(newResult, [&](OpOperand &use) {
        Operation *owner = use.getOwner();
        return isRetypedLoopOperand(use) ||
               (scope.contains(owner) && !keepOldCopy.contains(owner));
      });
    rewrite.remove(op);
    rewrite.insert(copy);
  }

  // The replacer recurses into nested encodings, so slice<{parent = #blocked}>
  // and the like are rewritten too. Applying it per op rewrites attribute
  // dictionaries and result types locally.
  for (Operation *op : rewrite)
    replacer.recursivelyReplaceElementsIn(op, /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);

  // arith.constant keeps its value as an inherent attribute (a property), which
  // the dictionary rewrite above does not reach. Reshape any rewritten constant
  // so its dense/splat value type stays consistent with the freshly rewritten
  // result type.
  for (Operation *op : rewrite) {
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
  // producers were rewritten as scoped ops, or rewired onto their duplicates,
  // so the ForOp operand and yield operand types have already moved; the
  // block-argument and result types are held on the ForOp itself and are
  // updated here to keep the loop signature consistent. Both are known safe to
  // move by the checks made before the rewrite began.
  for (auto &[forOp, indices] : forFixups) {
    for (unsigned i : indices) {
      BlockArgument arg = forOp.getRegionIterArg(i);
      arg.setType(replacer.replace(arg.getType()));
      Value res = forOp.getResult(i);
      res.setType(replacer.replace(res.getType()));
    }
  }
  return true;
}
} // end anonymous namespace

void RockSetReductionLayoutPass::runOnOperation() {
  // The `useReductionLayout` perfConfig knob is a tri-state gate:
  //   -1 (heuristic default): rewrite only convolution kernels (those carrying
  //      the `rock.conv_kernel` attribute).
  //    0 (off): disable the rewrite entirely; no kernel is touched.
  //    1 (on): force the rewrite on every kernel.
  // TODO: Investigate if this can be beneficial for non-convolution kernels.
  // https://amd-hub.atlassian.net/browse/AIROCMLIR-1049
  if (useReductionLayout == 0)
    return;
  bool forceAll = useReductionLayout == 1;
  ModuleOp mod = getOperation();

  // Analysis: map each gather-style load to the reduction dim of the dot
  // operand it feeds. Keyed on the load rather than on the dot operand because
  // one load can feed several operands, and the rewrite must run once per load.
  //
  // This walk must stay read-only. `rewriteGatherLoad` retypes ops in place --
  // including `scf.for` iter_args and results -- which is not safe to do while
  // a walk over the same region is in flight, so the rewrite happens afterwards
  // in a separate loop.
  llvm::MapVector<Operation *, unsigned> gatherLoads;
  auto record = [&](triton::DotOpInterface dot, Value operand,
                    OperandRole role) {
    std::optional<unsigned> kDim = getReductionDim(operand, role);
    if (!kDim)
      return;
    Operation *load = findFeedingLoad(operand);
    if (!load || !isGatherLoad(load, *kDim))
      return;
    LLVM_DEBUG(llvm::dbgs() << "rock-set-reduction-layout: gather load for "
                            << dot->getName() << " at " << dot.getLoc()
                            << ", reduction dim " << *kDim << "\n");
    // try_emplace keeps any value already recorded, so this also checks that a
    // load claimed by a second operand agrees. A load can be a gather for only
    // one reduction dim (see `isGatherLoad`), so a mismatch is impossible.
    auto it = gatherLoads.try_emplace(load, *kDim).first;
    assert(it->second == *kDim &&
           "one load claimed as a gather for two different reduction dims");
    (void)it;
  };
  mod.walk([&](triton::FuncOp func) {
    if (!forceAll && !func->hasAttr(rock::ConvKernelAttr::getMnemonic()))
      return;
    func.walk([&](triton::DotOpInterface dot) {
      record(dot, dot.getA(), OperandRole::A);
      record(dot, dot.getB(), OperandRole::B);
    });
  });
  if (gatherLoads.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-set-reduction-layout: no dot operand is fed by a "
                  "gather-style global load; nothing to redistribute\n");
    return;
  }

  for (auto [load, kDim] : gatherLoads)
    (void)rewriteGatherLoad(load, kDim);
}
