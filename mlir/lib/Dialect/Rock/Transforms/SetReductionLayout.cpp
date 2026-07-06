//===- SetReductionLayout.cpp - Redistribute reduction-operand load ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

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
  void runOnOperation() override;
};

// Walk from a dot operand back to the global load that produces it, through the
// ops the pipeline interposes between a gather load and tt.dot: the layout-only
// convert_layout / in_thread_transpose, and the local_alloc / local_load pair
// that stages an operand through shared memory. Returns null if the def chain
// does not find a tt.load / amdgpu.buffer_load.
Operation *findFeedingLoad(Value operand) {
  Operation *def = operand.getDefiningOp();
  while (def) {
    if (isa<triton::LoadOp, triton::amdgpu::BufferLoadOp>(def))
      return def;
    if (isa<triton::gpu::ConvertLayoutOp, triton::amdgpu::InThreadTransposeOp>(
            def)) {
      def = def->getOperand(0).getDefiningOp();
      continue;
    }

    if (auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(def)) {
      auto alloc =
          localLoad.getSrc().getDefiningOp<triton::gpu::LocalAllocOp>();
      if (!alloc || !alloc.getSrc())
        return nullptr;
      def = alloc.getSrc().getDefiningOp();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

// Redistribute a single blocked-encoded global load's warps onto its reduction
// (K) dim, given by kDim. The caller supplies kDim from the dot operand this
// load feeds.
// The rewrite is scoped to the load's own use-def slice
// (the load, the backward slice feeding its pointer/mask/other operands, and
// any in_thread_transpose consuming it) rather than applied module-wide: TTG
// encodings are uniqued by content, so a module-wide substitution keyed on the
// encoding would also rewrite unrelated values that merely happen to share it.
void rewriteGatherLoad(Operation *load, unsigned kDim) {
  MLIRContext *ctx = load->getContext();
  auto ty = dyn_cast<RankedTensorType>(load->getResult(0).getType());
  if (!ty)
    return;
  auto oldBlocked =
      dyn_cast<triton::gpu::BlockedEncodingAttr>(ty.getEncoding());
  if (!oldBlocked)
    return;
  ArrayRef<int64_t> shape = ty.getShape();

  // Only act on the "gather" operand (i.e., the reduction operand whose K is
  // the strided/slow axis)
  SmallVector<unsigned> order(oldBlocked.getOrder());
  if (order.empty() || order.back() != kDim)
    return;

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
    load->emitWarning("rock-set-reduction-layout: warps do not tile the "
                      "reduction dim; skipping");
    return;
  }

  auto newBlocked = triton::gpu::BlockedEncodingAttr::get(
      ctx, sizePerThread, threadsPerWarp, warpsPerCTA, order,
      oldBlocked.getCGALayout());
  if (newBlocked == oldBlocked) {
    LLVM_DEBUG(llvm::dbgs() << "rock-set-reduction-layout: load already in the "
                               "desired layout; skipping\n");
    return;
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

  // Gather the layout-connectivity class of this load: every value that must
  // move to the redistributed layout together.
  llvm::SetVector<Value> classValues;
  llvm::SetVector<Operation *> classOps;
  SmallVector<Value> worklist;
  auto enqueue = [&](Value v) {
    if (v && classValues.insert(v))
      worklist.push_back(v);
  };

  enqueue(load->getResult(0));
  for (Operation *user : load->getResult(0).getUsers())
    if (isa<triton::amdgpu::InThreadTransposeOp>(user))
      enqueue(user->getResult(0));

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
      continue; // block arguments have no defining op to walk
    }

    Operation *def = v.getDefiningOp();
    if (!def)
      continue;

    // We may have a scf.for result, reached from the result side: bridge to the
    // iter_arg and yielded value. Don't walk the loop's own operands or recurse
    // into its body; the carried result is retyped as an ordinary class value.
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      unsigned i = cast<OpResult>(v).getResultNumber();
      auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      enqueue(forOp.getInitArgs()[i]);
      enqueue(yield.getOperand(i));
      enqueue(forOp.getRegionIterArg(i));
      continue;
    }

    classOps.insert(def);
    for (Value operand : def->getOperands())
      enqueue(operand);
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
          return;
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
      // Make sure that the results of the loop are not used before updating the
      // type.
      assert(res.use_empty() &&
             "loop-carried reduction-layout slot has post-loop uses; retyping "
             "its result would corrupt them");
      res.setType(replacer.replace(res.getType()));
    }
  }
}
} // end anonymous namespace

void RockSetReductionLayoutPass::runOnOperation() {
  ModuleOp mod = getOperation();

  // Associate each dot operand with the global load that feeds it and the
  // reduction (K) dim implied by its operand position.
  llvm::MapVector<Operation *, unsigned> loadKDim;
  llvm::DenseSet<Operation *> conflicting;
  auto record = [&](Value operand, unsigned kDim) {
    Operation *load = findFeedingLoad(operand);
    if (!load)
      return;
    auto [it, inserted] = loadKDim.try_emplace(load, kDim);
    if (!inserted && it->second != kDim)
      conflicting.insert(load);
  };
  mod.walk([&](triton::DotOpInterface dot) {
    record(dot.getA(), /*kDim=*/1u);
    record(dot.getB(), /*kDim=*/0u);
  });
  if (loadKDim.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "rock-set-reduction-layout: no dot operand is fed "
                  "by a global load; nothing to redistribute\n");
    return;
  }

  // A single load that feeds two dots as different operands (conflicting
  // reduction dims) is ambiguous; leave it untouched rather than guess.
  for (auto [load, kDim] : loadKDim) {
    if (conflicting.contains(load)) {
      load->emitWarning("rock-set-reduction-layout: load feeds dot operands "
                        "with conflicting reduction dims; skipping");
      continue;
    }
    rewriteGatherLoad(load, kDim);
  }
}
