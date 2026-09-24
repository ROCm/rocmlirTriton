//===- SetITTReductionLayout.cpp - Every warp on K ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runs right after Triton's tritonamdgpu-in-thread-transpose, which widens a
// gather load's sizePerThread along K and redistributes its warps greedily,
// fastest dim first. With K the slow dim and a wide free dim, most warps land
// on the free dim and every thread owns many distinct K rows.
//
// When rock-incremental-pointer-arith marks the load
// rock.loop_variant_index_math, every one of those rows costs scalar work per
// iteration: either a non-power-of-two division sequence, when the index math
// is still recomputed inside the loop, or advancing the coordinates its carry
// path keeps across iterations. For those loads this pass puts every warp on
// K, which cuts the rows each thread owns by the warp count. For now it only
// does so when each thread owns exactly 16 K rows.
//
// Like in-thread-transpose, the load is rebuilt in the new layout between
// convert_layout ops; the remove-layout-conversions run that follows carries
// the layout back into the address computation.
//
//===----------------------------------------------------------------------===//

#include "WarpsOnK.h"

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKSETITTREDUCTIONLAYOUTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-set-itt-reduction-layout"

using namespace mlir;
using namespace mlir::rock;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace amdg = mlir::triton::amdgpu;

namespace {
struct RockSetITTReductionLayoutPass
    : public rock::impl::RockSetITTReductionLayoutPassBase<
          RockSetITTReductionLayoutPass> {
  void runOnOperation() override;
};

// The global load feeding `value` through layout conversions, if any.
tt::LoadOp findFeedingLoad(Value value) {
  while (auto cvt = value.getDefiningOp<ttg::ConvertLayoutOp>())
    value = cvt.getSrc();
  return value.getDefiningOp<tt::LoadOp>();
}

// The local_alloc owning the shared-memory buffer `memDesc` views, looking
// through memdesc_index and the scf.for iter_args/results a software pipeline
// threads its buffers through. Null when the buffer comes from anywhere else.
ttg::LocalAllocOp findRootAlloc(Value memDesc) {
  while (true) {
    if (auto alloc = memDesc.getDefiningOp<ttg::LocalAllocOp>())
      return alloc;
    if (auto index = memDesc.getDefiningOp<ttg::MemDescIndexOp>()) {
      memDesc = index.getSrc();
      continue;
    }
    if (auto arg = dyn_cast<BlockArgument>(memDesc)) {
      auto forOp = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
      OpOperand *init = forOp ? forOp.getTiedLoopInit(arg) : nullptr;
      if (!init)
        return nullptr;
      memDesc = init->get();
      continue;
    }
    if (auto forOp = memDesc.getDefiningOp<scf::ForOp>()) {
      memDesc = forOp.getTiedLoopInit(cast<OpResult>(memDesc))->get();
      continue;
    }
    return nullptr;
  }
}

// The shared-memory buffer `transpose` stages its result into, through
// local_store or local_alloc. Null when it has any other consumer, or stages
// into more than one buffer.
Value findStagingBuffer(amdg::InThreadTransposeOp transpose) {
  Value buffer;
  for (Operation *user : transpose->getUsers()) {
    ttg::LocalAllocOp alloc;
    if (auto store = dyn_cast<ttg::LocalStoreOp>(user))
      alloc = findRootAlloc(store.getDst());
    else
      alloc = dyn_cast<ttg::LocalAllocOp>(user);
    if (!alloc || (buffer && buffer != alloc.getResult()))
      return nullptr;
    buffer = alloc.getResult();
  }
  return buffer;
}

// Rebuild `load` in `encoding`, converting its tensor operands in and its
// result back out.
void relayoutLoad(tt::LoadOp load, ttg::BlockedEncodingAttr encoding) {
  OpBuilder b(load);
  Location loc = load.getLoc();
  SmallVector<Value> operands;
  for (Value operand : load->getOperands()) {
    if (auto ty = dyn_cast<RankedTensorType>(operand.getType()))
      operand = ttg::ConvertLayoutOp::create(
          b, loc, ty.cloneWithEncoding(encoding), operand);
    operands.push_back(operand);
  }
  Operation *newLoad = b.clone(*load);
  newLoad->setOperands(operands);
  auto oldTy = cast<RankedTensorType>(load.getResult().getType());
  newLoad->getResult(0).setType(oldTy.cloneWithEncoding(encoding));
  Value back =
      ttg::ConvertLayoutOp::create(b, loc, oldTy, newLoad->getResult(0));
  load.getResult().replaceAllUsesWith(back);
  load.erase();
}

// Rebuild `transpose` on a `encoding` source. Its #linear result follows the
// source layout; the local_store/local_alloc consumers accept any.
void relayoutTranspose(amdg::InThreadTransposeOp transpose,
                       ttg::BlockedEncodingAttr encoding) {
  OpBuilder b(transpose);
  Location loc = transpose.getLoc();
  RankedTensorType srcTy =
      transpose.getSrc().getType().cloneWithEncoding(encoding);
  Value src = ttg::ConvertLayoutOp::create(b, loc, srcTy, transpose.getSrc());
  auto linear = ttg::LinearEncodingAttr::get(
      b.getContext(), amdg::InThreadTransposeOp::deduceOutputLayout(
                          srcTy.getShape(), encoding));
  auto newTranspose = amdg::InThreadTransposeOp::create(
      b, loc, transpose.getType().cloneWithEncoding(linear), src);
  transpose.getResult().replaceAllUsesWith(newTranspose.getResult());
  transpose.erase();
}
} // end anonymous namespace

void RockSetITTReductionLayoutPass::runOnOperation() {
  // Every InThreadTransposeOp referring to the same buffer moves together.
  // This way we make sure that the prologue copy follows its in-loop copy.
  llvm::MapVector<Value, SmallVector<amdg::InThreadTransposeOp>> groups;
  getOperation().walk([&](amdg::InThreadTransposeOp transpose) {
    if (Value buffer = findStagingBuffer(transpose))
      groups[buffer].push_back(transpose);
  });

  // Process each group of InThreadTranspose ops.
  for (auto &entry : groups) {
    SmallVector<amdg::InThreadTransposeOp> &transposes = entry.second;
    auto srcTy = cast<RankedTensorType>(transposes.front().getSrc().getType());
    auto enc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!enc || srcTy.getRank() != 2) {
      LLVM_DEBUG(llvm::dbgs() << "rock-set-itt-reduction-layout: tensor is "
                                 "not rank-2 blocked; skipping\n");
      continue;
    }
    // InThreadTranspose ops only affect loads whose K is the slowest dim.
    unsigned kDim = enc.getOrder().back();

    llvm::SetVector<tt::LoadOp> loads;
    bool matched = true;
    for (amdg::InThreadTransposeOp transpose : transposes) {
      tt::LoadOp load = findFeedingLoad(transpose.getSrc());
      if (!load || transpose.getSrc().getType() != srcTy) {
        matched = false;
        break;
      }
      loads.insert(load);
    }
    if (!matched) {
      LLVM_DEBUG(llvm::dbgs()
                 << "rock-set-itt-reduction-layout: a transpose staging into "
                    "the buffer is not fed by a load, or its layout differs "
                    "from the others; skipping\n");
      continue;
    }
    if (llvm::none_of(loads, [](tt::LoadOp load) {
          return load->hasAttr(LoopVariantIndexMathAttr::getMnemonic());
        })) {
      LLVM_DEBUG(llvm::dbgs() << "rock-set-itt-reduction-layout: index math "
                                 "is not marked loop-variant; skipping\n");
      continue;
    }
    // Hack: 16 K rows per thread is the only count measured to gain (the rxl
    // encoder and decoder convolutions); leave the others alone.
    if (ttg::getElemsPerThread(srcTy)[kDim] != 16) {
      LLVM_DEBUG(llvm::dbgs() << "rock-set-itt-reduction-layout: threads do "
                                 "not own 16 K rows; skipping\n");
      continue;
    }

    // All checks out, now compute the new encoding and apply it to the loads.
    FailureOr<ttg::BlockedEncodingAttr> newEnc =
        computeLayoutWarpsOnK(enc, srcTy.getShape(), kDim);
    if (failed(newEnc)) {
      LLVM_DEBUG(llvm::dbgs() << "rock-set-itt-reduction-layout: warps do "
                                 "not tile K; skipping\n");
      continue;
    }
    if (*newEnc == enc) {
      LLVM_DEBUG(llvm::dbgs() << "rock-set-itt-reduction-layout: warps "
                                 "already on K; skipping\n");
      continue;
    }

    for (tt::LoadOp load : loads)
      relayoutLoad(load, *newEnc);
    for (amdg::InThreadTransposeOp transpose : transposes)
      relayoutTranspose(transpose, *newEnc);
  }
}
