//===- SetGatherWarps.cpp - Put gather warps on the reduction dim ---------===//
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
// rock.loop_variant_index_math, its index math keeps a non-power-of-two
// division inside the loop, so every one of those rows costs a scalar division
// sequence per iteration. For those loads this pass puts every warp on K,
// which cuts the rows each thread owns by the warp count.
//
// Like in-thread-transpose, the load is rebuilt in the new layout between
// convert_layout ops; the remove-layout-conversions run that follows carries
// the layout back into the address computation.
//
//===----------------------------------------------------------------------===//

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
#define GEN_PASS_DEF_ROCKSETGATHERWARPSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-set-gather-warps"

using namespace mlir;
using namespace mlir::rock;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace amdg = mlir::triton::amdgpu;

namespace {
struct RockSetGatherWarpsPass
    : public rock::impl::RockSetGatherWarpsPassBase<RockSetGatherWarpsPass> {
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
      if (!forOp || arg.getArgNumber() == 0) // 0 == induction variable
        return nullptr;
      memDesc = forOp.getInitArgs()[arg.getArgNumber() - 1];
      continue;
    }
    if (auto forOp = memDesc.getDefiningOp<scf::ForOp>()) {
      memDesc = forOp.getInitArgs()[cast<OpResult>(memDesc).getResultNumber()];
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

// `enc` with every warp on `kDim`, or null when that layout does not tile
// `shape` or is `enc` already.
ttg::BlockedEncodingAttr getWarpsOnK(ttg::BlockedEncodingAttr enc,
                                     ArrayRef<int64_t> shape, unsigned kDim) {
  SmallVector<unsigned> warpsPerCTA(enc.getWarpsPerCTA());
  unsigned numWarps = 1;
  for (unsigned warps : warpsPerCTA)
    numWarps *= warps;
  for (unsigned d = 0; d < warpsPerCTA.size(); ++d)
    warpsPerCTA[d] = d == kDim ? numWarps : 1;
  for (auto [d, size] : llvm::enumerate(shape)) {
    int64_t cover =
        enc.getSizePerThread()[d] * enc.getThreadsPerWarp()[d] * warpsPerCTA[d];
    if (size % cover != 0)
      return nullptr;
  }
  auto newEnc = ttg::BlockedEncodingAttr::get(
      enc.getContext(), enc.getSizePerThread(), enc.getThreadsPerWarp(),
      warpsPerCTA, enc.getOrder(), enc.getCGALayout());
  return newEnc == enc ? nullptr : newEnc;
}

RankedTensorType withEncoding(Type type, Attribute encoding) {
  auto ty = cast<RankedTensorType>(type);
  return RankedTensorType::get(ty.getShape(), ty.getElementType(), encoding);
}

// Rebuild `load` in `encoding`, converting its tensor operands in and its
// result back out.
void relayoutLoad(tt::LoadOp load, ttg::BlockedEncodingAttr encoding) {
  OpBuilder b(load);
  Location loc = load.getLoc();
  SmallVector<Value> operands;
  for (Value operand : load->getOperands()) {
    if (isa<RankedTensorType>(operand.getType()))
      operand = ttg::ConvertLayoutOp::create(
          b, loc, withEncoding(operand.getType(), encoding), operand);
    operands.push_back(operand);
  }
  Operation *newLoad = b.clone(*load);
  newLoad->setOperands(operands);
  Type oldTy = load.getResult().getType();
  newLoad->getResult(0).setType(withEncoding(oldTy, encoding));
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
  auto srcTy = withEncoding(transpose.getSrc().getType(), encoding);
  Value src = ttg::ConvertLayoutOp::create(b, loc, srcTy, transpose.getSrc());
  auto linear = ttg::LinearEncodingAttr::get(
      b.getContext(), amdg::InThreadTransposeOp::deduceOutputLayout(
                          srcTy.getShape(), encoding));
  auto newTranspose = amdg::InThreadTransposeOp::create(
      b, loc, withEncoding(transpose.getType(), linear), src);
  transpose.getResult().replaceAllUsesWith(newTranspose.getResult());
  transpose.erase();
}
} // end anonymous namespace

void RockSetGatherWarpsPass::runOnOperation() {
  // Every transpose staging into one buffer moves together, so that a
  // pipeline's prologue copy follows its in-loop copy.
  llvm::MapVector<Value, SmallVector<amdg::InThreadTransposeOp>> groups;
  getOperation().walk([&](amdg::InThreadTransposeOp transpose) {
    if (Value buffer = findStagingBuffer(transpose))
      groups[buffer].push_back(transpose);
  });

  for (auto &entry : groups) {
    SmallVector<amdg::InThreadTransposeOp> &transposes = entry.second;
    auto srcTy = cast<RankedTensorType>(transposes.front().getSrc().getType());
    auto enc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!enc || srcTy.getRank() != 2)
      continue;
    // in_thread_transpose only stages loads whose K is the slowest dim.
    unsigned kDim = enc.getOrder().back();

    llvm::SetVector<tt::LoadOp> loads;
    bool matched = llvm::all_of(transposes, [&](auto transpose) {
      tt::LoadOp load = findFeedingLoad(transpose.getSrc());
      if (!load || transpose.getSrc().getType() != srcTy)
        return false;
      loads.insert(load);
      return true;
    });
    if (!matched)
      continue;
    if (llvm::none_of(loads, [](tt::LoadOp load) {
          return load->hasAttr(LoopVariantIndexMathAttr::getMnemonic());
        })) {
      LLVM_DEBUG(llvm::dbgs() << "rock-set-gather-warps: index math is not "
                                 "marked loop-variant; skipping\n");
      continue;
    }
    ttg::BlockedEncodingAttr newEnc = getWarpsOnK(enc, srcTy.getShape(), kDim);
    if (!newEnc) {
      LLVM_DEBUG(llvm::dbgs() << "rock-set-gather-warps: warps already on K "
                                 "or do not tile it; skipping\n");
      continue;
    }

    for (tt::LoadOp load : loads)
      relayoutLoad(load, newEnc);
    for (amdg::InThreadTransposeOp transpose : transposes)
      relayoutTranspose(transpose, newEnc);
  }
}
