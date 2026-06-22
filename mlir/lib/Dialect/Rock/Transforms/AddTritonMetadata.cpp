//===- AddTritonMetadata.cpp - Attach Triton-bound rock metadata ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass runs right after rock-lower-stores, while the GEMM output store is
// still a rock.blockwise_store writing a transformed view of the kernel output.
// For each rock.blockwise_gemm consumed by such a store, it records whether the
// output tile is laid out row-major (N fast) or transposed/column-major (M
// fast) in global memory, as a discardable `rock.o_transposed` attribute.
//
// The attribute is later copied onto the lowered tt.dot / tt.dot_scaled by
// RockToTTIR, survives Triton's accelerate-matmul rewrite, and is finally
// consumed by rock-set-matmul-output-transpose to pick the accelerator result
// layout so the epilogue store coalesces.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKADDTRITONMETADATAPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-add-triton-metadata"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockAddTritonMetadataPass
    : public rock::impl::RockAddTritonMetadataPassBase<
          RockAddTritonMetadataPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// Follow `root` forward to every rock.blockwise_store that consumes it as its
// stored value. The walk crosses fusion ops (arith/math/rock.transform/etc.)
// and out of scf.for / scf.if regions via their yields. A single gemm result
// can be stored by more than one blockwise_store, so all of them are collected.
// Returns an empty list when no store is reachable (e.g. a chained-dot head
// whose result is fed into another gemm or returned).
static llvm::SmallSetVector<rock::BlockwiseStoreOp, 4>
findConsumerStores(Value root) {
  llvm::SmallSetVector<rock::BlockwiseStoreOp, 4> stores;
  SmallVector<Value> worklist{root};
  llvm::SmallPtrSet<Value, 16> seen;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;

    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();

      if (auto storeOp = dyn_cast<rock::BlockwiseStoreOp>(user)) {
        // Only count it when the value is the data being stored, not the
        // destination view.
        if (use.get() == storeOp.getSource())
          stores.insert(storeOp);
        continue;
      }

      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        // A yielded value becomes the tied parent result for scf.for / scf.if.
        if (isa<scf::ForOp, scf::IfOp>(yieldOp->getParentOp()))
          worklist.push_back(
              yieldOp->getParentOp()->getResult(use.getOperandNumber()));
        continue;
      }

      // Do not chase into another GEMM: that result keeps its own metadata.
      if (isa<rock::BlockwiseGemmOp>(user))
        continue;

      // Follow the value through any other (elementwise / cast / transform) op.
      worklist.append(user->result_begin(), user->result_end());
    }
  }
  return stores;
}

// Number of statically-known elements written by `storeOp` (0 if dynamic), used
// to pick the largest store when a gemm result is written by several of them.
static int64_t storeDestNumElements(rock::BlockwiseStoreOp storeOp) {
  // The dest is a stack of rock.transform views; trace it back to the kernel
  // argument it writes to and measure that underlying buffer.
  FailureOr<BlockArgument> destArg = rock::findBlockArgument(storeOp.getDest());
  if (failed(destArg))
    return 0;
  auto destType = dyn_cast<ShapedType>(destArg->getType());
  if (!destType || !destType.hasStaticShape())
    return 0;
  return destType.getNumElements();
}

// Decide whether the output written by `storeOp` is transposed (column-major:
// M is the fast/contiguous output dimension) by comparing the achievable
// vectorization along the fast N dimension (last) vs. the fast M dimension
// (second to last) of the (transformed) destination view. Returns failure when
// the destination has too few dims to reason about.
static FailureOr<bool> computeOTransposed(rock::BlockwiseStoreOp storeOp) {
  // The blockwise_gemm output is an M x N tile, written into `dest` (possibly
  // through shape-preserving fusion/views) as its two fastest-varying logical
  // dimensions. In the lowered IR `dest` is a higher-rank view of the kernel
  // output (e.g. carrying group / block-tiling dims), with M as the second-to-
  // last and N as the last dimension, so we read them relative to the rank.
  Value dest = storeOp.getDest();
  auto destType = dyn_cast<ShapedType>(dest.getType());
  if (!destType || destType.getRank() < 2)
    return failure();

  uint32_t nDim = destType.getRank() - 1;
  uint32_t mDim = destType.getRank() - 2;

  VectorizationResult nVec = getMaxVectorization(dest, nDim);
  VectorizationResult mVec = getMaxVectorization(dest, mDim);

  LLVM_DEBUG(llvm::dbgs() << "store dest vectorization: M(dim " << mDim
                          << ")=" << mVec.max << " N(dim " << nDim
                          << ")=" << nVec.max << "\n");

  // Transposed (column-major) only when M strictly vectorizes better than N.
  // Ties (e.g. both 1) default to row-major.
  return mVec.max > nVec.max;
}

void RockAddTritonMetadataPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping non-kernel func @" << func.getName() << "\n");
    return;
  }

  MLIRContext *ctx = &getContext();

  func.walk([&](rock::BlockwiseGemmOp gemmOp) {
    llvm::SmallSetVector<rock::BlockwiseStoreOp, 4> stores =
        findConsumerStores(gemmOp.getResult());
    if (stores.empty()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "No consumer store for gemm; leaving output layout default: "
          << gemmOp << "\n");
      return;
    }

    // A gemm result can be written by several stores; use the one writing the
    // largest tensor to memory as the representative for the output layout.
    rock::BlockwiseStoreOp storeOp;
    int64_t bestNumElements = -1;
    for (rock::BlockwiseStoreOp candidate : stores) {
      int64_t numElements = storeDestNumElements(candidate);
      if (numElements > bestNumElements) {
        bestNumElements = numElements;
        storeOp = candidate;
      }
    }

    FailureOr<bool> oTransposed = computeOTransposed(storeOp);
    if (failed(oTransposed))
      return;

    gemmOp->setDiscardableAttr(rock::OTransposedAttr::getNameStr(),
                               rock::OTransposedAttr::get(ctx, *oTransposed));
  });
}
