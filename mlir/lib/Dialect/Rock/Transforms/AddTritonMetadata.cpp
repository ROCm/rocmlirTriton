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
// fast) in global memory, as a discardable `rock.o_transposed` attribute. It
// also evaluates the OptimizeEpilogue store-tail heuristic and marks the kernel
// with `rock.prefer_lds_epilogue` when the automatic policy should retain the
// paced LDS epilogue.
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
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include <optional>

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
// to pick the layout representative when a gemm result is written by several
// stores. Note this measures the whole destination buffer, not the tile being
// stored, so it ranks outputs by how much of memory they cover rather than by
// per-store cost.
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

// Return the number of reduction-loop iterations feeding `gemmOp`. At this
// point in the Rock pipeline blockwise GEMMs have not been unrolled, so the
// enclosing constant scf.for is the source of truth for K depth.
static std::optional<int64_t> getReductionDepth(rock::BlockwiseGemmOp gemmOp) {
  scf::ForOp forOp = gemmOp->getParentOfType<scf::ForOp>();
  if (!forOp)
    return std::nullopt;

  std::optional<llvm::APInt> tripCount = forOp.getStaticTripCount();
  if (!tripCount || tripCount->isZero() || tripCount->getActiveBits() > 63)
    return std::nullopt;
  return static_cast<int64_t>(tripCount->getZExtValue());
}

// True when OptimizeEpilogue's register bypass would expose a store burst at
// least twice as deep as the matrix-core work available to hide it.
static bool preferLdsEpilogue(rock::BlockwiseGemmOp gemmOp,
                              rock::BlockwiseStoreOp storeOp,
                              int64_t blockSize) {
  auto storeType = dyn_cast<ShapedType>(storeOp.getSource().getType());
  if (!storeType || !storeType.hasStaticShape())
    return false;

  Type elemType = storeType.getElementType();
  if (!elemType.isIntOrFloat())
    return false;
  unsigned elemBits = elemType.getIntOrFloatBitWidth();
  if (elemBits != 16)
    return false;
  if (blockSize <= 0)
    return false;

  std::optional<int64_t> kDepth = getReductionDepth(gemmOp);
  if (!kDepth) {
    LLVM_DEBUG(llvm::dbgs() << "[optimize-epilogue] reduction depth unknown -> "
                               "keep register bypass\n");
    return false;
  }

  int64_t tileElems = storeType.getNumElements();
  constexpr double dwordx4Bytes = 16.0;
  double storesPerThread =
      (static_cast<double>(tileElems) * static_cast<double>(elemBits) / 8.0) /
      (dwordx4Bytes * static_cast<double>(blockSize));
  double exposure = storesPerThread / static_cast<double>(*kDepth);
  bool preferLds = exposure >= 2.0;

  LLVM_DEBUG(llvm::dbgs() << "[optimize-epilogue] tileElems=" << tileElems
                          << " elemBits=" << elemBits << " blockSize="
                          << blockSize << " storesPerThread=" << storesPerThread
                          << " kDepth=" << *kDepth << " E=" << exposure
                          << " -> " << (preferLds ? "LDS epilogue" : "bypass")
                          << "\n");
  return preferLds;
}

void RockAddTritonMetadataPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic())) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping non-kernel func @" << func.getName() << "\n");
    return;
  }

  MLIRContext *ctx = &getContext();
  func->removeAttr(rock::PreferLdsEpilogueAttr::getMnemonic());
  auto policyAttr = func->getAttrOfType<IntegerAttr>(
      rock::UseOptimizeEpilogueAttr::getMnemonic());
  int64_t policy = policyAttr ? policyAttr.getInt() : rock::kKnobDefault;
  func->removeAttr(rock::UseOptimizeEpilogueAttr::getMnemonic());
  bool useAutomaticPolicy = policy == rock::kKnobDefault;

  auto blockSizeAttr =
      func->getAttrOfType<IntegerAttr>(rock::BlockSizeAttr::getMnemonic());
  int64_t blockSize = blockSizeAttr ? blockSizeAttr.getInt() : 0;
  bool preferLds = false;

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

    // A gemm result can be written by several stores. The output layout needs a
    // single answer, so it is read off the store covering the largest tensor in
    // memory. Burst exposure instead depends on the tile each store writes, so
    // score them all.
    rock::BlockwiseStoreOp storeOp;
    int64_t bestNumElements = -1;
    for (rock::BlockwiseStoreOp candidate : stores) {
      int64_t numElements = storeDestNumElements(candidate);
      if (numElements > bestNumElements) {
        bestNumElements = numElements;
        storeOp = candidate;
      }
      if (useAutomaticPolicy)
        preferLds |= preferLdsEpilogue(gemmOp, candidate, blockSize);
    }

    FailureOr<bool> oTransposed = computeOTransposed(storeOp);
    if (succeeded(oTransposed))
      gemmOp->setDiscardableAttr(rock::OTransposedAttr::getNameStr(),
                                 rock::OTransposedAttr::get(ctx, *oTransposed));
  });

  if (preferLds)
    func->setDiscardableAttr(rock::PreferLdsEpilogueAttr::getMnemonic(),
                             UnitAttr::get(ctx));
}
