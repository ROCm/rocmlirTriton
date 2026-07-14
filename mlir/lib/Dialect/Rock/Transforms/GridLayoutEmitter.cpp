//===- GridLayoutEmitter.cpp - MLIR helper that contains the layout logic -===//
//
// Copyright 2020 The MLIR Authors.
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
// =============================================================================
//
// Helpers that map a flat block id onto the kernel grid by emitting the
// <group, m-block, n-block> triplet used by the generated gemm/attention
// kernels.
//
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "llvm/Support/Debug.h"

#include <cmath>

#include "GridLayoutEmitter.h"

#define DEBUG_TYPE "rock-grid-layout-emitter"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::arith;
using namespace mlir::rock::layout;

// based on
// https://github.com/HazyResearch/HipKittens/blob/7f6986b502396aa865c0c80625121daf7caa756d/include/common/util.cuh#L78
static Value rearrangeWorkgroupsForXCC(Location loc, PatternRewriter &b,
                                       Value bid, int64_t gridSize,
                                       int64_t numChiplets, int64_t chunkSize) {
  Type i32 = b.getIntegerType(32);
  Value numChipletsVal = b.createOrFold<ConstantIntOp>(loc, i32, numChiplets);
  Value chunkSizeVal = b.createOrFold<ConstantIntOp>(loc, i32, chunkSize);

  // Current XCD
  Value xcd = RemUIOp::create(b, loc, bid, numChipletsVal);

  // Largest full (numChiplets*chunkSize)-aligned block
  int64_t block = numChiplets * chunkSize;
  int64_t limit = (gridSize / block) * block;
  Value blockVal = b.createOrFold<ConstantIntOp>(loc, i32, block);
  Value limitVal = b.createOrFold<ConstantIntOp>(loc, i32, limit);

  // Local BID (within round-robin assignment)
  Value localBid = DivUIOp::create(b, loc, bid, numChipletsVal);
  Value chunkIdx = DivUIOp::create(b, loc, localBid, chunkSizeVal);
  Value posInChunk = RemUIOp::create(b, loc, localBid, chunkSizeVal);

  // New BID
  // newBid = chunkIdx * block + xcd * chunkSize + posInChunk;
  Value newBid = AddIOp::create(
      b, loc,
      AddIOp::create(b, loc, MulIOp::create(b, loc, chunkIdx, blockVal),
                     MulIOp::create(b, loc, xcd, chunkSizeVal)),
      posInChunk);

  // If bid beyond the last full block, leave unchanged
  // if (bid > limit) return bid;
  Value isBidLargerThanLastFullBlock =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, bid, limitVal);
  bid = arith::SelectOp::create(b, loc, isBidLargerThanLastFullBlock, bid,
                                newBid);

  return bid;
}

GridCoordinates rock::layout::makeGroupedGridLayout(PatternRewriter &b,
                                                    Location loc, Value bid,
                                                    GridLayoutInfo info,
                                                    StringRef arch) {
  // The swizzle below is a 1-D GROUP_SIZE_M super-grouping: groupSizeM (a.k.a.
  // GROUP_SIZE_M), the M-block band height, is its only knob. We pick it from
  // cache residency of the operands: LLC decides which operand is read from
  // DRAM once, L2 bounds the band height (see below).
  int64_t gridSize = info.gBlocks * info.mBlocks * info.nBlocks;
  int64_t concurrentWorkgroups =
      std::max(int64_t{1}, info.numCU / info.numChiplets);
  int64_t gridPerChiplet = std::max(int64_t{1}, gridSize / info.numChiplets);

  auto elemBytes = [](Type t) -> double {
    return llvm::divideCeil(t.getIntOrFloatBitWidth(), 8);
  };

  // A band is groupSizeM M-blocks spanning the *full* N width, swept
  // column-major. So each A row-block is reused across all of N (A is read once
  // overall), while B is re-read once per band -> mBlocks / groupSizeM passes.
  //
  // Two caches play different roles:
  //   * The last-level cache (Infinity Cache / MALL, or L2 when there is none)
  //     governs *DRAM traffic*: an operand that fits it is fetched from DRAM
  //     once regardless of how many times the swizzle re-reads it (later reads
  //     hit the LLC). So the read-once decision below tests LLC residency.
  //   * L2 (per-XCD scope) governs *reuse speed*: it bounds how tall a band can
  //     be while the live A row-blocks stay L2-resident across the N sweep.
  //
  // When exactly one operand fits the LLC, the other is the one that costs DRAM
  // traffic, so drive the band height to whichever extreme reads it once:
  //   * B fits LLC (and A does not): B is read from DRAM once no matter what.
  //     Use groupSizeM = 1 (row-major) to keep A's live footprint minimal so B
  //     keeps its cache residency (e.g. big A, tiny B).
  //   * A fits LLC (and B does not): A is read from DRAM once no matter what.
  //     Use groupSizeM = mBlocks (one band) so B is streamed exactly once; A is
  //     reused from the LLC across the sweep (e.g. conv, tiny A).
  // When both fit or neither fits, the A-vs-B tradeoff is real: make the band
  // as tall as L2 residency allows (fewer B passes) -- see the quadratic below.
  bool aFitsLLC = info.aTotalBytes <= info.llcBytes;
  bool bFitsLLC = info.bTotalBytes <= info.llcBytes;
  double aSlabBytes =
      info.mPerBlock * elemBytes(info.aInputType) * info.kPerBlock;
  double bSlabBytes =
      info.nPerBlock * elemBytes(info.bInputType) * info.kPerBlock;

  int64_t groupSizeM;
  if (info.gridGroupSize != 0) {
    // Explicit tuning override.
    groupSizeM = info.gridGroupSize;
    LLVM_DEBUG(llvm::dbgs() << "Setting groupSizeM by using tuning params to "
                            << groupSizeM << "\n");
  } else if (bFitsLLC && !aFitsLLC) {
    // B read from DRAM once regardless: stream A once, keep B's footprint
    // cached
    groupSizeM = 1;
    LLVM_DEBUG(llvm::dbgs() << "B fits LLC: groupSizeM = 1 (row-major)\n");
  } else if (aFitsLLC && !bFitsLLC) {
    // A read from DRAM once regardless: one band streams B once, A reused via
    // LLC
    groupSizeM = info.mBlocks;
    LLVM_DEBUG(llvm::dbgs()
               << "A fits LLC: groupSizeM = mBlocks (single band)\n");
  } else {
    // Make the band as tall as possible to cut B passes (mBlocks/groupSizeM),
    // bounded by keeping the live wave L2-resident. The sweep is column-major,
    // so the concurrent P workgroups span ~ceil(P/h) columns x h rows; L2 must
    // hold the h persistent A row-blocks plus the ~P/h active B col-blocks:
    //
    //     h*aSlab + (P/h)*bSlab <= L2.
    //
    // The tallest feasible h is the larger root of
    //     aSlab*h^2 - L2*h + P*bSlab = 0.
    // When the discriminant is negative no h holds both working sets at once (A
    // cannot stay resident across the sweep), so fall back to the footprint-
    // minimizing square wave h = sqrt(P*bSlab/aSlab). Both collapse to the
    // plain L2/aSlab band when the cache is roomy relative to concurrency (P <<
    // h).
    const double P = (double)concurrentWorkgroups;
    const double l2 = (double)info.l2Bytes;
    int64_t maxByL2;
    if (aSlabBytes <= 0.0) {
      maxByL2 = info.mBlocks;
    } else {
      double disc = l2 * l2 - 4.0 * aSlabBytes * P * bSlabBytes;
      double h = disc >= 0.0 ? (l2 + std::sqrt(disc)) / (2.0 * aSlabBytes)
                             : std::sqrt(P * bSlabBytes / aSlabBytes);
      maxByL2 = (int64_t)h;
    }
    groupSizeM = std::min(info.mBlocks, std::max(int64_t{1}, maxByL2));
    LLVM_DEBUG(llvm::dbgs()
               << "L2-bounded band height groupSizeM " << groupSizeM << "\n");
  }
  groupSizeM = std::min(std::max(int64_t{1}, groupSizeM),
                        std::max(int64_t{1}, info.mBlocks));

  // Currently the firmware will launch workgroups
  // in a round-robin fashion to each chiplet. However
  // we would want a group (>=1) of chiplets to perform
  // a spatially local tile.
  // Therefore, adjust bid to make every consecutive #groups of chiplets
  // be slowest changing in the grid.
  if (info.numChiplets > 1) {
    // Give each chiplet whole columns of the band so its groupSizeM A
    // row-blocks are reused across the columns it owns (and each column's
    // B-slab is reused down its groupSizeM rows). Columns per chunk ~ the
    // concurrent wave, so chunkSize stays a multiple of groupSizeM. Bounded by
    // the per-chiplet grid.
    int64_t colsPerChunk =
        std::max(int64_t{1}, concurrentWorkgroups / groupSizeM);
    int64_t chunkSize = std::min(groupSizeM * colsPerChunk, gridPerChiplet);
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, info.numChiplets,
                                    chunkSize);
  }

  Value mBlocksPerGroup =
      b.createOrFold<ConstantIntOp>(loc, b.getIntegerType(32), groupSizeM);
  Value blocksPerGroup = b.createOrFold<ConstantIntOp>(
      loc, b.getIntegerType(32), groupSizeM * info.nBlocks);
  Value mBlocksValue = b.createOrFold<ConstantIntOp>(loc, b.getIntegerType(32), info.mBlocks);

  // Compute g_block first and the bid in the actual group g_block
  Value mnBlocks =
      b.createOrFold<ConstantIntOp>(loc, b.getIntegerType(32), info.mBlocks * info.nBlocks);
  Value g_block = DivUIOp::create(b, loc, bid, mnBlocks);
  bid = RemUIOp::create(b, loc, bid, mnBlocks);

  // Group together the workgroups in g_block
  Value groupId = DivUIOp::create(b, loc, bid, blocksPerGroup);
  Value firstBidM = MulIOp::create(b, loc, groupId, mBlocksPerGroup);
  Value thisMBlocksPerGroup = MinUIOp::create(
      b, loc, SubIOp::create(b, loc, mBlocksValue, firstBidM), mBlocksPerGroup);
  Value m_block = AddIOp::create(
      b, loc, firstBidM, RemUIOp::create(b, loc, bid, thisMBlocksPerGroup));
  Value n_block =
      DivUIOp::create(b, loc, RemUIOp::create(b, loc, bid, blocksPerGroup),
                      thisMBlocksPerGroup);
  // no need to get splitKFactor here
  return {g_block, m_block, n_block};
}

AttnGridCoordinates rock::layout::makeGxNGridLayout(
    PatternRewriter &b, Location loc, Value bid, int64_t mBlocks, Value nIter,
    int64_t gridSize, StringRef arch, int64_t numChiplets, Value splitKV) {
  // Currently the firmware will launch workgroups
  // in a round-robin fashion to each chiplet. However
  // we would want a group (>=1) of chiplets to perform
  // a spatially local tile.
  // Therefore, adjust bid to make every consecutive #groups of chiplets
  // be slowest changing in the grid.
  if (numChiplets > 1) {
    int64_t chunkSize = std::max(int64_t{1}, gridSize / numChiplets);
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, numChiplets,
                                    chunkSize);
  }
  Value g1MBlockCountVal =
      b.createOrFold<ConstantIntOp>(loc, b.getIntegerType(32), mBlocks);

  Value gBlockIdx, mBlockIdx, splitKVIdx;
  if (splitKV) {
    Value noGSize = arith::MulIOp::create(b, loc, splitKV, g1MBlockCountVal);
    gBlockIdx = arith::DivUIOp::create(b, loc, bid, noGSize);
    mBlockIdx = arith::RemUIOp::create(b, loc, bid, g1MBlockCountVal);
    Value outerIdx = arith::DivUIOp::create(b, loc, bid, g1MBlockCountVal);
    splitKVIdx = arith::RemUIOp::create(b, loc, outerIdx, splitKV);
  } else {
    gBlockIdx = arith::DivUIOp::create(b, loc, bid, g1MBlockCountVal);
    mBlockIdx = arith::RemUIOp::create(b, loc, bid, g1MBlockCountVal);
    splitKVIdx = nullptr;
  }
  // braces for init of the base class: GridCoordinates
  return {{gBlockIdx, mBlockIdx, nIter}, splitKVIdx};
}
