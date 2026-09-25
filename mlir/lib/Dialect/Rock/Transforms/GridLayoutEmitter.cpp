//===- GridLayoutEmitter.cpp - MLIR helper that contains the layout logic -===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/Support/Debug.h"

#include "GridLayoutEmitter.h"

#define DEBUG_TYPE "rock-grid-layout-emitter"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::arith;
using namespace mlir::rock::layout;

static Value getI32(PatternRewriter &b, Location loc, OpFoldResult value) {
  if (std::optional<int64_t> cst = getConstantIntValue(value))
    return b.createOrFold<ConstantIntOp>(loc, b.getI32Type(), *cst);
  return cast<Value>(value);
}

/// `fn(lhs, rhs)` folded to a constant when both are constant, otherwise an
/// `OpTy` on their i32 values.
template <typename OpTy, typename Fn>
static OpFoldResult foldOrCreate(PatternRewriter &b, Location loc,
                                 OpFoldResult lhs, OpFoldResult rhs, Fn fn) {
  std::optional<int64_t> l = getConstantIntValue(lhs);
  std::optional<int64_t> r = getConstantIntValue(rhs);
  if (l && r)
    return b.getI64IntegerAttr(fn(*l, *r));
  return OpTy::create(b, loc, getI32(b, loc, lhs), getI32(b, loc, rhs))
      .getResult();
}

static OpFoldResult mul(PatternRewriter &b, Location loc, OpFoldResult lhs,
                        OpFoldResult rhs) {
  return foldOrCreate<MulIOp>(b, loc, lhs, rhs,
                              [](int64_t l, int64_t r) { return l * r; });
}
static OpFoldResult div(PatternRewriter &b, Location loc, OpFoldResult lhs,
                        OpFoldResult rhs) {
  return foldOrCreate<DivUIOp>(b, loc, lhs, rhs,
                               [](int64_t l, int64_t r) { return l / r; });
}
static OpFoldResult minUI(PatternRewriter &b, Location loc, OpFoldResult lhs,
                        OpFoldResult rhs) {
  return foldOrCreate<MinUIOp>(
      b, loc, lhs, rhs, [](int64_t l, int64_t r) { return std::min(l, r); });
}
static OpFoldResult maxUI(PatternRewriter &b, Location loc, OpFoldResult lhs,
                        OpFoldResult rhs) {
  return foldOrCreate<MaxUIOp>(
      b, loc, lhs, rhs, [](int64_t l, int64_t r) { return std::max(l, r); });
}

// based on
// https://github.com/HazyResearch/HipKittens/blob/7f6986b502396aa865c0c80625121daf7caa756d/include/common/util.cuh#L78
static Value rearrangeWorkgroupsForXCC(Location loc, PatternRewriter &b,
                                       Value bid, OpFoldResult gridSize,
                                       int64_t numChiplets,
                                       OpFoldResult chunkSize) {
  Type i32 = b.getIntegerType(32);
  Value numChipletsVal = b.createOrFold<ConstantIntOp>(loc, i32, numChiplets);
  Value chunkSizeVal = getI32(b, loc, chunkSize);

  // Current XCD
  Value xcd = RemUIOp::create(b, loc, bid, numChipletsVal);

  // Largest full (numChiplets*chunkSize)-aligned block
  OpFoldResult block =
      mul(b, loc, b.getI64IntegerAttr(numChiplets), chunkSize);
  OpFoldResult limit = mul(b, loc, div(b, loc, gridSize, block), block);
  Value blockVal = getI32(b, loc, block);
  Value limitVal = getI32(b, loc, limit);

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
  // Heuristic to compute groupSize
  // This also covers the cases where the output width is larger
  // than the input width
  int64_t bitWidthOut = info.outputType.getIntOrFloatBitWidth();
  int64_t bitWidthIn =
      std::min((int64_t)info.inputType.getIntOrFloatBitWidth(), bitWidthOut);
  int64_t groupSize = std::ceil(std::sqrt(info.numCU / info.numChiplets)) *
                      (bitWidthOut / bitWidthIn);
  // use gridGroupSize if it's not zero
  if (info.gridGroupSize != 0) {
    groupSize = info.gridGroupSize;
    LLVM_DEBUG(llvm::dbgs() << "Setting groupSize by using tuning params to "
                            << groupSize << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "Using heuristic to set groupSize to " << groupSize << "\n");
  }

  // Currently the firmware will launch workgroups
  // in a round-robin fashion to each chiplet. However
  // we would want a group (>=1) of chiplets to perform
  // a spatially local tile.
  // Therefore, adjust bid to make every consecutive #groups of chiplets
  // be slowest changing in the grid.
  if (info.numChiplets > 1) {
    OpFoldResult gridSize =
        mul(b, loc, mul(b, loc, info.gBlocks, info.mBlocks), info.nBlocks);
    OpFoldResult chunkSize = minUI(
        b, loc, b.getI64IntegerAttr(groupSize * groupSize),
        maxUI(b, loc, b.getI64IntegerAttr(1),
            div(b, loc, gridSize, b.getI64IntegerAttr(info.numChiplets))));
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, info.numChiplets,
                                    chunkSize);
  }

  Value mBlocksPerGroup = b.createOrFold<ConstantIntOp>(loc, b.getIntegerType(32), groupSize);
  Value blocksPerGroup = getI32(
      b, loc, mul(b, loc, b.getI64IntegerAttr(groupSize), info.nBlocks));
  Value mBlocksValue = getI32(b, loc, info.mBlocks);

  // Compute g_block first and the bid in the actual group g_block
  Value mnBlocks = getI32(b, loc, mul(b, loc, info.mBlocks, info.nBlocks));
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
    PatternRewriter &b, Location loc, Value bid, OpFoldResult mBlocks,
    Value nIter, OpFoldResult gridSize, StringRef arch, int64_t numChiplets,
    Value splitKV) {
  // Currently the firmware will launch workgroups
  // in a round-robin fashion to each chiplet. However
  // we would want a group (>=1) of chiplets to perform
  // a spatially local tile.
  // Therefore, adjust bid to make every consecutive #groups of chiplets
  // be slowest changing in the grid.
  if (numChiplets > 1) {
    OpFoldResult chunkSize =
        maxUI(b, loc, b.getI64IntegerAttr(1),
            div(b, loc, gridSize, b.getI64IntegerAttr(numChiplets)));
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, numChiplets,
                                    chunkSize);
  }
  Value g1MBlockCountVal = getI32(b, loc, mBlocks);

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
