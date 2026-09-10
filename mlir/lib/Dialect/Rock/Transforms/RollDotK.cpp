//===- RollDotK.cpp - roll a scalar-FMA dot's K into a loop ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites a `tt.dot` that lowers to scalar FMAs into an `scf.for` over
// segments of its K dimension, so that the fully unrolled `accumulators * K`
// FMAs become `accumulators * dotK` per loop body.
//
// The operands stay in the shared-memory buffer the pipeliner already gave
// them. A dot operand's shared encoding stores K as the slowest-varying
// dimension, so a K segment is a contiguous range of the buffer and the
// segmented view is just a reinterpretation: `[M, K]` becomes `[K / dotK, M,
// dotK]`, indexed by the induction variable. Nothing moves in memory.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKROLLDOTKPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-roll-dot-k"

using namespace mlir;
using namespace mlir::rock;

namespace ttg = mlir::triton::gpu;

namespace {

/// How many FMAs to aim to leave in a basic block. Below this the block stops
/// being what the backend struggles with: on the f32 convolution this pass was
/// built for, rolling to this width brings compile time to 1.7x rocMLIR, down
/// from 11x unrolled, which is the same overhead the f16 path already pays
/// without any oversized block. Going lower keeps shaving compile time but
/// buys progressively less, while each iteration has fewer FMAs to amortize
/// its loop overhead over.
constexpr int64_t kTargetBlockFMAs = 512;

struct RockRollDotKPass
    : public rock::impl::RockRollDotKPassBase<RockRollDotKPass> {
  using rock::impl::RockRollDotKPassBase<
      RockRollDotKPass>::RockRollDotKPassBase;
  void runOnOperation() override;
};

/// A `tt.dot` this pass can roll, with the shared-memory buffers its operands
/// are staged through already resolved.
struct RollableDot {
  triton::DotOp dot;
  /// The `[M, K]` and `[K, N]` buffers feeding A and B.
  TypedValue<ttg::MemDescType> aMem, bMem;
  /// Per-thread accumulator count, which rolling does not change.
  int64_t accs;
  int64_t k;
};

/// The shared-memory buffer a dot operand is loaded from, looking through the
/// layout-only conversions the pipeline puts between the load and the dot.
/// Null if the operand does not come from shared memory.
TypedValue<ttg::MemDescType> findStagingBuffer(Value operand) {
  Operation *def = operand.getDefiningOp();
  while (def) {
    if (auto load = dyn_cast<ttg::LocalLoadOp>(def)) {
      // An asynchronous load's token would have to be threaded into the loop.
      if (load.getToken())
        return {};
      return load.getSrc();
    }
    if (isa<ttg::ConvertLayoutOp>(def)) {
      def = def->getOperand(0).getDefiningOp();
      continue;
    }
    return {};
  }
  return {};
}

/// Maps each element of a shared-memory tile to its offset within the tile.
/// Indexed as `[d0 * shape[1] + d1]`.
SmallVector<int32_t> elementOffsets(ArrayRef<int64_t> shape, Attribute enc) {
  triton::LinearLayout ll = ttg::toLinearLayout(shape, enc);
  MLIRContext *ctx = enc.getContext();
  StringAttr kOffset = StringAttr::get(ctx, "offset");
  StringAttr kDim0 = StringAttr::get(ctx, "dim0");
  StringAttr kDim1 = StringAttr::get(ctx, "dim1");
  if (!ll.hasInDim(kOffset) || !ll.hasOutDim(kDim0) || !ll.hasOutDim(kDim1))
    return {};

  // A shared layout may carry input dimensions besides the offset ("block"
  // for a multi-CTA cluster); apply() insists on being given all of them.
  SmallVector<std::pair<StringAttr, int32_t>> ins;
  for (auto [name, size] : ll.getInDims())
    ins.emplace_back(name, 0);
  auto offsetIt = llvm::find_if(
      ins, [&](const auto &entry) { return entry.first == kOffset; });

  int64_t numElems = shape[0] * shape[1];
  SmallVector<int32_t> offsets(numElems, -1);
  for (int32_t off = 0; off < numElems; ++off) {
    offsetIt->second = off;
    int32_t d0 = -1, d1 = -1;
    for (auto [name, value] : ll.apply(ins)) {
      if (name == kDim0)
        d0 = value;
      else if (name == kDim1)
        d1 = value;
    }
    if (d0 < 0 || d1 < 0 || d0 >= shape[0] || d1 >= shape[1])
      return {};
    offsets[d0 * shape[1] + d1] = off;
  }
  // A layout that does not cover every element of the tile bijectively leaves
  // holes, and reasoning about segment boundaries would not be valid.
  if (llvm::is_contained(offsets, -1))
    return {};
  return offsets;
}

/// Whether viewing a `[M, K]`-shaped tile (with `kDim` naming which of the two
/// dimensions is K) as `nseg` consecutive `dotK`-wide tiles addresses exactly
/// the same bytes.
///
/// This holds when K is the slowest-varying dimension and the swizzling
/// pattern repeats per segment rather than straddling a segment boundary. It
/// is checked against the linear layout rather than by inspecting encoding
/// parameters, so an encoding this pass has not anticipated is rejected
/// instead of being silently mislowered.
bool segmentingPreservesAddresses(ArrayRef<int64_t> shape, Attribute enc,
                                  unsigned kDim, int64_t dotK) {
  assert(shape.size() == 2 && "expected a 2-D dot operand tile");
  int64_t nseg = shape[kDim] / dotK;

  SmallVector<int64_t> segShape(shape);
  segShape[kDim] = dotK;

  SmallVector<int32_t> wide = elementOffsets(shape, enc);
  SmallVector<int32_t> seg = elementOffsets(segShape, enc);
  if (wide.empty() || seg.empty())
    return false;

  int64_t segElems = segShape[0] * segShape[1];
  for (int64_t j = 0; j < nseg; ++j) {
    for (int64_t d0 = 0; d0 < segShape[0]; ++d0) {
      for (int64_t d1 = 0; d1 < segShape[1]; ++d1) {
        SmallVector<int64_t, 2> wideIdx{d0, d1};
        wideIdx[kDim] += j * dotK;
        int32_t want = j * segElems + seg[d0 * segShape[1] + d1];
        if (wide[wideIdx[0] * shape[1] + wideIdx[1]] != want)
          return false;
      }
    }
  }
  return true;
}

/// Recognizes a dot this pass can roll, without yet deciding whether it
/// should be. Returns nullopt when any gating condition fails.
std::optional<RollableDot> matchDot(triton::DotOp dot) {
  auto reject = [&](StringRef why) -> std::optional<RollableDot> {
    LLVM_DEBUG(llvm::dbgs()
               << "not rolling dot at " << dot.getLoc() << ": " << why << "\n");
    return std::nullopt;
  };

  auto dTy = dot.getD().getType();
  // The same test the AMD backend uses to route a dot to convertAMDFMADot: a
  // blocked result encoding is exactly the scalar-FMA path, while WMMA and
  // MFMA dots carry their own encodings and cost one instruction per tile.
  if (!isa<ttg::BlockedEncodingAttr>(dTy.getEncoding()))
    return reject("its result encoding is not blocked, so it does not lower "
                  "to scalar FMAs");

  auto aTy = dot.getA().getType();
  auto bTy = dot.getB().getType();
  if (aTy.getRank() != 2 || bTy.getRank() != 2)
    return reject("its operands are not both rank 2");
  if (!aTy.getElementType().isF32() || !bTy.getElementType().isF32() ||
      !dTy.getElementType().isF32())
    return reject("it is not an f32 dot");

  TypedValue<ttg::MemDescType> aMem = findStagingBuffer(dot.getA());
  TypedValue<ttg::MemDescType> bMem = findStagingBuffer(dot.getB());
  if (!aMem || !bMem)
    return reject("an operand is not staged through shared memory by a "
                  "synchronous local_load");

  // The staged buffers must be the whole tile the dot consumes, so that
  // segmenting them is a pure reinterpretation.
  auto aMemTy = aMem.getType();
  auto bMemTy = bMem.getType();
  if (aMemTy.getShape() != aTy.getShape() ||
      bMemTy.getShape() != bTy.getShape())
    return reject("a staged buffer is not the whole tile the dot consumes");
  // memdesc_reinterpret rejects subviews.
  if (aMemTy.getShape() != aMemTy.getAllocShape() ||
      bMemTy.getShape() != bMemTy.getAllocShape())
    return reject("a staged buffer is a subview, which memdesc_reinterpret "
                  "rejects");

  int64_t k = aTy.getShape()[1];
  if (k != bTy.getShape()[0] || !llvm::isPowerOf2_64(k))
    return reject("its operands disagree on K, or K is not a power of two");

  RollableDot cand;
  cand.dot = dot;
  cand.aMem = aMem;
  cand.bMem = bMem;
  cand.k = k;
  cand.accs = ttg::getTotalElemsPerThread(dTy);
  return cand;
}

/// The widest segment that gets a dot's loop body under `budget`, or 0 if the
/// dot's full K already fits, in which case rolling would shrink nothing.
int64_t chooseDotK(const RollableDot &cand, int64_t budget) {
  // The body can never hold fewer than `accs` FMAs, and each iteration
  // amortizes its overhead over `accs` FMAs, so a dot with few accumulators
  // has nothing to gain here.
  if (cand.accs >= budget)
    return 0;

  int64_t dotK = cand.k;
  while (dotK > 1 && cand.accs * dotK > budget)
    dotK /= 2;
  return dotK < cand.k ? dotK : 0;
}

/// Replaces `cand.dot` with a loop over `cand.k / dotK` narrower dots.
LogicalResult rollDot(const RollableDot &cand, int64_t dotK) {
  triton::DotOp dot = cand.dot;
  Location loc = dot.getLoc();
  assert(dotK > 0 && dotK < cand.k && cand.k % dotK == 0 &&
         "dotK must be a proper divisor of the dot's K");
  int64_t nseg = cand.k / dotK;

  auto aTy = dot.getA().getType();
  auto bTy = dot.getB().getType();
  auto aMemTy = cand.aMem.getType();
  auto bMemTy = cand.bMem.getType();

  auto canSegment = [&](StringRef which, ArrayRef<int64_t> shape, Attribute enc,
                        unsigned kDim) {
    if (segmentingPreservesAddresses(shape, enc, kDim, dotK))
      return true;
    LLVM_DEBUG(llvm::dbgs()
               << "  operand " << which
               << " cannot be segmented at dotK=" << dotK
               << ": its shared encoding does not store K as the "
                  "slowest-varying dimension with a per-segment swizzling "
                  "pattern, or its linear layout is not one this pass can "
                  "reason about\n");
    return false;
  };
  // A is [M, K] and B is [K, N], so K is dimension 1 of A and 0 of B.
  if (!canSegment("A", aMemTy.getShape(), aMemTy.getEncoding(), /*kDim=*/1) ||
      !canSegment("B", bMemTy.getShape(), bMemTy.getEncoding(), /*kDim=*/0))
    return failure();

  OpBuilder b(dot);

  auto segmentView = [&](TypedValue<ttg::MemDescType> mem,
                         ArrayRef<int64_t> segShape) -> Value {
    auto memTy = mem.getType();
    SmallVector<int64_t> segmentedShape{nseg};
    llvm::append_range(segmentedShape, segShape);
    auto ty = ttg::MemDescType::get(segmentedShape, memTy.getElementType(),
                                    memTy.getEncoding(), memTy.getMemorySpace(),
                                    memTy.getMutableMemory());
    return ttg::MemDescReinterpretOp::create(b, loc, ty, mem);
  };

  SmallVector<int64_t, 2> aSegShape{aTy.getShape()[0], dotK};
  SmallVector<int64_t, 2> bSegShape{dotK, bTy.getShape()[1]};
  Value aSeg = segmentView(cand.aMem, aSegShape);
  Value bSeg = segmentView(cand.bMem, bSegShape);

  // memdesc_index takes an i32, so the loop is built on i32 to feed it
  // directly. This matches the loops the pipeliner emits.
  Value lb = arith::ConstantIntOp::create(b, loc, 0, 32);
  Value ub = arith::ConstantIntOp::create(b, loc, nseg, 32);
  Value step = arith::ConstantIntOp::create(b, loc, 1, 32);
  auto loop = scf::ForOp::create(b, loc, lb, ub, step, ValueRange{dot.getC()});

  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loop.getBody());
    Value j = loop.getInductionVar();
    Value acc = loop.getRegionIterArg(0);

    auto loadSegment = [&](Value seg, ArrayRef<int64_t> segShape,
                           RankedTensorType operandTy) -> Value {
      auto segTy = cast<ttg::MemDescType>(seg.getType());
      auto viewTy = ttg::MemDescType::get(
          segShape, segTy.getElementType(), segTy.getEncoding(),
          segTy.getMemorySpace(), segTy.getMutableMemory());
      Value view = ttg::MemDescIndexOp::create(b, loc, viewTy, seg, j);
      // Load straight into the dot-operand layout; the narrower tile keeps the
      // encoding, which does not depend on shape.
      auto tensorTy = RankedTensorType::get(
          segShape, operandTy.getElementType(), operandTy.getEncoding());
      return ttg::LocalLoadOp::create(b, loc, tensorTy, view);
    };

    Value a = loadSegment(aSeg, aSegShape, aTy);
    Value bVal = loadSegment(bSeg, bSegShape, bTy);
    Value acc2 = triton::DotOp::create(b, loc, acc.getType(), a, bVal, acc,
                                       dot.getInputPrecision(),
                                       dot.getMaxNumImpreciseAcc());
    scf::YieldOp::create(b, loc, ValueRange{acc2});
  }

  dot.getResult().replaceAllUsesWith(loop.getResult(0));
  dot.erase();
  return success();
}

void RockRollDotKPass::runOnOperation() {
  ModuleOp mod = getOperation();

  mod.walk([&](FunctionOpInterface func) {
    if (func.isExternal())
      return;

    StringRef funcArch = arch;
    StringAttr archAttr;
    if (funcArch.empty()) {
      if (FailureOr<StringAttr> onFunc = rock::getArchOnFunc(func);
          succeeded(onFunc)) {
        archAttr = *onFunc;
        funcArch = archAttr.getValue();
      } else if (auto target =
                     mod->getAttrOfType<StringAttr>(ttg::AttrTargetName)) {
        funcArch = target.getValue();
        funcArch.consume_front("hip:");
      }
    }
    if (funcArch.empty() || !rock::isRDNA(funcArch)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "skipping " << func.getName() << ": not an RDNA target\n");
      return;
    }

    // The scheduler works a basic block at a time, so several chained dots in
    // one block (as rock-decompose-nonpow2-k emits) share the cost and have to
    // be judged together. Drain dots sit in their own scf.if regions and so
    // land in their own blocks, which is what we want.
    llvm::MapVector<Block *, SmallVector<RollableDot>> byBlock;
    func.walk([&](triton::DotOp dot) {
      if (std::optional<RollableDot> cand = matchDot(dot))
        byBlock[dot->getBlock()].push_back(*cand);
    });

    for (auto &[block, cands] : byBlock) {
      int64_t blockFMAs = 0;
      for (const RollableDot &cand : cands)
        blockFMAs += cand.accs * cand.k;
      if (blockFMAs <= kTargetBlockFMAs) {
        LLVM_DEBUG(llvm::dbgs()
                   << "leaving a block of " << cands.size()
                   << " dot(s) alone: " << blockFMAs << " FMAs is within the "
                   << kTargetBlockFMAs << " target\n");
        continue;
      }

      // Rolling moves a dot's FMAs into the loop body, which is a block of its
      // own, so each rolled body is sized against the whole target rather than
      // a share of it. What stays unrolled keeps sharing this block, which is
      // what `residual` tracks.
      int64_t residual = blockFMAs;
      auto tryRoll = [&](const RollableDot &cand, int64_t dotK) {
        LLVM_DEBUG(llvm::dbgs() << "rolling dot with K=" << cand.k << " accs="
                                << cand.accs << " into dotK=" << dotK << "\n");
        if (failed(rollDot(cand, dotK))) {
          LLVM_DEBUG(llvm::dbgs() << "  left unrolled\n");
          return;
        }
        residual -= cand.accs * cand.k;
      };

      SmallVector<const RollableDot *> alreadyFit;
      for (const RollableDot &cand : cands) {
        int64_t dotK = chooseDotK(cand, kTargetBlockFMAs);
        if (dotK == 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "dot with K=" << cand.k << " accs=" << cand.accs
                     << " is already within the target unrolled; deferring in "
                        "case the block it shares still overflows\n");
          alreadyFit.push_back(&cand);
          continue;
        }
        tryRoll(cand, dotK);
      }

      // A dot whose full K already fits the target has nothing to shrink on its
      // own, yet several such dots still overflow the block they share. Halving
      // one's K keeps its body under the target and takes it out of this block,
      // so do that until what is left unrolled fits.
      for (const RollableDot *cand : alreadyFit) {
        if (residual <= kTargetBlockFMAs)
          break;
        if (cand->k > 1)
          tryRoll(*cand, cand->k / 2);
      }
      if (residual > kTargetBlockFMAs)
        LLVM_DEBUG(llvm::dbgs()
                   << "block still holds " << residual
                   << " unrolled FMAs, above the " << kTargetBlockFMAs
                   << " target: nothing left that rolling can move out\n");
    }
  });
}

} // namespace
