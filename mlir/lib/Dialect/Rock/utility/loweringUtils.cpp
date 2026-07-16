//===- loweringUtils.cpp - Rock utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <numeric>
#include <optional>
using namespace mlir;
using namespace mlir::rock;

#define DEBUG_TYPE "rock-lowering-utils"

bool mlir::rock::isWrWAtomicKernel(StringRef arch, Type dataType,
                                   bool requiredPadding) {
  return isFastAtomicAddSupported(arch, dataType) &&
         (dataType.isF32() || dataType.isF16()) && !requiredPadding;
}

bool mlir::rock::is4GBMemoryType(ShapedType type) {
  if (!type.hasStaticShape())
    return true;
  int64_t elemBytes;
  if (auto shapedElemTy = dyn_cast<ShapedType>(type.getElementType()))
    elemBytes = (shapedElemTy.getNumElements() *
                 shapedElemTy.getElementTypeBitWidth()) /
                8;
  else
    elemBytes = type.getElementTypeBitWidth() / 8;

  return (type.getNumElements() * elemBytes) >
         (int64_t)std::numeric_limits<uint32_t>::max();
}

bool mlir::rock::isValidKBlocks(int64_t kBlocks, int64_t N) {
  return kBlocks >= 1 && N % kBlocks == 0;
}

// Padding budget for stream-K feasibility: never waste more than this fraction
// of the partition dimension on zero-filled tiles just to enable the
// decomposition. Chosen to stay below the smallest tail stream-K is offered
// for: the tuner only requests stream-K once the data-parallel work imbalance
// is >= 1.20 (computeOptimalStreamKMultiples), i.e. a wasted-work floor of
// (1.20 - 1) / 1.20 ~= 16.7%. Capping the padding at 1/8 keeps the overhead
// comfortably under that reclaimable tail so enabling stream-K stays a net win.
static constexpr double kMaxStreamKPadFraction = 0.125;

// Minimum fraction of the requested persistent grid (targetP = streamKMultiple
// * numCU) a plan must actually fill. Since P = floor(targetP / others) *
// others, a coarse partition (large `others`) can land P as low as
// ~span/(span+1) of targetP -- down to ~2/3 when span == 2 -- leaving that many
// CUs idle. Reject any partition dim that fills less than this fraction.
static constexpr double kMinStreamKFillFraction = 0.75;

FailureOr<StreamKPlan>
mlir::rock::computeStreamKPlanForDim(StreamKPartDim partDim, int64_t g,
                                     int64_t mBlocks, int64_t nBlocks,
                                     int64_t numCU, int64_t streamKMultiple) {
  if (streamKMultiple < 1)
    return failure();

  int64_t numBlocks, others;
  switch (partDim) {
  case StreamKPartDim::G:
    numBlocks = g;
    others = mBlocks * nBlocks;
    break;
  case StreamKPartDim::M:
    numBlocks = mBlocks;
    others = g * nBlocks;
    break;
  case StreamKPartDim::N:
    numBlocks = nBlocks;
    others = g * mBlocks;
    break;
  }
  // `others` is a product of two block counts, all validated > 0 above, so it
  // can never be non-positive here.
  assert(others > 0 && "stream-K: `others` must be positive");

  const int64_t targetP = streamKMultiple * numCU;
  // Largest whole number of blocks per data-parallel wave that keeps the
  // persistent grid P = span*others at or below the requested target. `span`
  // must be >= 2: the remainder needs a *proper* nonzero divisor rb of span
  // (1 <= rb < span) so splitK = span/rb is integral and a K-split remainder
  // actually exists; span == 1 admits no such rb, and span == 0 means `others`
  // alone already overshoots the target. It must also be <= numBlocks so at
  // least one full wave fits along this dim.
  const int64_t span = targetP / others; // floor
  if (span < 2 || span > numBlocks)
    return failure();
  const int64_t P = span * others;
  const int64_t rem = numBlocks % span;

  // rb ("remainder blocks") is the number of partition-dim blocks left over
  // after the full data-parallel waves are peeled off: the ragged tail
  //   paddedNumBlocks = numWaves * span + rb,   1 <= rb < span
  // that is handled by K-splitting instead of as an underfilled wave. rb must
  // be a nonzero *proper divisor* of span, so the per-tile split-K factor
  // splitK = span/rb is integral and the tail fans out into rb*splitK = span
  // sub-gemms -- exactly one more P-sized grid. Choose the rb needing the least
  // padding to reach residue rb; break ties toward the largest rb (=> smallest
  // split-K => least atomic contention).
  int64_t bestPad = -1, bestRb = 0;
  for (int64_t rb = 1; rb < span; ++rb) {
    if (span % rb != 0)
      continue;
    // Smallest pad >= 0 that moves the current tail `rem` forward to `rb`.
    // rb - rem is in (-span, span), so a single +span makes it non-negative.
    int64_t pad = (rb - rem + span) % span;
    if (numBlocks + pad < span + rb)
      pad += span; // keep at least one full data-parallel wave
    if (bestPad < 0 || pad < bestPad || (pad == bestPad && rb > bestRb)) {
      bestPad = pad;
      bestRb = rb;
    }
  }
  if (bestPad < 0)
    return failure();

  const int64_t paddedNumBlocks = numBlocks + bestPad;
  StreamKPlan plan;
  plan.partDim = partDim;
  plan.P = P;
  plan.span = span;
  plan.numWaves = (paddedNumBlocks - bestRb) / span;
  plan.remBlocks = bestRb;
  plan.splitK = span / bestRb;
  plan.padBlocks = bestPad;
  return plan;
}

FailureOr<StreamKPlan> mlir::rock::chooseStreamKPlan(int64_t g, int64_t mBlocks,
                                                     int64_t nBlocks,
                                                     int64_t numCU,
                                                     int64_t streamKMultiple) {
  // Priority order N -> M -> G, used as the final tie-break: the loop keeps the
  // first (highest-priority) plan among those of equal padding overhead.
  const StreamKPartDim order[3] = {StreamKPartDim::N, StreamKPartDim::M,
                                   StreamKPartDim::G};
  FailureOr<StreamKPlan> best = failure();
  double bestOverhead = 0.0;
  for (StreamKPartDim dim : order) {
    FailureOr<StreamKPlan> plan = computeStreamKPlanForDim(
        dim, g, mBlocks, nBlocks, numCU, streamKMultiple);
    if (failed(plan))
      continue;
    // Groups can't be padded (that would add real batches), so only accept a
    // G-partition when it needs no padding.
    if (dim == StreamKPartDim::G && plan->padBlocks > 0)
      continue;
    int64_t numBlocks = (dim == StreamKPartDim::G)   ? g
                        : (dim == StreamKPartDim::M) ? mBlocks
                                                     : nBlocks;
    double overhead = double(plan->padBlocks) / double(numBlocks);
    if (overhead > kMaxStreamKPadFraction)
      continue;
    // Reject partitions that leave the persistent grid substantially
    // underfilled: a large `others` forces span low (down to 2), so P can sit
    // well below targetP and idle a chunk of the CUs. Gate on the fill fraction
    // P/targetP.
    const int64_t targetP = streamKMultiple * numCU;
    if (static_cast<double>(plan->P) < kMinStreamKFillFraction * targetP)
      continue;
    if (failed(best) || overhead < bestOverhead) {
      best = plan;
      bestOverhead = overhead;
    }
  }
  return best;
}

LogicalResult mlir::rock::calculateKBlockNum(const int64_t batchSize,
                                             const GemmSize &gemmSize,
                                             int64_t MPerBlock,
                                             int64_t NPerBlock,
                                             int64_t KPerBlock, int64_t KPack,
                                             int64_t num_cu, int64_t &nKBlock) {
  const int64_t gemmM = gemmSize.m;
  const int64_t gemmN = gemmSize.n;
  const int64_t gemmK = gemmSize.k;

  int64_t gemmKBlock = 1;

  assert(gemmM > 0 && gemmN > 0 && gemmK > 0);
  assert(MPerBlock > 0 && NPerBlock > 0 && KPerBlock > 0 && KPack > 0 &&
         batchSize > 0);

  if ((gemmM % MPerBlock != 0) || (gemmN % NPerBlock != 0) ||
      (gemmK % (KPerBlock * KPack) != 0))
    return failure();

  const int64_t gridSize =
      gemmSize.g * (gemmM / MPerBlock) * (gemmN / NPerBlock);
  const int64_t maxGridSize = 20 * num_cu;

  gemmKBlock = std::max(maxGridSize / gridSize, static_cast<int64_t>(1));
  gemmKBlock = std::min(gemmKBlock, batchSize);

  for (; gemmKBlock > 1; --gemmKBlock) {
    if (!isValidKBlocks(gemmKBlock, batchSize))
      continue;

    if (gemmK % (gemmKBlock * KPerBlock * KPack) != 0)
      continue;

    break;
  }
  // not more than n
  gemmKBlock = std::min(batchSize, gemmKBlock);
  // not less than 1
  gemmKBlock = std::max((int64_t)1, gemmKBlock);

  nKBlock = gemmKBlock;
  return success();
}

bool mlir::rock::isEveryElementWrittenBwdData(ArrayRef<int64_t> strideDims,
                                              ArrayRef<int64_t> dilationDims,
                                              ArrayRef<int64_t> filterDims) {
  bool result = true;
  for (const auto &[stride, dilation, filterSize] :
       zip(strideDims, dilationDims, filterDims)) {
    if (!(dilation == 1 && stride <= filterSize))
      result = false;
  }
  return result;
}

SmallVector<int64_t>
mlir::rock::backwardDataKernelIds(ArrayRef<int64_t> strideDims,
                                  ArrayRef<int64_t> dilationDims,
                                  ArrayRef<int64_t> filterDims) {
  assert(strideDims.size() == dilationDims.size());
  SmallVector<int64_t, 5> gcdStrideDilations;
  for (const auto &[stride, dilation] : zip(strideDims, dilationDims))
    gcdStrideDilations.push_back(std::gcd(stride, dilation));

  SmallVector<int64_t, 5> filTilda;
  for (const auto &[stride, gcdSD] : zip(strideDims, gcdStrideDilations))
    filTilda.push_back(stride / gcdSD);

  // Populate the kernel IDs according to the current backward data convolution
  // algorithm implementation.
  llvm::SmallVector<int64_t> kernelIds;
  int64_t subproduct = 1;
  int64_t product;
  for (size_t i = 1; i < filterDims.size(); i++)
    subproduct *= filTilda[i];
  product = subproduct * filTilda[0];
  for (int64_t kernelId = 0; kernelId < product; ++kernelId) {
    // gemmK size is different for each GEMM
    SmallVector<int64_t, 3> iTilda;
    int64_t divisor = 1;
    iTilda.resize(filterDims.size());
    switch (filterDims.size()) {
    default:
      llvm_unreachable("Only 2-D and 3-D have been implemented.");
      break;
    case 3:
      divisor = filTilda[2];
      iTilda[2] = kernelId % divisor;
      [[fallthrough]];
    case 2:
      iTilda[1] = (kernelId % subproduct) / divisor;
      iTilda[0] = kernelId / subproduct;
    }

    // gemmK must be > 0, otherwise this kernel has no filter slice to run.
    int64_t gemmKproduct = 1;
    for (size_t i = 0; i < filterDims.size(); i++) {
      if (iTilda[i] >= filterDims[i]) {
        gemmKproduct = 0;
        break;
      }
      gemmKproduct *= llvm::divideCeil(filterDims[i] - iTilda[i], filTilda[i]);
    }
    if (gemmKproduct > 0) {
      kernelIds.push_back(kernelId);
    }
  }

  return kernelIds;
}

FailureOr<ArrayAttr> mlir::rock::getLoadRegsAsTileViews(
    OpBuilder &b, Location loc, Value globalBuffer, StringRef dName,
    ArrayRef<int64_t> bidGridLengths, int64_t kPerBlock, int64_t dPerBlock,
    bool isKFirst) {
  SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
  if (dName != "m" && dName != "n") {
    return emitError(loc, "expected dName to be m or n but got " + dName);
  }
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  StringRef otherBlockDim = dName == "m" ? "n_block" : "m_block";

  ShapedType matrixType = cast<ShapedType>(globalBuffer.getType());
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
  // For matrix B (isKFirst=true): k at index 1, d at index 2
  // For matrix A (isKFirst=false): k at index 2, d at index 1
  int64_t kGlobal = isKFirst ? matrixShape[1] : matrixShape[2];
  int64_t dGlobal = isKFirst ? matrixShape[2] : matrixShape[1];

  int64_t kIters = kGlobal / kPerBlock;

  std::string dIterName = llvm::formatv("{0}_iter", dName);

  std::string firstDim = dIterName;
  int firstDimLen = dPerBlock;
  std::string secondDim = "k_iter";
  int secondDimLen = kPerBlock;
  if (isKFirst) {
    std::swap(firstDim, secondDim);
    std::swap(firstDimLen, secondDimLen);
  }

  TopDownTMBuilder toGlobalIdx(b,
                               {"k_loop", bidGridOrder[0], bidGridOrder[1],
                                bidGridOrder[2], firstDim, secondDim},
                               {kIters, bidGridLengths[0], bidGridLengths[1],
                                bidGridLengths[2], firstDimLen, secondDimLen},
                               loc);

  toGlobalIdx.passThrough({"g"}, {0}, {"g_block"});
  // For matrix B (isKFirst): source is [g, k, n], k at index 1, n at index 2
  // For matrix A (!isKFirst): source is [g, m, k], m at index 1, k at index 2
  int kLowerIdx = isKFirst ? 1 : 2;
  int dLowerIdx = isKFirst ? 2 : 1;
  toGlobalIdx.unmerge("k", kLowerIdx, {"k_loop", "k_iter"},
                      {kIters, kPerBlock});
  toGlobalIdx.unmerge(dName, dLowerIdx, {thisBlockDim, dIterName},
                      {dGlobal / dPerBlock, dPerBlock});

  toGlobalIdx.ignore(otherBlockDim);
  TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();
  return b.getArrayAttr({toGlobalIdxAttr});
}

Value mlir::rock::normalizeMatrix(Value matrix, OpBuilder &b, Location loc,
                                  bool doTranspose, StringRef firstDim,
                                  StringRef secondDim) {
  auto matrixType = cast<ShapedType>(matrix.getType());
  bool addGroup = matrixType.getShape().size() != 3;
  if (!addGroup && !doTranspose)
    return matrix;
  OpBuilder::InsertionGuard guard(b);
  if (auto *defOp = matrix.getDefiningOp())
    b.setInsertionPointAfter(defOp);
  SmallVector<StringRef, 3> bottomNames;
  if (!addGroup)
    bottomNames.push_back("gemmG");
  if (doTranspose)
    bottomNames.append({secondDim, firstDim});
  else
    bottomNames.append({firstDim, secondDim});
  BottomUpTMBuilder normalizer(b, bottomNames, matrixType.getShape(), loc);

  if (addGroup)
    normalizer.addDim("gemmG", 0, 1);
  else
    normalizer.passThrough(normalizer.startName(0));

  normalizer.passThrough({firstDim, secondDim}, {1, 2}, {firstDim, secondDim});
  TransformMapAttr normalizeAttr = normalizer.get();
  return TransformOp::create(b, loc, matrix, normalizeAttr);
}

Value mlir::rock::padVector(Value vector, OpBuilder &b, Location loc,
                            StringRef firstDim, int64_t firstDimPad) {
  if (firstDimPad == 0)
    return vector;
  OpBuilder::InsertionGuard guard(b);
  if (auto *defOp = vector.getDefiningOp())
    b.setInsertionPointAfter(defOp);
  ArrayRef<int64_t> shape = cast<ShapedType>(vector.getType()).getShape();
  assert(shape.size() == 2);
  BottomUpTMBuilder padder(b, {"gemmG", firstDim}, shape, loc);
  padder.passThrough("gemmG");
  SmallString<8> paddedName;
  (firstDim + Twine("Pad")).toVector(paddedName);
  padder.pad(paddedName, firstDim, 0, firstDimPad);
  TransformMapAttr padAttr = padder.get();
  return TransformOp::create(b, loc, vector, padAttr);
}

Value mlir::rock::padMatrix(Value matrix, OpBuilder &b, Location loc,
                            StringRef firstDim, int64_t firstDimPad,
                            StringRef secondDim, int64_t secondDimPad) {
  if (firstDimPad == 0 && secondDimPad == 0)
    return matrix;
  OpBuilder::InsertionGuard guard(b);
  if (auto *defOp = matrix.getDefiningOp())
    b.setInsertionPointAfter(defOp);
  ArrayRef<int64_t> shape = cast<ShapedType>(matrix.getType()).getShape();
  BottomUpTMBuilder padder(b, {"gemmG", firstDim, secondDim}, shape, loc);
  padder.passThrough("gemmG");
  if (firstDimPad == 0) {
    padder.passThrough(firstDim);
  } else {
    SmallString<8> paddedName;
    (firstDim + Twine("Pad")).toVector(paddedName);
    padder.pad(paddedName, firstDim, 0, firstDimPad);
  }
  if (secondDimPad == 0) {
    padder.passThrough(secondDim);
  } else {
    SmallString<8> paddedName;
    (secondDim + Twine("Pad")).toVector(paddedName);
    padder.pad(paddedName, secondDim, 0, secondDimPad);
  }
  TransformMapAttr padAttr = padder.get();
  return TransformOp::create(b, loc, matrix, padAttr);
}

Value mlir::rock::splitKFoldOperand(OpBuilder &b, Location loc, Value operand,
                                    int64_t splitK, unsigned kDim,
                                    unsigned nonKDim, StringRef preservedName) {
  auto type = cast<RankedTensorType>(operand.getType());
  ArrayRef<int64_t> shape = type.getShape();
  assert(shape.size() == 3 && "expected a rank-3 [G, *, *] gemm operand");
  assert(kDim >= 1 && kDim <= 2 && "kDim must be 1 or 2 (G is dim 0)");
  assert(nonKDim >= 1 && nonKDim <= 2 && "nonKDim must be 1 or 2 (G is dim 0)");
  assert(nonKDim != kDim && "nonKDim must differ from kDim");
  assert(splitK >= 1 && "splitK must be positive");
  int64_t operandK = shape[kDim];
  assert(operandK % splitK == 0 &&
         "operand K must be divisible by splitK after padding");

  // The unmerge inserts the split sub-dim (gemmKSplit) at `kDim`, so the
  // preserved dim shifts right by one only if it currently sits after `kDim`.
  unsigned newNonKDim = nonKDim < kDim ? nonKDim : nonKDim + 1;

  SmallVector<StringRef, 3> inNames(3);
  inNames[0] = "gemmG";
  inNames[kDim] = "gemmK";
  inNames[nonKDim] = preservedName;

  // Bottom: unmerge gemmK -> (gemmKSplit, gemmK); the preserved dim moves to
  // newNonKDim to make room for the extra split sub-dim.
  BottomUpTMBuilder unmerge(b, inNames, shape, loc);
  unmerge.passThrough({"gemmG", preservedName}, {0, newNonKDim},
                      {"gemmG", preservedName});
  unmerge.unmerge({"gemmKSplit", "gemmK"}, {kDim, kDim + 1}, "gemmK",
                  {splitK, operandK / splitK});
  TransformMapAttr unmergeAttr = unmerge.get();

  // Above: merge (gemmG, gemmKSplit) -> gemmG; restore the original K/preserved
  // positions.
  BottomUpTMBuilder merge = BottomUpTMBuilder::above(unmerge, unmergeAttr);
  merge.merge("gemmG", 0, {"gemmG", "gemmKSplit"});
  merge.passThrough({"gemmK", preservedName}, {kDim, nonKDim},
                    {"gemmK", preservedName});
  TransformMapAttr mergeAttr = merge.get();

  return rock::transform(b, operand, b.getArrayAttr({mergeAttr, unmergeAttr}));
}

Value mlir::rock::splitKFoldOutputView(OpBuilder &b, Location loc, Value view,
                                       int64_t splitK) {
  auto type = cast<RankedTensorType>(view.getType());
  ArrayRef<int64_t> shape = type.getShape();
  assert(shape.size() == 3 && "expected a rank-3 [G, M, N] output view");
  assert(splitK >= 1 && "splitK must be positive");
  int64_t G = shape[0], M = shape[1], N = shape[2];

  TopDownTMBuilder merge(b, {"gemmG", "gemmM", "gemmN"}, {G * splitK, M, N},
                         loc);
  merge.merge({"gemmG", "gemmKSplit"}, {0, 1}, "gemmG", {G, splitK});
  merge.passThrough({"gemmM", "gemmN"}, {2, 3}, {"gemmM", "gemmN"});
  TransformMapAttr mergeAttr = merge.get();

  TopDownTMBuilder ignore = TopDownTMBuilder::below(merge, mergeAttr);
  ignore.ignore("gemmKSplit");
  ignore.passThrough({"gemmG", "gemmM", "gemmN"}, {0, 1, 2},
                     {"gemmG", "gemmM", "gemmN"});
  TransformMapAttr ignoreAttr = ignore.get();

  return rock::transform(b, view, b.getArrayAttr({mergeAttr, ignoreAttr}));
}

Value mlir::rock::scaleFusionAddendDown(OpBuilder &b, Location loc, Value other,
                                        int64_t factor) {
  Type otherElmType = cast<ShapedType>(other.getType()).getElementType();
  Value factorValue = createConstantFloatOp(
      b, loc, other.getType(), otherElmType, static_cast<float>(factor));
  return b.createOrFold<arith::DivFOp>(loc, other, factorValue);
}

LogicalResult mlir::rock::regularizeOutputFusionForKReduction(Value gemmResult,
                                                              int64_t factor,
                                                              RewriterBase &b) {
  // A K reduction spread across `factor` atomic-add partials applies every
  // fused `add/sub gemmOut, other` `factor` times, so divide `other` by
  // `factor` to keep the total unchanged. mul/div/neg/type-conversions
  // distribute over the partial sum and need no change (checkValidOutputFusion
  // only collects the add/sub ops here).
  SmallVector<std::tuple<Operation *, int>> adds;
  if (failed(checkValidOutputFusion(gemmResult, adds)))
    return gemmResult.getDefiningOp()->emitOpError(
        "has invalid output fusion for a K reduction");

  for (auto [arithOp, gemmOutIndex] : adds) {
    assert(arithOp->getNumOperands() == 2);
    assert(gemmOutIndex == 0 || gemmOutIndex == 1);
    b.setInsertionPoint(arithOp);
    Value gemmOut = arithOp->getOperand(gemmOutIndex);
    int otherIndex = (gemmOutIndex == 0) ? 1 : 0;
    Value otherBySplitk = scaleFusionAddendDown(
        b, arithOp->getLoc(), arithOp->getOperand(otherIndex), factor);
    if (isa<arith::AddFOp>(arithOp)) {
      b.replaceOpWithNewOp<arith::AddFOp>(arithOp, gemmOut, otherBySplitk);
    } else if (isa<arith::SubFOp>(arithOp)) {
      if (gemmOutIndex == 0)
        b.replaceOpWithNewOp<arith::SubFOp>(arithOp, gemmOut, otherBySplitk);
      else
        b.replaceOpWithNewOp<arith::SubFOp>(arithOp, otherBySplitk, gemmOut);
    } else {
      return failure();
    }
  }
  return success();
}

FailureOr<BlockArgument> mlir::rock::findBlockArgument(Value value) {
  auto maybeBlockArg = dyn_cast_or_null<BlockArgument>(value);
  while (!maybeBlockArg) {
    // Keep going until the operation that defines the value is a
    // view-like operation
    if (auto viewOp =
            dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp())) {
      value = viewOp.getViewSource();
    } else {
      return failure();
    }
    maybeBlockArg = dyn_cast_or_null<BlockArgument>(value);
  }

  return maybeBlockArg;
}

// Helper function to get attributes from parents
template <typename RetAttrType>
static FailureOr<RetAttrType> getAttrFromOpOrParents(
    Operation *op, StringRef opAttr,
    std::optional<StringRef> maybeDialectAttr = std::nullopt) {
  StringRef dialectAttr = maybeDialectAttr.value_or(opAttr);
  Operation *func = getParentFuncOp(op);
  RetAttrType attr;
  auto getAnyAttr = [&](ArrayRef<StringRef> attrNames, Operation *op) {
    for (StringRef attrName : attrNames) {
      if (!attr) {
        attr = op->getAttrOfType<RetAttrType>(attrName);
      } else {
        return;
      }
    }
  };

  // First check for the attribute on the op
  getAnyAttr({opAttr}, op);
  if (!attr) {
    // If that fails then try checking for the attribute on the func
    getAnyAttr({opAttr, dialectAttr}, func);
  }

  // If there is no desired attribute on the func, then check the nearest parent
  // with a symbol table (covers both ModuleOp and gpu::GPUModuleOp)
  if (!attr) {
    if (auto symbolTableOp = func->getParentWithTrait<OpTrait::SymbolTable>()) {
      getAnyAttr({opAttr, dialectAttr}, symbolTableOp);
      if (attr)
        return attr;
    }
  }

  if (!attr) {
    return failure();
  }
  return attr;
}

FailureOr<IntegerAttr> mlir::rock::getGridSize(Operation *op) {
  return getAttrFromOpOrParents<IntegerAttr>(op,
                                             rock::GridSizeAttr::getMnemonic());
}

FailureOr<IntegerAttr> mlir::rock::getBlockSize(Operation *op) {
  return getAttrFromOpOrParents<IntegerAttr>(
      op, rock::BlockSizeAttr::getMnemonic());
}

FailureOr<SetVector<StoreOp>>
mlir::rock::traceRootOutputToStoreOps(Value output) {
  SetVector<StoreOp> stores;

  // output should be the result of the kernel (gemm, attention, etc.)
  // Find rock.store operations that use output as their source,
  // tracing through fusion ops (arith.*, math.*) to reach the stores.
  SmallVector<Value> worklist;
  worklist.push_back(output);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (auto storeOp = dyn_cast<StoreOp>(owner)) {
        // Only the stored-value operand consumes the traced output.
        if (use.get() == storeOp.getSource())
          stores.insert(storeOp);
      } else if (isForwardTraceOp(owner)) {
        for (Value result : owner->getResults())
          worklist.push_back(result);
      }
    }
  }

  if (!stores.empty())
    return stores;

  LLVM_DEBUG(
      llvm::dbgs() << "traceRootOutputToStoreOps: no rock.store ops found!\n");
  return failure();
}

FailureOr<SmallVector<BlockArgument>>
mlir::rock::traceRootOutputToArgs(Value output, func::FuncOp func) {
  if (func.getNumArguments() == 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "traceRootOutputToArgs: no function arguments\n");
    return failure();
  }

  FailureOr<SetVector<StoreOp>> maybeStores = traceRootOutputToStoreOps(output);
  if (failed(maybeStores))
    return failure();

  SetVector<BlockArgument> args;
  auto funcArgs = func.getArguments();

  for (auto storeOp : maybeStores.value()) {
    // The dest operand of rock.store can be traced to a function argument
    Value dest = storeOp.getDest();
    FailureOr<BlockArgument> destArg = findBlockArgument(dest);
    if (succeeded(destArg)) {
      for (auto arg : funcArgs) {
        if (destArg.value() == arg)
          args.insert(arg);
      }
    }
  }

  if (!args.empty())
    return SmallVector<BlockArgument>(args.begin(), args.end());

  LLVM_DEBUG(llvm::dbgs() << "traceRootOutputToArgs: no arguments found!\n");
  return failure();
}

ArrayAttr
mlir::rock::computeOutputLseTransforms(OpBuilder &b, Location loc,
                                       int64_t mPerBlock,
                                       ArrayRef<int64_t> bidGridLengths) {
  // Create views as gridwise sub-tile of LSE
  TopDownTMBuilder toMatrixLSE(
      b, {"g_block", "m_block", "m_iter"},
      {bidGridLengths[0], bidGridLengths[1], mPerBlock}, loc);

  toMatrixLSE.passThrough({"gemmG"}, {0}, {"g_block"});
  toMatrixLSE.unmerge("gemmM", 1, {"m_block", "m_iter"},
                      {bidGridLengths[1], mPerBlock});

  TransformMapAttr toMatrixLSEAttr = toMatrixLSE.get();

  // Before returning the output view, if necessary, swap back the
  // threadid/iter dimensions on both the M/N axis.
  SmallVector<Attribute> transformAttrs{toMatrixLSEAttr};

  return b.getArrayAttr(transformAttrs);
}

llvm::FailureOr<ArrayAttr>
mlir::rock::computeOutputTransforms(OpBuilder &b, Location loc,
                                    int64_t mPerBlock, int64_t nPerBlock,
                                    ArrayRef<int64_t> bidGridLengths) {
  // Create views as gridwise sub-tile of C
  TopDownTMBuilder toMatrixC(
      b, {"g_block", "m_block", "n_block", "m_iter", "n_iter"},
      {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], mPerBlock,
       nPerBlock},
      loc);

  toMatrixC.passThrough({"gemmG"}, {0}, {"g_block"});
  toMatrixC.unmerge("gemmM", 1, {"m_block", "m_iter"},
                    {bidGridLengths[1], mPerBlock});
  toMatrixC.unmerge("gemmN", 2, {"n_block", "n_iter"},
                    {bidGridLengths[2], nPerBlock});

  TransformMapAttr toMatrixCAttr = toMatrixC.get();

  // Before returning the output view, if necessary, swap back the
  // threadid/iter dimensions on both the M/N axis.
  SmallVector<Attribute> transformAttrs{toMatrixCAttr};

  return b.getArrayAttr(transformAttrs);
}

Type mlir::rock::getAccType(Type elemA, Type elemB) {
  OpBuilder b(elemA.getContext());

  Type accType;
  if (isa<FloatType>(elemA) && isa<FloatType>(elemB)) {
    accType = b.getF32Type();
  } else if (isa<IntegerType>(elemA) && isa<IntegerType>(elemB)) {
    accType = b.getI32Type();
  } else {
    llvm_unreachable("not expected type");
  }
  return accType;
}

// This function will process a tile of gemm input into LDS (or register)
// buffer in a way it could be fed to blockwise_gemm op
Value mlir::rock::loadTile(PatternRewriter &rewriter, Location loc, Value in,
                           Value kIter, StringRef dName,
                           rock::layout::GridCoordinates gridCoords,
                           int64_t kPerBlock, int64_t dPerBlock, bool isKFirst,
                           SmallVector<int64_t, 3> &bidGridLengths,
                           rock::CacheModifier cache) {
  FailureOr<ArrayAttr> maybeBufferViews = getLoadRegsAsTileViews(
      rewriter, loc, in, dName, bidGridLengths, kPerBlock, dPerBlock, isKFirst);
  assert(succeeded(maybeBufferViews));
  ArrayAttr bufferViews = maybeBufferViews.value();

  // Compute the tile result type by applying the tiling transforms to
  // determine the output shape, then taking the last two dimensions.
  Value wrappedSource = transform(rewriter, in, bufferViews);
  auto sourceType = cast<RankedTensorType>(wrappedSource.getType());
  auto sourceShape = sourceType.getShape();
  auto resultType = RankedTensorType::get(sourceShape.take_back(2),
                                          sourceType.getElementType());

  // Create a LoadMarkerOp placeholder. LowerLoads will later convert this
  // into an actual BlockwiseLoadOp by tracing back through the source chain.
  // We pass the original (un-transformed) input as source and carry the
  // tiling transforms as metadata in extraViews.
  auto markerOp =
      LoadMarkerOp::create(rewriter, loc, resultType, in, bufferViews,
                           ValueRange{kIter, gridCoords.g_block,
                                      gridCoords.m_block, gridCoords.n_block},
                           cache);
  return markerOp.getResult();
}

// This function creates a zero-initialized accumulator tensor
Value mlir::rock::createZeroAccBuffer(PatternRewriter &rewriter, Location loc,
                                      ArrayRef<int64_t> shape, Type accType) {
  auto tensorType = RankedTensorType::get(shape, accType);
  auto zeroAttr = rewriter.getZeroAttr(tensorType);
  return arith::ConstantOp::create(rewriter, loc, tensorType, zeroAttr);
}

Value mlir::rock::insertBroadcast(OpBuilder &b, Location loc, Value inp,
                                  ArrayRef<int64_t> outShape) {
  ArrayRef<int64_t> inpShape = cast<ShapedType>(inp.getType()).getShape();
  bool broadcastDone = false;
  rock::BottomUpTMBuilder broadcastDims(b, inpShape, loc);
  for (unsigned int i = 0; i < outShape.size(); i++) {
    if (inpShape[i] == 1 && outShape[i] != 1) {
      broadcastDims.broadcast({i}, {outShape[i]});
      broadcastDone = true;
    } else {
      broadcastDims.passThrough({i}, {i});
    }
  }
  if (!broadcastDone)
    return inp;
  return rock::TransformOp::create(b, loc, inp, broadcastDims.get());
}

bool mlir::rock::isFusionOp(Operation *op) {
  if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
    return false;
  // Exclude zero-operand ops like arith.constant — they don't participate
  // in data-flow fusion chains.
  return op->getNumOperands() > 0 && op->getNumResults() == 1;
}

bool mlir::rock::isForwardTraceOp(Operation *op) {
  return isFusionOp(op) || isa<ViewLikeOpInterface>(op) || isa<ReduceOp>(op);
}

FusionInfo mlir::rock::collectFusionInfo(Value root) {
  DenseSet<Value> chainValues;
  chainValues.insert(root);

  // Pass 1: flood-fill all values reachable through fusion ops from root.
  // Must be done before checking operands, because use-list iteration order
  // is not guaranteed to follow program order — a downstream op (e.g. addf)
  // may be visited before an upstream op (e.g. mulf), causing the upstream
  // result to be missing from chainValues when the downstream op's operands
  // are inspected.
  SmallVector<Value> worklist;
  worklist.push_back(root);
  SmallVector<Operation *> fusionOps;
  DenseSet<Operation *> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (!(isFusionOp(owner) || isa<ViewLikeOpInterface>(owner)) ||
          !visited.insert(owner).second)
        continue;
      if (isFusionOp(owner))
        fusionOps.push_back(owner);

      for (Value result : owner->getResults()) {
        chainValues.insert(result);
        worklist.push_back(result);
      }
    }
  }

  // Pass 2: now that chainValues is complete, find operands outside the chain.
  DenseMap<Value, Value> extraInputs;
  for (Operation *op : fusionOps) {
    for (Value operand : op->getOperands()) {
      if (!chainValues.count(operand))
        extraInputs.try_emplace(operand, operand);
    }
  }

  return {extraInputs, chainValues, fusionOps};
}

DenseMap<Value, Value> mlir::rock::collectFusionExtraInputs(Value root) {
  return collectFusionInfo(root).extraInputs;
}

void mlir::rock::replaceFusionExtraInputs(
    Value root, const DenseMap<Value, Value> &inputMap) {
  if (inputMap.empty())
    return;
  SmallVector<Value> worklist;
  worklist.push_back(root);
  DenseSet<Operation *> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (!isFusionOp(owner) || visited.count(owner))
        continue;
      visited.insert(owner);

      // Replace extra input operands with their padded versions.
      for (OpOperand &operand : owner->getOpOperands()) {
        auto it = inputMap.find(operand.get());
        if (it != inputMap.end() && it->second != it->first)
          operand.set(it->second);
      }

      // Continue through results.
      for (Value result : owner->getResults())
        worklist.push_back(result);
    }
  }
}

LogicalResult mlir::rock::setStoreMethodAndPrefill(OpBuilder &builder,
                                                   StoreOp storeOp,
                                                   StoreMethod newStoreMethod) {
  StoreMethod existing = storeOp.getStoreMethod();
  if (newStoreMethod == StoreMethod::AtomicAdd &&
      existing == StoreMethod::AtomicMax)
    return storeOp->emitError(
        "incompatible store methods: can't set atomic_add on atomic_max store");
  if (newStoreMethod == StoreMethod::AtomicMax &&
      existing == StoreMethod::AtomicAdd)
    return storeOp->emitError(
        "incompatible store methods: can't set atomic_max on atomic_add store");

  if (newStoreMethod == StoreMethod::Set)
    return success();

  storeOp.setStoreMethodAttr(builder.getAttr<StoreMethodAttr>(newStoreMethod));

  auto func = storeOp->getParentOfType<func::FuncOp>();
  if (!func)
    return storeOp->emitError("store op not inside a function");

  FailureOr<BlockArgument> destArg = findBlockArgument(storeOp.getDest());
  if (failed(destArg))
    return storeOp->emitError(
        "can't trace store destination to function argument");

  auto elementType = cast<ShapedType>(destArg->getType()).getElementType();
  bool isMax = (newStoreMethod == StoreMethod::AtomicMax);
  Attribute prefillValue;
  if (auto floatTy = dyn_cast<FloatType>(elementType)) {
    if (isMax)
      prefillValue = builder.getFloatAttr(
          floatTy,
          APFloat::getInf(floatTy.getFloatSemantics(), /*Negative=*/true));
    else
      prefillValue = builder.getFloatAttr(floatTy, 0.0);
  } else if (auto intTy = dyn_cast<IntegerType>(elementType)) {
    if (isMax)
      prefillValue = builder.getIntegerAttr(
          intTy, APInt::getSignedMinValue(intTy.getWidth()));
    else
      prefillValue = builder.getIntegerAttr(intTy, 0);
  } else {
    return storeOp->emitError("expecting float or int element type");
  }

  func.setArgAttr(destArg->getArgNumber(), PrefillAttr::getMnemonic(),
                  prefillValue);
  return success();
}

void mlir::rock::propagateOutputType(Value oldRoot, Value newRoot) {
  auto newRootType = dyn_cast<RankedTensorType>(newRoot.getType());
  if (!newRootType)
    return;

  // worklist items: (oldValue whose uses to scan, newValue to substitute)
  SmallVector<std::pair<Value, Value>> worklist;
  worklist.push_back({oldRoot, newRoot});
  DenseSet<Operation *> visited;

  while (!worklist.empty()) {
    auto [oldVal, newVal] = worklist.pop_back_val();
    auto newShape = cast<RankedTensorType>(newVal.getType()).getShape();

    for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
      Operation *owner = use.getOwner();
      if (!isFusionOp(owner))
        continue;

      // Always replace the operand, even if we've already visited this op.
      // A fusion op can use the same value for multiple operands
      // (e.g. arith.addf %x, %x).
      use.set(newVal);

      if (visited.count(owner))
        continue;
      visited.insert(owner);

      // Update each result: preserve element type, adopt the new shape.
      for (OpResult result : owner->getResults()) {
        auto oldType = dyn_cast<RankedTensorType>(result.getType());
        if (!oldType)
          continue;
        if (oldType.getShape() != newShape) {
          auto updatedType =
              RankedTensorType::get(newShape, oldType.getElementType());
          result.setType(updatedType);
        }
        // Continue propagating through this result's downstream uses.
        worklist.push_back({result, result});
      }
    }
  }
}

FailureOr<OutputsAndFusionInputs>
mlir::rock::traceOutputsAndFusionInputs(Value rootOut) {
  auto maybeStores = rock::traceRootOutputToStoreOps(rootOut);
  if (failed(maybeStores))
    return failure();

  OutputsAndFusionInputs info;
  info.stores = maybeStores.value();
  for (auto storeOp : info.stores)
    info.outputViews.push_back(storeOp.getDest());

  // Collect extra fusion inputs (operands of fusion ops that are not in the
  // gemm-result chain, e.g. the second operand of arith.addf).
  info.fusionInputMap = rock::collectFusionExtraInputs(rootOut);
  return info;
}

arith::NarrowTypeEmulationConverter rock::create4BitTypeConverter() {
  arith::NarrowTypeEmulationConverter typeConverter(/*targetBitwidth=*/8);
  memref::populateMemRefNarrowTypeEmulationConversions(typeConverter);
  typeConverter.addSourceMaterialization([](OpBuilder &builder, Type type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });
  typeConverter.addTargetMaterialization([](OpBuilder &builder, Type type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });
  return typeConverter;
}

void mlir::rock::markAsNotApplicable(Operation *op) {
  assert(op && "markAsNotApplicable: op must be non-null");
  ModuleOp moduleOp =
      isa<ModuleOp>(op) ? cast<ModuleOp>(op) : op->getParentOfType<ModuleOp>();
  assert(moduleOp && "markAsNotApplicable: op must be inside a ModuleOp");
  moduleOp->setAttr(rock::NotApplicableAttr::getMnemonic(),
                    UnitAttr::get(op->getContext()));
}
