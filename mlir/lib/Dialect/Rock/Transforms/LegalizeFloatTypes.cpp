//===- LegalizeFloatTypes.cpp - non-TT_Float -> integer legalization ------===//
//
// Copyright 2026 The MLIR Authors.
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
//===----------------------------------------------------------------------===//
//
// Triton's tt.load only supports TT_Float types. Float types outside that set
// (e.g. f8E8M0FNU, f4E2M1FN) are rewritten to integer types:
//   - 8-bit types (f8E8M0FNU) -> i8 (same shape, same size)
//   - 4-bit types (f4E2M1FN)  -> first to i4, then packed to i8 with dimension
//     halving along the K dimension (or D if K is not has maxVectorization <
//     2).
//
// The packing uses getMaxVectorization() to find which dimension has
// maxVectorization >= 2, halves it, and rewrites ALL transforms in the chain.
//
// The original float type and packing direction (matrixAKPack/matrixBKPack) are
// saved on BlockwiseGemmOp for RockToTTIR to set on tt.dot_scaled.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Rock/utility/tritonUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/WalkResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLEGALIZEFLOATTYPESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-legalize-float-types"

using namespace mlir;
using namespace mlir::rock;

namespace {

static bool isNonTTFloat(Type t,
                         std::optional<unsigned> bitWidth = std::nullopt) {
  if (auto tType = dyn_cast<RankedTensorType>(t))
    t = tType.getElementType();
  if (!isa<FloatType>(t) || rock::isTTFloat(t))
    return false;
  if (bitWidth)
    return t.getIntOrFloatBitWidth() == *bitWidth;
  return t.getIntOrFloatBitWidth() == 8 || t.getIntOrFloatBitWidth() == 4;
}

//===----------------------------------------------------------------------===//
// Type conversion: unsupported float -> integer of same bit width
//===----------------------------------------------------------------------===//

/// Bitcast non-TT_Float types to integer types of the same bit width
/// (e.g. f8E8M0FNU -> i8).
static Type convertElementType(Type elemType, MLIRContext *ctx) {
  if (!isNonTTFloat(elemType))
    return elemType;
  return IntegerType::get(ctx, elemType.getIntOrFloatBitWidth());
}

static bool is4Bit(Type t) {
  return t.isIntOrFloat() && t.getIntOrFloatBitWidth() == 4;
}

static Type convertType(Type type, MLIRContext *ctx) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Type newElem = convertElementType(tensorType.getElementType(), ctx);
    if (newElem == tensorType.getElementType())
      return type;
    return RankedTensorType::get(tensorType.getShape(), newElem,
                                 tensorType.getEncoding());
  }
  return convertElementType(type, ctx);
}

/// Record original element types on every BlockwiseGemmOp before the type
/// conversion rewrites them.
static void recordOrigTypesOnGemm(func::FuncOp funcOp) {
  funcOp.walk([](BlockwiseGemmOp gemmOp) {
    Type aElem =
        cast<ShapedType>(gemmOp.getMatrixA().getType()).getElementType();
    Type bElem =
        cast<ShapedType>(gemmOp.getMatrixB().getType()).getElementType();
    if (isNonTTFloat(aElem))
      gemmOp.setMatrixAOrigElemTypeAttr(TypeAttr::get(aElem));
    if (isNonTTFloat(bElem))
      gemmOp.setMatrixBOrigElemTypeAttr(TypeAttr::get(bElem));
  });
}

//===----------------------------------------------------------------------===//
// 4-bit packing: i4 -> i8 with dimension halving
//===----------------------------------------------------------------------===//

/// A collected transform chain: the sequence of TransformOps from the block arg
/// up to the tile load, plus the root block arg and the blockwise_load op.
struct OperandInput {
  BlockArgument rootArg;
  BlockwiseLoadOp loadOp;
};

/// Walk backwards from `val` through fusion ops and blockwise_loads to
/// collect all inputs.
/// The expected IR pattern is:
///   block_arg -> Transform -> ... -> Transform -> BlockwiseLoad
///     -> FusionOp -> ... -> FusionOp -> BlockwiseGemm
static FailureOr<SmallVector<OperandInput>> collectOperandInputs(Value val) {
  SmallVector<OperandInput> inputs;
  SmallVector<Value> worklist{val};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;

    Operation *defOp = cur.getDefiningOp();
    if (!defOp) {
      emitError(cur.getLoc(), "reached block arg without blockwise_load");
      return failure();
    }

    if (auto loadOp = dyn_cast<BlockwiseLoadOp>(defOp)) {
      Value source = loadOp.getSource();
      OperandInput input;
      input.loadOp = loadOp;
      Value root;
      SmallVector<TransformOp> transforms;
      std::tie(root, std::ignore) = rock::untransform(source, transforms);
      auto blockArg = dyn_cast<BlockArgument>(root);
      if (!blockArg) {
        loadOp.emitError("transform chain root is not a block argument");
        return failure();
      }
      if (is4Bit(cast<ShapedType>(blockArg.getType()).getElementType())) {
        input.rootArg = blockArg;
        inputs.push_back(input);
      }
      continue;
    }

    if (rock::isFusionOp(defOp)) {
      for (Value operand : defOp->getOperands())
        if (isa<RankedTensorType>(operand.getType()))
          worklist.push_back(operand);
      continue;
    }

    // This walk is searching for i4 block arguments that feed through
    // transform chains into blockwise_loads.  Constants (e.g. inlined scale
    // or zero-point literals) can appear as operands of fusion
    // ops in dequant chains, but they are leaf values with no further operands
    // to trace, and they can never be i4 block arguments that need packing.
    if (isa<arith::ConstantOp>(defOp))
      continue;

    defOp->emitError("unexpected op in 4-bit transform chain walk");
    return failure();
  }
  return inputs;
}

/// Given a TransformMapAttr and a dimension index, halve the size of that
/// dimension in the upper bounds and adjust the transform op that produces it.
/// Returns the new TransformMapAttr and the index of the halved lower dimension
/// (for propagation to the next level).
/// `targetLowerDim` is the pre-computed lower dim index from traceHalvingPath,
/// used as the authoritative target for all transform types.
static FailureOr<std::pair<TransformMapAttr, int64_t>>
halveDimInMap(MLIRContext *ctx, TransformMapAttr mapAttr, int64_t upperDimIdx,
              int64_t targetLowerDim) {
  if (targetLowerDim < 0)
    return failure();
  SmallVector<int64_t> newUpperBounds(mapAttr.getUpperBounds().asArrayRef());
  SmallVector<int64_t> newLowerBounds(mapAttr.getLowerBounds().asArrayRef());
  if (newUpperBounds[upperDimIdx] % 2 != 0)
    return failure();
  newUpperBounds[upperDimIdx] /= 2;

  // Find which TransformAttr owns this upper dimension.
  SmallVector<TransformAttr> newOps;
  int64_t halvedLowerDim = -1;
  for (TransformAttr trAttr : mapAttr.getOps()) {
    auto upperDims = trAttr.getUpperDims();
    bool ownsThisDim = false;
    int subIdx = -1;
    for (auto [i, ud] : llvm::enumerate(upperDims)) {
      if (ud == (uint32_t)upperDimIdx) {
        ownsThisDim = true;
        subIdx = i;
        break;
      }
    }
    if (!ownsThisDim) {
      newOps.push_back(trAttr);
      continue;
    }

    switch (trAttr.getType()) {
    case TransformType::Unmerge: {
      // Unmerge{a, b, ...}["x","y",...]->["z"]: halve params[subIdx].
      SmallVector<int64_t> newParams(trAttr.getParams());
      if (newParams[subIdx] % 2 != 0)
        return failure();
      newParams[subIdx] /= 2;
      // Propagate to the lower dimension: recalculate the product of all
      // params (which equals the lower bound for that dim).
      int64_t lowerDim = targetLowerDim;
      int64_t newProduct = 1;
      for (int64_t p : newParams)
        newProduct *= p;
      newLowerBounds[lowerDim] = newProduct;
      halvedLowerDim = lowerDim;
      newOps.push_back(
          TransformAttr::get(ctx, trAttr.getType(), newParams,
                             trAttr.getUpperNames(), trAttr.getUpperDims(),
                             trAttr.getLowerNames(), trAttr.getLowerDims()));
      break;
    }
    case TransformType::PassThrough: {
      int64_t lowerDim = targetLowerDim;
      if (newLowerBounds[lowerDim] % 2 != 0)
        return failure();
      newLowerBounds[lowerDim] /= 2;
      halvedLowerDim = lowerDim;
      newOps.push_back(trAttr);
      break;
    }
    case TransformType::Merge: {
      // Merge takes multiple lower dims into one upper dim. Use the
      // pre-computed target lower dim from traceHalvingPath to identify
      // which of the Merge's lower dims to halve.
      SmallVector<int64_t> newParams(trAttr.getParams());
      int64_t targetIdx = -1;
      for (auto [i, ld] : llvm::enumerate(trAttr.getLowerDims())) {
        if ((int64_t)ld == targetLowerDim) {
          targetIdx = i;
          break;
        }
      }
      if (targetIdx < 0 || targetIdx >= (int64_t)trAttr.getLowerDims().size() ||
          newParams[targetIdx] % 2 != 0)
        return failure();
      newParams[targetIdx] /= 2;
      int64_t lowerDim = trAttr.getLowerDims()[targetIdx];
      if (newLowerBounds[lowerDim] % 2 != 0)
        return failure();
      newLowerBounds[lowerDim] /= 2;
      halvedLowerDim = lowerDim;
      newOps.push_back(
          TransformAttr::get(ctx, trAttr.getType(), newParams,
                             trAttr.getUpperNames(), trAttr.getUpperDims(),
                             trAttr.getLowerNames(), trAttr.getLowerDims()));
      break;
    }
    case TransformType::ConstDim:
    case TransformType::AddDim: {
      // AddDim/ConstDim don't lead to a kernel argument!
      return failure();
    }
    case TransformType::Broadcast: {
      // Broadcast replicates from a size-1 lower dim. There is no real
      // contiguous data to pack, so halving through Broadcast is invalid.
      return failure();
    }
    case TransformType::Pad: {
      // Pad adds padding around a dimension. Halving the padded (upper) size
      // requires adjusting the unpadded (lower) bound accordingly.
      SmallVector<int64_t> newParams(trAttr.getParams());
      int64_t lowerDim = targetLowerDim;
      int64_t padBefore = newParams[subIdx * 2];
      int64_t padAfter = newParams[subIdx * 2 + 1];

      // We must be able to divide the padding params by two
      if (padBefore % 2 != 0 || padAfter % 2 != 0)
        return failure();

      newParams[subIdx * 2] = padBefore / 2;
      newParams[subIdx * 2 + 1] = padAfter / 2;
      int64_t newUpper = newUpperBounds[upperDimIdx]; // already halved above
      int64_t newLower =
          newUpper - newParams[subIdx * 2] - newParams[subIdx * 2 + 1];
      newLowerBounds[lowerDim] = newLower;
      halvedLowerDim = lowerDim;
      newOps.push_back(
          TransformAttr::get(ctx, trAttr.getType(), newParams,
                             trAttr.getUpperNames(), trAttr.getUpperDims(),
                             trAttr.getLowerNames(), trAttr.getLowerDims()));
      break;
    }
    case TransformType::Slice: {
      // Slice selects [start, end) from a lower dim. Halving the upper size
      // means halving the slice length, keeping the start the same.
      SmallVector<int64_t> newParams(trAttr.getParams());
      int64_t start = newParams[subIdx * 2];
      int64_t end = newParams[subIdx * 2 + 1];

      // TODO: restricted to start=0 for now
      if (start != 0 || end % 2 != 0)
        return failure();

      int64_t newEnd = start + (end - start) / 2;
      newParams[subIdx * 2 + 1] = newEnd;
      int64_t lowerDim = targetLowerDim;
      if (newLowerBounds[lowerDim] % 2 != 0)
        return failure();
      newLowerBounds[lowerDim] /= 2;
      halvedLowerDim = lowerDim;
      newOps.push_back(
          TransformAttr::get(ctx, trAttr.getType(), newParams,
                             trAttr.getUpperNames(), trAttr.getUpperDims(),
                             trAttr.getLowerNames(), trAttr.getLowerDims()));
      break;
    }
    case TransformType::Embed: {
      // Implement if we need f4 convolution
      emitError(UnknownLoc::get(ctx),
                "Embed transform type not supported for dimension halving");
      return failure();
    }
    }
  }

  auto newMap = TransformMapAttr::get(newOps, newUpperBounds, newLowerBounds);
  return std::make_pair(newMap, halvedLowerDim);
}

/// Given a TransformMapAttr and an upper dim index, determine which lower dim
/// that upper dim maps to. For most transform types this is unambiguous.
/// For Merge, `mergeLowerIdx` specifies which of the Merge's lower dims to
/// pick (default -1 means use the last one, i.e. the old assumption).
/// Returns the lower dim index, or -1 on failure.
static int64_t traceDimThroughMap(TransformMapAttr mapAttr, int64_t upperDimIdx,
                                  int64_t mergeLowerIdx = -1) {
  for (TransformAttr trAttr : mapAttr.getOps()) {
    auto upperDims = trAttr.getUpperDims();
    int subIdx = -1;
    for (auto [i, ud] : llvm::enumerate(upperDims)) {
      if (ud == (uint32_t)upperDimIdx) {
        subIdx = i;
        break;
      }
    }
    if (subIdx < 0)
      continue;

    switch (trAttr.getType()) {
    case TransformType::Unmerge:
      return trAttr.getLowerDims()[0];
    case TransformType::PassThrough:
    case TransformType::Pad:
    case TransformType::Slice:
      return trAttr.getLowerDims()[subIdx];
    case TransformType::Merge:
    case TransformType::AddDim:
    case TransformType::ConstDim:
    case TransformType::Broadcast:
    case TransformType::Embed:
      return -1;
    }
  }
  return -1;
}

/// Pre-compute the halving path through a transform chain.
///
/// Walks the chain from the outermost transform (near blockwise_load) down to
/// the block arg. At each level, determines which lower dim the halved upper
/// dim maps to. For Merge transforms (which have multiple lower dims), tries
/// each candidate and picks the one whose path reaches the last dimension of
/// the bottom-most 1D block arg - since that is stride-1 in physical memory.
///
/// Returns a vector of (TransformOp, lowerDim) pairs. The second element is
/// the actual lower dim index in the map's lower bounds for every transform.
/// Recursively search for a path from `curDim` at `chainIdx` down to the
/// stride-1 upper dim (`targetUpperDim`) of the bottom Unmerge.
/// For Merge transforms, tries each lower dim candidate.
/// Appends (TransformOp, lowerDim) pairs to `result`; the second element is
/// always the actual lower dim index in the map's lower bounds.
static bool
findHalvingPath(MutableArrayRef<TransformOp> chain, int64_t targetUpperDim,
                size_t chainIdx, int64_t curDim,
                SmallVector<std::pair<TransformOp, int64_t>> &result) {
  if (chainIdx == chain.size() - 1) {
    if (curDim != targetUpperDim)
      return false;
    TransformMapAttr bottomMap = chain[chainIdx].getTransform();
    int64_t lowerDim = traceDimThroughMap(bottomMap, curDim);
    result.push_back({chain[chainIdx], lowerDim});
    return true;
  }

  TransformOp trOp = chain[chainIdx];
  TransformMapAttr mapAttr = trOp.getTransform();

  for (TransformAttr trAttr : mapAttr.getOps()) {
    int subIdx = -1;
    for (auto [i, ud] : llvm::enumerate(trAttr.getUpperDims())) {
      if (ud == (uint32_t)curDim) {
        subIdx = i;
        break;
      }
    }
    if (subIdx < 0)
      continue;

    if (trAttr.getType() == TransformType::Merge) {
      for (int64_t lowerDim : trAttr.getLowerDims()) {
        SmallVector<std::pair<TransformOp, int64_t>> candidate(result);
        candidate.push_back({trOp, lowerDim});
        if (findHalvingPath(chain, targetUpperDim, chainIdx + 1, lowerDim,
                            candidate)) {
          result = std::move(candidate);
          return true;
        }
      }
      return false;
    }

    int64_t lowerDim = traceDimThroughMap(mapAttr, curDim);
    if (lowerDim < 0)
      return false;
    result.push_back({trOp, lowerDim});
    return findHalvingPath(chain, targetUpperDim, chainIdx + 1, lowerDim,
                           result);
  }
  return false;
}

/// Pre-compute the halving path through a transform chain.
///
/// Walks the chain from the outermost transform (near blockwise_load) down to
/// the block arg. At each level, determines which lower dim the halved upper
/// dim maps to. For Merge transforms (which have multiple lower dims), tries
/// each candidate and picks the one whose path reaches the last upper dim of
/// the bottom Unmerge -- since that is stride-1 in physical memory.
///
/// Returns a vector of (TransformOp, lowerDim) pairs. The second element is
/// the actual lower dim index in the map's lower bounds for every transform.
static FailureOr<SmallVector<std::pair<TransformOp, int64_t>>>
traceHalvingPath(OperandInput input, int64_t topDimIdx) {
  SmallVector<std::pair<TransformOp, int64_t>> path;

  SmallVector<TransformOp> chain;
  Value currValue = input.loadOp.getSource();
  while (auto trOp = currValue.getDefiningOp<TransformOp>()) {
    chain.push_back(trOp);
    currValue = trOp.getInput();
  }

  auto rootType = cast<RankedTensorType>(input.rootArg.getType());
  if (rootType.getRank() != 1)
    return input.rootArg.getOwner()->getParentOp()->emitError(
        "expected 1D block arg for halving path trace");

  if (chain.empty())
    return emitError(input.loadOp.getLoc(), "empty transform chain");

  // The bottom transform (closest to block arg) is always an Unmerge from 1D
  // to ND. Its LAST upper dim is stride-1 (row-major order). Since all upper
  // dims map to the same lower dim 0, we check that the path arrives at the
  // last upper dim of this bottom Unmerge.
  int64_t targetUpperDim = -1;
  TransformMapAttr bottomMap = chain.back().getTransform();
  for (TransformAttr trAttr : bottomMap.getOps()) {
    if (trAttr.getType() == TransformType::Unmerge) {
      targetUpperDim = trAttr.getUpperDims().back();
      break;
    }
  }
  if (targetUpperDim < 0)
    return emitError(input.loadOp.getLoc(),
                     "bottom transform has no Unmerge to 1D");

  if (!findHalvingPath(chain, targetUpperDim, 0, topDimIdx, path))
    return emitError(input.loadOp.getLoc(),
                     "could not trace halving path to stride-1 dimension");
  return path;
}

/// Rewrite all transforms in a chain, propagating the /2 from the outermost
/// (closest to blockwise_load) dimension down to the raw block arg.
/// Also changes element type from i4 to i8.
///
/// Uses a pre-computed halving path (from traceHalvingPath) to correctly
/// handle Merge transforms where the stride-1 lower dim may not be the last.
static LogicalResult rewriteTransformChain(MLIRContext *ctx, OperandInput input,
                                           int64_t topDimIdx) {
  // Pre-compute the halving path to determine which Merge lower dim to use.
  auto maybePath = traceHalvingPath(input, topDimIdx);
  if (failed(maybePath))
    return failure();
  auto &path = maybePath.value();

  Type i8Ty = IntegerType::get(ctx, 8);
  int64_t curDimIdx = topDimIdx;

  Value currValue = input.loadOp.getSource();
  size_t pathIdx = 0;
  while (auto trOp = currValue.getDefiningOp<TransformOp>()) {
    TransformMapAttr oldMap = trOp.getTransform();

    if (pathIdx >= path.size())
      return trOp.emitError("halving path too short for transform chain");
    assert(path[pathIdx].first == trOp && "path/chain mismatch");
    int64_t pathLowerDim = path[pathIdx].second;

    auto result = halveDimInMap(ctx, oldMap, curDimIdx, pathLowerDim);
    if (failed(result))
      return trOp.emitError("failed to halve dimension ") << curDimIdx;
    auto [newMap, nextDimIdx] = *result;
    trOp.setTransformAttr(newMap);

    auto oldResultType = cast<RankedTensorType>(trOp.getResult().getType());
    trOp.getResult().setType(
        RankedTensorType::get(newMap.getUpperBounds().asArrayRef(), i8Ty,
                              oldResultType.getEncoding()));

    curDimIdx = nextDimIdx;
    currValue = trOp.getInput();
    ++pathIdx;
  }

  // Update the block argument type.
  auto oldArgType = cast<RankedTensorType>(input.rootArg.getType());
  SmallVector<int64_t> newArgShape(oldArgType.getShape());
  if (newArgShape[curDimIdx] % 2 != 0)
    return failure();

  newArgShape[curDimIdx] /= 2;
  input.rootArg.setType(
      RankedTensorType::get(newArgShape, i8Ty, oldArgType.getEncoding()));
  return success();
}

/// Insert a three-step broadcast transform composition between a halved 1-D
/// block arg and the existing transform chain:
///
///   [flat/2]  -- AddDim -->  [flat/2, 1]
///             -- Broadcast -->  [flat/2, 2]
///             -- Merge -->  [flat]
///
/// When we halve the block arg from tensor<N x i4> to tensor<N/2 x i8>, two
/// i4 values are packed into each i8 byte.  The rest of the transform chain
/// still expects a logical tensor of size N (one element per original i4
/// value).  This broadcast restores the original flat size by mapping every
/// pair of consecutive logical indices (2j, 2j+1) to the same physical byte
/// at index j.  A subsequent shift constant then selects the correct 4-bit
/// half (lower or upper bits) from each byte.
static Value insertSubByteBroadcast(OpBuilder &builder, MLIRContext *ctx,
                                    Location loc, Value halvedArg,
                                    int64_t halfSize, int64_t fullSize) {
  auto addDimMap = TransformMapAttr::get(
      {TransformAttr::get(ctx, TransformType::PassThrough, {}, {"raw"},
                          {uint32_t(0)}, {"raw"}, {uint32_t(0)}),
       TransformAttr::get(ctx, TransformType::AddDim, {1}, {"dup"},
                          {uint32_t(1)}, {}, {})},
      {halfSize, 1}, {halfSize});
  Value step1 = TransformOp::create(builder, loc, halvedArg, addDimMap);

  auto broadcastMap = TransformMapAttr::get(
      {TransformAttr::get(ctx, TransformType::PassThrough, {}, {"raw"},
                          {uint32_t(0)}, {"raw"}, {uint32_t(0)}),
       TransformAttr::get(ctx, TransformType::Broadcast, {1}, {"dup"},
                          {uint32_t(1)}, {"dup"}, {uint32_t(1)})},
      {halfSize, 2}, {halfSize, 1});
  Value step2 = TransformOp::create(builder, loc, step1, broadcastMap);

  auto mergeMap = TransformMapAttr::get(
      {TransformAttr::get(ctx, TransformType::Merge, {halfSize, 2}, {"flat"},
                          {uint32_t(0)}, {"raw", "dup"},
                          {uint32_t(0), uint32_t(1)})},
      {fullSize}, {halfSize, 2});
  return TransformOp::create(builder, loc, step2, mergeMap);
}

/// Figure out which tile axis (K or D) to use for sub-byte extraction.
/// The block arg is a flat 1-D tensor whose last element is "stride-1" (i.e.
/// adjacent elements in the flat buffer are adjacent in memory).  The transform
/// chain reshapes that flat buffer into a 6-D tile.  We need to know which
/// tile axis (K or D) ends up being contiguous in the flat buffer, because
/// that is the axis along which two consecutive i4 values share the same i8
/// byte, and therefore the axis along which we alternate the shift pattern.
static FailureOr<int64_t> findSubBytePackingDim(
    MutableArrayRef<TransformOp> chain, int64_t kDimIdx, int64_t dDimIdx,
    SmallVector<std::pair<TransformOp, int64_t>> &pathOut, Operation *emitLoc) {
  int64_t targetUpperDim = -1;
  TransformMapAttr bottomMap = chain.back().getTransform();
  for (TransformAttr trAttr : bottomMap.getOps()) {
    if (trAttr.getType() == TransformType::Unmerge) {
      targetUpperDim = trAttr.getUpperDims().back();
      break;
    }
  }
  if (targetUpperDim < 0)
    return emitLoc->emitError(
        "bottom transform has no Unmerge for stride-1 trace");

  if (findHalvingPath(chain, targetUpperDim, 0, kDimIdx, pathOut))
    return kDimIdx;
  pathOut.clear();
  if (findHalvingPath(chain, targetUpperDim, 0, dDimIdx, pathOut))
    return dDimIdx;
  return emitLoc->emitError(
      "cannot determine stride-1 source dim for sub-byte arg");
}

/// Check whether a transform attribute involves `lowerDim` in its lower dims.
static bool transformTouchesLowerDim(TransformAttr trAttr, int64_t lowerDim) {
  for (auto ld : trAttr.getLowerDims())
    if ((int64_t)ld == lowerDim)
      return true;
  return false;
}

/// Compute the effective stride factor from a halving path.
///
/// The sub-byte shift pattern has a base period of 2 (alternating every
/// element).  However, when the halving path passes through a Merge, multiple
/// consecutive tile positions map to the same flat block-arg index, which
/// stretches the period.  The stride factor is the product of the "tail sizes"
/// at each Merge along the path.
///
/// Example: Merge{2, 2} following lower dim 0 has tail size = product of
/// param sizes after index 0 = 2, so the sub-byte period doubles from 2 to 4,
/// giving the pattern [0, 0, 4, 4, 0, 0, 4, 4, ...].
static FailureOr<int64_t> computeSubByteStrideFactor(
    MutableArrayRef<std::pair<TransformOp, int64_t>> path) {
  int64_t strideFactor = 1;
  for (auto &[trOp, lowerDim] : path) {
    for (TransformAttr trAttr : trOp.getTransform().getOps()) {
      if (!transformTouchesLowerDim(trAttr, lowerDim))
        continue;

      switch (trAttr.getType()) {
      case TransformType::PassThrough:
      case TransformType::Unmerge:
      case TransformType::Broadcast:
      case TransformType::AddDim:
      case TransformType::ConstDim:
        break;
      case TransformType::Merge: {
        auto lowerDims = trAttr.getLowerDims();
        for (size_t i = 0; i < lowerDims.size(); ++i) {
          if ((int64_t)lowerDims[i] == lowerDim) {
            auto params = trAttr.getParams();
            int64_t tailSize = 1;
            for (size_t j = i + 1; j < params.size(); ++j)
              tailSize *= params[j];
            strideFactor *= tailSize;
            break;
          }
        }
        break;
      }
      case TransformType::Pad:
      case TransformType::Slice:
        // Pad/Slice only add/remove elements at the boundary without
        // changing the stride relationship within the real data region.
        break;
      case TransformType::Embed:
        return trOp.emitError("unsupported transform type (Embed) on traced "
                              "dimension in sub-byte stride factor "
                              "computation");
      }
    }
  }
  return strideFactor;
}

/// Build the 2-D shift constant for sub-byte extraction.
///
/// For a tile of shape [D, K], the shift at position (d, k) is determined by
/// which sub-byte (lower or upper bits) that tile element corresponds to.
/// The base formula is:  shift = ((pos / strideFactor) % 2) * 4
/// where `pos` is the position along `loadAxisIdx` (0 for D, 1 for K).
///
/// The constant is tagged with a "rock.sub_byte_shift" dictionary attribute
/// containing {axis, halfPeriod, v0, v1} so that RockToTTIR can lower it to
/// Triton-native ops without pattern-matching the dense blob.
///
/// This static path is only valid when `strideFactor < axisLen`, i.e. the
/// nibble alternation can be expressed entirely within a single tile.  When
/// `strideFactor >= axisLen` the in-tile pattern is degenerate (all zeros)
/// and the alternation must be driven by the outer K-loop induction variable;
/// that case is handled separately by `buildSubByteShiftDynamic`.
static Value buildSubByteShiftConstant(OpBuilder &builder, Location loc,
                                       ArrayRef<int64_t> tileShape,
                                       RankedTensorType i8TileType,
                                       int64_t loadAxisIdx,
                                       int64_t strideFactor) {
  SmallVector<Attribute> shiftValues;
  shiftValues.reserve(tileShape[0] * tileShape[1]);
  for (int64_t d = 0; d < tileShape[0]; ++d)
    for (int64_t k = 0; k < tileShape[1]; ++k) {
      int64_t pos = (loadAxisIdx == 0) ? d : k;
      int64_t subByteIdx = (pos / strideFactor) % 2;
      shiftValues.push_back(builder.getI8IntegerAttr(subByteIdx * 4));
    }
  auto constOp = arith::ConstantOp::create(
      builder, loc, DenseElementsAttr::get(i8TileType, shiftValues));

  SmallVector<NamedAttribute> attrs = {
      builder.getNamedAttr("axis", builder.getI64IntegerAttr(loadAxisIdx)),
      builder.getNamedAttr("halfPeriod",
                           builder.getI64IntegerAttr(strideFactor)),
      builder.getNamedAttr("v0", builder.getI8IntegerAttr(0)),
      builder.getNamedAttr("v1", builder.getI8IntegerAttr(4)),
  };
  constOp->setAttr("rock.sub_byte_shift", builder.getDictionaryAttr(attrs));
  return constOp;
}

/// Build a loop-variant tile of nibble-shift values for the case where
/// `strideFactor` is at least the tile axis length.  In that regime every
/// element in a single tile shares the *same* shift (low or high nibble),
/// determined purely by the enclosing K-loop iv:
///
///   shift = ((iv * tileExtent) / strideFactor) % 2 * 4
///
/// The result is `tt.splat`-broadcast to a `tensor<DxKxi8>`.  Because the
/// result has a real SSA dependency on the iv it cannot be hoisted out of the
/// scf.for by LICM/canonicalization, which is why we emit the dynamic ops
/// here instead of relying on the marker-and-late-lower mechanism used for
/// the static case.
static FailureOr<Value> buildSubByteShiftDynamic(OpBuilder &builder,
                                                 Location loc,
                                                 RankedTensorType i8TileType,
                                                 int64_t strideFactor,
                                                 int64_t tileExtent,
                                                 Operation *anchor) {
  auto forOp = anchor->getParentOfType<scf::ForOp>();
  if (!forOp)
    return emitError(loc, "sub-byte load with broadcast period >= tile extent "
                          "must be inside an scf.for whose iv is the K-loop "
                          "counter");

  Type i32 = builder.getI32Type();
  Value iv = forOp.getInductionVar();
  if (iv.getType() != i32) {
    auto ivIntTy = dyn_cast<IntegerType>(iv.getType());
    if (!ivIntTy)
      return emitError(loc, "sub-byte K-loop iv must be an integer; got ")
             << iv.getType();
    if (ivIntTy.getWidth() > 32)
      iv = arith::TruncIOp::create(builder, loc, i32, iv);
    else
      iv = arith::ExtSIOp::create(builder, loc, i32, iv);
  }

  Value tileExtentVal = arith::ConstantOp::create(
      builder, loc, builder.getI32IntegerAttr(tileExtent));
  Value origin = arith::MulIOp::create(builder, loc, iv, tileExtentVal);
  Value halfPeriodVal = arith::ConstantOp::create(
      builder, loc, builder.getI32IntegerAttr(strideFactor));
  Value div = arith::DivUIOp::create(builder, loc, origin, halfPeriodVal);
  Value two = arith::ConstantOp::create(builder, loc,
                                        builder.getI32IntegerAttr(2));
  Value mod = arith::RemUIOp::create(builder, loc, div, two);
  Value four = arith::ConstantOp::create(builder, loc,
                                         builder.getI32IntegerAttr(4));
  Value shiftI32 = arith::MulIOp::create(builder, loc, mod, four);
  Value shiftI8 =
      arith::TruncIOp::create(builder, loc, builder.getI8Type(), shiftI32);
  Value shifts =
      triton::SplatOp::create(builder, loc, i8TileType, shiftI8);
  return shifts;
}

/// Replace arith.extui / arith.extsi users of a loaded i8 tile with sub-byte
/// extraction logic:
///
///   extui  ->  (loaded >> shifts) & 0x0F
///   extsi  ->  ((loaded >> shifts) & 0x0F) << 4  >> s 4   (sign-extend)
static void replaceExtUsersWithSubByteExtract(OpBuilder &builder, Location loc,
                                              Value loadResult,
                                              RankedTensorType i8TileType,
                                              Value shifts, Value mask) {
  for (auto *user :
       llvm::make_early_inc_range(loadResult.getUsers())) {
    if (isa<arith::ExtUIOp>(user)) {
      builder.setInsertionPoint(user);
      Value shifted =
          arith::ShRUIOp::create(builder, loc, loadResult, shifts);
      Value subByte = arith::AndIOp::create(builder, loc, shifted, mask);
      user->getResult(0).replaceAllUsesWith(subByte);
      user->erase();
    } else if (isa<arith::ExtSIOp>(user)) {
      builder.setInsertionPoint(user);
      Value shifted =
          arith::ShRUIOp::create(builder, loc, loadResult, shifts);
      Value subByte = arith::AndIOp::create(builder, loc, shifted, mask);
      Value four = arith::ConstantOp::create(
          builder, loc,
          DenseElementsAttr::get(i8TileType, builder.getI8IntegerAttr(4)));
      Value shl = arith::ShLIOp::create(builder, loc, subByte, four);
      Value sext = arith::ShRSIOp::create(builder, loc, shl, four);
      user->getResult(0).replaceAllUsesWith(sext);
      user->erase();
    }
  }
}

/// Handle a single OperandInput whose block arg is i4 but whose GEMM operand
/// is wider (e.g. f16, behind a dequant fusion chain).
///
/// 1. Halves the block arg from tensor<N x i4> to tensor<N/2 x i8>.
/// 2. Inserts broadcast transforms so each byte is read at two positions.
/// 3. Determines the correct tile axis and stride for sub-byte extraction.
/// 4. Replaces arith.extui/extsi users with shift-and-mask logic.
static LogicalResult processSubByteInput(OpBuilder &builder, MLIRContext *ctx,
                                         OperandInput &input, bool isA,
                                         int64_t sourceRank, Type i8Ty) {
  int64_t kDimIdx = isA ? (sourceRank - 1) : (sourceRank - 2);
  int64_t dDimIdx = isA ? (sourceRank - 2) : (sourceRank - 1);

  // Build the transform chain from load source down to block arg.
  Value cur = input.loadOp.getSource();
  TransformOp bottomTransform;
  SmallVector<TransformOp> chain;
  while (auto trOp = cur.getDefiningOp<TransformOp>()) {
    chain.push_back(trOp);
    bottomTransform = trOp;
    cur = trOp.getInput();
  }
  if (!bottomTransform)
    return input.loadOp.emitError("no transform chain for sub-byte block arg");

  // Find which tile dim (K or D) maps to stride-1 in the block arg.
  SmallVector<std::pair<TransformOp, int64_t>> path;
  auto halveDimOrErr =
      findSubBytePackingDim(chain, kDimIdx, dDimIdx, path, bottomTransform);
  if (failed(halveDimOrErr))
    return failure();
  int64_t halveDimIdx = *halveDimOrErr;

  // Halve the block arg: tensor<N x i4> -> tensor<N/2 x i8>.
  auto oldArgType = cast<RankedTensorType>(input.rootArg.getType());
  int64_t flatSize = oldArgType.getShape()[0];
  if (flatSize % 2 != 0)
    return input.loadOp.emitError("sub-byte block arg flat size must be even");
  input.rootArg.setType(
      RankedTensorType::get({flatSize / 2}, i8Ty, oldArgType.getEncoding()));

  // Insert broadcast transforms between the halved arg and the existing chain.
  builder.setInsertionPoint(bottomTransform);
  Location loc = input.loadOp.getLoc();
  Value broadcastOut = insertSubByteBroadcast(builder, ctx, loc,
                                              input.rootArg,
                                              flatSize / 2, flatSize);
  bottomTransform->setOperand(0, broadcastOut);

  // Update the load result type from i4 to i8 (shape stays the same).
  auto loadResultType =
      cast<RankedTensorType>(input.loadOp.getResult().getType());
  auto tileShape = loadResultType.getShape();
  auto i8TileType = RankedTensorType::get(tileShape, i8Ty);
  input.loadOp.getResult().setType(i8TileType);

  // Build the shift and mask constants, then replace ext users.
  int64_t loadAxisIdx = halveDimIdx - (sourceRank - 2);
  auto strideFactorOrErr = computeSubByteStrideFactor(path);
  if (failed(strideFactorOrErr))
    return failure();
  int64_t strideFactor = *strideFactorOrErr;

  // When strideFactor exceeds the tile axis extent, the in-tile shift pattern
  // is degenerate (always 0) and the nibble alternation must be driven by the
  // outer K-loop iv.  Emit a loop-variant `tt.splat` of an arith-computed
  // scalar shift inside the loop body so that LICM cannot hoist it out (which
  // would silently revert to always-low-nibble behavior).
  Value shifts;
  int64_t axisLen = tileShape[loadAxisIdx];
  if (strideFactor >= axisLen) {
    builder.setInsertionPoint(input.loadOp);
    auto dynamicShiftsOrErr = buildSubByteShiftDynamic(
        builder, loc, i8TileType, strideFactor, axisLen, input.loadOp);
    if (failed(dynamicShiftsOrErr))
      return failure();
    shifts = *dynamicShiftsOrErr;
  } else {
    shifts = buildSubByteShiftConstant(builder, loc, tileShape, i8TileType,
                                       loadAxisIdx, strideFactor);
  }
  Value mask = arith::ConstantOp::create(
      builder, loc,
      DenseElementsAttr::get(i8TileType, builder.getI8IntegerAttr(15)));
  replaceExtUsersWithSubByteExtract(builder, loc, input.loadOp.getResult(),
                                    i8TileType, shifts, mask);
  return success();
}

/// Pack 4-bit (i4) kernel arguments to i8 with dimension halving.
///
/// For each BlockwiseGemmOp, traces matrixA/matrixB backwards through
/// blockwise_loads and transform chains, then halves the stride-1 dimension
/// (K preferred, then D) and converts i4 -> i8.
///
/// Two paths depending on whether the GEMM operand itself is 4-bit:
///
///   Direct 4-bit GEMM:  Halves dimensions through the entire transform chain
///     via rewriteTransformChain.  Sets kPack attributes.
///
///   Sub-byte (i4 behind a dequant chain):  Handled by processSubByteInput —
///     halves only the block arg, inserts broadcast transforms, and replaces
///     arith.extui/extsi with sub-byte shift-and-mask extraction.
static LogicalResult pack4BitKernelArgs(func::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  Type i8Ty = IntegerType::get(ctx, 8);
  DenseSet<BlockArgument> processedArgs;

  WalkResult walkResult = funcOp.walk([&](BlockwiseGemmOp gemmOp) {
    auto processOperand = [&](Value operand, bool isA) -> LogicalResult {
      auto tensorType = cast<RankedTensorType>(operand.getType());
      bool gemmOperandIs4Bit = is4Bit(tensorType.getElementType());

      auto maybeInputs = collectOperandInputs(operand);
      if (failed(maybeInputs))
        return gemmOperandIs4Bit ? failure() : success();
      SmallVector<OperandInput> &inputs = maybeInputs.value();
      if (inputs.empty()) {
        if (gemmOperandIs4Bit)
          return gemmOp.emitError("no inputs found for 4-bit operand");
        return success();
      }

      std::optional<bool> kPack = std::nullopt;
      for (OperandInput &input : inputs) {
        bool localKPack = true;
        if (processedArgs.contains(input.rootArg))
          return gemmOp.emitError(
              "duplicate block arg across transform chains");
        if (!input.loadOp)
          return gemmOp.emitError("transform chain has no blockwise_load");

        Value source = input.loadOp.getSource();
        int64_t sourceRank = cast<ShapedType>(source.getType()).getRank();
        if (sourceRank != 6)
          return gemmOp.emitError("source rank must be six");
        int64_t kDimIdx = isA ? (sourceRank - 1) : (sourceRank - 2);
        int64_t dDimIdx = isA ? (sourceRank - 2) : (sourceRank - 1);

        if (gemmOperandIs4Bit) {
          // Direct 4-bit GEMM: halve via vectorization + rewriteTransformChain.
          std::optional<int64_t> halveDimIdx = std::nullopt;
          VectorizationResult kVectorRes =
              getMaxVectorization(source, kDimIdx);
          if (kVectorRes.max > 1)
            halveDimIdx = kDimIdx;

          if (!halveDimIdx.has_value()) {
            localKPack = false;
            VectorizationResult dVectorRes =
                getMaxVectorization(source, dDimIdx);
            if (dVectorRes.max > 1)
              halveDimIdx = dDimIdx;
          }
          if (!halveDimIdx.has_value())
            return gemmOp.emitError(
                "max vectorization of both D and K is 1");

          if (kPack.has_value() && kPack.value() != localKPack)
            return gemmOp.emitError(
                "all inputs must agree whether to either pack along K or not");
          if (!kPack.has_value())
            kPack = localKPack;

          if (failed(rewriteTransformChain(ctx, input, halveDimIdx.value())))
            return failure();

          auto newSrcType =
              cast<RankedTensorType>(input.loadOp.getSource().getType());
          int64_t srcRank = newSrcType.getRank();
          SmallVector<int64_t> newResShape;
          newResShape.push_back(newSrcType.getShape()[srcRank - 2]);
          newResShape.push_back(newSrcType.getShape()[srcRank - 1]);
          input.loadOp.getResult().setType(
              RankedTensorType::get(newResShape, i8Ty));
        } else {
          // Sub-byte path: i4 behind a dequant fusion chain.
          if (failed(processSubByteInput(builder, ctx, input, isA, sourceRank,
                                         i8Ty)))
            return failure();
        }
        processedArgs.insert(input.rootArg);
      }

      if (gemmOperandIs4Bit) {
        if (!kPack.has_value())
          return gemmOp.emitError("kPack is undefined");
        if (isA)
          gemmOp.setMatrixAKPackAttr(builder.getBoolAttr(kPack.value()));
        else
          gemmOp.setMatrixBKPackAttr(builder.getBoolAttr(kPack.value()));
      }
      return success();
    };

    if (failed(processOperand(gemmOp.getMatrixA(), /*isA=*/true)) ||
        failed(processOperand(gemmOp.getMatrixB(), /*isA=*/false)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted())
    return failure();

  // Flip any remaining 4-bit tensor results to i8 (element type only).
  // Fusion ops on 4-bit data are handled later by fixup4BitFusionOps.
  funcOp.walk([&](Operation *op) {
    if (rock::isFusionOp(op))
      return;
    for (OpResult result : op->getResults()) {
      auto tType = dyn_cast<RankedTensorType>(result.getType());
      if (!tType || !is4Bit(tType.getElementType()))
        continue;
      result.setType(
          RankedTensorType::get(tType.getShape(), i8Ty, tType.getEncoding()));
    }
  });

  // rewriteTransformChain mutates block arg types (i4 -> i8 with halved
  // shapes), but the FunctionType attribute doesn't auto-update, so sync it.
  SmallVector<Type> newArgTypes;
  for (auto arg : funcOp.getArguments())
    newArgTypes.push_back(arg.getType());
  funcOp.setFunctionType(
      FunctionType::get(ctx, newArgTypes, funcOp.getResultTypes()));
  return success();
}

//===----------------------------------------------------------------------===//
// Fusion op fix-up: wrap broken fusion ops with arith.bitcast
//===----------------------------------------------------------------------===//

/// Original tensor types for each operand/result of a fusion op, captured
/// before the type conversion walk rewrites them to integer.
struct FusionOpInfo {
  SmallVector<Type> origOperandTypes;
  SmallVector<Type> origResultTypes;
};

/// Scan all fusion ops and record the original tensor types for any
/// operand/result whose element type is a non-TT float.
static DenseMap<Operation *, FusionOpInfo>
recordFusionOpTypes(func::FuncOp funcOp) {
  DenseMap<Operation *, FusionOpInfo> fusionInfoMap;
  funcOp.walk([&](Operation *op) {
    if (!isFusionOp(op))
      return;
    FusionOpInfo info;
    bool hasNonTT = false;
    for (Value operand : op->getOperands()) {
      info.origOperandTypes.push_back(operand.getType());
      if (auto tType = dyn_cast<RankedTensorType>(operand.getType()))
        if (isNonTTFloat(tType.getElementType()))
          hasNonTT = true;
    }
    for (OpResult result : op->getResults()) {
      info.origResultTypes.push_back(result.getType());
      if (auto tType = dyn_cast<RankedTensorType>(result.getType()))
        if (isNonTTFloat(tType.getElementType()))
          hasNonTT = true;
    }
    if (hasNonTT)
      fusionInfoMap[op] = std::move(info);
  });
  return fusionInfoMap;
}

/// After the blanket type conversion (Step 1), fusion ops that had 8-bit
/// non-TT float operands/results now illegally carry integer types.
/// Fix them by restoring the original float types and inserting
/// arith.bitcast ops at the boundaries (i8 -> f8 on inputs, f8 -> i8 on
/// outputs).
///
/// Note: arith.truncf/extf involving f8E8M0FNU and f4E2M1FN are expanded to
/// integer arithmetic by arith-expand AFTER this pass runs (in the kernel
/// pipeline, between SerializeHostFuncs and RockToTTIR).
static LogicalResult
fixup8BitFusionOps(MLIRContext *ctx,
                   const DenseMap<Operation *, FusionOpInfo> &fusionInfoMap) {
  OpBuilder builder(ctx);

  for (auto &[op, info] : fusionInfoMap) {
    bool has8bit = false;
    for (Type t : info.origOperandTypes)
      if (isNonTTFloat(t, 8))
        has8bit = true;
    for (Type t : info.origResultTypes)
      if (isNonTTFloat(t, 8))
        has8bit = true;
    if (!has8bit)
      continue;

    Location loc = op->getLoc();

    // Fix operands: i8 -> original float
    builder.setInsertionPoint(op);
    for (unsigned i = 0; i < info.origOperandTypes.size(); ++i) {
      auto origTensorType =
          dyn_cast<RankedTensorType>(info.origOperandTypes[i]);
      if (!origTensorType || !isNonTTFloat(origTensorType, 8))
        continue;

      Value operand = op->getOperand(i);
      Value bc = arith::BitcastOp::create(builder, loc, origTensorType,
                                          operand);
      op->setOperand(i, bc);
    }

    // Fix results: restore float type, bitcast back to i8
    for (unsigned i = 0; i < info.origResultTypes.size(); ++i) {
      auto origTensorType =
          dyn_cast<RankedTensorType>(info.origResultTypes[i]);
      if (!origTensorType || !isNonTTFloat(origTensorType, 8))
        continue;

      OpResult result = op->getResult(i);
      Type intType = result.getType();

      result.setType(origTensorType);

      builder.setInsertionPointAfter(op);
      Value bc = arith::BitcastOp::create(builder, loc, intType, result);
      result.replaceAllUsesExcept(bc, bc.getDefiningOp());
    }
  }
  return success();
}

/// After 4-bit packing (pack4BitKernelArgs), fusion ops that originally
/// operated on f4 data have a shape mismatch: their operands are packed i8
/// (halved shape) but their result types still carry the unpacked shape.
/// Fix them by unpacking operands to f4 (nibble extraction), restoring the
/// fusion op to operate on f4, then repacking the result to i8.
static LogicalResult
fixup4BitFusionOps(func::FuncOp funcOp, MLIRContext *ctx,
                   const DenseMap<Operation *, FusionOpInfo> &fusionInfoMap) {
  OpBuilder builder(ctx);
  Type i4Ty = IntegerType::get(ctx, 4);
  Type i8Ty = IntegerType::get(ctx, 8);

  for (auto &[op, info] : fusionInfoMap) {
    bool has4bit = false;
    for (Type t : info.origOperandTypes)
      if (isNonTTFloat(t, 4))
        has4bit = true;
    for (Type t : info.origResultTypes)
      if (isNonTTFloat(t, 4))
        has4bit = true;
    if (!has4bit)
      continue;

    Location loc = op->getLoc();

    // After packing, the operands coming from blockwise_load have the packed
    // shape (halved, i8). We unpack each into low/high f4 nibbles.
    SmallVector<Value> lowOperands, highOperands;
    builder.setInsertionPoint(op);

    for (unsigned i = 0; i < info.origOperandTypes.size(); ++i) {
      auto origTensorType =
          dyn_cast<RankedTensorType>(info.origOperandTypes[i]);
      Value operand = op->getOperand(i);

      if (!origTensorType || !isNonTTFloat(origTensorType, 4)) {
        lowOperands.push_back(operand);
        highOperands.push_back(operand);
        continue;
      }

      // operand.getType() is the packed type (e.g. tensor<64x32xi8>)
      auto packedType = cast<RankedTensorType>(operand.getType());
      auto f4ElemType = origTensorType.getElementType();
      auto i4TensorType = RankedTensorType::get(packedType.getShape(), i4Ty,
                                                 packedType.getEncoding());
      auto f4TensorType = RankedTensorType::get(
          packedType.getShape(), f4ElemType, packedType.getEncoding());

      auto cst4 = arith::ConstantOp::create(
          builder, loc,
          DenseElementsAttr::get(packedType, builder.getI8IntegerAttr(4)));
      auto cst0F = arith::ConstantOp::create(
          builder, loc,
          DenseElementsAttr::get(packedType, builder.getI8IntegerAttr(15)));

      Value lowI8 = arith::AndIOp::create(builder, loc, operand, cst0F);
      Value lowI4 = arith::TruncIOp::create(builder, loc, i4TensorType, lowI8);
      Value lowF4 =
          arith::BitcastOp::create(builder, loc, f4TensorType, lowI4);

      Value highI8 = arith::ShRUIOp::create(builder, loc, operand, cst4);
      Value highI4 =
          arith::TruncIOp::create(builder, loc, i4TensorType, highI8);
      Value highF4 =
          arith::BitcastOp::create(builder, loc, f4TensorType, highI4);

      lowOperands.push_back(lowF4);
      highOperands.push_back(highF4);
    }

    // Build clone result types: for f4 results, use the packed operand shape
    // (from the already-packed blockwise_load output) with the original f4
    // element type. We can't use op->getResult().getType() because packing
    // only updates transforms/loads, not the fusion op's result types.
    RankedTensorType packedRefType;
    for (unsigned i = 0; i < info.origOperandTypes.size(); ++i) {
      if (isNonTTFloat(info.origOperandTypes[i], 4)) {
        packedRefType = cast<RankedTensorType>(op->getOperand(i).getType());
        break;
      }
    }

    SmallVector<Type> cloneResultTypes;
    for (unsigned i = 0; i < info.origResultTypes.size(); ++i) {
      auto origTensorType =
          dyn_cast<RankedTensorType>(info.origResultTypes[i]);
      if (origTensorType && isNonTTFloat(origTensorType, 4)) {
        cloneResultTypes.push_back(RankedTensorType::get(
            packedRefType.getShape(), origTensorType.getElementType(),
            packedRefType.getEncoding()));
      } else {
        cloneResultTypes.push_back(op->getResult(i).getType());
      }
    }

    // Clone the op for low nibble.
    builder.setInsertionPoint(op);
    Operation *lowClone = builder.clone(*op);
    for (unsigned i = 0; i < lowOperands.size(); ++i)
      lowClone->setOperand(i, lowOperands[i]);
    for (unsigned i = 0; i < cloneResultTypes.size(); ++i)
      lowClone->getResult(i).setType(cloneResultTypes[i]);

    // Clone the op for high nibble.
    Operation *highClone = builder.clone(*op);
    for (unsigned i = 0; i < highOperands.size(); ++i)
      highClone->setOperand(i, highOperands[i]);
    for (unsigned i = 0; i < cloneResultTypes.size(); ++i)
      highClone->getResult(i).setType(cloneResultTypes[i]);

    // Repack each result: f4 -> i4 -> i8 (low | high<<4).
    builder.setInsertionPointAfter(highClone);
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      auto origTensorType =
          dyn_cast<RankedTensorType>(info.origResultTypes[i]);
      if (!origTensorType || !isNonTTFloat(origTensorType, 4)) {
        op->getResult(i).replaceAllUsesWith(lowClone->getResult(i));
        continue;
      }

      auto i4PackedType = RankedTensorType::get(packedRefType.getShape(), i4Ty,
                                                packedRefType.getEncoding());
      auto i8PackedType = RankedTensorType::get(packedRefType.getShape(), i8Ty,
                                                packedRefType.getEncoding());

      Value resLowI4 = arith::BitcastOp::create(builder, loc, i4PackedType,
                                                  lowClone->getResult(i));
      Value resHighI4 = arith::BitcastOp::create(builder, loc, i4PackedType,
                                                   highClone->getResult(i));
      Value resLowI8 =
          arith::ExtUIOp::create(builder, loc, i8PackedType, resLowI4);
      Value resHighI8 =
          arith::ExtUIOp::create(builder, loc, i8PackedType, resHighI4);

      auto cst4 = arith::ConstantOp::create(
          builder, loc,
          DenseElementsAttr::get(i8PackedType, builder.getI8IntegerAttr(4)));
      Value shifted = arith::ShLIOp::create(builder, loc, resHighI8, cst4);
      Value packed = arith::OrIOp::create(builder, loc, resLowI8, shifted);

      op->getResult(i).replaceAllUsesWith(packed);
    }

    op->erase();
  }
  return success();
}

/// Legalize all non-TT_Float types to integer types of the same bit width
/// throughout the kernel function (no shape changes).
static LogicalResult convertKernel(func::FuncOp funcOp, MLIRContext *ctx) {
  // Save original types on BlockwiseGemmOp BEFORE converting.
  recordOrigTypesOnGemm(funcOp);

  // Record original types on fusion ops BEFORE converting.
  auto fusionInfoMap = recordFusionOpTypes(funcOp);

  // Step 1: Simple type swap (f8->i8, f4->i4) with no shape changes.
  SmallVector<Type> newArgTypes;
  for (auto arg : funcOp.getArguments()) {
    Type newType = convertType(arg.getType(), ctx);
    if (newType != arg.getType())
      arg.setType(newType);
    newArgTypes.push_back(arg.getType());
  }

  funcOp.walk([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      Type newType = convertType(result.getType(), ctx);
      if (newType != result.getType())
        result.setType(newType);
    }
  });

  SmallVector<Type> newResultTypes;
  for (Type t : funcOp.getResultTypes())
    newResultTypes.push_back(convertType(t, ctx));
  funcOp.setFunctionType(FunctionType::get(ctx, newArgTypes, newResultTypes));

  // Step 1.5: Fix up 8-bit fusion ops that were broken by the type swap.
  if (failed(fixup8BitFusionOps(ctx, fusionInfoMap)))
    return failure();

  // Step 2: Pack 4-bit types (i4) to i8 with dimension halving.
  if (failed(pack4BitKernelArgs(funcOp, ctx)))
    return failure();

  // Step 3: Fix up 4-bit fusion ops that were broken by the packing.
  return fixup4BitFusionOps(funcOp, ctx, fusionInfoMap);
}

//===----------------------------------------------------------------------===//
// GPU wrapper conversion
//===----------------------------------------------------------------------===//

/// Legalize a memref type with a non-TT_Float or integer i4 element type to
/// its i8 equivalent: f8E8M0FNU -> i8 (same shape), f4E2M1FN / i4 -> i8
/// (halved last dim).
static std::optional<MemRefType> convertMemRefType(MemRefType memrefType,
                                                   Type i8Ty) {
  Type elemType = memrefType.getElementType();
  if (!isNonTTFloat(elemType) && !elemType.isInteger(4))
    return std::nullopt;
  SmallVector<int64_t> newShape(memrefType.getShape());
  if (elemType.getIntOrFloatBitWidth() == 4) {
    assert(newShape.back() % 2 == 0 && "The last dimension must be even");
    newShape.back() /= 2;
  }
  return MemRefType::get(newShape, i8Ty, memrefType.getLayout(),
                         memrefType.getMemorySpace());
}

/// For the GPU wrapper: legalize all memref-typed values with non-TT_Float
/// element types to packed i8.  This covers gpu.alloc, gpu.memcpy, gpu.dealloc,
/// and bufferization.to_tensor so that the LLVM lowering computes correct byte
/// counts for sub-byte types.
static void convertWrapper(func::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  Type i8Ty = IntegerType::get(ctx, 8);

  // Step 1: Reinterpret each function arg with a non-TT_Float memref type
  // and replace all interior uses.  The function signature stays unchanged
  // so the caller (main) doesn't need updating yet - EmulateNarrowTypes
  // handles that later.
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  for (auto arg : funcOp.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    if (!memrefType)
      continue;
    auto newType = convertMemRefType(memrefType, i8Ty);
    if (!newType)
      continue;
    auto castOp = UnrealizedConversionCastOp::create(builder, funcOp.getLoc(),
                                                     *newType, arg);
    arg.replaceAllUsesExcept(castOp.getResult(0), castOp);
  }

  // Step 2: Legalize result types of all ops (gpu.alloc, etc.) that still
  // carry a non-TT_Float memref element type.
  funcOp.walk([&](Operation *op) {
    if (isa<UnrealizedConversionCastOp>(op))
      return;
    for (OpResult result : op->getResults()) {
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      if (!memrefType)
        continue;
      if (auto newType = convertMemRefType(memrefType, i8Ty))
        result.setType(*newType);
    }
  });

  // Step 3: Fix bufferization.to_tensor result types to match their
  // (now converted) buffer operand types.
  funcOp.walk([&](bufferization::ToTensorOp toTensorOp) {
    auto bufType = cast<MemRefType>(toTensorOp.getBuffer().getType());
    auto resultType = cast<RankedTensorType>(toTensorOp.getResult().getType());
    if (bufType.getElementType() != resultType.getElementType() ||
        bufType.getShape() != resultType.getShape()) {
      toTensorOp.getResult().setType(cast<bufferization::TensorLikeType>(
          RankedTensorType::get(bufType.getShape(), bufType.getElementType())));
    }
  });
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

/// Return true if funcOp contains a call to a function with rock.kernel.
static bool callsKernel(func::FuncOp funcOp) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return false;
  auto res = funcOp.walk([&](func::CallOp callOp) -> WalkResult {
    auto callee = moduleOp.lookupSymbol<func::FuncOp>(callOp.getCallee());
    if (callee && callee->hasAttr(rock::KernelAttr::getMnemonic())) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res.wasInterrupted();
}

struct RockLegalizeFloatTypesPass
    : public rock::impl::RockLegalizeFloatTypesPassBase<
          RockLegalizeFloatTypesPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockLegalizeFloatTypesPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  MLIRContext *ctx = &getContext();

  if (funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    if (failed(convertKernel(funcOp, ctx)))
      return signalPassFailure();
  } else if (callsKernel(funcOp)) {
    convertWrapper(funcOp, ctx);
  }
}
