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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Rock/utility/tritonUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/WalkResult.h"

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

static bool isNonTTFloat(Type t) {
  return isa<FloatType>(t) && !rock::isTTFloat(t) &&
         (t.getIntOrFloatBitWidth() == 8 || t.getIntOrFloatBitWidth() == 4);
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

/// Pack 4-bit (i4) kernel arguments to i8 with dimension halving.
/// For each BlockwiseGemmOp, traces matrixA/matrixB backwards through
/// blockwise_loads and transform chains, then halves the stride-1 dimension
/// (K preferred, then D) and converts i4 -> i8.
static LogicalResult pack4BitKernelArgs(func::FuncOp funcOp, MLIRContext *ctx) {
  OpBuilder builder(ctx);
  Type i8Ty = IntegerType::get(ctx, 8);
  DenseSet<BlockArgument> processedArgs;

  WalkResult walkResult = funcOp.walk([&](BlockwiseGemmOp gemmOp) {
    auto processOperand = [&](Value operand, bool isA) -> LogicalResult {
      auto tensorType = cast<RankedTensorType>(operand.getType());
      if (!is4Bit(tensorType.getElementType()))
        return success();

      auto maybeInputs = collectOperandInputs(operand);
      if (failed(maybeInputs))
        return failure();
      SmallVector<OperandInput> &inputs = maybeInputs.value();
      if (inputs.empty())
        return gemmOp.emitError("no inputs found for 4-bit operand");

      std::optional<bool> kPack = std::nullopt;
      for (OperandInput &input : inputs) {
        bool localKPack = true;
        if (processedArgs.contains(input.rootArg)) {
          // This is not supported for now. It could be tricky if one transform
          // chain is `packable` in D and the other in K!
          return gemmOp.emitError(
              "duplicate block arg across transform chains");
        }
        if (!input.loadOp)
          return gemmOp.emitError("transform chain has no blockwise_load");
        Value source = input.loadOp.getSource();

        // The outermost transform's upper view is the 6D source i.e. (k_loop,
        // g_block, m_block, n_block, k_iter, n_iter) for B of blockwise_load.
        // The tile dims are the last 2 dims of this view. matrixA (MxK): tile
        // dim 1 = K -> source dim (rank-1) matrixB (KxN): tile dim 0 = K ->
        // source dim (rank-2)
        int64_t sourceRank = cast<ShapedType>(source.getType()).getRank();
        if (sourceRank != 6)
          return gemmOp.emitError("source rank must be six");
        int64_t kDimIdx = isA ? (sourceRank - 1) : (sourceRank - 2);
        int64_t dDimIdx = isA ? (sourceRank - 2) : (sourceRank - 1);

        // Try K first, then D.
        std::optional<int64_t> halveDimIdx = std::nullopt;

        VectorizationResult kVectorRes = getMaxVectorization(source, kDimIdx);

        if (kVectorRes.max > 1)
          halveDimIdx = kDimIdx;

        if (!halveDimIdx.has_value()) {
          localKPack = false;
          VectorizationResult dVectorRes = getMaxVectorization(source, dDimIdx);
          if (dVectorRes.max > 1)
            halveDimIdx = dDimIdx;
        }
        if (!halveDimIdx.has_value())
          return gemmOp.emitError("max vectorization of both D and K is 1");

        if (kPack.has_value() && kPack.value() != localKPack)
          return gemmOp.emitError(
              "all inputs must agree whether to either pack along K or not");

        if (!kPack.has_value())
          kPack = localKPack;

        if (failed(rewriteTransformChain(ctx, input, halveDimIdx.value())))
          return failure();
        processedArgs.insert(input.rootArg);

        auto newSrcType =
            cast<RankedTensorType>(input.loadOp.getSource().getType());
        int64_t srcRank = newSrcType.getRank();
        SmallVector<int64_t> newResShape;
        newResShape.push_back(newSrcType.getShape()[srcRank - 2]);
        newResShape.push_back(newSrcType.getShape()[srcRank - 1]);
        input.loadOp.getResult().setType(
            RankedTensorType::get(newResShape, i8Ty));
      }

      if (!kPack.has_value())
        return gemmOp.emitError("kPack is undefined");

      if (isA)
        gemmOp.setMatrixAKPackAttr(builder.getBoolAttr(kPack.value()));
      else
        gemmOp.setMatrixBKPackAttr(builder.getBoolAttr(kPack.value()));
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
  // Fusions (arith/math ops) on 4-bit data are not supported: f4E2M1FN is a
  // storage format consumed directly by MFMA and has no hardware arithmetic.
  WalkResult fusionCheck = funcOp.walk([&](Operation *op) {
    for (OpResult result : op->getResults()) {
      auto tType = dyn_cast<RankedTensorType>(result.getType());
      if (!tType || !is4Bit(tType.getElementType()))
        continue;
      if (rock::isFusionOp(op)) {
        op->emitError("fusion ops on 4-bit types are not supported; "
                      "f4E2M1FN is a storage-only format with no "
                      "hardware arithmetic");
        return WalkResult::interrupt();
      }
      result.setType(
          RankedTensorType::get(tType.getShape(), i8Ty, tType.getEncoding()));
    }
    return WalkResult::advance();
  });
  if (fusionCheck.wasInterrupted())
    return failure();

  // rewriteTransformChain mutates block arg types (i4 -> i8 with halved
  // shapes), but the FunctionType attribute doesn't auto-update, so sync it.
  SmallVector<Type> newArgTypes;
  for (auto arg : funcOp.getArguments())
    newArgTypes.push_back(arg.getType());
  funcOp.setFunctionType(
      FunctionType::get(ctx, newArgTypes, funcOp.getResultTypes()));
  return success();
}

/// Legalize all non-TT_Float types to integer types of the same bit width
/// throughout the kernel function (no shape changes).
static LogicalResult convertKernel(func::FuncOp funcOp, MLIRContext *ctx) {
  // Save original types on BlockwiseGemmOp BEFORE converting.
  recordOrigTypesOnGemm(funcOp);

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

  // Step 2: Pack 4-bit types (i4) to i8 with dimension halving.
  return pack4BitKernelArgs(funcOp, ctx);
}

//===----------------------------------------------------------------------===//
// GPU wrapper conversion
//===----------------------------------------------------------------------===//

/// Legalize a memref type with a non-TT_Float element type to its i8
/// equivalent: f8E8M0FNU -> i8 (same shape), f4E2M1FN -> i8 (halved last dim).
static std::optional<MemRefType> convertMemRefType(MemRefType memrefType,
                                                   Type i8Ty) {
  if (!isNonTTFloat(memrefType.getElementType()))
    return std::nullopt;
  SmallVector<int64_t> newShape(memrefType.getShape());
  if (memrefType.getElementType().getIntOrFloatBitWidth() == 4) {
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
