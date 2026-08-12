//===- GridwiseAttnToBlockwise - MLIR Rock ops lowering passes -----===//
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
// ============================================================
//
// This pass converts rock.gridwise_attention
// into block- ops
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/fusionUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Rock/utility/tritonUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "GridLayoutEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGRIDWISEATTNTOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gridwise-attn-to-blockwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockGridwiseAttnToBlockwisePass
    : public rock::impl::RockGridwiseAttnToBlockwisePassBase<
          RockGridwiseAttnToBlockwisePass> {
  void runOnOperation() override;
};

} // end anonymous namespace

// Pick cache modifiers for the K and V loads in attention / gemm+gemm. Q always
// stays cached (reused across the whole nLoop). K/V are only reused across the
// seqQ tiles (reuse factor = gemm0MBlocks; GQA repeats fold into seqQ, split-KV
// just partitions seqK), so when seqQ is skinny (decode) they are read once:
// stream them (CS) under cache pressure (Q+K+V don't fit in the LLC). K and V
// are decided independently (kReloads/vReloads): an operand that reloads data
// (non-injective view, e.g. a conv im2col gemm0) relies on caching and is never
// streamed.
//
// NOTE: runs at the load_marker stage assuming a single Q/K/V; consider moving
// after LowerLoads (which materializes the actual blockwise loads) for fusion.
static std::pair<rock::CacheModifier, rock::CacheModifier>
chooseAttentionKVCacheModifiers(StringRef arch, Type qElemType,
                                int64_t qNumElems, Type kElemType,
                                int64_t kNumElems, bool kReloads,
                                Type vElemType, int64_t vNumElems,
                                bool vReloads, int64_t gemm0MBlocks) {
  const int64_t llcBytes = rock::getLastLevelCacheSize(arch);
  auto bytesOf = [](int64_t numElems, Type elemType) -> int64_t {
    return llvm::divideCeil(numElems * elemType.getIntOrFloatBitWidth(), 8);
  };
  const int64_t footprintBytes = bytesOf(qNumElems, qElemType) +
                                 bytesOf(kNumElems, kElemType) +
                                 bytesOf(vNumElems, vElemType);

  constexpr int64_t kSkinnyBlockThreshold = 2;
  const bool stream =
      footprintBytes > llcBytes && gemm0MBlocks < kSkinnyBlockThreshold;

  rock::CacheModifier cacheK = rock::CacheModifier::NONE;
  rock::CacheModifier cacheV = rock::CacheModifier::NONE;
  if (stream && !kReloads)
    cacheK = rock::CacheModifier::CS;
  if (stream && !vReloads)
    cacheV = rock::CacheModifier::CS;
  return {cacheK, cacheV};
}

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// GridwiseAttention lowering.
//===----------------------------------------------------------------------===//

struct GridwiseAttentionRewritePattern
    : public OpRewritePattern<GridwiseAttentionOp> {
  using OpRewritePattern<GridwiseAttentionOp>::OpRewritePattern;

  // Helper function to broadcast a 1D row vector [M] to 2D [M, N]
  // using expand_dims + broadcast pattern that lowers to Triton's
  // tt.expand_dims + tt.broadcast
  Value broadcastRowTo2D(PatternRewriter &rewriter, Location loc,
                         Value rowVector, int64_t numCols) const {
    auto rowType = cast<RankedTensorType>(rowVector.getType());
    int64_t numRows = rowType.getShape()[0];
    auto resultType =
        RankedTensorType::get({numRows, numCols}, rowType.getElementType());
    return expandDimAndBroadcast(rewriter, loc, rowVector, /*axis=*/1,
                                 resultType);
  }

  // Concatenate `chunks` (each [rows, nPerBlockG1]) along the column (N) axis
  // into a single [rows, nPerBlockG1 * chunks.size()] tile, preserving order
  // (chunk c -> columns [c*nPerBlockG1, (c+1)*nPerBlockG1)). The chunk count is
  // a power of two, so we fold pairs with tt.join + tt.trans + tt.reshape.
  Value concatChunksAlongN(PatternRewriter &rewriter, Location loc,
                           ArrayRef<Value> chunks, int64_t rows,
                           int64_t nPerBlockG1) const {
    assert(!chunks.empty());
    assert(llvm::isPowerOf2_64(chunks.size()) &&
           "chunk count must be a power of two for tt.join folding");
    if (chunks.size() == 1)
      return chunks.front();

    SmallVector<Value> level(chunks.begin(), chunks.end());
    int64_t cols = nPerBlockG1;
    auto transOrder = rewriter.getDenseI32ArrayAttr({0, 2, 1});
    while (level.size() > 1) {
      SmallVector<Value> next;
      next.reserve(level.size() / 2);
      for (size_t i = 0; i < level.size(); i += 2) {
        // join: [rows, cols] x [rows, cols] -> [rows, cols, 2]
        Value joined =
            triton::JoinOp::create(rewriter, loc, level[i], level[i + 1]);
        // trans: [rows, cols, 2] -> [rows, 2, cols]
        Value transed =
            triton::TransOp::create(rewriter, loc, joined, transOrder);
        // reshape: [rows, 2, cols] -> [rows, 2*cols] (block order preserved)
        Value reshaped = triton::ReshapeOp::create(
            rewriter, loc, SmallVector<int64_t>{rows, cols * 2}, transed);
        next.push_back(reshaped);
      }
      level = std::move(next);
      cols *= 2;
    }
    return level.front();
  }

  // If both `a` and `b` equal -inf, then `a - b` is NaN and any value derived
  // from that subtraction (for example, `exp2(a - b)`) is poisoned. Return
  // zero in that case and `original` otherwise.
  Value selectZeroIfBothNegInf(PatternRewriter &rewriter, Location loc, Value a,
                               Value b, Value original) const {
    assert(a.getType() == b.getType() && b.getType() == original.getType());
    auto shapedType = cast<ShapedType>(a.getType());
    Type elementType = shapedType.getElementType();
    Value negInf = createConstantFloatOp(
        rewriter, loc, shapedType, elementType,
        -std::numeric_limits<float>::infinity(), APFloat::opOK);
    Value zero = createConstantFloatOp(rewriter, loc, shapedType, elementType,
                                       0.0, APFloat::opOK);
    Value aIsNegInf = arith::CmpFOp::create(
        rewriter, loc, arith::CmpFPredicate::OEQ, a, negInf);
    Value bIsNegInf = arith::CmpFOp::create(
        rewriter, loc, arith::CmpFPredicate::OEQ, b, negInf);
    Value bothNegInf =
        arith::AndIOp::create(rewriter, loc, aIsNegInf, bIsNegInf);
    return arith::SelectOp::create(rewriter, loc, bothNegInf, zero, original);
  }

  // This function computes exp(gemm0 - rowmax_j)
  Value expSubstractMaxFromGemm0(PatternRewriter &rewriter, Location loc,
                                Value softmaxInput,
                                Value softmaxMax,
                                Value maxRow) const {
    // TODO: arith.maxnumf?
    Value maxRowNew = arith::MaximumFOp::create(rewriter, loc, maxRow, softmaxMax);
    
    // Broadcast maxRowNew from [M] to [M, N] to match softmaxInput shape
    auto inputShape = cast<RankedTensorType>(softmaxInput.getType()).getShape();
    Value maxRowBroadcast = broadcastRowTo2D(rewriter, loc, maxRowNew, inputShape[1]);
    
    Value gemm0SubMaxRow =
        arith::SubFOp::create(rewriter, loc, softmaxInput, maxRowBroadcast);
    Value softmaxExp =
        math::Exp2Op::create(rewriter, loc, gemm0SubMaxRow);

    return selectZeroIfBothNegInf(rewriter, loc, softmaxInput, maxRowBroadcast,
                                  softmaxExp);
  }

  // This updates the row sum according to the following
  // formula:
  // li = exp(m_{j-1} - m_{j}) * l_{j-1} + rowsum(Pij)
  // where
  // l is the rowsum accumulator
  // m is the rowmax accmulator
  // P is exp(gemm0 - rowmax_j)
  std::tuple<Value, Value, Value> updateRowSum(PatternRewriter &rewriter, Location loc,
                    Value softmaxSum, Value softmaxMax, Value sumRow, Value maxRow) const {
    // TODO: arith.maxnumf?
    Value maxRowNew = arith::MaximumFOp::create(rewriter, loc, maxRow, softmaxMax);

    Value maxRowDiff =
        arith::SubFOp::create(rewriter, loc, maxRow, maxRowNew);
    // A fully masked row keeps both maxima at -inf. Use exp2(0) = 1 for the
    // previous-sum scale so the zero accumulated sum remains zero.
    maxRowDiff =
        selectZeroIfBothNegInf(rewriter, loc, maxRow, maxRowNew, maxRowDiff);

    Value maxRowDiffExp =
        math::Exp2Op::create(rewriter, loc, maxRowDiff);

    Value sumRowNew = maxRowDiffExp;
    sumRowNew =
        arith::MulFOp::create(rewriter, loc, sumRowNew, sumRow);
    sumRowNew = arith::AddFOp::create(rewriter, loc, sumRowNew,
                                            softmaxSum);
    return {maxRowDiffExp, sumRowNew, maxRowNew};
  }

  // This computes LSE (log-sum-exp)
  // Note that this happens at the end of the kernel, so m and l are not running
  // sum/max anymore. They are the final values.
  // input = gemm0 output
  // x = input/log(2) -> we divide by log(2) to be able to use exp2()
  // m = max x
  // l = sum exp2(x-m)
  // We want to compute log(sum e^x), therefore we do:
  // log(l*exp2(m)) = (log2(l) + m)*log(2) -> we use exp2() for "m", because we
  // need to use the same exp function used for "l"
  Value computeLse(PatternRewriter &rewriter, Location loc, Type lseType,
                   Value sumRow, Value maxRow) const {
    auto lseElemType = cast<ShapedType>(lseType).getElementType();
    Value ln2Const = createConstantFloatOp(
        rewriter, loc, lseType, lseElemType, 0.69314718f,
        lseElemType.getIntOrFloatBitWidth() >= 32 ? APFloat::opOK
                                                  : APFloat::opInexact);

    // convert to LSE type (need full tensor type, not just element type)
    auto sumRowType = cast<RankedTensorType>(sumRow.getType());
    auto destTensorType = RankedTensorType::get(sumRowType.getShape(), lseElemType);
    Value sumRowCasted =
        createTypeConversionOp(rewriter, loc, sumRow, destTensorType);
    Value maxRowCasted =
        createTypeConversionOp(rewriter, loc, maxRow, destTensorType);
    
    // lse_i = (log2(l_i) + m_i)*log(2)
    // Migraphx expects LSE to be log
    Value log2Li = math::Log2Op::create(rewriter, loc, sumRowCasted);
    Value maxRowLog2 = maxRowCasted;
    Value lseLog2 = arith::AddFOp::create(rewriter, loc, log2Li, maxRowLog2);
    return arith::MulFOp::create(rewriter, loc, lseLog2, ln2Const);
  }

  // This is the out of loop scaling of attention output
  // where its divided by the accumulated rowsum
  Value scaleFinalOutput(PatternRewriter &rewriter, Location loc,
                        Value attentionAcc,
                        Value sumRow) const {
    // Broadcast sumRow from [M] to [M, N] to match attentionAcc shape
    auto accType = cast<RankedTensorType>(attentionAcc.getType());
    Value sumRowBroadcast = broadcastRowTo2D(rewriter, loc, sumRow, accType.getShape()[1]);

    // Cast broadcast to match accumulator element type if needed (e.g. f16 -> f32)
    sumRowBroadcast = createTypeConversionOp(rewriter, loc, sumRowBroadcast, accType);

    Value scaledOutput =
        arith::DivFOp::create(rewriter, loc, attentionAcc, sumRowBroadcast);
    Type outputElementType = accType.getElementType();
    Value zero = createConstantFloatOp(rewriter, loc, accType,
                                       outputElementType, 0.0, APFloat::opOK);
    Value isZeroSum = arith::CmpFOp::create(
        rewriter, loc, arith::CmpFPredicate::OEQ, sumRowBroadcast, zero);
    return arith::SelectOp::create(rewriter, loc, isZeroSum, zero,
                                   scaledOutput);
  }

  // This function does the corrections to row-based tiled reductions
  // according to flash attention 2 algorithm :
  // https://arxiv.org/pdf/2205.14135.pdf
  //
  // The shapes expected by the functions:
  // gemm0OutBufferMaxView.shape = [g0.Mpt, g0.Npt]
  // gemm1OutThreadwiseView.shape = [g1.Mpt=g0.Mpt, g1.Npt]
  // attentionOutAccBuffer.shape = [g1.Mpt=g0.Mpt, g1.Npt]
  //
  // This function will do the following logic :
  //
  // maxRowBufferNew = max(maxRowBuffer, gemm0OutBufferMaxView[:,0])
  // expMaxDiff = exp(maxRowBuffer - maxRowBufferNew)
  // attentionOutAccBufferMaxScaled = if not first iter ? attentionOutAccBuffer
  // / expMaxDiff : attentionOutAccBuffer attentionOutAccBufferMaxScaled +=
  // gemm1OutThreadwiseView [STORE] attentionOutAccBuffer =
  // attentionOutAccBufferMaxScaled
  Value createAttentionRowStateCorrections(PatternRewriter &rewriter,
                                          Location loc,
                                          Value gemm1Out,
                                          Value attentionAcc,
                                          Value expMaxDiffRow) const {
    // Broadcast expMaxDiffRow from [M] to [M, N] to match attentionAcc shape
    auto accType = cast<RankedTensorType>(attentionAcc.getType());
    Value expMaxDiffBroadcast = broadcastRowTo2D(rewriter, loc, expMaxDiffRow, accType.getShape()[1]);

    // Cast broadcast to match accumulator element type if needed (e.g. f16 -> f32)
    expMaxDiffBroadcast = createTypeConversionOp(rewriter, loc, expMaxDiffBroadcast, accType);

    Value scaledAttentionAcc =
        arith::MulFOp::create(rewriter, loc, attentionAcc, expMaxDiffBroadcast);
    Value newAttentionAcc = arith::AddFOp::create(
        rewriter, loc, scaledAttentionAcc, gemm1Out);

    return newAttentionAcc;
  }

  // This function will create a tile view that has the unpadded
  // coordinates if there were any padding involved in the gemm operands.
  ArrayAttr unpadTileView(PatternRewriter &rewriter, Location loc,
                          ArrayAttr tileView, int64_t prePadDim1,
                          int64_t prePadDim2) const {
    ArrayRef<int64_t> paddedShape = getLowerShape(tileView);
    assert(paddedShape.size() == 3);
    TopDownTMBuilder viewBuilder{
        rewriter, {"g", "paddedDim1", "paddedDim2"}, paddedShape, loc};
    viewBuilder.passThrough("g");
    // paddedShape is G x M x N
    viewBuilder.pad(
        {"paddedDim1", "paddedDim2"},
        {0, paddedShape[1] - prePadDim1, 0, paddedShape[2] - prePadDim2});
    TransformMapAttr padMap = viewBuilder.get();

    return prependUpperViews(rewriter, tileView,
                             rewriter.getArrayAttr({padMap}));
  }

  ArrayAttr outputViewToIndex(PatternRewriter &rewriter, Location loc,
                              ArrayAttr tileView) const {
    ArrayRef<int64_t> shape = getLowerShape(tileView);
    TopDownTMBuilder viewBuilder{
        rewriter, {"gemmG", "gemmN", "gemmM"}, shape, loc};
        
    viewBuilder.unmerge("raw", 0, {"gemmG", "gemmN", "gemmM"}, shape);

    return prependUpperViews(rewriter, tileView,
                             rewriter.getArrayAttr({viewBuilder.get()}));
  }

  ArrayAttr outputViewToN(PatternRewriter &rewriter, Location loc,
                          ArrayAttr tileView) const {
    ArrayRef<int64_t> shape = getLowerShape(tileView);
    TopDownTMBuilder viewBuilder{
        rewriter, {"gemmG", "gemmM", "gemmN"}, shape, loc};
    viewBuilder.ignore("gemmG");
    viewBuilder.ignore("gemmM");
    viewBuilder.passThrough({"gemmN"}, {0}, {"gemmN"});

    return prependUpperViews(rewriter, tileView,
                             rewriter.getArrayAttr({viewBuilder.get()}));
  }

  ArrayAttr outputViewToM(PatternRewriter &rewriter, Location loc,
                          ArrayAttr tileView) const {
    ArrayRef<int64_t> shape = getLowerShape(tileView);
    TopDownTMBuilder viewBuilder{
        rewriter, {"gemmG", "gemmM", "gemmN"}, shape, loc};
    viewBuilder.ignore("gemmG");
    viewBuilder.ignore("gemmN");
    viewBuilder.passThrough({"gemmM"}, {0}, {"gemmM"});

    return prependUpperViews(rewriter, tileView,
                             rewriter.getArrayAttr({viewBuilder.get()}));
  }

  // If padding is used in the kernel, this means the first gemm
  // will be done in a larger matrix. In typical, gemm kernels
  // the padded region in the output will just contain zeros. However,
  // attention kernel will perform softmax normalization on rows.
  // Therefore, having zeros -- zero not being the minimum representable
  // value in the element type -- going to affect all the values
  // post normalization. Therefore, this function creates a transforming
  // for loop that overwrites out of bounds values of first gemm output
  // to be negative infinity.
  Value createFirstGemmNegInfPadding(PatternRewriter &rewriter, Location loc,
                                     layout::GridCoordinates gridCoords,
                                     Value fakeTensor, Value firstGemmResult,
                                     Value negInfTensor,
                                     ArrayAttr tileView) const {
    ArrayAttr indexView = outputViewToIndex(rewriter, loc, tileView);
    fakeTensor = transform(rewriter, fakeTensor, indexView);

    auto transformsToPtrOp = TransformsToPtrOp::create(
        rewriter, loc, fakeTensor,
        ValueRange{gridCoords.g_block, gridCoords.m_block, gridCoords.n_block});
    Value maskTensor = transformsToPtrOp.getMask();
    return arith::SelectOp::create(rewriter, loc, maskTensor, firstGemmResult, negInfTensor);
  }

  enum class OutOfScopeType { KVCache, Causal, PrefixCausal, SlidingWindow };

  Value setGemm0OutputOutOfScope(PatternRewriter &rewriter, Location loc,
                                 OutOfScopeType outOfScopeType,
                                 layout::GridCoordinates gridCoords,
                                 Value firstGemmResult, Value fakeTensorM,
                                 Value fakeTensorN, Value negInfTensor,
                                 ArrayAttr tileView, bool enabled,
                                 Value nLoopIV, Value gemm0NBlocksLastIter,
                                 Value currentSeqLen, Value prefixOffset,
                                 Value slidingWindowLowerBound) const {
    if (enabled) {
      ArrayAttr nView = outputViewToN(rewriter, loc, tileView);
      ArrayAttr mView = outputViewToM(rewriter, loc, tileView);
      Value fakeTensorNview = transform(rewriter, fakeTensorN, nView);
      Value fakeTensorMview = transform(rewriter, fakeTensorM, mView);

      auto transformsToPtrN = TransformsToPtrOp::create(
          rewriter, loc, fakeTensorNview,
          ValueRange{gridCoords.g_block, gridCoords.m_block,
                     gridCoords.n_block});
      Value nIndex = transformsToPtrN.getPointers();

      auto transformsToPtrM = TransformsToPtrOp::create(
          rewriter, loc, fakeTensorMview,
          ValueRange{gridCoords.g_block, gridCoords.m_block,
                     gridCoords.n_block});
      Value mIndex = transformsToPtrM.getPointers();

      // For KVCache, we only need to mask on the last iteration, but for causal
      // masking we need to mask on every iteration.
      bool needsLastIterCheck = (outOfScopeType == OutOfScopeType::KVCache);

      // Use a lambda to generate the masking logic.
      auto generateMaskingLogic = [&](OpBuilder &b) {
        Value isInvalid;

        switch (outOfScopeType) {
        case OutOfScopeType::KVCache: {
        assert(currentSeqLen != nullptr);
        auto splatType =
            RankedTensorType::get(cast<ShapedType>(mIndex.getType()).getShape(),
                                  currentSeqLen.getType());
        Value currentSeqLenSplat = triton::SplatOp::create(rewriter, loc, splatType, currentSeqLen);
        // pointerTensor is mIndex
        isInvalid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                          nIndex, currentSeqLenSplat);
        break;
        }
        case OutOfScopeType::Causal: {
        // pointerTensor is nIndex
        isInvalid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                          nIndex, mIndex);
        break;
        }
        case OutOfScopeType::PrefixCausal: {
        // Prefix causal: mask when key_pos > (query_pos + prefix_offset).
        // This is used for prefix attention where:
        // - A prefix of tokens (0..prefix_offset) is always visible
        // - Anything after the prefix, standard causal masking applies
        assert(prefixOffset != nullptr);
        auto splatType =
            RankedTensorType::get(cast<ShapedType>(mIndex.getType()).getShape(),
                                  prefixOffset.getType());
        Value prefixOffsetSplat = triton::SplatOp::create(rewriter, loc, splatType, prefixOffset);

        // Compute query_pos + prefix_offset
        Value threshold =
            arith::AddIOp::create(b, loc, mIndex, prefixOffsetSplat);
        isInvalid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ugt,
                                          nIndex, threshold);
        break;
        }
        case OutOfScopeType::SlidingWindow: {
          // Sliding window: mask when key_pos < max(0, currentSeqLen -
          // windowSize). slidingWindowLowerBound is precomputed as
          // max(0, currentSeqLen - windowSize). The key position is nIndex in
          // the rocmlirTriton (N-loop) lowering.
          assert(slidingWindowLowerBound != nullptr);
          auto splatType = RankedTensorType::get(
              cast<ShapedType>(nIndex.getType()).getShape(),
              slidingWindowLowerBound.getType());
          Value lowerBoundSplat = triton::SplatOp::create(
              rewriter, loc, splatType, slidingWindowLowerBound);
          isInvalid = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult,
                                            nIndex, lowerBoundSplat);
          break;
        }
        }
        
        return arith::SelectOp::create(b, loc, isInvalid, negInfTensor, firstGemmResult);
      };

      if (needsLastIterCheck) {
        auto isLastIteration =
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                  nLoopIV, gemm0NBlocksLastIter);
        return arith::SelectOp::create(rewriter, loc, isLastIteration, generateMaskingLogic(rewriter), firstGemmResult);
      } else {
        // For causal masking, apply on every iteration
        return generateMaskingLogic(rewriter);
      }
    }
    // if not enabled, return input
    return firstGemmResult;
  }

  /// Undo GQA transforms for tensors of the fusion between first gemm and
  /// second gemm
  ArrayAttr undoGQATransforms(PatternRewriter &rewriter, Location loc,
                              GridwiseAttentionOp op,
                              ArrayRef<int64_t> unpaddedShape) const {
    ArrayAttr gqaTransform = nullptr;
    if (op.getNumRepeatsGQAAttr()) {
      SmallVector<StringRef> startNames = {"gemmG", "seqLenQ", "seqLenKV"};
      int64_t numRepeats = op.getNumRepeatsGQAAttr().getInt();

      assert(unpaddedShape.size() == 3);
      int64_t gemmG = unpaddedShape[0];
      int64_t seqLenQ = unpaddedShape[1];
      int64_t seqLenKV = unpaddedShape[2];
      assert(seqLenQ % numRepeats == 0);

      // (gemmG, seqLenQ*numRepeats, seqLenKV) -> (gemmG, numRepeats, seqLenQ,
      // seqLenKV)
      rock::TopDownTMBuilder unmerge(rewriter, startNames,
                                     {gemmG, seqLenQ, seqLenKV});
      unmerge.merge({"seqLenQ", "numRepeats"}, {2, 1}, "seqLenQ",
                    {seqLenQ / numRepeats, numRepeats});
      unmerge.passThrough({"gemmG", "seqLenKV"}, {0, 3}, {"gemmG", "seqLenKV"});
      auto unmergeAttr = unmerge.get();

      // (gemmG, numRepeats, seqLenQ, seqLenKV) -> (gemmG*numRepeats, seqLenQ,
      // seqLenKV)
      auto merger = rock::TopDownTMBuilder::below(unmerge, unmergeAttr);
      merger.unmerge("gemmG", 0, {"gemmG", "numRepeats"}, {gemmG, numRepeats});
      merger.passThrough({"seqLenQ", "seqLenKV"}, {1, 2},
                         {"seqLenQ", "seqLenKV"});
      auto mergerAttr = merger.get();

      SmallVector<Attribute> transformAttrs{unmergeAttr, mergerAttr};
      gqaTransform = rewriter.getArrayAttr(transformAttrs);
    }
    return gqaTransform;
  }

  // Transform GEMM0 output buffer for splitKV > 1 to match preSoftmaxBody
  // expectations. The preSoftmaxBody was created with splitKV baked into the
  // shapes, but GEMM0 computes without splitKV. This transform expands the
  // shapes at the fusion boundary.
  static ArrayAttr
  createSplitKVTransformsForGemm0Out(OpBuilder &builder, Location loc,
                                     ArrayRef<int64_t> gemm0OutShape,
                                     int64_t splitKV) {
    assert(splitKV > 1 && "split-kv must be greater than one");

    // GEMM0 output is [B*H, SeqQ, SeqK]
    // Need to transform to [B*H*splitKV, SeqQ, SeqK/splitKV] for fusion
    assert(gemm0OutShape.size() == 3 && "GEMM0 output must be 3D");
    assert(gemm0OutShape[2] % splitKV == 0 &&
           "SeqK must be divisible by splitKV");

    int64_t seqK = gemm0OutShape[2];
    int64_t seqKChunk = seqK / splitKV;

    // Step 1: Unmerge seqK: [B*H, SeqQ, SeqK] -> [B*H, SeqQ, splitKV,
    // SeqK/splitKV]
    rock::BottomUpTMBuilder unmergeSeqK(builder, {"batch", "seqQ", "seqK"},
                                        gemm0OutShape, loc);
    unmergeSeqK.unmerge({"splitKV", "seqK_chunk"}, {2, 3}, "seqK",
                        {splitKV, seqKChunk});
    unmergeSeqK.passThrough({"batch", "seqQ"}, {0, 1}, {"batch", "seqQ"});
    auto unmergeSeqKAttr = unmergeSeqK.get();

    // Step 2: Merge batch+splitKV: [B*H, SeqQ, splitKV, SeqK/splitKV] ->
    // [B*H*splitKV, SeqQ, SeqK/splitKV]
    auto merge = rock::BottomUpTMBuilder::above(unmergeSeqK, unmergeSeqKAttr);
    merge.merge("batch", 0, {"batch", "splitKV"});
    merge.passThrough({"seqQ", "seqK_chunk"}, {1, 2}, {"seqQ", "seqK_chunk"});
    auto mergeAttr = merge.get();

    return builder.getArrayAttr({mergeAttr, unmergeSeqKAttr});
  }

  std::tuple<Value, Value, Value, Value, Value, Value>
  getNLoopInfo(PatternRewriter &rewriter, Location loc,
               layout::AttnGridCoordinates gridCoordsGemm0,
               Value currentSeqLenTensor, Value prefixOffsetTensor,
               int64_t gemm0M, int64_t gemm0N, int64_t gemm0MPerBlock,
               int64_t gemm0NPerBlock, int64_t splitKV, bool isCausal,
               bool isKVCache, bool isPrefixCausal, int64_t slidingWindowSize,
               IntegerAttr numRepeatsGQA = nullptr) const {
    Value gemm0NBlocksLastIter;
    Value currentSeqLen;
    Value prefixOffset;
    Value effectiveSeqLen;
    Value slidingWindowLowerBound;
    Value start, end;

    // Lambda to load a 1D tensor value (used for currentSeqLen and
    // prefixOffset)
    auto loadTensorValue = [&](Value tensor) -> Value {
      assert(tensor && "tensor must be non-null");

      auto resultType = RankedTensorType::get({1},
                                          cast<ShapedType>(tensor.getType()).getElementType());

      // add dim 1 for load_marker to make sense
      ArrayRef<int64_t> inpShape =
          cast<ShapedType>(tensor.getType()).getShape();
      assert(inpShape.size() == 1 && "Expected rank to be one");
      SmallVector<StringRef> startNames = {"gemmG"};
      rock::BottomUpTMBuilder addDim(rewriter, startNames, inpShape);
      addDim.addDim("dummy", 1, 1);
      addDim.passThrough(ArrayRef<uint32_t>{0}, ArrayRef<uint32_t>{0});
      auto addDimAttr = addDim.get();
      Value tensorAddDim =
          rock::TransformOp::create(rewriter, loc, tensor, addDimAttr);

      // Create a LoadMarkerOp placeholder for a scalar-like load.
      ArrayAttr emptyViews = rewriter.getArrayAttr({});
      auto markerOp = LoadMarkerOp::create(
          rewriter, loc, resultType, tensorAddDim, emptyViews,
          ValueRange{gridCoordsGemm0.g_block}, rock::CacheModifier::NONE,
          /*reductionTileAxes=*/nullptr);

      return triton::UnsplatOp::create(rewriter, loc, markerOp);
    };

    // This is needed for KV Cache/Causal/Prefix Causal/Sliding Window masking
    // support
    if (isCausal || isKVCache || isPrefixCausal || slidingWindowSize > 0) {
      Value one = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), 1);
      Value constGemm0NPerBlock = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), gemm0NPerBlock);
      int64_t gemm0NBlocks = gemm0N / gemm0NPerBlock;
      Value constGemm0NBlocks = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), gemm0NBlocks);

      if (isKVCache) {
        currentSeqLen = loadTensorValue(currentSeqLenTensor);
        effectiveSeqLen = currentSeqLen;
      }

      // Compute sliding window lower bound: max(0, currentSeqLen - windowSize).
      if (slidingWindowSize > 0) {
        assert(currentSeqLen != nullptr &&
               "sliding window requires currentSeqLen (KV-cache)");
        Value constWindowSize = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), slidingWindowSize);
        Value zeroConst = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), 0);
        Value lowerBound = arith::SubIOp::create(rewriter, loc, currentSeqLen,
                                                 constWindowSize);
        slidingWindowLowerBound =
            arith::MaxSIOp::create(rewriter, loc, lowerBound, zeroConst);
      }

      if (isCausal || isPrefixCausal) {
        // Compute the last Q position in the block.
        // (mIndex + 1) * MPerBlock - 1.
        Value mIndex = gridCoordsGemm0.m_block;
        Value constGemm0MPerBlock = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), gemm0MPerBlock);
        Value mIndexPlusOne = arith::AddIOp::create(rewriter, loc, mIndex, one);
        Value nextBlockStart = arith::MulIOp::create(
            rewriter, loc, mIndexPlusOne, constGemm0MPerBlock);
        Value maxRowOfBlock =
            arith::SubIOp::create(rewriter, loc, nextBlockStart, one);
        if (numRepeatsGQA) {
          Value constNumRepeatsGQA =
              rewriter.createOrFold<arith::ConstantIntOp>(
                  loc, rewriter.getI32Type(), numRepeatsGQA.getInt());
          maxRowOfBlock = rewriter.createOrFold<arith::DivUIOp>(
              loc, maxRowOfBlock, constNumRepeatsGQA);
        }

        if (isPrefixCausal) {
          prefixOffset = loadTensorValue(prefixOffsetTensor);
          maxRowOfBlock =
              arith::AddIOp::create(rewriter, loc, maxRowOfBlock, prefixOffset);
        }

        effectiveSeqLen = maxRowOfBlock;
        if (isKVCache)
          effectiveSeqLen = arith::MinUIOp::create(rewriter, loc, currentSeqLen,
                                                   effectiveSeqLen);

        // For prefix causal, adding prefix_offset can push maxRowOfBlock beyond
        // gemm0N. Similarly, when gemm0M > gemm0N, the last query position can
        // exceed the key sequence length. In both cases, bound by gemm0N - 1.
        if (gemm0M > gemm0N || isPrefixCausal) {
          // Bound by actual K dimension (key sequence length)
          Value gemm0NMinusOne = rewriter.createOrFold<arith::ConstantIntOp>(
              loc, rewriter.getI32Type(), gemm0N - 1);
          effectiveSeqLen = arith::MinUIOp::create(
              rewriter, loc, effectiveSeqLen, gemm0NMinusOne);
        }
      } else {
        effectiveSeqLen = currentSeqLen;
      }

      // compute end index
      Value numerator = arith::AddIOp::create(rewriter, loc, effectiveSeqLen,
                                              constGemm0NPerBlock);
      end = rewriter.createOrFold<arith::DivUIOp>(loc, numerator,
                                                  constGemm0NPerBlock);

      // Clamp the trip count to the physical K/V allocation. Causal modes are
      // already bounded by effectiveSeqLen; pure KV cache needs this because
      // Triton's AMD buffer lowering does not use num_records to bound reads.
      end = arith::MinUIOp::create(rewriter, loc, end, constGemm0NBlocks);

      Value zero = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), 0);

      // start index is zero unless split-kv is enabled
      start = zero;
      if (splitKV != 1) {
        // here, "end" now means number of iterations in total, we need to split
        // those iterations into split-kv blocks.
        // see runEarlyExit() for details about early exit.
        Value constSplitKV = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), splitKV);
        Value constSplitKVM1 = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), splitKV - 1);
        Value splitKVNumerator =
            arith::AddIOp::create(rewriter, loc, end, constSplitKVM1);
        Value gemm0NIterations = rewriter.createOrFold<arith::DivUIOp>(
            loc, splitKVNumerator, constSplitKV);

        // if split-kv is enabled, we need to compute the start and end indices.
        start = arith::MulIOp::create(
            rewriter, loc, gridCoordsGemm0.split_block, gemm0NIterations);
        Value splitPlusOne = arith::AddIOp::create(
            rewriter, loc, gridCoordsGemm0.split_block, one);
        Value endSplitKV = arith::MulIOp::create(rewriter, loc, splitPlusOne,
                                                 gemm0NIterations);
        end = arith::MinUIOp::create(rewriter, loc, end, endSplitKV);
      }

      // Adjust start for sliding window: skip N-blocks that are entirely below
      // the window. All positions in those blocks would be masked to -inf
      // anyway, so we can avoid the loads and GEMMs altogether.
      if (slidingWindowSize > 0) {
        Value slidingWindowStart = rewriter.createOrFold<arith::DivUIOp>(
            loc, slidingWindowLowerBound, constGemm0NPerBlock);
        // start/slidingWindowStart are non-negative iteration indices derived
        // from unsigned division; use unsigned max to stay consistent with the
        // surrounding DivUIOp/MinUIOp arithmetic.
        start =
            arith::MaxUIOp::create(rewriter, loc, start, slidingWindowStart);
      }

      // compute last iteration of the block, this will be used later in
      // setGemm0OutputOutOfScope()
      gemm0NBlocksLastIter =
          rewriter.createOrFold<arith::SubIOp>(loc, end, one);
    } else if (splitKV != 1) {
      // if split-kv is enabled, we need to compute the start and end indices.
      // this is the code for the case where kv-cache and causal are not
      // enabled. the logic is easier, but note that some blocks will early
      // exit, see runEarlyExit() for details.
      // Use ceiling division so each split gets at least 1 iteration when
      // gemm0N < gemm0NPerBlock * splitKV (e.g. seq_len_k=64, block=64,
      // splitKV=2).
      int64_t nIterPerSplit =
          llvm::divideCeil(llvm::divideCeil(gemm0N, gemm0NPerBlock), splitKV);
      Value gemm0NIterations = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), nIterPerSplit);
      Value one = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), 1);
      start = arith::MulIOp::create(rewriter, loc, gridCoordsGemm0.split_block,
                                    gemm0NIterations);
      Value splitPlusOne = arith::AddIOp::create(
          rewriter, loc, gridCoordsGemm0.split_block, one);
      end =
          arith::MulIOp::create(rewriter, loc, splitPlusOne, gemm0NIterations);
    } else {
      start = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), 0);
      int64_t gemm0NBlocks = gemm0N / gemm0NPerBlock;
      end = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), gemm0NBlocks);
    }
    return std::make_tuple(start, end, gemm0NBlocksLastIter, currentSeqLen,
                           prefixOffset, slidingWindowLowerBound);
  }

  // Helper function to determine if early exit optimization is possible.
  // Early exit requires splitKV > 1 and at least one of: padding in gemm0M,
  // causal masking, or KV cache.
  static bool isEarlyExitPossible(int64_t splitKV, int64_t gemm0NPerBlock,
                                  std::optional<APInt> prePadG0N, bool isCausal,
                                  bool isKVCache) {
    // We have no work to do if (1) and (2 || 3) conditions are true:
    // 1. split-kv > 1
    // 2. there's padding in gemm0M && (at least) the last block in split-kv
    // dimension has nothing to do
    // 3. (kvcache || causal) && (end <= start)
    // - Note, causal could be set true here for prefix causal, or just
    //   regular causal.
    if (splitKV == 1)
      return false;

    bool earlyExitDueToPadding =
        prePadG0N.has_value() &&
        (prePadG0N.value().getSExtValue() >= gemm0NPerBlock);
    bool earlyExitDueToCausalOrKVCache = isCausal || isKVCache;

    return earlyExitDueToPadding || earlyExitDueToCausalOrKVCache;
  }

  // Helper function to compute the 'someWorkToDo' condition used for early
  // exit optimization.
  FailureOr<Value> computeIfWorkToDo(PatternRewriter &rewriter, Location loc,
                                     Value start, Value end, int64_t splitKV,
                                     int64_t gemm0NPerBlock,
                                     std::optional<APInt> prePadG0N,
                                     bool isCausal, bool isKVCache) const {
    if (!isEarlyExitPossible(splitKV, gemm0NPerBlock, prePadG0N, isCausal,
                             isKVCache))
      return failure();

    // Determine which condition applies to generate the appropriate runtime
    // check
    bool earlyExitDueToPadding =
        prePadG0N.has_value() &&
        (prePadG0N.value().getSExtValue() >= gemm0NPerBlock);
    bool earlyExitDueToCausalOrKVCache = isCausal || isKVCache;

    Value someWorkToDo;
    // For dynamic kernels, no need to check padding condition. start/end
    // checks can handle padding as well.
    if (earlyExitDueToCausalOrKVCache) {
      // If end is less than (or equal) start, then we can early exit the
      // split KV loop.
      someWorkToDo = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ugt, end, start);
    } else if (earlyExitDueToPadding) {
      Value constGemm0NPerBlock = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), gemm0NPerBlock);
      Value prePadNValue = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), prePadG0N.value().getSExtValue());
      Value startIteration =
          arith::MulIOp::create(rewriter, loc, start, constGemm0NPerBlock);

      // If startIteration is less than prePadMValue, then there is work to do
      someWorkToDo =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult,
                                startIteration, prePadNValue);
    }

    return someWorkToDo;
  }

  std::optional<scf::IfOp> runEarlyExit(PatternRewriter &rewriter, Location loc,
                                        Value start, Value end, int64_t splitKV,
                                        int64_t gemm0NPerBlock,
                                        std::optional<APInt> prePadG0N,
                                        bool isCausal, bool isKVCache,
                                        ArrayRef<Value> elseYieldValues) const {
    assert((elseYieldValues.size() == 1 || elseYieldValues.size() == 2) &&
           "early exit if must yield outAcc (length 1) or outAcc and lseOut "
           "(length 2)");

    FailureOr<Value> maybeSomeWorkToDo =
        computeIfWorkToDo(rewriter, loc, start, end, splitKV, gemm0NPerBlock,
                          std::move(prePadG0N), isCausal, isKVCache);

    if (failed(maybeSomeWorkToDo))
      return std::nullopt;

    SmallVector<Type> resultTypes;
    for (Value v : elseYieldValues)
      resultTypes.push_back(v.getType());

    scf::IfOp ifb =
        scf::IfOp::create(rewriter, loc, resultTypes, maybeSomeWorkToDo.value(),
                          /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&ifb.getThenRegion().front());

    rewriter.setInsertionPointToStart(&ifb.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, elseYieldValues);
    rewriter.setInsertionPointToStart(&ifb.getThenRegion().front());

    return ifb;
  }

  // Retile a ranked-tensor splat constant to targetShape in place.
  // Returns success (no-op) for non-tensor constants, failure for non-splat
  // tensors so the caller can emit a diagnostic.
  static LogicalResult retileSplatConstant(arith::ConstantOp constOp,
                                           ArrayRef<int64_t> targetShape) {
    auto origType = dyn_cast<RankedTensorType>(constOp.getType());
    if (!origType)
      return success();
    auto splatAttr = dyn_cast<SplatElementsAttr>(constOp.getValue());
    if (!splatAttr)
      return failure();
    auto newType =
        RankedTensorType::get(targetShape, origType.getElementType());
    constOp.setValueAttr(
        SplatElementsAttr::get(newType, splatAttr.getSplatValue<Attribute>()));
    constOp.getResult().setType(newType);
    return success();
  }

  // Map external splat constants referenced by the region body to
  // tile-shaped replacements. Constants like scale factors or mask fill
  // values may be defined at function scope and captured by closure.
  LogicalResult mapExternalSplatConstants(PatternRewriter &rewriter,
                                          Location loc, GridwiseAttentionOp op,
                                          Block &block,
                                          RankedTensorType tileType,
                                          IRMapping &mapping) const {
    for (Operation &bodyOp : block.without_terminator()) {
      for (Value operand : bodyOp.getOperands()) {
        if (operand.getParentBlock() == &block)
          continue;
        if (mapping.contains(operand))
          continue;
        auto constOp = operand.getDefiningOp<arith::ConstantOp>();
        if (!constOp)
          continue;
        if (!isa<RankedTensorType>(constOp.getType()))
          continue;
        auto clonedConst = cast<arith::ConstantOp>(rewriter.clone(*constOp));
        if (failed(retileSplatConstant(clonedConst, tileType.getShape())))
          return op->emitOpError()
                 << "non-splat constant in preSoftmaxBody cannot be tiled";
        mapping.map(operand, clonedConst.getResult());
      }
    }
    return success();
  }

  // Apply pre-softmax element-wise fusions to the first GEMM (Q*K) output.
  // The body is expected to be purely elementwise (no rock.transform ops)
  // after regularization by RockRegularizeInterGemmFusionPass. Block arg 0 is
  // the QK product; remaining args are extra inputs loaded via LoadMarkerOps.
  FailureOr<Value> postProcessFirstGemm(PatternRewriter &rewriter, Location loc,
                                        GridwiseAttentionOp op,
                                        layout::GridCoordinates gridCoords,
                                        Value srcGemm0OutBuffer,
                                        ArrayAttr gemm0OutViews) const {
    Region &region = op.getPreSoftmaxBody();
    if (region.empty())
      return srcGemm0OutBuffer;

    Block &block = region.front();

    if (block.without_terminator().empty())
      return srcGemm0OutBuffer;

    IRMapping mapping;
    auto tileType = cast<RankedTensorType>(srcGemm0OutBuffer.getType());

    if (block.getNumArguments() == 0)
      return op->emitOpError()
             << "preSoftmaxBody block must have at least one argument";
    BlockArgument qkArg = block.getArgument(0);

    Value gemm0Mapped = srcGemm0OutBuffer;
    Type regionElemType = cast<ShapedType>(qkArg.getType()).getElementType();
    if (tileType.getElementType() != regionElemType) {
      auto castType =
          RankedTensorType::get(tileType.getShape(), regionElemType);
      gemm0Mapped =
          createTypeConversionOp(rewriter, loc, gemm0Mapped, castType);
    }
    mapping.map(qkArg, gemm0Mapped);

    ValueRange extraInputs = op.getPreSoftmaxElemWiseInputs();
    unsigned numExtraArgs = block.getNumArguments() - 1;
    if (extraInputs.size() != numExtraArgs)
      return op->emitOpError()
             << "preSoftmaxBody has " << numExtraArgs
             << " non-QK block argument(s) but op has " << extraInputs.size()
             << " preSoftmaxElemWiseInputs";

    for (unsigned i = 0; i < extraInputs.size(); ++i) {
      Value globalInput = extraInputs[i];
      Value root;
      ArrayAttr globalInputMaps;
      std::tie(root, globalInputMaps, std::ignore) =
          untransform(rewriter, globalInput);

      auto resultType = RankedTensorType::get(
          tileType.getShape(),
          cast<ShapedType>(globalInput.getType()).getElementType());

      SmallVector<Attribute> allViews(gemm0OutViews.begin(),
                                      gemm0OutViews.end());
      if (!globalInputMaps.empty())
        allViews.append(globalInputMaps.begin(), globalInputMaps.end());
      ArrayAttr otherInputMap = rewriter.getArrayAttr(allViews);

      // These tiles are fused into the first gemm's output and then flow
      // through the softmax into the second gemm, so which axis they end up
      // reduced over is not known here.
      auto markerOp = LoadMarkerOp::create(
          rewriter, loc, resultType, root, otherInputMap,
          ValueRange{gridCoords.g_block, gridCoords.m_block,
                     gridCoords.n_block},
          rock::CacheModifier::NONE, /*reductionTileAxes=*/nullptr);

      mapping.map(block.getArgument(i + 1), markerOp.getResult());
    }

    if (failed(mapExternalSplatConstants(rewriter, loc, op, block, tileType,
                                         mapping)))
      return failure();

    // Clone elementwise ops from the body, fixing up result types to use
    // tile-level shapes while preserving element types.
    for (Operation &bodyOp : block.without_terminator()) {
      Operation *cloned = rewriter.clone(bodyOp, mapping);

      if (auto constOp = dyn_cast<arith::ConstantOp>(cloned)) {
        if (failed(retileSplatConstant(constOp, tileType.getShape())))
          return op->emitOpError()
                 << "non-splat constant in preSoftmaxBody cannot be tiled";
      }

      if (cloned->getNumResults() == 1 && cloned->getNumOperands() > 0) {
        auto operandTy =
            dyn_cast<RankedTensorType>(cloned->getOperand(0).getType());
        auto origResultTy =
            dyn_cast<RankedTensorType>(cloned->getResult(0).getType());
        if (!operandTy || !origResultTy) {
          LLVM_DEBUG(llvm::dbgs()
                     << "We expect first operand and result to be tensors\n");
          return failure();
        }
        cloned->getResult(0).setType(RankedTensorType::get(
            operandTy.getShape(), origResultTy.getElementType()));
      }
    }

    auto yieldOp = cast<rock::YieldOp>(block.getTerminator());
    return mapping.lookup(yieldOp.getOperand(0));
  }

  LogicalResult matchAndRewrite(GridwiseAttentionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    TypedValue<ShapedType> inQ = op.getQueries();
    ArrayRef<int64_t> qShape = cast<ShapedType>(inQ.getType()).getShape();
    Type elemTypeQ = cast<ShapedType>(inQ.getType()).getElementType();
    FailureOr<Type> maybeElemTypeQLoad = getInputFusionElementType(inQ);
    if (failed(maybeElemTypeQLoad))
      return op->emitOpError()
             << "Could not determine the underlying data type of Q";
    Type elemTypeQLoad = maybeElemTypeQLoad.value();

    TypedValue<ShapedType> inK = op.getKeys();
    ArrayRef<int64_t> kShape = cast<ShapedType>(inK.getType()).getShape();
    Type elemTypeK = cast<ShapedType>(inK.getType()).getElementType();
    FailureOr<Type> maybeElemTypeKLoad = getInputFusionElementType(inK);
    if (failed(maybeElemTypeKLoad))
      return op->emitOpError()
             << "Could not determine the underlying data type of K";
    Type elemTypeKLoad = maybeElemTypeKLoad.value();

    TypedValue<ShapedType> inV = op.getValues();
    Type elemTypeV = inV.getType().getElementType();
    FailureOr<Type> maybeElemTypeVLoad = getInputFusionElementType(inV);
    if (failed(maybeElemTypeVLoad))
      return op->emitOpError()
             << "Could not determine the underlying data type of V";
    Type elemTypeVLoad = maybeElemTypeVLoad.value();

    // Get output shape - transpose dims 1 and 2 of the result type
    auto outType = cast<ShapedType>(op.getResult().getType());
    ArrayRef<int64_t> outShape = outType.getShape();
    Type elemTypeOut = outType.getElementType();

    Value lse = op.getLse();

    Value currentSeqLenTensor = op.getCurrentSeqLen();
    Value prefixOffsetTensor = op.getPrefixOffset();
    bool isKVCache = currentSeqLenTensor != nullptr;
    bool isCausal = op.getCausal();
    bool isPrefixCausal = isCausal && prefixOffsetTensor;
    int64_t splitKV = op.getSplitKV();

    // Gemm0 out is casted to be softmaxType (if null, it's casted to elemTypeV)
    Type elemTypeSoftmax = op.getSoftmaxType().value_or(elemTypeV);

    int64_t gemm0G = qShape[0];
    int64_t gemm0M = qShape[1];
    int64_t gemm0K = qShape[2];
    int64_t gemm0N = kShape[2];

    int64_t gemm1M = outShape[1];
    int64_t gemm1N = outShape[2];

    GemmParamsAttr gemm0TuningParams = op.getParams0();
    GemmParamsAttr gemm1TuningParams = op.getParams1();
    int64_t gemm0KPerBlock = gemm0TuningParams.getKPerBlock();
    int64_t gemm0MPerBlock = gemm0TuningParams.getMPerBlock();
    int64_t gemm0NPerBlock = gemm0TuningParams.getNPerBlock();
    int64_t gemm0MBlocks = gemm0M / gemm0MPerBlock;
    assert(gemm0M % gemm0MPerBlock == 0);
    int64_t gemm0NBlocks = gemm0N / gemm0NPerBlock;
    assert(gemm0N % gemm0NPerBlock == 0);

    // Get current workgroup ID.
    Value bid =
        triton::GetProgramIdOp::create(rewriter, op.getLoc(), triton::ProgramIDDim::X);

    // Calculate different size derivations
    int64_t gemm1KPerBlock = gemm1TuningParams.getKPerBlock();
    assert(gemm0NPerBlock == gemm1KPerBlock);
    int64_t gemm1MPerBlock = gemm1TuningParams.getMPerBlock();
    int64_t gemm1NPerBlock = gemm1TuningParams.getNPerBlock();
    assert(gemm1N % gemm1NPerBlock == 0 &&
           "gemm1N must be a multiple of gemm1NPerBlock");
    // The head dim (gemm1N) is processed in `gemm1NChunks`
    // compile-time-constant chunks of `gemm1NPerBlock`. Each chunk runs its own
    // V load + second GEMM (so only one nPerBlockG1-wide V tile is staged in
    // LDS at a time); the per-chunk output tiles are concatenated back into the
    // full [gemm1MPerBlock, gemm1N] output tile before the single store.
    int64_t gemm1NChunks = gemm1N / gemm1NPerBlock;
    // The chunks are folded back together with pairwise tt.join in
    // concatChunksAlongN, so the chunk count must be a power of two. gemm1N is
    // padded to a power of two (AttnToGridwise.cpp) and gemm1NPerBlock is a
    // power of two, so this holds; assert to catch any future regression.
    assert(llvm::isPowerOf2_64(gemm1NChunks) &&
           "gemm1NChunks must be a power of two for tt.join folding");

    // params related to how we load Q
    bool prefetchQTile = gemm0K == gemm0KPerBlock;

    int64_t gemm1MBlocks = gemm1M / gemm1MPerBlock;
    assert(gemm1M % gemm1MPerBlock == 0);
    SmallVector<int64_t, 3> gemm0BidGridLengths = {gemm0G, gemm0MBlocks,
                                                   gemm0NBlocks};
    // Whether the K/V loads reload data (non-injective view: conv im2col,
    // broadcast, ...). Such operands rely on caching and are never streamed.
    FailureOr<bool> maybeKReloads = rock::isInputNonInjective(inK);
    FailureOr<bool> maybeVReloads = rock::isInputNonInjective(inV);
    if (failed(maybeKReloads))
      return op->emitOpError("could not trace K to determine load injectivity");
    if (failed(maybeVReloads))
      return op->emitOpError("could not trace V to determine load injectivity");
    bool kReloads = maybeKReloads.value();
    bool vReloads = maybeVReloads.value();

    LLVM_DEBUG(llvm::dbgs()
               << "elemTypeQLoad: " << elemTypeQLoad << "\n"
               << "elemTypeKLoad: " << elemTypeKLoad << "\n"
               << "elemTypeVLoad: " << elemTypeVLoad << "\n"
               << "kReloads: " << (kReloads ? "yes" : "no") << "\n"
               << "vReloads: " << (vReloads ? "yes" : "no") << "\n");

    // Compute output transforms for this load
    FailureOr<ArrayAttr> maybeGemm0OutTileView = computeOutputTransforms(
        rewriter, loc, gemm0MPerBlock, gemm0NPerBlock, gemm0BidGridLengths);

    if (failed(maybeGemm0OutTileView))
      return op->emitError("Failed to compute output transforms");

    auto gemm0OutTileView = maybeGemm0OutTileView.value();

    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    // We need two different grid lengths because the V input tensor and the
    // output tensor have different shapes:
    // - V tensor shape: [gemm0G, seqK, headDim] - splitKV is NOT in the batch
    //   dim, and the head dim is indexed by chunk (n_block in [0,
    //   gemm1NChunks))
    // - Output tensor shape: [gemm0G * splitKV, seqQ, headDim] - splitKV IS in
    //   the batch dim (each split writes to a separate slice of the output),
    //   and the concatenated full-N tile is stored as a single N block
    // Therefore, loadTile for V uses gemm1BidGridLengths, while the output
    // store transforms use gemm1BidGridLengthsForStore.
    SmallVector<int64_t, 3> gemm1BidGridLengths = {gemm0G, gemm1MBlocks,
                                                   gemm1NChunks};
    SmallVector<int64_t, 3> gemm1BidGridLengthsForStore = {gemm0G * splitKV, gemm1MBlocks, 1};

    // if splitKV == 1, we define nullptr, and makeGxNGridLayout() will use
    // fewer instructions
    Value splitKVConst =
        (splitKV > 1) ? rewriter.createOrFold<ConstantIntOp>(loc, rewriter.getI32Type(), splitKV)
                      : nullptr;

    auto maybeGridSize = rock::getGridSize(op);
    if (failed(maybeGridSize))
      return op->emitError("Failed to get grid_size");

    int64_t gridSize = maybeGridSize->getInt();
        
    auto arch = rock::getArchValue(op);

    // Cache hint for the K/V loads: stream them when seqQ is skinny (decode)
    // and the KV cache doesn't fit in the LLC. Q is always kept cached.
    auto [cacheK, cacheV] = chooseAttentionKVCacheModifiers(
        arch, elemTypeQLoad, inQ.getType().getNumElements(), elemTypeKLoad,
        inK.getType().getNumElements(), kReloads, elemTypeVLoad,
        inV.getType().getNumElements(), vReloads, gemm0MBlocks);

    auto gridCoordsGemm0mIter0 = layout::makeGxNGridLayout(
        rewriter, loc, bid, gemm0MBlocks,
        rewriter.createOrFold<arith::ConstantIntOp>(loc, rewriter.getI32Type(),
                                                    0),
        gridSize, arch, rock::getNumChipletsValue(op), splitKVConst);

    auto blockMTensorType =
        RankedTensorType::get({gemm0MPerBlock}, elemTypeSoftmax);
    Value maxRow = createConstantFloatOp(
        rewriter, loc, blockMTensorType, elemTypeSoftmax,
        -std::numeric_limits<float>::infinity(), APFloat::opOK);
    Value sumRow = createConstantFloatOp(rewriter, loc, blockMTensorType,
                                         elemTypeSoftmax, 0.0, APFloat::opOK);
    Value zero = rewriter.createOrFold<ConstantIntOp>(loc, rewriter.getI32Type(), 0);

    Value gemm0NBlocksLastIter;
    Value currentSeqLen;
    Value prefixOffset;
    Value slidingWindowLowerBound;
    Value start, end;
    int64_t slidingWindowSize =
        static_cast<int64_t>(op.getSlidingWindowSize().value_or(0));
    // get nLoop
    std::tie(start, end, gemm0NBlocksLastIter, currentSeqLen, prefixOffset,
             slidingWindowLowerBound) =
        getNLoopInfo(rewriter, loc, gridCoordsGemm0mIter0, currentSeqLenTensor,
                     prefixOffsetTensor, gemm0M, gemm0N, gemm0MPerBlock,
                     gemm0NPerBlock, splitKV, isCausal, isKVCache,
                     isPrefixCausal, slidingWindowSize,
                     op.getNumRepeatsGQAAttr());

    // Early exit: Skip all computation when there's no work but always write
    // output. The IfOp returns (outAcc, lseOut?) so the code after the if
    // always has defined values; the else branch yields zero-initialized
    // tensors.
    SmallVector<Value> earlyExitElseValues;
    auto initOutAcc = rock::createZeroAccBuffer(
        rewriter, loc, {gemm1MPerBlock, gemm1N}, elemTypeOut);
    earlyExitElseValues.push_back(initOutAcc);
    RankedTensorType lseType;
    if (lse) {
      auto lseElemType = cast<ShapedType>(lse.getType()).getElementType();
      lseType = RankedTensorType::get({gemm0MPerBlock}, lseElemType);
      Value initLseOut = createConstantFloatOp(
          rewriter, loc, lseType, lseElemType,
          -std::numeric_limits<float>::infinity(), APFloat::opOK);
      earlyExitElseValues.push_back(initLseOut);
    }
    std::optional<scf::IfOp> earlyExitIf = runEarlyExit(
        rewriter, loc, start, end, splitKV, gemm0NPerBlock, op.getPrePadG0N(),
        isCausal, isKVCache, earlyExitElseValues);

    // If gemm0K is equal to gemm0KPerBlock that means
    // effectively there is no K loop. Therefore, we
    // can prefetch the Q tile into regs outside of the
    // loop.
    Value loadedQ;
    // TODO(rocmlirTriton): do this in an independent pass, hoist loads out of
    // the loop if possible
    if (prefetchQTile) {
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: gemm0K is equal to gemm0KPerBlock\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: Prefetching Q tile into regs...\n");

      // it is fine m iteration to be zero as it irrelevant to Q tensor
      // as the first gemm is Kt x Qt.
      auto gridCoordsGemm0LoadQ = layout::makeGxNGridLayout(
          rewriter, loc, bid, gemm0MBlocks, zero, gridSize, arch,
          rock::getNumChipletsValue(op), splitKVConst);

      loadedQ = rock::loadTile(
          rewriter, loc, inQ, /*kiter=*/zero, "m", gridCoordsGemm0LoadQ,
          gemm0KPerBlock, gemm0MPerBlock,
          /*isKFirst=*/false, gemm0BidGridLengths, rock::CacheModifier::NONE);
    }

    Type accType = rock::getAccType(elemTypeQ, elemTypeK);
    Type gemm1AccType = rock::getAccType(elemTypeV, elemTypeV);

    // Create initial accumulator for nLoop with the shape of the final output
    // tile. The second GEMM multiplies softmax output (cast to elemTypeV) by V,
    // so its accumulator type may differ from the first GEMM's (e.g. f32 vs i32
    // when Q/K are i8 but V is f16).
    SmallVector<Value> nLoopInitArgs;
    nLoopInitArgs.reserve(gemm1NChunks + 2);
    for (int64_t chunk = 0; chunk < gemm1NChunks; ++chunk)
      nLoopInitArgs.push_back(rock::createZeroAccBuffer(
          rewriter, loc, {gemm1MPerBlock, gemm1NPerBlock}, gemm1AccType));
    nLoopInitArgs.push_back(maxRow);
    nLoopInitArgs.push_back(sumRow);

    Value one = rewriter.createOrFold<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), 1);
    scf::ForOp nLoopOp =
        scf::ForOp::create(rewriter, loc, start, end, one, nLoopInitArgs);
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(nLoopOp.getBody());
      int64_t kIterationsGemm0 = gemm0K / gemm0KPerBlock;
      // Convert loop IV to i32 for grid layout and load operations
      Value nLoopIV = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), nLoopOp.getInductionVar());
      // Get the iteration arguments: one accumulator per head-dim chunk,
      // followed by the shared softmax row state (maxRow, sumRow).
      SmallVector<Value> attentionAccs;
      attentionAccs.reserve(gemm1NChunks);
      for (int64_t chunk = 0; chunk < gemm1NChunks; ++chunk)
        attentionAccs.push_back(nLoopOp.getRegionIterArg(chunk));

      maxRow = nLoopOp.getRegionIterArg(gemm1NChunks);
      sumRow = nLoopOp.getRegionIterArg(gemm1NChunks + 1);

      layout::GridCoordinates gridCoordsGemm0 = layout::makeGxNGridLayout(
          rewriter, loc, bid, gemm0MBlocks, nLoopIV, gridSize, arch,
          rock::getNumChipletsValue(op), splitKVConst);
      Value initAcc = rock::createZeroAccBuffer(
          rewriter, loc, {gemm0MPerBlock, gemm0NPerBlock}, accType);

      Value endKLoop =
          rewriter.createOrFold<arith::ConstantIntOp>(loc, rewriter.getI32Type(), kIterationsGemm0);
      scf::ForOp kLoopOp = scf::ForOp::create(rewriter, loc, zero, endKLoop,
                                              one, ValueRange{initAcc});
      {
        PatternRewriter::InsertionGuard kLoopGuard(rewriter);
        rewriter.setInsertionPointToStart(kLoopOp.getBody());
        Value kLoopIV = kLoopOp.getInductionVar();
        Value accArg = kLoopOp.getRegionIterArg(0);

        // if gemm0K is equal to gemm0KPerBlock, the Q tile
        // is already prefetched into regs. See above.
        if (!prefetchQTile) {
          loadedQ =
              rock::loadTile(rewriter, loc, inQ, /*kiter=*/kLoopIV, "m",
                             gridCoordsGemm0, gemm0KPerBlock, gemm0MPerBlock,
                             /*isKFirst=*/false, gemm0BidGridLengths,
                             rock::CacheModifier::NONE);
        }

        Value loadedK =
            rock::loadTile(rewriter, loc, inK, /*kiter=*/kLoopIV, "n",
                           gridCoordsGemm0, gemm0KPerBlock, gemm0NPerBlock,
                           /*isKFirst=*/true, gemm0BidGridLengths, cacheK);

        Value newAcc = BlockwiseGemmOp::create(
            rewriter, loc, loadedQ, loadedK, accArg,
            /*matrixScaleA=*/nullptr,
            /*matrixScaleB=*/nullptr, /*quantBlockSize=*/nullptr,
            /*matrixAOrigElemType=*/nullptr, /*matrixBOrigElemType=*/nullptr,
            /*matrixAKPack=*/nullptr, /*matrixBKPack=*/nullptr);

        // Yield the new accumulator
        scf::YieldOp::create(rewriter, loc, ValueRange{newAcc});
      }
      Value firstGemmResult = kLoopOp.getResult(0);

      int64_t prePadG0M = gemm0M;
      if (op.getPrePadG0M().has_value()) {
        prePadG0M = op.getPrePadG0M().value().getSExtValue();
      }
      int64_t prePadG0N = gemm0N;
      if (op.getPrePadG0N().has_value()) {
        prePadG0N = op.getPrePadG0N().value().getSExtValue();
      }
      ArrayAttr gemm0OutTileViewUnPadded =
          unpadTileView(rewriter, loc, gemm0OutTileView, prePadG0M, prePadG0N);

      // undo Grouped-Query Attention (GQA) transforms
      // This is needed because the preSoftmaxElementWise inputs (if any), don't
      // have the GQA transformed applied to them. So, we undo the transform to
      // the output of the first GEMM. See postProcessFirstGemm() to understand
      // the transforms done to preSoftmaxElementWise inputs.
      ArrayRef<int64_t> unpaddedShape = getLowerShape(gemm0OutTileViewUnPadded);
      ArrayAttr undoGQA = undoGQATransforms(rewriter, loc, op, unpaddedShape);

      // undo the GQA transforms for postProcessFirstGemm()
      if (undoGQA)
        gemm0OutTileViewUnPadded =
            prependUpperViews(rewriter, gemm0OutTileViewUnPadded, undoGQA);

      // Apply splitKV transforms if needed
      // This transforms the GEMM0 output from [B*H, SeqQ, SeqK] to
      // [B*H*splitKV, SeqQ, SeqK/splitKV] to match the preSoftmax inputs.
      if (splitKV > 1 && op.getPreSoftmaxHasSplitKVTransforms()) {
        ArrayAttr splitKVTransforms = createSplitKVTransformsForGemm0Out(
            rewriter, loc, unpaddedShape, splitKV);
        gemm0OutTileViewUnPadded = prependUpperViews(
            rewriter, gemm0OutTileViewUnPadded, splitKVTransforms);
      }

      // Align the preSoftmaxElementWise (if any) to
      // be performed on the output of the first gemm.
      auto maybeFirstGemmResult =
          postProcessFirstGemm(rewriter, loc, op, gridCoordsGemm0,
                               firstGemmResult, gemm0OutTileViewUnPadded);
      if (failed(maybeFirstGemmResult))
        return op->emitError("Failed to post process first GEMM output");
      firstGemmResult = maybeFirstGemmResult.value();

      Value maxRowDiffExp, softmaxExp;
      // Softmax
      if (op.getEnableSoftmax()) {
        // convert firstGemmResult to elemTypeSoftmax
        auto firstGemmType = cast<RankedTensorType>(firstGemmResult.getType());
        auto softmaxInputType = RankedTensorType::get(firstGemmType.getShape(), elemTypeSoftmax);
        Value softmaxInput = createTypeConversionOp(rewriter, loc, firstGemmResult, softmaxInputType);

        // Scale gemm0 output by (1/ln2)
        // So that we can use exp2 instead of exp.
        Value ln2Recip = createConstantFloatOp(
            rewriter, loc, softmaxInput.getType(), elemTypeSoftmax, 1.44269504f,
            elemTypeSoftmax.getIntOrFloatBitWidth() >= 32 ? APFloat::opOK
                                                          : APFloat::opInexact);
        softmaxInput = arith::MulFOp::create(rewriter, loc, softmaxInput, ln2Recip);

        // fakeTensor is needed to generate the views and indices+mask with
        // TransformsToPtrOp It represents the Q*K matrix (that is never written
        // to global memory)
        ArrayRef<int64_t> lowerShape = getLowerShape(gemm0OutTileViewUnPadded);
        assert(lowerShape.size() == 3);
        int64_t tensorSize =
            std::accumulate(lowerShape.begin(), lowerShape.end(), 1LL,
                            std::multiplies<int64_t>());
        Value fakeTensor =
            rock::createZeroAccBuffer(rewriter, loc, {tensorSize}, accType);
        Value fakeTensorM =
            rock::createZeroAccBuffer(rewriter, loc, {lowerShape[1]}, accType);
        Value fakeTensorN =
            rock::createZeroAccBuffer(rewriter, loc, {lowerShape[2]}, accType);
        Value negInfTensor = createConstantFloatOp(
            rewriter, loc, softmaxInput.getType(),
            cast<ShapedType>(softmaxInput.getType()).getElementType(),
            -std::numeric_limits<float>::infinity(), APFloat::opOK);

        // Handle padding
        bool hasPadding =
            op.getPrePadG0M().has_value() || op.getPrePadG0N().has_value();
        if (hasPadding) {
          softmaxInput = createFirstGemmNegInfPadding(
              rewriter, loc, gridCoordsGemm0, fakeTensor, softmaxInput,
              negInfTensor, gemm0OutTileViewUnPadded);
        }

        // Negative Infinite for extra values based on masking type
        // KV cache masking is independent of causal masking - it masks out
        // positions beyond currentSeqLen (padding). Apply it whenever KV
        // cache is enabled, regardless of causal/prefix-causal mode.
        softmaxInput = setGemm0OutputOutOfScope(
            rewriter, loc, OutOfScopeType::KVCache, gridCoordsGemm0,
            softmaxInput, fakeTensorM, fakeTensorN, negInfTensor,
            gemm0OutTileViewUnPadded, isKVCache, nLoopIV, gemm0NBlocksLastIter,
            currentSeqLen,
            /*prefixOffset=*/nullptr, /*slidingWindowLowerBound=*/nullptr);

        // Sliding window masking: mask when key < max(0, currentSeqLen -
        // windowSize). Independent of causal masking and applied on every
        // iteration (like causal), alongside KV-cache masking.
        softmaxInput = setGemm0OutputOutOfScope(
            rewriter, loc, OutOfScopeType::SlidingWindow, gridCoordsGemm0,
            softmaxInput, fakeTensorM, fakeTensorN, negInfTensor,
            gemm0OutTileViewUnPadded, slidingWindowSize > 0, nLoopIV,
            gemm0NBlocksLastIter,
            /*currentSeqLen=*/nullptr,
            /*prefixOffset=*/nullptr, slidingWindowLowerBound);

        // Causal masking: either prefix-causal or standard causal
        // Prefix causal: mask when key > (query + offset).
        // This combines causal masking with a prefix offset
        softmaxInput = setGemm0OutputOutOfScope(
            rewriter, loc, OutOfScopeType::PrefixCausal, gridCoordsGemm0,
            softmaxInput, fakeTensorM, fakeTensorN, negInfTensor,
            gemm0OutTileViewUnPadded, isPrefixCausal, nLoopIV,
            gemm0NBlocksLastIter,
            /*currentSeqLen=*/nullptr, prefixOffset,
            /*slidingWindowLowerBound=*/nullptr);

        // Standard causal masking: mask when key > query
        softmaxInput = setGemm0OutputOutOfScope(
            rewriter, loc, OutOfScopeType::Causal, gridCoordsGemm0,
            softmaxInput, fakeTensorM, fakeTensorN, negInfTensor,
            gemm0OutTileViewUnPadded, isCausal && !isPrefixCausal, nLoopIV,
            gemm0NBlocksLastIter,
            /*currentSeqLen=*/nullptr,
            /*prefixOffset=*/nullptr, /*slidingWindowLowerBound=*/nullptr);

        IntegerAttr reductionAxis = rewriter.getIndexAttr(1);

        auto softmaxShape = cast<ShapedType>(softmaxInput.getType()).getShape();
        assert(softmaxShape.size() == 2);

        // Softmax max reduction
        Value softmaxMax = BlockwiseReduceOp::create(
            rewriter, loc, softmaxInput, reductionAxis,
            rewriter.getAttr<rock::ReduceMethodAttr>(rock::ReduceMethod::Max));

        softmaxExp = expSubstractMaxFromGemm0(rewriter, loc, softmaxInput,
                                 softmaxMax, maxRow);

        // Softmax sum reduction
        Value softmaxSum = BlockwiseReduceOp::create(
            rewriter, loc, softmaxExp, reductionAxis,
            rewriter.getAttr<rock::ReduceMethodAttr>(rock::ReduceMethod::Sum));

        std::tie(maxRowDiffExp, sumRow, maxRow) = updateRowSum(rewriter, loc, softmaxSum, softmaxMax, sumRow, maxRow);
      }

      // Emit blockwise GEMM 1.
      auto gemm0Out = op.getEnableSoftmax() ? softmaxExp : firstGemmResult;
      // Convert gemm0Out to match V's element type for the second gemm
      Type gemm0OutElemType =
          cast<RankedTensorType>(gemm0Out.getType()).getElementType();
      if (gemm0OutElemType != elemTypeV) {
        auto gemm0OutType = cast<RankedTensorType>(gemm0Out.getType());
        auto destType =
            RankedTensorType::get(gemm0OutType.getShape(), elemTypeV);
        gemm0Out = createTypeConversionOp(rewriter, loc, gemm0Out, destType);
      }

      // For attention (softmax enabled), each nLoop iteration starts gemm1
      // from zero because flash-attention corrections handle cross-iteration
      // accumulation. For gemm_gemm (no softmax), we accumulate directly
      // into attentionAcc across nLoop iterations.
      // Process the head dim one nPerBlockG1-wide chunk at a time. Each chunk
      // loads its own V tile (n_block = chunk) and runs an independent second
      // GEMM into its own accumulator; the softmax row state is shared.
      SmallVector<Value> nLoopYields;
      nLoopYields.reserve(gemm1NChunks + 2);
      for (int64_t chunk = 0; chunk < gemm1NChunks; ++chunk) {
        Value gemm1InitAcc;
        if (op.getEnableSoftmax()) {
          gemm1InitAcc = rock::createZeroAccBuffer(
              rewriter, loc, {gemm1MPerBlock, gemm1NPerBlock}, gemm1AccType);
        } else {
          gemm1InitAcc = attentionAccs[chunk];
        }

        Value chunkIdx = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), chunk);
        auto gridCoordsGemm1 = layout::makeGxNGridLayout(
            rewriter, loc, bid, gemm1MBlocks, chunkIdx, gridSize, arch,
            rock::getNumChipletsValue(op), splitKVConst);

        Value loadedV =
            rock::loadTile(rewriter, loc, inV,
                           /*kIter=*/nLoopIV, "n", gridCoordsGemm1,
                           gemm1KPerBlock, gemm1NPerBlock,
                           /*isKFirst=*/true, gemm1BidGridLengths, cacheV);

        Value gemm1Out = BlockwiseGemmOp::create(
            rewriter, loc, gemm0Out, loadedV, gemm1InitAcc,
            /*matrixScaleA=*/nullptr, /*matrixScaleB=*/nullptr,
            /*quantBlockSize=*/nullptr,
            /*matrixAOrigElemType=*/nullptr, /*matrixBOrigElemType=*/nullptr,
            /*matrixAKPack=*/nullptr, /*matrixBKPack=*/nullptr);

        // Apply flash attention correction (per chunk, shared row state)
        if (op.getEnableSoftmax()) {
          attentionAccs[chunk] = createAttentionRowStateCorrections(
              rewriter, loc, gemm1Out, attentionAccs[chunk], maxRowDiffExp);
        } else {
          attentionAccs[chunk] = gemm1Out;
        }
        nLoopYields.push_back(attentionAccs[chunk]);
      }

      // Yield the updated per-chunk accumulators, then maxRow, sumRow.
      nLoopYields.push_back(maxRow);
      nLoopYields.push_back(sumRow);
      scf::YieldOp::create(rewriter, loc, nLoopYields);
    }
    SmallVector<Value> outAccs;
    outAccs.reserve(gemm1NChunks);
    for (int64_t chunk = 0; chunk < gemm1NChunks; ++chunk)
      outAccs.push_back(nLoopOp.getResult(chunk));
    maxRow = nLoopOp.getResult(gemm1NChunks);
    sumRow = nLoopOp.getResult(gemm1NChunks + 1);

    if (op.getEnableSoftmax()) {
      for (int64_t chunk = 0; chunk < gemm1NChunks; ++chunk)
        outAccs[chunk] =
            scaleFinalOutput(rewriter, loc, outAccs[chunk], sumRow);
    }

    // Concatenate the per-chunk [gemm1MPerBlock, gemm1NPerBlock] output tiles
    // into the full [gemm1MPerBlock, gemm1N] tile consumed by the single store.
    Value outAcc = concatChunksAlongN(rewriter, loc, outAccs, gemm1MPerBlock,
                                      gemm1NPerBlock);

    if (cast<ShapedType>(outAcc.getType()).getElementType() != elemTypeOut) {
        auto outAccTensorType = cast<RankedTensorType>(outAcc.getType());
        auto destType = RankedTensorType::get(outAccTensorType.getShape(), elemTypeOut);
        outAcc = createTypeConversionOp(rewriter, loc, outAcc, destType);
    }
    Value lseOut;
    if (lse) {
      // it must be guaranteed by the verifier
      assert(op.getEnableSoftmax());
      lseOut = computeLse(rewriter, loc, lseType, sumRow, maxRow);
    }

    // When early exit is enabled, the IfOp returns (outAcc, lseOut?). Yield
    // from the then region and then take results from the if.
    if (earlyExitIf.has_value()) {
      rewriter.setInsertionPointToEnd(&earlyExitIf->getThenRegion().front());
      if (lse)
        scf::YieldOp::create(rewriter, loc, ValueRange{outAcc, lseOut});
      else
        scf::YieldOp::create(rewriter, loc, ValueRange{outAcc});
      rewriter.setInsertionPointAfter(*earlyExitIf);
      outAcc = earlyExitIf->getResult(0);
      if (lse)
        lseOut = earlyExitIf->getResult(1);
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: early exit enabled - "
                 << "output writes will execute unconditionally\n");
    }

    // Note that we don't use splitKV here because that dimension belongs to the
    // batch size already for output tensors
    auto gridCoordsGemm1 = layout::makeGxNGridLayout(
        rewriter, loc, bid, gemm1MBlocks, zero, gridSize, arch,
        rock::getNumChipletsValue(op));

    // Compute output transforms - use grid lengths with splitKV for output
    // The store consumes the full concatenated [gemm1MPerBlock, gemm1N] tile
    // as a single N block (gemm1BidGridLengthsForStore's n_block length is 1),
    // so the output transforms must span the full head dim (gemm1N), not the
    // per-chunk gemm1NPerBlock width used inside the GEMM1 chunk loop.
    FailureOr<ArrayAttr> maybeOutputViews = computeOutputTransforms(
        rewriter, loc, gemm1MPerBlock, gemm1N, gemm1BidGridLengthsForStore);

    if (failed(maybeOutputViews)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to compute output transforms\n");
      return failure();
    }

    ArrayAttr idToMatrixCMaps = maybeOutputViews.value();

    // Create StoreMarkerOp to mark the tile with output transforms for later
    // store lowering. The result type is the full tensor type so that fusion
    // ops can operate on it directly.
    auto storeMarkerOp = StoreMarkerOp::create(
        rewriter, loc, op.getResult().getType(), outAcc, idToMatrixCMaps,
        ValueRange{gridCoordsGemm1.g_block, gridCoordsGemm1.m_block,
                   gridCoordsGemm1.n_block});

    if (lse) {
      ArrayAttr lseMap = computeOutputLseTransforms(
          rewriter, loc, gemm1MPerBlock, gemm1BidGridLengthsForStore);

      auto lseStoreMarkerOp = StoreMarkerOp::create(
          rewriter, loc, op.getLse().getType(), lseOut, lseMap,
          ValueRange{gridCoordsGemm1.g_block, gridCoordsGemm1.m_block});
      rewriter.replaceOp(op, ValueRange{storeMarkerOp, lseStoreMarkerOp});
    } else {
      rewriter.replaceOp(op, storeMarkerOp.getResult());
    }

    return success();
  }
};

} // end anonymous namespace

void RockGridwiseAttnToBlockwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<GridwiseAttentionOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         affine::AffineDialect, scf::SCFDialect,
                         math::MathDialect, triton::TritonDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseAttentionRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
