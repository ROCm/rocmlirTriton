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
    Type elemType = rowType.getElementType();
    
    // Step 1: Expand dims [M] -> [M, 1]
    auto expandedType = RankedTensorType::get({numRows, 1}, elemType);
    Value expanded = triton::ExpandDimsOp::create(rewriter, loc, expandedType,
                                                    rowVector, 1);
    
    // Step 2: Broadcast [M, 1] -> [M, N]
    auto resultType = RankedTensorType::get({numRows, numCols}, elemType);
    return triton::BroadcastOp::create(rewriter, loc, resultType, expanded);
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

    return softmaxExp;
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

    return arith::DivFOp::create(rewriter, loc, attentionAcc, sumRowBroadcast);
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

  enum class OutOfScopeType { KVCache, Causal, PrefixCausal };

  Value setGemm0OutputOutOfScope(
      PatternRewriter &rewriter, Location loc, OutOfScopeType outOfScopeType,
      layout::GridCoordinates gridCoords, Value firstGemmResult,
      Value fakeTensorM, Value fakeTensorN, Value negInfTensor, ArrayAttr tileView, bool enabled,
      Value nLoopIV, Value gemm0NBlocksLastIter, Value currentSeqLen,
      Value prefixOffset) const {
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

  std::tuple<Value, Value, Value, Value, Value>
  getNLoopInfo(PatternRewriter &rewriter, Location loc,
               layout::AttnGridCoordinates gridCoordsGemm0,
               Value currentSeqLenTensor, Value prefixOffsetTensor,
               int64_t gemm0M, int64_t gemm0N, int64_t gemm0MPerBlock,
               int64_t gemm0NPerBlock, int64_t splitKV, bool isCausal,
               bool isKVCache, bool isPrefixCausal,
               IntegerAttr numRepeatsGQA = nullptr) const {
    Value gemm0NBlocksLastIter;
    Value currentSeqLen;
    Value prefixOffset;
    Value effectiveSeqLen;
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
      auto markerOp =
          LoadMarkerOp::create(rewriter, loc, resultType, tensorAddDim,
                               emptyViews, ValueRange{gridCoordsGemm0.g_block});

      return triton::UnsplatOp::create(rewriter, loc, markerOp);
    };

    // This is needed for KV Cache/Causal/Prefix Causal masking support
    if (isCausal || isKVCache || isPrefixCausal) {
      if (isKVCache) {
        currentSeqLen = loadTensorValue(currentSeqLenTensor);
        effectiveSeqLen = currentSeqLen;
      }

      if (isCausal || isPrefixCausal) {
        // Compute the last Q position in the block.
        // (nIndex + 1) * NPerBlock - 1.
        Value mIndex = gridCoordsGemm0.m_block;
        Value constGemm0MPerBlock = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), gemm0MPerBlock);
        Value one = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), 1);
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
          assert(isCausal && "isPrefixCausal requires isCausal");
          // For prefix causal: effective seq len = maxRowOfBlock + offset
          // This determines how many M-blocks we need to process
          prefixOffset = loadTensorValue(prefixOffsetTensor);
          maxRowOfBlock =
              arith::AddIOp::create(rewriter, loc, maxRowOfBlock, prefixOffset);
        }

        if (effectiveSeqLen) {
          // if effectiveSeqLen is set, it means KV Cache is enabled,
          // so we need to take the minimum of currentSeqLen and maxRowOfBlock
          maxRowOfBlock = arith::MinUIOp::create(rewriter, loc, currentSeqLen,
                                                 maxRowOfBlock);
        }

        // For prefix causal, adding prefix_offset can push maxRowOfBlock beyond
        // gemm0N. Similarly, when gemm0M > gemm0N, the last query position can
        // exceed the key sequence length. In both cases, bound by gemm0N - 1.
        if (gemm0M > gemm0N || isPrefixCausal) {
          // Bound by actual K dimension (key sequence length)
          Value gemm0NMinusOne = rewriter.createOrFold<arith::ConstantIntOp>(
              loc, rewriter.getI32Type(), gemm0N - 1);
          maxRowOfBlock = arith::MinUIOp::create(rewriter, loc, maxRowOfBlock,
                                                 gemm0NMinusOne);
        }

        effectiveSeqLen = maxRowOfBlock;
      }

      // compute end index
      Value constGemm0NPerBlock = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), gemm0NPerBlock);
      Value numerator = arith::AddIOp::create(rewriter, loc, effectiveSeqLen,
                                              constGemm0NPerBlock);
      end = rewriter.createOrFold<arith::DivUIOp>(loc, numerator,
                                                  constGemm0NPerBlock);
      Value one = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), 1);
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
                           prefixOffset);
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
    // The early-exit IfOp yields one accumulator per gemm1-N sub-tile
    // (gemm1NBlocks values, == 1 for attention / small GEG) optionally
    // followed by lseOut when softmax + LSE is enabled.
    assert(!elseYieldValues.empty() &&
           "early exit if must yield at least one outAcc value");

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

      auto markerOp = LoadMarkerOp::create(
          rewriter, loc, resultType, root, otherInputMap,
          ValueRange{gridCoords.g_block, gridCoords.m_block,
                     gridCoords.n_block});

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
    // Attention keeps the fused single-tile invariant (head dim stays small).
    // GEG/CEG may have a large output-N and tile it across an inner unrolled
    // gemm1-N loop; gemm1NBlocks is then > 1.
    // if (op.getEnableSoftmax()) {
    //   assert(gemm1N == gemm1NPerBlock &&
    //          "Attention requires gemm1NPerBlock == gemm1N");
    // } else {
    //   assert(gemm1N % gemm1NPerBlock == 0 &&
    //          "non-attention gemm1N must be a multiple of gemm1NPerBlock");
    // }
    int64_t gemm1NBlocks = gemm1N / gemm1NPerBlock;

    // params related to how we load Q
    bool prefetchQTile = gemm0K == gemm0KPerBlock;

    int64_t gemm1MBlocks = gemm1M / gemm1MPerBlock;
    assert(gemm1M % gemm1MPerBlock == 0);
    SmallVector<int64_t, 3> gemm0BidGridLengths = {gemm0G, gemm0MBlocks,
                                                   gemm0NBlocks};
    LLVM_DEBUG(llvm::dbgs()
               << "elemTypeQLoad: " << elemTypeQLoad << "\n"
               << "elemTypeKLoad: " << elemTypeKLoad << "\n"
               << "elemTypeVLoad: " << elemTypeVLoad << "\n");
    
    // Compute output transforms for this load
    FailureOr<ArrayAttr> maybeGemm0OutTileView = computeOutputTransforms(
        rewriter, loc, gemm0MPerBlock, gemm0NPerBlock, gemm0BidGridLengths);

    if (failed(maybeGemm0OutTileView))
      return op->emitError("Failed to compute output transforms");

    auto gemm0OutTileView = maybeGemm0OutTileView.value();

    SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
    // We need two different grid lengths because the V input tensor and the
    // output tensor have different shapes when splitKV > 1:
    // - V tensor shape: [gemm0G, seqK, headDim] - splitKV is NOT in the batch dim
    // - Output tensor shape: [gemm0G * splitKV, seqQ, headDim] - splitKV IS in
    //   the batch dim (each split writes to a separate slice of the output)
    // Therefore, loadTile for V uses gemm1BidGridLengths, while the output
    // store transforms use gemm1BidGridLengthsForStore.
    SmallVector<int64_t, 3> gemm1BidGridLengths = {gemm0G, gemm1MBlocks,
                                                   gemm1NBlocks};
    SmallVector<int64_t, 3> gemm1BidGridLengthsForStore = {
        gemm0G * splitKV, gemm1MBlocks, gemm1NBlocks};

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
    Value start, end;
    // get nLoop
    std::tie(start, end, gemm0NBlocksLastIter, currentSeqLen, prefixOffset) =
        getNLoopInfo(rewriter, loc, gridCoordsGemm0mIter0, currentSeqLenTensor,
                     prefixOffsetTensor, gemm0M, gemm0N, gemm0MPerBlock,
                     gemm0NPerBlock, splitKV, isCausal, isKVCache,
                     isPrefixCausal, op.getNumRepeatsGQAAttr());

    // Early exit: Skip all computation when there's no work but always write
    // output. The IfOp returns (outAcc_0, ..., outAcc_{gemm1NBlocks-1}, lseOut?)
    // so the code after the if always has defined values; the else branch
    // yields zero-initialized tensors for every gemm1-N sub-tile.
    SmallVector<Value> earlyExitElseValues;
    for (int64_t nb = 0; nb < gemm1NBlocks; ++nb) {
      Value initOutAcc = rock::createZeroAccBuffer(
          rewriter, loc, {gemm1MPerBlock, gemm1NPerBlock}, elemTypeOut);
      earlyExitElseValues.push_back(initOutAcc);
    }
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

      loadedQ =
          rock::loadTile(rewriter, loc, inQ, /*kiter=*/zero, "m",
                         gridCoordsGemm0LoadQ, gemm0KPerBlock, gemm0MPerBlock,
                         /*isKFirst=*/false, gemm0BidGridLengths);
    }

    Type accType = rock::getAccType(elemTypeQ, elemTypeK);
    Type gemm1AccType = rock::getAccType(elemTypeV, elemTypeV);

    // Create one initial accumulator per gemm1-N sub-tile. For attention
    // (gemm1NBlocks == 1) this is the single existing accumulator; for GEG/CEG
    // we thread `gemm1NBlocks` accs so the inner unrolled gemm1-N loop can
    // accumulate one register-resident sub-tile per iteration without touching
    // LDS. The second GEMM may have a different acc type (e.g. f32 vs i32
    // when Q/K are i8 but V is f16).
    SmallVector<Value> nLoopIterArgs;
    nLoopIterArgs.reserve(gemm1NBlocks + 2);
    for (int64_t nb = 0; nb < gemm1NBlocks; ++nb) {
      nLoopIterArgs.push_back(rock::createZeroAccBuffer(
          rewriter, loc, {gemm1MPerBlock, gemm1NPerBlock}, gemm1AccType));
    }
    nLoopIterArgs.push_back(maxRow);
    nLoopIterArgs.push_back(sumRow);

    Value one = rewriter.createOrFold<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), 1);
    scf::ForOp nLoopOp =
        scf::ForOp::create(rewriter, loc, start, end, one, nLoopIterArgs);
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(nLoopOp.getBody());
      int64_t kIterationsGemm0 = gemm0K / gemm0KPerBlock;
      // Convert loop IV to i32 for grid layout and load operations
      Value nLoopIV = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), nLoopOp.getInductionVar());
      // Get the iteration arguments. Per-sub-tile accumulators come first,
      // followed by maxRow/sumRow (used only when softmax is enabled, but
      // threaded through unconditionally to keep the iter_arg layout simple).
      SmallVector<Value> accs;
      accs.reserve(gemm1NBlocks);
      for (int64_t nb = 0; nb < gemm1NBlocks; ++nb)
        accs.push_back(nLoopOp.getRegionIterArg(nb));

      maxRow = nLoopOp.getRegionIterArg(gemm1NBlocks);
      sumRow = nLoopOp.getRegionIterArg(gemm1NBlocks + 1);

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
                             /*isKFirst=*/false, gemm0BidGridLengths);
        }

        Value loadedK =
            rock::loadTile(rewriter, loc, inK, /*kiter=*/kLoopIV, "n",
                           gridCoordsGemm0, gemm0KPerBlock, gemm0NPerBlock,
                           /*isKFirst=*/true, gemm0BidGridLengths);

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
            softmaxInput, fakeTensorM, fakeTensorN, negInfTensor, gemm0OutTileViewUnPadded,
            isKVCache, nLoopIV, gemm0NBlocksLastIter, currentSeqLen,
            /*prefixOffset=*/nullptr);

        // Causal masking: either prefix-causal or standard causal
        // Prefix causal: mask when key > (query + offset).
        // This combines causal masking with a prefix offset
        softmaxInput = setGemm0OutputOutOfScope(
            rewriter, loc, OutOfScopeType::PrefixCausal, gridCoordsGemm0,
            softmaxInput, fakeTensorM, fakeTensorN, negInfTensor, gemm0OutTileViewUnPadded,
            isPrefixCausal, nLoopIV, gemm0NBlocksLastIter,
            /*currentSeqLen=*/nullptr, prefixOffset);

        // Standard causal masking: mask when key > query
        softmaxInput = setGemm0OutputOutOfScope(
            rewriter, loc, OutOfScopeType::Causal, gridCoordsGemm0,
            softmaxInput, fakeTensorM, fakeTensorN, negInfTensor, gemm0OutTileViewUnPadded,
            isCausal && !isPrefixCausal, nLoopIV, gemm0NBlocksLastIter,
            /*currentSeqLen=*/nullptr,
            /*prefixOffset=*/nullptr);

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

      // Statically unrolled inner loop over the gemm1-N sub-tiles. For
      // attention this iterates once (gemm1NBlocks == 1) and the IR is
      // identical to before. For GEG/CEG we run one BlockwiseGemmOp per
      // sub-tile against the same P, reusing the gemm0 work.
      SmallVector<Value> newAccs;
      newAccs.reserve(gemm1NBlocks);
      for (int64_t nb = 0; nb < gemm1NBlocks; ++nb) {
        // For softmax/attention, each nLoop iteration starts gemm1 from zero
        // because flash-attention corrections handle cross-iteration
        // accumulation. For GEG (no softmax), we accumulate directly into
        // accs[nb] across nLoop iterations.
        Value gemm1InitAcc;
        if (op.getEnableSoftmax()) {
          gemm1InitAcc = rock::createZeroAccBuffer(
              rewriter, loc, {gemm1MPerBlock, gemm1NPerBlock}, gemm1AccType);
        } else {
          gemm1InitAcc = accs[nb];
        }

        Value nBlockConst = rewriter.createOrFold<arith::ConstantIntOp>(
            loc, rewriter.getI32Type(), nb);
        auto gridCoordsGemm1Tile = layout::makeGxNGridLayout(
            rewriter, loc, bid, gemm1MBlocks, nBlockConst, gridSize, arch,
            rock::getNumChipletsValue(op), splitKVConst);

        Value loadedV =
            rock::loadTile(rewriter, loc, inV,
                           /*kIter=*/nLoopIV, "n", gridCoordsGemm1Tile,
                           gemm1KPerBlock, gemm1NPerBlock,
                           /*isKFirst=*/true, gemm1BidGridLengths);

        Value gemm1Out = BlockwiseGemmOp::create(
            rewriter, loc, gemm0Out, loadedV, gemm1InitAcc,
            /*matrixScaleA=*/nullptr, /*matrixScaleB=*/nullptr,
            /*quantBlockSize=*/nullptr,
            /*matrixAOrigElemType=*/nullptr, /*matrixBOrigElemType=*/nullptr,
            /*matrixAKPack=*/nullptr, /*matrixBKPack=*/nullptr);

        // Apply flash attention correction. Only valid when softmax is
        // enabled, which guarantees gemm1NBlocks == 1 here.
        if (op.getEnableSoftmax()) {
          newAccs.push_back(createAttentionRowStateCorrections(
              rewriter, loc, gemm1Out, accs[nb], maxRowDiffExp));
        } else {
          newAccs.push_back(gemm1Out);
        }
      }

      // Yield the updated accumulators followed by maxRow/sumRow.
      SmallVector<Value> yieldVals(newAccs.begin(), newAccs.end());
      yieldVals.push_back(maxRow);
      yieldVals.push_back(sumRow);
      scf::YieldOp::create(rewriter, loc, yieldVals);
    }
    SmallVector<Value> outAccs;
    outAccs.reserve(gemm1NBlocks);
    for (int64_t nb = 0; nb < gemm1NBlocks; ++nb)
      outAccs.push_back(nLoopOp.getResult(nb));
    maxRow = nLoopOp.getResult(gemm1NBlocks);
    sumRow = nLoopOp.getResult(gemm1NBlocks + 1);

    if (op.getEnableSoftmax()) {
      // Softmax invariant: gemm1NBlocks == 1.
      outAccs[0] = scaleFinalOutput(rewriter, loc, outAccs[0], sumRow);
    }
    for (int64_t nb = 0; nb < gemm1NBlocks; ++nb) {
      if (cast<ShapedType>(outAccs[nb].getType()).getElementType() !=
          elemTypeOut) {
        auto outAccTensorType = cast<RankedTensorType>(outAccs[nb].getType());
        auto destType =
            RankedTensorType::get(outAccTensorType.getShape(), elemTypeOut);
        outAccs[nb] =
            createTypeConversionOp(rewriter, loc, outAccs[nb], destType);
      }
    }
    Value lseOut;
    if (lse) {
      // it must be guaranteed by the verifier
      assert(op.getEnableSoftmax());
      lseOut = computeLse(rewriter, loc, lseType, sumRow, maxRow);
    }

    // When early exit is enabled, the IfOp returns
    // (outAcc_0, ..., outAcc_{gemm1NBlocks-1}, lseOut?). Yield from the then
    // region and then take results from the if.
    if (earlyExitIf.has_value()) {
      rewriter.setInsertionPointToEnd(&earlyExitIf->getThenRegion().front());
      SmallVector<Value> thenYieldVals(outAccs.begin(), outAccs.end());
      if (lse)
        thenYieldVals.push_back(lseOut);
      scf::YieldOp::create(rewriter, loc, thenYieldVals);
      rewriter.setInsertionPointAfter(*earlyExitIf);
      for (int64_t nb = 0; nb < gemm1NBlocks; ++nb)
        outAccs[nb] = earlyExitIf->getResult(nb);
      if (lse)
        lseOut = earlyExitIf->getResult(gemm1NBlocks);
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: early exit enabled - "
                 << "output writes will execute unconditionally\n");
    }

    // Compute output transforms - use grid lengths with splitKV for output.
    // The output view's n_block axis now has gemm1NBlocks lanes, so each
    // store_marker addresses a disjoint sub-tile via its constant n_block.
    FailureOr<ArrayAttr> maybeOutputViews = computeOutputTransforms(
        rewriter, loc, gemm1MPerBlock, gemm1NPerBlock,
        gemm1BidGridLengthsForStore);

    if (failed(maybeOutputViews)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to compute output transforms\n");
      return failure();
    }

    ArrayAttr idToMatrixCMaps = maybeOutputViews.value();

    // Note that we don't use splitKV here because that dimension belongs to
    // the batch size already for output tensors.
    SmallVector<Value> storeMarkers;
    storeMarkers.reserve(gemm1NBlocks);
    for (int64_t nb = 0; nb < gemm1NBlocks; ++nb) {
      Value nBlockConst = rewriter.createOrFold<arith::ConstantIntOp>(
          loc, rewriter.getI32Type(), nb);
      auto gridCoordsGemm1Store = layout::makeGxNGridLayout(
          rewriter, loc, bid, gemm1MBlocks, nBlockConst, gridSize, arch,
          rock::getNumChipletsValue(op));

      // Create StoreMarkerOp to mark the tile with output transforms for later
      // store lowering. The result type is the full tensor type so that
      // fusion ops can operate on it directly. Multiple store_markers with
      // distinct n_block constants will be lowered by LowerStores into
      // disjoint blockwise_stores covering the full gemm1-N axis.
      auto storeMarkerOp = StoreMarkerOp::create(
          rewriter, loc, op.getResult().getType(), outAccs[nb], idToMatrixCMaps,
          ValueRange{gridCoordsGemm1Store.g_block,
                     gridCoordsGemm1Store.m_block,
                     gridCoordsGemm1Store.n_block});
      storeMarkers.push_back(storeMarkerOp.getResult());
    }

    if (lse) {
      // LSE has no gemm1-N axis; it is keyed on (g_block, m_block) only and
      // only exists when softmax is enabled (which forces gemm1NBlocks == 1).
      assert(gemm1NBlocks == 1);
      auto gridCoordsGemm1Lse = layout::makeGxNGridLayout(
          rewriter, loc, bid, gemm1MBlocks, zero, gridSize, arch,
          rock::getNumChipletsValue(op));
      ArrayAttr lseMap = computeOutputLseTransforms(
          rewriter, loc, gemm1MPerBlock, gemm1BidGridLengthsForStore);

      auto lseStoreMarkerOp = StoreMarkerOp::create(
          rewriter, loc, op.getLse().getType(), lseOut, lseMap,
          ValueRange{gridCoordsGemm1Lse.g_block, gridCoordsGemm1Lse.m_block});
      rewriter.replaceOp(op,
                         ValueRange{storeMarkers[0], lseStoreMarkerOp});
      return success();
    }

    if (gemm1NBlocks == 1) {
      // Single-tile path (attention without LSE, or small GEG): byte-identical
      // to the previous lowering.
      rewriter.replaceOp(op, storeMarkers.front());
      return success();
    }

    // Multi-tile GEG/CEG path: each store_marker writes a disjoint sub-tile of
    // the output. RockLowerStores walks rock.store ops and traces back to a
    // single store_marker per store, so we need one rock.store per marker.
    // All clones write to the *same* destination view (so source/dest shapes
    // agree); only the last clone's result feeds func.return, while the
    // earlier clones have unused results. rock.store has no Pure trait and
    // no memory-effect interface, so RemoveDeadValuesPass preserves them.
    rock::StoreOp consumerStore = nullptr;
    for (Operation *user : op.getResult().getUsers()) {
      if (auto storeOp = dyn_cast<rock::StoreOp>(user)) {
        if (consumerStore)
          return op->emitOpError(
              "expected exactly one rock.store consumer of the gridwise "
              "result when tiling gemm1-N; multi-store fusion is not yet "
              "supported");
        consumerStore = storeOp;
      } else {
        return op->emitOpError(
            "tiling gemm1-N requires a direct rock.store consumer; found "
            "intervening op which is not yet supported");
      }
    }
    if (!consumerStore)
      return op->emitOpError(
          "tiling gemm1-N requires at least one rock.store consumer");

    rewriter.setInsertionPoint(consumerStore);
    Value finalResult;
    for (int64_t nb = 0; nb < gemm1NBlocks; ++nb) {
      auto cloned = rock::StoreOp::create(
          rewriter, consumerStore.getLoc(), consumerStore.getResult().getType(),
          storeMarkers[nb], consumerStore.getDest(),
          consumerStore.getStoreMethodAttr());
      if (nb == gemm1NBlocks - 1)
        finalResult = cloned.getResult();
    }
    rewriter.replaceOp(consumerStore, finalResult);
    rewriter.eraseOp(op);
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
