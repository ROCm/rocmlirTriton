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

    auto tileShape = cast<ShapedType>(firstGemmResult.getType()).getShape();
    auto pointerTensorType = RankedTensorType::get(tileShape, rewriter.getI32Type());
    auto maskTensorType = RankedTensorType::get(tileShape, rewriter.getI1Type());
    auto transformsToPtrOp = TransformsToPtrOp::create(
        rewriter, loc, pointerTensorType, maskTensorType, fakeTensor, ValueRange{gridCoords.g_block, gridCoords.m_block,
            gridCoords.n_block});
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

      auto tileShape = cast<ShapedType>(firstGemmResult.getType()).getShape();
      auto pointerTensorType =
          RankedTensorType::get(tileShape, rewriter.getI32Type());
      auto maskTensorType =
          RankedTensorType::get(tileShape, rewriter.getI1Type());

      auto transformsToPtrN = TransformsToPtrOp::create(
          rewriter, loc, pointerTensorType, maskTensorType, fakeTensorNview,
          ValueRange{gridCoords.g_block, gridCoords.m_block,
                     gridCoords.n_block});
      Value nIndex = transformsToPtrN.getPointers();

      auto transformsToPtrM = TransformsToPtrOp::create(
          rewriter, loc, pointerTensorType, maskTensorType, fakeTensorMview,
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
        Value numerator =
            arith::AddIOp::create(rewriter, loc, end, constSplitKVM1);
        Value gemm0NIterations =
            rewriter.createOrFold<arith::DivUIOp>(loc, numerator, constSplitKV);

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

  // Trace the chain of rock.transform ops on the QK block argument and
  // collect them for skipping. These transforms reshape the GEMM result in
  // the original full-tensor shape space (e.g. adding a unit batch dim)
  // and are meaningless in tile space. Only pure reshape transforms
  // (PassThrough, AddDim, Unmerge, Merge) are allowed; anything else
  // (Pad, Broadcast, etc.) is rejected.
  LogicalResult skipQKArgTransformChain(
      GridwiseAttentionOp op, BlockArgument qkArg, Value gemm0Mapped,
      DenseSet<Operation *> &opsToSkip, IRMapping &mapping,
      SmallVectorImpl<TransformMapAttr> &qkTransformAttrs) const {
    Value chainEnd = qkArg;
    while (chainEnd.hasOneUse()) {
      Operation *user = *chainEnd.getUsers().begin();
      auto transformOp = dyn_cast<TransformOp>(user);
      if (!transformOp)
        break;
      for (TransformAttr tr : transformOp.getTransform().getOps()) {
        switch (tr.getType()) {
        case TransformType::PassThrough:
        case TransformType::AddDim:
        case TransformType::Unmerge:
        case TransformType::Merge:
          break;
        default:
          return op->emitOpError()
                 << "preSoftmaxBody QK argument has a non-reshape "
                    "transform; only PassThrough, AddDim, Unmerge, and "
                    "Merge are supported, but found: "
                 << transformOp.getTransform();
        }
      }
      opsToSkip.insert(user);
      qkTransformAttrs.push_back(transformOp.getTransform());
      chainEnd = transformOp.getResult();
    }
    if (!chainEnd.use_empty()) {
      for (Operation *user : chainEnd.getUsers()) {
        if (isa<TransformOp>(user))
          return op->emitOpError()
                 << "preSoftmaxBody QK argument has a transform chain "
                    "that is not purely linear (multi-use or branching); "
                    "this is not supported";
      }
    }
    mapping.map(chainEnd, gemm0Mapped);
    return success();
  }

  // Map external splat constants referenced by the region body to
  // tile-shaped replacements. Constants like scale factors or mask fill
  // values may be defined at function scope and captured by closure.
  LogicalResult mapExternalSplatConstants(PatternRewriter &rewriter,
                                          Location loc,
                                          GridwiseAttentionOp op, Block &block,
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
        auto origType = dyn_cast<RankedTensorType>(constOp.getType());
        if (!origType)
          continue;
        auto splatAttr = dyn_cast<SplatElementsAttr>(constOp.getValue());
        if (!splatAttr)
          return op->emitOpError()
                 << "non-splat constant in preSoftmaxBody cannot be tiled";
        auto newType = RankedTensorType::get(tileType.getShape(),
                                             origType.getElementType());
        Value newConst = arith::ConstantOp::create(
            rewriter, loc,
            SplatElementsAttr::get(newType,
                                   splatAttr.getSplatValue<Attribute>()));
        mapping.map(operand, newConst);
      }
    }
    return success();
  }

  // Apply pre-softmax element-wise fusions to the first GEMM (Q*K) output.
  // Clones the ops from the op's preSoftmaxBody region into the current IR,
  // mapping block arguments to tile-level values: arg0 is the gemm0 result
  // and remaining args are extra element-wise inputs loaded via LoadMarkerOps.
  // Returns the (possibly fused) result, or the original buffer if no fusions
  // are present.
  FailureOr<Value> postProcessFirstGemm(PatternRewriter &rewriter, Location loc,
                                        GridwiseAttentionOp op,
                                        layout::GridCoordinates gridCoords,
                                        Value srcGemm0OutBuffer,
                                        ArrayAttr gemm0OutViews) const {
    Region &region = op.getPreSoftmaxBody();
    if (region.empty())
      return srcGemm0OutBuffer;

    Block &block = region.front();

    // If there are no fusion ops in the region, nothing to do.
    if (block.without_terminator().empty())
      return srcGemm0OutBuffer;

    // Build the mapping from block arguments to tile-level values.
    IRMapping mapping;
    auto tileType = cast<RankedTensorType>(srcGemm0OutBuffer.getType());

    // Block arg 0 is always the QK product (enforced by
    // ElementwiseRegionFinder in TosaToRock). Extra element-wise inputs
    // follow at args 1..N.
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

    // Collect transform ops from the body that should be folded into
    // LoadMarkerOp views (for extra inputs) or simply skipped (for the QK
    // arg) rather than cloned with stale full-tensor shape metadata.
    DenseSet<Operation *> bodyTransformOpsToSkip;
    SmallVector<TransformMapAttr> qkTransformAttrs;

    if (failed(skipQKArgTransformChain(op, qkArg, gemm0Mapped,
                                       bodyTransformOpsToSkip, mapping,
                                       qkTransformAttrs)))
      return failure();

    // The QK transforms above reshape the GEMM output into the body's
    // working shape (e.g. AddDim turns [2,5,5] into [2,1,5,5]). Extra-input
    // body transforms also end in that working shape, but gemm0OutViews
    // only reach the GEMM output shape. To close the gap in the view chain
    // (gemm0OutViews -> ??? -> body transforms), we insert the inverse of
    // each QK transform. For example, if the QK transform is:
    //   AddDim: lower=[2,5,5] -> upper=[2,1,5,5]
    // its inverse is:
    //   ConstDim: lower=[2,1,5,5] -> upper=[2,5,5]
    // which maps the body's 4-D working coords back to the 3-D GEMM output
    // coords that gemm0OutViews expect.
    SmallVector<Attribute> invertedQKAttrs;
    for (TransformMapAttr qkAttr : qkTransformAttrs) {
      TransformMapAttr inv = invertTransformMap(rewriter, qkAttr, loc);
      if (!inv)
        return op->emitOpError()
               << "failed to invert QK argument transform: " << qkAttr;
      invertedQKAttrs.push_back(inv);
    }

    ValueRange extraInputs = op.getPreSoftmaxElemWiseInputs();
    unsigned numExtraArgs = block.getNumArguments() - 1;
    if (extraInputs.size() != numExtraArgs)
      return op->emitOpError()
             << "preSoftmaxBody has " << numExtraArgs
             << " non-QK block argument(s) but op has "
             << extraInputs.size() << " preSoftmaxElemWiseInputs";

    for (unsigned i = 0; i < extraInputs.size(); ++i) {
      Value globalInput = extraInputs[i];
      Value root;
      ArrayAttr globalInputMaps;
      std::tie(root, globalInputMaps, std::ignore) =
          untransform(rewriter, globalInput);

      auto resultType = RankedTensorType::get(
          tileType.getShape(),
          cast<ShapedType>(globalInput.getType()).getElementType());

      // Trace the chain of rock.transform ops on the block argument.
      // These transforms bridge from the gemm0 output shape to the actual
      // input tensor shape (e.g. broadcast [1,64,64]->[1,1,1] then reshape
      // [1,1,1]->[1] for a scalar input). They must be part of the
      // LoadMarkerOp's views so the view chain reaches the input tensor.
      BlockArgument blockArg = block.getArgument(i + 1);
      Value chainEnd = blockArg;
      SmallVector<Attribute> bodyTransformAttrs;

      // The hasOneUse() condition means we only follow purely linear chains
      // where each value feeds exactly one consumer. This works for all
      // bodies produced by the current pipeline (ElementwiseRegionFinder
      // generates linear reshape chains, converted to rock.transform by
      // ViewToTransform). It will not handle:
      //   - An intermediate transform result consumed by both the next
      //     transform and an elementwise op (multi-use). The chain would
      //     stop early, leaving later transforms to be incorrectly cloned
      //     with mismatched shape metadata.
      //   - A value consumed by two different TransformOps (branching).
      while (chainEnd.hasOneUse()) {
        Operation *user = *chainEnd.getUsers().begin();
        auto transformOp = dyn_cast<TransformOp>(user);
        if (!transformOp)
          break;
        bodyTransformAttrs.push_back(transformOp.getTransform());
        bodyTransformOpsToSkip.insert(user);
        chainEnd = transformOp.getResult();
      }

      // If the chain stopped at a multi-use value that still has a
      // TransformOp consumer, we have either a multi-use intermediate
      // or a branching chain. Both would leave transforms unfolded and
      // incorrectly cloned with mismatched shape metadata.
      if (!chainEnd.use_empty()) {
        for (Operation *user : chainEnd.getUsers()) {
          if (isa<TransformOp>(user))
            return op->emitOpError()
                   << "preSoftmaxBody block argument " << (i + 1)
                   << " has a transform chain that is not purely linear "
                      "(multi-use or branching); this is not supported";
        }
      }

      // Build the complete view chain for the LoadMarkerOp.  The chain
      // (applied in reverse array order) maps from the raw input tensor
      // up to the tile shape:
      //   tile <- gemm0OutViews <- invertedQK <- bodyTrans <- globalInputMaps <- raw
      // Each segment is optional and contributes nothing when empty.
      SmallVector<Attribute> allViews(gemm0OutViews.begin(),
                                      gemm0OutViews.end());
      if (!bodyTransformAttrs.empty()) {
        allViews.append(invertedQKAttrs.begin(), invertedQKAttrs.end());
        for (auto it = bodyTransformAttrs.rbegin();
             it != bodyTransformAttrs.rend(); ++it)
          allViews.push_back(*it);
      }
      if (!globalInputMaps.empty())
        allViews.append(globalInputMaps.begin(), globalInputMaps.end());
      ArrayAttr otherInputMap = rewriter.getArrayAttr(allViews);

      auto markerOp = LoadMarkerOp::create(
          rewriter, loc, resultType, root, otherInputMap,
          ValueRange{gridCoords.g_block, gridCoords.m_block,
                     gridCoords.n_block});

      // Map the end of the transform chain (or the block arg itself if no
      // transforms) so that downstream uses get the load_marker result.
      mapping.map(chainEnd, markerOp.getResult());
    }

    if (failed(mapExternalSplatConstants(rewriter, loc, op, block, tileType,
                                         mapping)))
      return failure();

    // Clone ops from the region body (except the terminator and any
    // transform ops already folded into LoadMarkerOp views), fixing up
    // result types to use tile-level shapes while preserving element types
    // (important for casts like arith.extf / arith.truncf).
    for (Operation &bodyOp : block.without_terminator()) {
      if (bodyTransformOpsToSkip.contains(&bodyOp))
        continue;
      Operation *cloned = rewriter.clone(bodyOp, mapping);

      // Splat constants defined inside the region body must also be resized.
      if (auto constOp = dyn_cast<arith::ConstantOp>(cloned)) {
        auto origType = dyn_cast<RankedTensorType>(constOp.getType());
        if (!origType)
          continue;
        auto splatAttr = dyn_cast<SplatElementsAttr>(constOp.getValue());
        if (!splatAttr)
          return op->emitOpError()
                 << "non-splat constant in preSoftmaxBody cannot be tiled";
        auto newType = RankedTensorType::get(tileType.getShape(),
                                             origType.getElementType());
        constOp.setValueAttr(SplatElementsAttr::get(
            newType, splatAttr.getSplatValue<Attribute>()));
        constOp.getResult().setType(newType);
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

    // The yield operand (mapped) is the fused result.
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
    assert(gemm1N == gemm1NPerBlock &&
           "Current limitation, gemm1NPerBlock has to be equal to gemm1N");

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
    SmallVector<int64_t, 3> gemm1BidGridLengths = {gemm0G, gemm1MBlocks, 1};
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
    auto gridCoordsGemm0mIter0 = layout::makeGxNGridLayout(
        rewriter, loc, bid, gemm0MBlocks,
        rewriter.createOrFold<arith::ConstantIntOp>(loc, rewriter.getI32Type(),
                                                    0),
        gridSize, arch, rock::getNumCUValue(op), splitKVConst);

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
    // output. The IfOp returns (outAcc, lseOut?) so the code after the if
    // always has defined values; the else branch yields zero-initialized
    // tensors.
    SmallVector<Value> earlyExitElseValues;
    auto initOutAcc = rock::createZeroAccBuffer(
        rewriter, loc, {gemm1MPerBlock, gemm1NPerBlock}, elemTypeOut);
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
    // TODO(roctriton): do this in an independent pass, hoist loads out of the loop if possible
    if (prefetchQTile) {
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: gemm0K is equal to gemm0KPerBlock\n");
      LLVM_DEBUG(llvm::dbgs()
                 << "rock.attention: Prefetching Q tile into regs...\n");

      // it is fine m iteration to be zero as it irrelevant to Q tensor
      // as the first gemm is Kt x Qt.
      auto gridCoordsGemm0LoadQ = layout::makeGxNGridLayout(
          rewriter, loc, bid, gemm0MBlocks, zero, gridSize, arch,
          rock::getNumCUValue(op), splitKVConst);

      loadedQ =
          rock::loadTile(rewriter, loc, inQ, /*kiter=*/zero, "m",
                         gridCoordsGemm0LoadQ, gemm0KPerBlock, gemm0MPerBlock,
                         /*isKFirst=*/false, gemm0BidGridLengths);
    }

    Type accType = rock::getAccType(elemTypeQ, elemTypeK);
    Type gemm1AccType = rock::getAccType(elemTypeV, elemTypeV);

    // Create initial accumulator for nLoop with the shape of the final output
    // tile. The second GEMM multiplies softmax output (cast to elemTypeV) by V,
    // so its accumulator type may differ from the first GEMM's (e.g. f32 vs i32
    // when Q/K are i8 but V is f16).
    Value initNLoopAcc = rock::createZeroAccBuffer(
        rewriter, loc, {gemm1MPerBlock, gemm1NPerBlock}, gemm1AccType);

    Value one = rewriter.createOrFold<arith::ConstantIntOp>(
        loc, rewriter.getI32Type(), 1);
    scf::ForOp nLoopOp =
        scf::ForOp::create(rewriter, loc, start, end, one,
                           ValueRange{initNLoopAcc, maxRow, sumRow});
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(nLoopOp.getBody());
      int64_t kIterationsGemm0 = gemm0K / gemm0KPerBlock;
      // Convert loop IV to i32 for grid layout and load operations
      Value nLoopIV = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getI32Type(), nLoopOp.getInductionVar());
      // Get the iteration arguments
      Value attentionAcc = nLoopOp.getRegionIterArg(0);

      maxRow = nLoopOp.getRegionIterArg(1);
      sumRow = nLoopOp.getRegionIterArg(2);

      layout::GridCoordinates gridCoordsGemm0 = layout::makeGxNGridLayout(
          rewriter, loc, bid, gemm0MBlocks, nLoopIV, gridSize, arch,
          rock::getNumCUValue(op), splitKVConst);
      Value initAcc = rock::createZeroAccBuffer(
          rewriter, loc, {gemm0MPerBlock, gemm0NPerBlock}, accType);

      Value endKLoop =
          rewriter.createOrFold<arith::ConstantIntOp>(loc, rewriter.getI32Type(), kIterationsGemm0);
      scf::ForOp kLoopOp = scf::ForOp::create(rewriter, loc, zero, endKLoop,
                                              one, ValueRange{initAcc});
      {
        PatternRewriter::InsertionGuard guard(rewriter);
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

        // TODO(roctriton): scaled gemm
        Value newAcc = BlockwiseGemmOp::create(
            rewriter, loc, accArg.getType(), loadedQ, loadedK, accArg,
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
      int64_t splitKV = op.getSplitKV();
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
        auto softmaxTensorType = RankedTensorType::get({softmaxShape[0]}, elemTypeSoftmax);

        // Softmax max reduction
        Value softmaxMax = BlockwiseReduceOp::create(
            rewriter, loc, softmaxTensorType, softmaxInput, reductionAxis, rewriter.getAttr<rock::ReduceMethodAttr>(rock::ReduceMethod::Max));

        softmaxExp = expSubstractMaxFromGemm0(rewriter, loc, softmaxInput,
                                 softmaxMax, maxRow);

        // Softmax sum reduction
        Value softmaxSum = BlockwiseReduceOp::create(
            rewriter, loc, softmaxTensorType, softmaxExp, reductionAxis, rewriter.getAttr<rock::ReduceMethodAttr>(rock::ReduceMethod::Sum));
            
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
      Value gemm1InitAcc;
      if (op.getEnableSoftmax()) {
        gemm1InitAcc = rock::createZeroAccBuffer(
            rewriter, loc, {gemm1MPerBlock, gemm1NPerBlock}, gemm1AccType);
      } else {
        gemm1InitAcc = attentionAcc;
      }

      auto gridCoordsGemm1 = layout::makeGxNGridLayout(
          rewriter, loc, bid, gemm1MBlocks, zero, gridSize, arch,
          rock::getNumCUValue(op), splitKVConst);

      Value loadedV = rock::loadTile(rewriter, loc, inV,
                                     /*kIter=*/nLoopIV, "n", gridCoordsGemm1,
                                     gemm1KPerBlock, gemm1NPerBlock,
                                     /*isKFirst=*/true, gemm1BidGridLengths);

      // TODO(roctriton): scaled gemm
      Value gemm1Out = BlockwiseGemmOp::create(
          rewriter, loc, gemm1InitAcc.getType(), gemm0Out, loadedV,
          gemm1InitAcc, /*matrixScaleA=*/nullptr, /*matrixScaleB=*/nullptr,
          /*quantBlockSize=*/nullptr,
          /*matrixAOrigElemType=*/nullptr, /*matrixBOrigElemType=*/nullptr,
          /*matrixAKPack=*/nullptr, /*matrixBKPack=*/nullptr);

      // Apply flash attention correction
      if (op.getEnableSoftmax()) {
        attentionAcc = createAttentionRowStateCorrections(
            rewriter, loc, gemm1Out, attentionAcc, maxRowDiffExp);
      } else {
        attentionAcc = gemm1Out;
      }
        
        // Yield the updated accumulators (attentionAcc, maxRow, sumRow)
        scf::YieldOp::create(rewriter, loc, ValueRange{attentionAcc, maxRow, sumRow});
    }
    Value outAcc = nLoopOp.getResult(0);
    maxRow = nLoopOp.getResult(1);
    sumRow = nLoopOp.getResult(2);

    if (op.getEnableSoftmax()) {
        outAcc = scaleFinalOutput(rewriter, loc, outAcc,
                            sumRow);
    }
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
    auto gridCoordsGemm1 =
        layout::makeGxNGridLayout(rewriter, loc, bid, gemm1MBlocks, zero,
                                  gridSize, arch, rock::getNumCUValue(op));

    // Compute output transforms - use grid lengths with splitKV for output
    FailureOr<ArrayAttr> maybeOutputViews = computeOutputTransforms(
        rewriter, loc, gemm1MPerBlock, gemm1NPerBlock, gemm1BidGridLengthsForStore);

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
