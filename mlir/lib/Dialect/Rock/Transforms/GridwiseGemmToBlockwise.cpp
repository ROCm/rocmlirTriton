//===- GridwiseGemmToBlockwise - MLIR Rock ops lowering passes -----===//
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
// ============================================================
//
// This pass converts rock.gridwise_gemm
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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "GridLayoutEmitter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <optional>
#include <tuple>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKGRIDWISEGEMMTOBLOCKWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-gridwise-gemm-to-blockwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockGridwiseGemmToBlockwisePass
    : public rock::impl::RockGridwiseGemmToBlockwisePassBase<
          RockGridwiseGemmToBlockwisePass> {
  void runOnOperation() override;
};

} // end anonymous namespace

static scf::ForOp createMainLoop(PatternRewriter &rewriter, Location loc,
                                 Value end, ValueRange iterArgs) {
  Value one = rewriter.createOrFold<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), 1);
  Value start = rewriter.createOrFold<arith::ConstantIntOp>(
      loc, rewriter.getI32Type(), 0);
  scf::ForOp loopOp =
      scf::ForOp::create(rewriter, loc, start, end, one, iterArgs);
  return loopOp;
}

// Pick cache modifiers for the A and B GEMM operand loads based on reuse: A is
// reused nBlocks times, B mBlocks times. If a dimension is skinny (one block),
// the operand along the *other* dimension has no reuse, so stream it (CS) to
// avoid evicting the reused operand -- but only under cache pressure (both
// operands don't fit in the LLC), otherwise nothing is evicted anyway.
//
// An operand whose load reloads data (non-injective view: conv im2col, a
// broadcast, ...) relies on caching for its repeated reads, so it is never
// streamed regardless of skinniness (aReloads/bReloads).
//
// NOTE: This runs at the load_marker stage and assumes a single A and B. Once
// fusion is involved there may be several inputs; consider moving this
// heuristic after LowerLoads (which materializes the actual blockwise loads).
static std::pair<rock::CacheModifier, rock::CacheModifier>
chooseGemmLoadCacheModifiers(StringRef arch, Type aElemType, Type bElemType,
                             int64_t G, int64_t M, int64_t N, int64_t K,
                             int64_t mBlocks, int64_t nBlocks, bool aReloads,
                             bool bReloads) {
  const int64_t llcBytes = rock::getLastLevelCacheSize(arch);
  auto bytesOf = [](int64_t numElems, Type elemType) -> int64_t {
    return llvm::divideCeil(numElems * elemType.getIntOrFloatBitWidth(), 8);
  };
  const int64_t aBytes = bytesOf(G * M * K, aElemType);
  const int64_t bBytes = bytesOf(G * N * K, bElemType);

  constexpr int64_t kSkinnyBlockThreshold = 2;
  const bool cachePressure = (aBytes + bBytes) > llcBytes;

  rock::CacheModifier cacheA = rock::CacheModifier::NONE;
  rock::CacheModifier cacheB = rock::CacheModifier::NONE;
  if (cachePressure) {
    if (mBlocks < kSkinnyBlockThreshold && !bReloads) // M skinny -> stream B
      cacheB = rock::CacheModifier::CS;
    if (nBlocks < kSkinnyBlockThreshold && !aReloads) // N skinny -> stream A
      cacheA = rock::CacheModifier::CS;
  }

  return {cacheA, cacheB};
}

// Emit the K reduction for one (m, n) output tile of `op` and return the final
// accumulator tile.
//
// `kPerBlock` is a power of two, but K need not be a multiple of it. The
// power-of-two-tiled region [0, kMain), where kMain = (K / kPerBlock) *
// kPerBlock, is handled by the main reduction loop -- one blockwise_gemm of
// width kPerBlock per iteration, which is exactly the fast path emitted when K
// already divides evenly. The leftover tail [kMain, K) is *peeled*: it is split
// into power-of-two K segments (rock::decomposePow2) and each segment is
// contracted once, after the loop, threading the same register accumulator.
// This keeps the common case untouched and pays the multi-segment cost only on
// the single tail, instead of widening every iteration to a non-power-of-two
// kPerBlock.
static Value emitKReductionLoop(PatternRewriter &b, Location loc,
                                GridwiseGemmOp op, Value matA, Value matB,
                                Value scaleA, Value scaleB, Value initAcc,
                                rock::layout::GridCoordinates gridCoords,
                                SmallVector<int64_t, 3> &bidGridLengths,
                                rock::CacheModifier cacheA,
                                rock::CacheModifier cacheB,
                                int64_t quantKPerBlock) {
  int64_t K = cast<ShapedType>(matA.getType()).getShape()[2];
  GemmParamsAttr params = op.getParams();
  int64_t kPerBlock = params.getKPerBlock();
  int64_t mPerBlock = params.getMPerBlock();
  int64_t nPerBlock = params.getNPerBlock();
  bool isScaledGemm = scaleA && scaleB;

  int64_t kIterations = K / kPerBlock;
  int64_t kMain = kIterations * kPerBlock;
  int64_t kRem = K - kMain;

  // Slice the contiguous K range [start, start + len) out of a gridwise operand
  // (dim 2 of A [G, M, K], dim 1 of B [G, K, N]). A range that spans the whole
  // K dimension returns the operand untouched, so the power-of-two path emits
  // no transforms at all.
  auto sliceK = [&](Value mat, unsigned kDim, int64_t start,
                    int64_t len) -> Value {
    int64_t kSize = cast<ShapedType>(mat.getType()).getShape()[kDim];
    if (start == 0 && len == kSize)
      return mat;
    return sliceBlockedDims(b, loc, mat, /*sliceDims=*/{kDim}, /*blocks=*/{1},
                            /*tiles=*/{kSize}, {Pow2Segment{start, len}});
  };

  // Emit the A/B (and, for scaled GEMMs, scale) loads and the blockwise_gemm
  // for one K tile of width `segLen` at K block `kIter` of the given operand
  // views, reducing into `acc`.
  auto contract = [&](Value aView, Value bView, Value kIter, int64_t segLen,
                      Value acc) -> Value {
    Value loadedB =
        rock::loadTile(b, loc, bView, kIter, "n", gridCoords, segLen, nPerBlock,
                       /*isKFirst=*/true, bidGridLengths, cacheB);
    Value loadedA =
        rock::loadTile(b, loc, aView, kIter, "m", gridCoords, segLen, mPerBlock,
                       /*isKFirst=*/false, bidGridLengths, cacheA);

    Value loadedScaleA, loadedScaleB;
    if (isScaledGemm) {
      // Note we load with dName="m" here because the shape of scaleB is [B, N,
      // K] instead of [B, K, N]. This is because tt.dot_scaled expected scaleB
      // transposed. Scales share the reuse pattern of the operand they scale,
      // so reuse the same cache modifier chosen for A/B above.
      loadedScaleB =
          rock::loadTile(b, loc, scaleB, kIter, "n", gridCoords, quantKPerBlock,
                         nPerBlock, /*isKFirst=*/false, bidGridLengths, cacheB);
      loadedScaleA =
          rock::loadTile(b, loc, scaleA, kIter, "m", gridCoords, quantKPerBlock,
                         mPerBlock, /*isKFirst=*/false, bidGridLengths, cacheA);
    }

    // Emit blockwise GEMM. This will load data from LDS (or registers) and
    // compute the MMA at the same time.
    return BlockwiseGemmOp::create(
        b, loc, loadedA, loadedB, acc, loadedScaleA, loadedScaleB,
        op.getQuantBlockSizeAttr(),
        /*matrixAOrigElemType=*/nullptr, /*matrixBOrigElemType=*/nullptr,
        /*matrixAKPack=*/nullptr, /*matrixBKPack=*/nullptr);
  };

  // Main reduction loop over [0, kMain).
  Value acc = initAcc;
  if (kMain > 0) {
    Value matAMain = sliceK(matA, /*kDim=*/2, /*start=*/0, /*len=*/kMain);
    Value matBMain = sliceK(matB, /*kDim=*/1, /*start=*/0, /*len=*/kMain);
    Value nIterations =
        ConstantIntOp::create(b, loc, b.getI32Type(), kIterations);
    scf::ForOp loopOp = createMainLoop(b, loc, nIterations, ValueRange{acc});
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());
      Value newAcc = contract(matAMain, matBMain, loopOp.getInductionVar(),
                              kPerBlock, loopOp.getRegionIterArg(0));
      scf::YieldOp::create(b, loc, ValueRange{newAcc});
    }
    acc = loopOp.getResult(0);
  }

  // Peel the leftover tail [kMain, K) into power-of-two segments and contract
  // each once. decomposePow2 is empty for kRem == 0, so a K that is a multiple
  // of kPerBlock emits nothing here.
  if (kRem > 0) {
    Value kZero = ConstantIntOp::create(b, loc, b.getI32Type(), 0);
    for (Pow2Segment seg : decomposePow2(kRem)) {
      Value aTail = sliceK(matA, /*kDim=*/2, kMain + seg.offset, seg.length);
      Value bTail = sliceK(matB, /*kDim=*/1, kMain + seg.offset, seg.length);
      acc = contract(aTail, bTail, kZero, seg.length, acc);
    }
  }

  return acc;
}

namespace {

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//
struct GridwiseGemmRewritePattern : public OpRewritePattern<GridwiseGemmOp> {
  using OpRewritePattern<GridwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GridwiseGemmOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Obtain data types of inputs.
    auto elementTypeA = op.getA().getType().getElementType();
    auto maybeElementTypeALoad = getInputFusionElementType(op.getA());
    if (failed(maybeElementTypeALoad))
      return op->emitOpError()
             << "Could not determine the underlying data type of A";
    auto elementTypeALoad = maybeElementTypeALoad.value();

    auto elementTypeB = op.getB().getType().getElementType();
    auto maybeElementTypeBLoad = getInputFusionElementType(op.getB());
    if (failed(maybeElementTypeBLoad))
      return op->emitOpError()
             << "Could not determine the underlying data type of B";
    auto elementTypeBLoad = maybeElementTypeBLoad.value();
    auto elemTypeOut = op.getResult().getType().getElementType();
    auto maybeElemTypeOutStore = getOutputFusionElementType(op.getResult());
    if (failed(maybeElemTypeOutStore))
      return op->emitOpError()
             << "Could not determine the underlying data type of output";

    auto elemTypeOutStore = maybeElemTypeOutStore.value();

    auto scaleA = op.getScaleA();
    auto scaleB = op.getScaleB();
    bool hasScaleA = scaleA != nullptr;
    bool hasScaleB = scaleB != nullptr;
    bool isScaledGemm = hasScaleA && hasScaleB;

    // Prepare some useful constants.
    Value matA = op.getA();
    Value matB = op.getB();

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> aShape, bShape;
    aShape = op.getA().getType().getShape();
    bShape = op.getB().getType().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t M = aShape[1];
    int64_t K = aShape[2];
    int64_t N = bShape[2];

    // Obtain critical tuning parameters.
    StringRef arch = rock::getArchValue(op);
    uint32_t blockSize = rock::getBlockSize(op).value().getInt();
    uint32_t gridSize = rock::getGridSize(op).value().getInt();
    GemmParamsAttr tuningParams = op.getParams();
    int64_t kpack = tuningParams.getKpack();
    int64_t kPerBlock = tuningParams.getKPerBlock();
    int64_t mPerBlock = tuningParams.getMPerBlock();
    int64_t nPerBlock = tuningParams.getNPerBlock();
    int64_t mBlocks = M / mPerBlock;
    int64_t nBlocks = N / nPerBlock;
    std::optional<int64_t> quantBlockSize = op.getQuantBlockSize();
    int64_t quantKPerBlock = 0;
    if (quantBlockSize.has_value() && kPerBlock % quantBlockSize.value() != 0) {
      rock::markAsNotApplicable(op);
      return op->emitOpError()
             << "kPerBlock is not a multiple of quantBlockSize";
    }
    if (quantBlockSize.has_value())
      quantKPerBlock = kPerBlock / quantBlockSize.value();

    LLVM_DEBUG(llvm::dbgs() << "gridSize: " << gridSize << "\n"
                            << "blockSize: " << blockSize << "\n"
                            << "elementTypeALoad: " << elementTypeALoad << "\n"
                            << "elementTypeBLoad: " << elementTypeBLoad << "\n"
                            << "\n"
                            << "kPerBlock: " << kPerBlock << "\n"
                            << "mPerBlock: " << mPerBlock << "\n"
                            << "nPerBlock: " << nPerBlock << "\n");
    SmallVector<int64_t, 3> bidGridLengths = {G, mBlocks, nBlocks};

    // Get current workgroup ID.
    Value bid =
        triton::GetProgramIdOp::create(b, op.getLoc(), triton::ProgramIDDim::X);

    // Compute grid coordinates
    int64_t gridGroupSize = tuningParams.getGridGroupSize();
    auto gridCoords = layout::makeGroupedGridLayout(
        b, loc, bid,
        {G, mBlocks, nBlocks, rock::getNumCUValue(op),
         rock::getNumChipletsValue(op), elementTypeALoad, elemTypeOutStore,
         gridGroupSize},
        arch);

    int64_t numWaves = tuningParams.getNumWaves();
    int64_t numCTAs = tuningParams.getNumCTAs();

    // Whether an operand's load reloads data (non-injective view: conv im2col,
    // broadcast, ...). Such operands rely on caching and are never streamed.
    FailureOr<bool> maybeAReloads = rock::isInputNonInjective(matA);
    FailureOr<bool> maybeBReloads = rock::isInputNonInjective(matB);
    if (failed(maybeAReloads))
      return op->emitOpError("could not trace A to determine load injectivity");
    if (failed(maybeBReloads))
      return op->emitOpError("could not trace B to determine load injectivity");
    bool aReloads = maybeAReloads.value();
    bool bReloads = maybeBReloads.value();

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << M << "\n"
               << "N: " << N << "\n"
               << "K: " << K << "\n"
               << "G: " << G << "\n"
               << "mPerBlock: " << mPerBlock << "\n"
               << "nPerBlock: " << nPerBlock << "\n"
               << "kPerBlock: " << kPerBlock << "\n"
               << "kpack: " << kpack << "\n"
               << "mBlocks = M / mPerBlock: " << mBlocks << "\n"
               << "nBlocks = N / nPerBlock: " << nBlocks << "\n"
               << "numWaves: " << numWaves << "\n"
               << "numCTAs: " << numCTAs << "\n"
               << "aReloads: " << (aReloads ? "yes" : "no") << "\n"
               << "bReloads: " << (bReloads ? "yes" : "no") << "\n");

    Type accType = rock::getAccType(elementTypeA, elementTypeB);
    Value initAcc =
        rock::createZeroAccBuffer(b, loc, {mPerBlock, nPerBlock}, accType);

    // Choose cache modifiers for the operands based on data reuse: a skinny
    // GEMM streams the large (low-reuse) operand to avoid evicting the one that
    // is actually reused across workgroups.
    auto [cacheA, cacheB] = chooseGemmLoadCacheModifiers(
        arch, elementTypeALoad, elementTypeBLoad, G, M, N, K, mBlocks, nBlocks,
        aReloads, bReloads);

    // Scaled GEMMs keep K padded to a multiple of the (power-of-two) kPerBlock,
    // so the tail-peeling path is never exercised for them. Reject the
    // unexpected case rather than peeling scale loads with a bogus tail index.
    if (isScaledGemm && (K % kPerBlock != 0)) {
      rock::markAsNotApplicable(op);
      return op->emitOpError("a K that is not a multiple of kPerBlock is not "
                             "supported for scaled gemm");
    }

    // Reduce over K with a power-of-two main loop and a peeled power-of-two
    // tail (see emitKReductionLoop).
    Value loopResult = emitKReductionLoop(
        b, loc, op, matA, matB, scaleA, scaleB, initAcc, gridCoords,
        bidGridLengths, cacheA, cacheB, quantKPerBlock);

    // Compute output transforms
    FailureOr<ArrayAttr> maybeOutputViews =
        computeOutputTransforms(b, loc, mPerBlock, nPerBlock, bidGridLengths);

    if (failed(maybeOutputViews)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to compute output transforms\n");
      return failure();
    }

    ArrayAttr idToMatrixCMaps = maybeOutputViews.value();

    // convert to expected output type
    auto outAccTensorType = cast<RankedTensorType>(loopResult.getType());
    auto destType =
        RankedTensorType::get(outAccTensorType.getShape(), elemTypeOut);
    loopResult = createTypeConversionOp(b, loc, loopResult, destType);

    // Create StoreMarkerOp to mark the tile with output transforms for later
    // store lowering. The result type is the full tensor type so that fusion
    // ops can operate on it directly.
    auto storeMarkerOp = StoreMarkerOp::create(
        b, loc, op.getResult().getType(), loopResult, idToMatrixCMaps,
        ValueRange{gridCoords.g_block, gridCoords.m_block, gridCoords.n_block});

    b.replaceOp(op, storeMarkerOp);
    return success();
  }
};

} // end anonymous namespace

void RockGridwiseGemmToBlockwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<rock::GridwiseGemmOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                         affine::AffineDialect, scf::SCFDialect,
                         triton::TritonDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<GridwiseGemmRewritePattern>(ctx);
  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    signalPassFailure();
  }
}
