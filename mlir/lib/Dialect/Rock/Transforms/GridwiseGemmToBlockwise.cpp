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
#include "llvm/ADT/bit.h"
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

//===----------------------------------------------------------------------===//
// Non-power-of-two K tile peeling
//===----------------------------------------------------------------------===//

namespace {
/// A power-of-two segment [offset, offset + length) of the per-block K tile.
struct KSegment {
  int64_t offset;
  int64_t length;
};
} // namespace

/// Decompose `n` into a minimal sequence of power-of-two segments (largest
/// first) that exactly cover [0, n). A power-of-two `n` yields a single
/// {0, n} segment, so the split is a no-op for that (common) case. For example
/// kPerBlock = 48 -> {{0, 32}, {32, 16}}.
static SmallVector<KSegment> decomposeKPow2(int64_t n) {
  SmallVector<KSegment> segs;
  int64_t off = 0;
  for (int64_t rem = n; rem > 0;) {
    int64_t seg = static_cast<int64_t>(llvm::bit_floor<uint64_t>(rem));
    segs.push_back({off, seg});
    off += seg;
    rem -= seg;
  }
  return segs;
}

/// Build a view of the rank-3 gemm operand `operand` in which the contraction
/// dimension `kDimIdx` (of size `kIters * kPerBlock`) is restructured into
/// (k_loop, k_iter) = (kIters, kPerBlock), the k_iter sub-dim is sliced to
/// `seg`, and the two are re-merged. The resulting K dimension has size
/// `kIters * seg.length`, and k_loop index `l` of it maps onto original K
/// indices `l * kPerBlock + seg.offset + i`.
///
/// Feeding this view to the regular `loadTile` (with kPerBlock = seg.length)
/// therefore loads, for outer K-loop iteration `l`, exactly the K sub-tile
/// [l*kPerBlock + seg.offset, l*kPerBlock + seg.offset + seg.length) -- a
/// power-of-two-wide slab -- while keeping the outer loop trip count kIters
/// unchanged.
static Value sliceOperandKSegment(OpBuilder &b, Location loc, Value operand,
                                  unsigned kDimIdx, int64_t kIters,
                                  int64_t kPerBlock, KSegment seg) {
  auto type = cast<RankedTensorType>(operand.getType());
  ArrayRef<int64_t> shape = type.getShape();
  unsigned rank = shape.size();

  SmallVector<std::string> baseStore, loopStore, iterStore;
  for (unsigned i = 0; i < rank; ++i) {
    baseStore.push_back(("d" + Twine(i)).str());
    loopStore.push_back(("d" + Twine(i) + "l").str());
    iterStore.push_back(("d" + Twine(i) + "i").str());
  }
  SmallVector<StringRef> baseNames(baseStore.begin(), baseStore.end());

  // Layer 1: restructure the K dim into (k_loop, k_iter); pass the rest.
  BottomUpTMBuilder l1(b, baseNames, shape, loc);
  {
    unsigned up = 0;
    for (unsigned i = 0; i < rank; ++i) {
      if (i == kDimIdx) {
        l1.unmerge({StringRef(loopStore[i]), StringRef(iterStore[i])},
                   {up, up + 1}, StringRef(baseStore[i]), {kIters, kPerBlock});
        up += 2;
      } else {
        l1.passThrough({StringRef(baseStore[i])}, {up},
                       {StringRef(baseStore[i])});
        up += 1;
      }
    }
  }
  TransformMapAttr a1 = l1.get();

  // Layer 2: slice the k_iter sub-dim to the segment; pass the rest.
  BottomUpTMBuilder l2 = BottomUpTMBuilder::above(l1, a1);
  SmallVector<StringRef> names2;
  l2.getStartNames(names2);
  StringRef kIterName(iterStore[kDimIdx]);
  SmallVector<StringRef> passNames;
  for (StringRef nm : names2)
    if (nm != kIterName)
      passNames.push_back(nm);
  l2.passThrough(passNames);
  l2.slice({kIterName}, {kIterName}, {seg.offset}, {seg.offset + seg.length});
  TransformMapAttr a2 = l2.get();

  // Layer 3: re-merge (k_loop, k_iter) back into the K dimension.
  BottomUpTMBuilder l3 = BottomUpTMBuilder::above(l2, a2);
  {
    unsigned up = 0;
    for (unsigned i = 0; i < rank; ++i) {
      if (i == kDimIdx) {
        l3.merge(StringRef(baseStore[i]), up,
                 {StringRef(loopStore[i]), StringRef(iterStore[i])});
      } else {
        l3.passThrough({StringRef(baseStore[i])}, {up},
                       {StringRef(baseStore[i])});
      }
      up += 1;
    }
  }
  TransformMapAttr a3 = l3.get();

  return rock::transform(b, operand, b.getArrayAttr({a3, a2, a1}));
}

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

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

    // Emit loop with iter_args for the accumulator
    int64_t kIterations = K / kPerBlock;
    Value nIterations =
        ConstantIntOp::create(b, loc, b.getI32Type(), kIterations);

    // Peel a non-power-of-two per-block K tile into power-of-two segments (e.g.
    // kPerBlock = 48 -> {32, 16}). The Triton layouts produced downstream
    // require power-of-two tensor shapes, so each K-loop iteration contracts
    // the segments in sequence, accumulating into the same per-workgroup tile
    // (an in-register reduction along K -- no atomics, output side unchanged).
    // A power-of-two kPerBlock yields a single {0, kPerBlock} segment, leaving
    // the emitted IR identical to the un-peeled path.
    SmallVector<KSegment> kSegs = decomposeKPow2(kPerBlock);
    bool peelK = kSegs.size() > 1;
    if (peelK && isScaledGemm) {
      rock::markAsNotApplicable(op);
      return op->emitOpError("non-power-of-two kPerBlock is not supported for "
                             "scaled gemm");
    }

    // Pre-build the loop-invariant per-segment K-sliced operand views. A single
    // (power-of-two) segment reuses the operands directly, so their views are
    // left untouched.
    SmallVector<Value> segMatA, segMatB;
    if (peelK) {
      for (KSegment seg : kSegs) {
        segMatA.push_back(sliceOperandKSegment(b, loc, matA, /*kDimIdx=*/2,
                                               kIterations, kPerBlock, seg));
        segMatB.push_back(sliceOperandKSegment(b, loc, matB, /*kDimIdx=*/1,
                                               kIterations, kPerBlock, seg));
      }
    } else {
      segMatA.push_back(matA);
      segMatB.push_back(matB);
    }

    scf::ForOp loopOp = createMainLoop(b, loc, nIterations, ValueRange{initAcc});
    Value loopResult;
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());
      Value iv = loopOp.getInductionVar();
      Value accArg = loopOp.getRegionIterArg(0);

      // Choose cache modifiers for the operands based on data reuse: a skinny
      // GEMM streams the large (low-reuse) operand to avoid evicting the one
      // that is actually reused across workgroups.
      auto [cacheA, cacheB] = chooseGemmLoadCacheModifiers(
          arch, elementTypeALoad, elementTypeBLoad, G, M, N, K, mBlocks,
          nBlocks, aReloads, bReloads);

      // Contract each power-of-two K segment in turn, threading the accumulator
      // so every segment of this K-loop iteration reduces into the same tile.
      // (A power-of-two kPerBlock has a single full-width segment; scaled gemms
      // are restricted to that case above.)
      Value acc = accArg;
      for (auto [segIdx, seg] : llvm::enumerate(kSegs)) {
        // Load from global memory to registers
        Value loadedB = rock::loadTile(
            b, loc, segMatB[segIdx], /*kiter=*/iv, "n", gridCoords, seg.length,
            nPerBlock, /*isKFirst=*/true, bidGridLengths, cacheB);
        Value loadedA = rock::loadTile(
            b, loc, segMatA[segIdx], /*kiter=*/iv, "m", gridCoords, seg.length,
            mPerBlock, /*isKFirst=*/false, bidGridLengths, cacheA);

        Value loadedScaleA, loadedScaleB;
        if (isScaledGemm) {
          // Note we load with dName="m" here because the shape of scaleB is [B,
          // N, K] instead of [B, K, N]. This is because tt.dot_scaled expected
          // scaleB transposed
          // Scales share the reuse pattern of the operand they scale, so reuse
          // the same cache modifier chosen for A/B above.
          loadedScaleB =
              rock::loadTile(b, loc, scaleB, /*kiter=*/iv, "n", gridCoords,
                             quantKPerBlock, nPerBlock,
                             /*isKFirst=*/false, bidGridLengths, cacheB);
          loadedScaleA =
              rock::loadTile(b, loc, scaleA, /*kiter=*/iv, "m", gridCoords,
                             quantKPerBlock, mPerBlock,
                             /*isKFirst=*/false, bidGridLengths, cacheA);
        }

        // Emit blockwise GEMM. This will load data from LDS (or registers) and
        // compute the MMA at the same time
        acc = BlockwiseGemmOp::create(
            b, loc, loadedA, loadedB, acc, loadedScaleA, loadedScaleB,
            op.getQuantBlockSizeAttr(),
            /*matrixAOrigElemType=*/nullptr, /*matrixBOrigElemType=*/nullptr,
            /*matrixAKPack=*/nullptr, /*matrixBKPack=*/nullptr);
      }

      // Yield the new accumulator
      scf::YieldOp::create(b, loc, ValueRange{acc});
      loopResult = loopOp.getResult(0);
    }

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
