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

    LLVM_DEBUG(llvm::dbgs() << "M: " << M << "\n"
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
                            << "numCTAs: " << numCTAs << "\n");

    Type accType = rock::getAccType(elementTypeA, elementTypeB);
    Value initAcc =
        rock::createZeroAccBuffer(b, loc, {mPerBlock, nPerBlock}, accType);

    // Emit loop with iter_args for the accumulator
    int64_t kIterations = K / kPerBlock;
    Value nIterations =
        ConstantIntOp::create(b, loc, b.getI32Type(), kIterations);

    scf::ForOp loopOp = createMainLoop(b, loc, nIterations, ValueRange{initAcc});
    Value loopResult;
    {
      PatternRewriter::InsertionGuard guard(b);
      b.setInsertionPointToStart(loopOp.getBody());
      Value iv = loopOp.getInductionVar();
      Value accArg = loopOp.getRegionIterArg(0);

      // Load from global memory to registers
      Value loadedB =
          rock::loadTile(b, loc, matB, /*kiter=*/iv, "n", gridCoords, kPerBlock,
                         nPerBlock, /*isKFirst=*/true, bidGridLengths);
      Value loadedA =
          rock::loadTile(b, loc, matA, /*kiter=*/iv, "m", gridCoords, kPerBlock,
                         mPerBlock, /*isKFirst=*/false, bidGridLengths);

      Value loadedScaleA, loadedScaleB;
      if (isScaledGemm) {
        // Note we load with dName="m" here because the shape of scaleB is [B,
        // N, K] instead of [B, K, N]. This is because tt.dot_scaled expected
        // scaleB transposed
        loadedScaleB = rock::loadTile(b, loc, scaleB, /*kiter=*/iv, "n",
                                      gridCoords, quantKPerBlock, nPerBlock,
                                      /*isKFirst=*/false, bidGridLengths);
        loadedScaleA = rock::loadTile(b, loc, scaleA, /*kiter=*/iv, "m",
                                      gridCoords, quantKPerBlock, mPerBlock,
                                      /*isKFirst=*/false, bidGridLengths);
      }

      // Emit blockwise GEMM. This will load data from LDS (or registers) and
      // compute the MMA at the same time
      Value newAcc = BlockwiseGemmOp::create(
          b, loc, accArg.getType(), loadedA, loadedB, accArg, loadedScaleA,
          loadedScaleB, op.getQuantBlockSizeAttr(),
          /*matrixAOrigElemType=*/nullptr, /*matrixBOrigElemType=*/nullptr,
          /*matrixAKPack=*/nullptr, /*matrixBKPack=*/nullptr);

      // Yield the new accumulator
      scf::YieldOp::create(b, loc, ValueRange{newAcc});
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
