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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
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

// Per-field perf-config validators. A violation is treated as a hard
// diagnostic; `markAsNotApplicable` is reserved for arch-feature mismatches.
static bool isPositivePowerOfTwo(int64_t v) {
  return v > 0 && llvm::isPowerOf2_64(static_cast<uint64_t>(v));
}

static LogicalResult validatePositivePowerOfTwo(Operation *op, StringRef name,
                                                int64_t value) {
  if (!isPositivePowerOfTwo(value))
    return op->emitError() << name << "=" << value
                           << " must be a positive power of two";
  return success();
}

static LogicalResult validatePositiveValue(Operation *op, StringRef name,
                                           int64_t value) {
  if (value <= 0)
    return op->emitError() << name << "=" << value << " must be positive";
  return success();
}

static LogicalResult validateNumCTAs(Operation *op, int64_t numCTAs) {
  if (numCTAs < 1)
    return op->emitError() << "numCTAs=" << numCTAs << " must be >= 1";
  if (!isPositivePowerOfTwo(numCTAs))
    return op->emitError() << "numCTAs=" << numCTAs
                           << " must be a positive power of two";
  StringRef arch = rock::getArchValue(op);
  int64_t maxNumCTAs = rock::getMaxNumCTAs(arch);
  if (numCTAs > maxNumCTAs)
    return op->emitError() << "numCTAs=" << numCTAs << " exceeds max ("
                           << maxNumCTAs << ") for " << arch;
  if (numCTAs != 1 && !rock::supportsMultiCTALaunch(arch))
    return op->emitError() << "numCTAs=" << numCTAs
                           << " but multi-CTA launch is not supported on "
                           << arch;
  return success();
}

static LogicalResult validateKpack(Operation *op, int64_t kpack) {
  StringRef arch = rock::getArchValue(op);
  if (kpack < 1)
    return op->emitError() << "kpack=" << kpack << " must be positive";
  int64_t maxKpack = rock::getMaxKpack(arch);
  if (kpack > maxKpack)
    return op->emitError() << "kpack=" << kpack << " exceeds max (" << maxKpack
                           << ") for " << arch;
  return success();
}

static LogicalResult validateNumWaves(Operation *op, int64_t numWaves) {
  if (!isPositivePowerOfTwo(numWaves))
    return op->emitError() << "numWaves=" << numWaves
                           << " must be a positive power of two";
  int64_t waveSize = rock::getWaveSize(rock::getArchValue(op));
  int64_t maxNumWaves = rock::maxHardwareWorkgroupSize / waveSize;
  if (numWaves > maxNumWaves)
    return op->emitError() << "numWaves=" << numWaves
                           << " * waveSize=" << waveSize
                           << " exceeds max workgroup size ("
                           << rock::maxHardwareWorkgroupSize << ")";
  return success();
}

static LogicalResult validateMatrixInstrNonkdim(Operation *op,
                                                int64_t matrixInstrNonkdim) {
  if (matrixInstrNonkdim != 0 && !isPositivePowerOfTwo(matrixInstrNonkdim))
    return op->emitError()
           << "matrixInstrNonkdim=" << matrixInstrNonkdim
           << " must be 0 (heuristic) or a positive power of two";
  return success();
}

static LogicalResult validateSplitKFactor(Operation *op, int64_t splitKFactor) {
  if (splitKFactor < 1)
    return op->emitError() << "splitKFactor=" << splitKFactor
                           << " must be >= 1";
  if (isa<AttentionOp>(op) && splitKFactor != 1)
    return op->emitError() << "splitKFactor=" << splitKFactor
                           << " must be 1 for attention";
  return success();
}

static LogicalResult validateNumStages(Operation *op, int64_t numStages) {
  if (numStages < 1)
    return op->emitError() << "numStages=" << numStages << " must be >= 1";
  return success();
}

static LogicalResult validateWavesPerEU(Operation *op, int64_t wavesPerEU) {
  if (wavesPerEU < 0)
    return op->emitError() << "wavesPerEU=" << wavesPerEU << " must be >= 0";
  StringRef arch = rock::getArchValue(op);
  int64_t maxWavesPerEU = rock::getMaxWavesPerEU(arch);
  if (wavesPerEU > maxWavesPerEU)
    return op->emitError() << "wavesPerEU=" << wavesPerEU << " exceeds max ("
                           << maxWavesPerEU << ") for " << arch;
  return success();
}

static LogicalResult validateGridGroupSize(Operation *op,
                                           int64_t gridGroupSize) {
  if (gridGroupSize < 0)
    return op->emitError() << "gridGroupSize=" << gridGroupSize
                           << " must be >= 0";
  return success();
}

static LogicalResult validateNPerBlockG1(Operation *op, int64_t nPerBlockG1) {
  if (nPerBlockG1 != 0 && !isPositivePowerOfTwo(nPerBlockG1))
    return op->emitError() << "nPerBlockG1=" << nPerBlockG1
                           << " must be 0 (untiled) or a positive power of two";
  return success();
}

LogicalResult
mlir::rock::validatePerfConfig(Operation *op,
                               RockTuningParamAttrInterface params,
                               bool requirePow2MN, bool requirePow2K) {
  auto validateMN =
      requirePow2MN ? validatePositivePowerOfTwo : validatePositiveValue;
  if (failed(validateMN(op, "mPerBlock", params.getMPerBlock())))
    return failure();
  if (failed(validateMN(op, "nPerBlock", params.getNPerBlock())))
    return failure();
  if (auto gemmGemmParams = dyn_cast<GemmGemmParamsAttr>(params))
    if (failed(validateNPerBlockG1(op, gemmGemmParams.getNPerBlockG1())))
      return failure();
  auto validateK =
      requirePow2K ? validatePositivePowerOfTwo : validatePositiveValue;
  if (failed(validateK(op, "kPerBlock", params.getKPerBlock())))
    return failure();
  if (failed(validateKpack(op, params.getKpack())))
    return failure();
  if (failed(validateNumCTAs(op, params.getNumCTAs())))
    return failure();
  if (failed(validateNumWaves(op, params.getNumWaves())))
    return failure();
  if (failed(validateMatrixInstrNonkdim(op, params.getMatrixInstrNonkdim())))
    return failure();
  if (failed(validateSplitKFactor(op, params.getSplitKFactor())))
    return failure();
  if (failed(validateNumStages(op, params.getNumStages())))
    return failure();
  if (failed(validateWavesPerEU(op, params.getWavesPerEU())))
    return failure();
  if (failed(validateGridGroupSize(op, params.getGridGroupSize())))
    return failure();
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
                      {dynAwareDiv(dGlobal, dPerBlock), dPerBlock});

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

StringRef mlir::rock::getRuntimeGemmDimName(RuntimeGemmDim dim) {
  switch (dim) {
  case RuntimeGemmDim::G:
    return "G";
  case RuntimeGemmDim::M:
    return "M";
  case RuntimeGemmDim::N:
    return "N";
  case RuntimeGemmDim::K:
    return "K";
  }
  llvm_unreachable("unhandled RuntimeGemmDim");
}

unsigned mlir::rock::getRuntimeGemmDimIndex(unsigned numArgs,
                                            RuntimeGemmDim dim) {
  assert(numArgs >= kNumRuntimeGemmDims &&
         "argument list is too short to hold the runtime gemm dimensions");
  return numArgs - kNumRuntimeGemmDims + static_cast<unsigned>(dim);
}

/// Keys of the `rock.dyn_grid_size` dictionary.
static constexpr StringLiteral kMPerBlockKey = "mPerBlock";
static constexpr StringLiteral kGnBlocksKey = "gnBlocks";

DictionaryAttr mlir::rock::makeDynGridSizeAttr(Builder &b,
                                               DynGridSize gridSize) {
  return b.getDictionaryAttr(
      {b.getNamedAttr(kMPerBlockKey, b.getI64IntegerAttr(gridSize.mPerBlock)),
       b.getNamedAttr(kGnBlocksKey, b.getI64IntegerAttr(gridSize.gnBlocks))});
}

std::optional<DynGridSize> mlir::rock::getDynGridSize(Attribute attr) {
  auto dict = dyn_cast_if_present<DictionaryAttr>(attr);
  if (!dict)
    return std::nullopt;
  auto mPerBlock = dict.getAs<IntegerAttr>(kMPerBlockKey);
  auto gnBlocks = dict.getAs<IntegerAttr>(kGnBlocksKey);
  if (!mPerBlock || !gnBlocks)
    return std::nullopt;
  return DynGridSize{mPerBlock.getInt(), gnBlocks.getInt()};
}

/// The static and dynamic grid attributes are alternatives, and
/// `rock-tensor-to-triton-ptr` takes the static one when both are present. A
/// kernel that ended up with both would therefore launch on whichever was
/// written first rather than fail, so catch that here instead.
///
/// Rewriting the same kind is allowed: a backward-data convolution holds
/// several gemms in one function and each of them publishes the grid.
static void assertNoOtherGridSizeKind(func::FuncOp funcOp, StringRef other) {
  assert(!funcOp->hasAttr(other) &&
         "a kernel carries either a static or a dynamic grid size, not both");
  (void)funcOp;
  (void)other;
}

void mlir::rock::setGridSize(func::FuncOp funcOp, Builder &b,
                             int64_t gridSize) {
  assertNoOtherGridSizeKind(funcOp, DynGridSizeAttr::getMnemonic());
  assert(gridSize > 0 && "a grid must hold at least one block");
  funcOp->setAttr(GridSizeAttr::getMnemonic(), b.getI32IntegerAttr(gridSize));
}

void mlir::rock::setDynGridSize(func::FuncOp funcOp, Builder &b,
                                DynGridSize gridSize) {
  assertNoOtherGridSizeKind(funcOp, GridSizeAttr::getMnemonic());
  assert(gridSize.mPerBlock > 0 && gridSize.gnBlocks > 0 &&
         "both factors of a dynamic grid must be positive, since the launch is "
         "their product");
  funcOp->setAttr(DynGridSizeAttr::getMnemonic(),
                  makeDynGridSizeAttr(b, gridSize));
}

/// Whether `transforms` only reshuffles coordinates, so that the view and the
/// buffer underneath it hold the same number of elements. That equality is what
/// relates a dynamic extent to a buffer's element count.
static bool preservesElementCount(ArrayRef<TransformMapAttr> transforms) {
  for (TransformMapAttr map : transforms) {
    ArrayRef<int64_t> upperBounds = map.getUpperBounds().asArrayRef();
    for (TransformAttr transform : map.getOps()) {
      switch (transform.getType()) {
      case TransformType::PassThrough:
      case TransformType::Unmerge:
      case TransformType::Merge:
        break;
      case TransformType::AddDim:
        // A unit dimension mapped to nothing leaves the count alone; a wider
        // one repeats the data underneath it.
        for (uint32_t upperDim : transform.getUpperDims())
          if (upperBounds[upperDim] != 1)
            return false;
        break;
      case TransformType::ConstDim:
        // ConstDim is parameterized by [value, length] pairs. Pinning a
        // dimension of length one selects the only element there is, which is
        // the inverse of the unit AddDim above; a longer one picks out one
        // slice of several and drops the rest.
        for (size_t i = 1, e = transform.getParams().size(); i < e; i += 2)
          if (transform.getParams()[i] != 1)
            return false;
        break;
      case TransformType::Pad:
      case TransformType::Slice:
      case TransformType::Embed:
      case TransformType::Broadcast:
        return false;
      }
    }
  }
  return true;
}

/// The kernel argument `view` is an element-count-preserving view of.
static FailureOr<BlockArgument>
findViewedArgument(Operation *gemmOp, StringRef dimName, Value view) {
  SmallVector<TransformMapAttr> transforms;
  auto [root, unused] = untransform(view, transforms);
  if (!preservesElementCount(transforms))
    return gemmOp->emitOpError()
           << "cannot recover the runtime value of dimension " << dimName
           << " because the operand is not an element-count-preserving view of "
              "a kernel argument";

  auto blockArg = dyn_cast<BlockArgument>(root);
  if (!blockArg)
    return gemmOp->emitOpError()
           << "cannot recover the runtime value of dimension " << dimName
           << " because the operand does not trace back to a kernel argument";
  return blockArg;
}

/// The lone dynamic extent of `shape`. With two unknowns in one view the
/// buffer's element count is one equation short of pinning either of them down.
static FailureOr<unsigned> findLoneDynamicDim(Operation *gemmOp, StringRef what,
                                              ArrayRef<int64_t> shape) {
  if (llvm::count_if(shape, ShapedType::isDynamic) != 1)
    return gemmOp->emitOpError()
           << "cannot recover the runtime extents of " << what
           << " because it does not have exactly one dynamic extent";
  return static_cast<unsigned>(std::distance(
      shape.begin(), llvm::find_if(shape, ShapedType::isDynamic)));
}

/// Product of every extent of `shape` other than `skipped`.
static int64_t productOfOtherDims(ArrayRef<int64_t> shape, unsigned skipped) {
  int64_t product = 1;
  for (auto [index, dim] : llvm::enumerate(shape))
    if (index != skipped)
      product *= dim;
  return product;
}

/// Work out how a caller can obtain dimension `dimIndex` of gemm operand
/// `operand`.
static FailureOr<ExtentRecipe> buildExtentRecipe(Operation *gemmOp,
                                                 StringRef dimName,
                                                 Value operand,
                                                 unsigned dimIndex) {
  ArrayRef<int64_t> shape = cast<ShapedType>(operand.getType()).getShape();
  if (!ShapedType::isDynamic(shape[dimIndex]))
    return ExtentRecipe{shape[dimIndex], 0, 1};

  if (failed(findLoneDynamicDim(gemmOp, dimName, shape)))
    return failure();

  FailureOr<BlockArgument> blockArg =
      findViewedArgument(gemmOp, dimName, operand);
  if (failed(blockArg))
    return failure();

  return ExtentRecipe{ShapedType::kDynamic, blockArg->getArgNumber(),
                      productOfOtherDims(shape, dimIndex)};
}

/// What each axis of a gemm view means. A is G x M x K, B is G x K x N and the
/// result is G x M x N once the gemm has been normalized. An axis that no
/// runtime argument carries is `std::nullopt`, so that finding an unknown
/// extent there fails loudly instead of being attributed to the wrong
/// dimension.
using GemmAxes = std::array<std::optional<RuntimeGemmDim>, 3>;
static constexpr GemmAxes kAAxes = {RuntimeGemmDim::G, RuntimeGemmDim::M,
                                    RuntimeGemmDim::K};
static constexpr GemmAxes kBAxes = {RuntimeGemmDim::G, RuntimeGemmDim::K,
                                    RuntimeGemmDim::N};
static constexpr GemmAxes kOutAxes = {RuntimeGemmDim::G, RuntimeGemmDim::M,
                                      RuntimeGemmDim::N};
/// Attention's output is G x M x O, where O is the head dimension of the
/// values. That is the second gemm's N, which the four runtime arguments do not
/// cover, so it must stay static.
static constexpr GemmAxes kAttentionOutAxes = {RuntimeGemmDim::G,
                                               RuntimeGemmDim::M, std::nullopt};

/// The gemm-shaped operands of the one op in a kernel that carries runtime
/// dimensions.
///
/// For a two-gemm op such as `rock.attention` these describe the *first* gemm:
/// its G and M size the grid and its N is the second gemm's K, so between them
/// the four runtime arguments name every extent of both gemms except the value
/// head dimension.
struct GemmLikeOp {
  Operation *op;
  /// The G x M x K matrix.
  Value a;
  /// The G x K x N matrix.
  Value b;
  /// The op's output, whose axes `resultAxes` names.
  Value result;
  GemmAxes resultAxes;
};

/// Find the single gemm-like op of `funcOp`. The G, M, N and K arguments are
/// identified by position alone, so numbering them for several such ops at once
/// would be ambiguous.
static FailureOr<GemmLikeOp> findLoneGemmLikeOp(func::FuncOp funcOp) {
  SmallVector<GemmLikeOp> found;
  funcOp.walk([&](Operation *op) {
    if (auto gemmOp = dyn_cast<GemmOp>(op)) {
      found.push_back(
          {op, gemmOp.getA(), gemmOp.getB(), gemmOp.getResult(), kOutAxes});
    } else if (auto attnOp = dyn_cast<AttentionOp>(op)) {
      found.push_back({op, attnOp.getQueries(), attnOp.getKeys(),
                       attnOp.getResult(), kAttentionOutAxes});
    }
  });

  if (found.size() != 1)
    return funcOp.emitOpError()
           << "dynamic shapes are only supported for kernels with exactly one "
              "rock.gemm or rock.attention, found "
           << found.size();

  if (auto attnOp = dyn_cast<AttentionOp>(found.front().op)) {
    if (attnOp.getQTransposed() || attnOp.getOTransposed())
      return attnOp.emitOpError()
             << "dynamic shapes are not supported for a transposed Q or O yet";
  }
  return found.front();
}

FailureOr<SmallVector<ExtentRecipe>>
mlir::rock::buildGemmExtentRecipes(func::FuncOp funcOp) {
  FailureOr<GemmLikeOp> root = findLoneGemmLikeOp(funcOp);
  if (failed(root))
    return failure();

  Value a = root->a, b = root->b;
  const std::pair<Value, unsigned> sources[kNumRuntimeGemmDims] = {
      {a, 0}, {a, 1}, {b, 2}, {a, 2}};

  SmallVector<ExtentRecipe> recipes;
  for (auto [index, source] : llvm::enumerate(sources)) {
    StringRef dimName =
        getRuntimeGemmDimName(static_cast<RuntimeGemmDim>(index));
    FailureOr<ExtentRecipe> recipe =
        buildExtentRecipe(root->op, dimName, source.first, source.second);
    if (failed(recipe))
      return failure();
    recipes.push_back(*recipe);
  }
  return recipes;
}

FailureOr<SmallVector<DynamicArgExtent>>
mlir::rock::getDynamicArgExtents(func::FuncOp funcOp) {
  FailureOr<GemmLikeOp> root = findLoneGemmLikeOp(funcOp);
  if (failed(root))
    return failure();

  /// One gemm view of a kernel argument.
  struct View {
    /// Value whose shape carries the G/M/N/K extents.
    Value gemmShaped;
    /// Value that traces back to the kernel argument. This is `gemmShaped` for
    /// the operands, but for the output the extents live on the gemm result
    /// while the buffer is the store destination, which may be flattened.
    Value buffer;
    GemmAxes axes;
  };

  Value a = root->a, b = root->b;
  SmallVector<View> views = {{a, a, kAAxes}, {b, b, kBAxes}};

  FailureOr<SetVector<StoreOp>> stores =
      traceRootOutputToStoreOps(root->result);
  if (failed(stores))
    return root->op->emitOpError()
           << "cannot size a dynamic output because the gemm result does not "
              "reach a rock.store";
  for (StoreOp storeOp : *stores) {
    // The result's extents only describe the destination if the views on both
    // sides of the store preserve the element count.
    SmallVector<TransformMapAttr> sourceTransforms;
    untransform(storeOp.getSource(), sourceTransforms);
    if (!preservesElementCount(sourceTransforms))
      return root->op->emitOpError()
             << "cannot size a dynamic output because what is stored is not an "
                "element-count-preserving view of the gemm result";
    views.push_back({root->result, storeOp.getDest(), root->resultAxes});
  }

  SmallVector<DynamicArgExtent> extents;
  llvm::SmallDenseSet<unsigned> described;
  for (const View &view : views) {
    ArrayRef<int64_t> shape =
        cast<ShapedType>(view.gemmShaped.getType()).getShape();
    if (llvm::none_of(shape, ShapedType::isDynamic))
      continue;

    FailureOr<unsigned> dynamicDim =
        findLoneDynamicDim(root->op, "a gemm view", shape);
    if (failed(dynamicDim))
      return failure();

    std::optional<RuntimeGemmDim> dim = view.axes[*dynamicDim];
    if (!dim)
      return root->op->emitOpError()
             << "axis " << *dynamicDim
             << " of a gemm view is dynamic, but no runtime argument carries "
                "that dimension";
    FailureOr<BlockArgument> blockArg =
        findViewedArgument(root->op, getRuntimeGemmDimName(*dim), view.buffer);
    if (failed(blockArg))
      return failure();

    if (described.insert(blockArg->getArgNumber()).second)
      extents.push_back({blockArg->getArgNumber(), *dim,
                         productOfOtherDims(shape, *dynamicDim)});
  }

  // Anything left over could only be sized by guesswork, which would silently
  // compute the wrong thing rather than fail.
  for (BlockArgument arg : funcOp.getArguments()) {
    auto shapedType = dyn_cast<ShapedType>(arg.getType());
    if (!shapedType || shapedType.hasStaticShape())
      continue;
    if (!described.contains(arg.getArgNumber()))
      return funcOp.emitOpError()
             << "argument " << arg.getArgNumber()
             << " has a dynamic extent that is not a view of the gemm, so its "
                "size cannot be derived";
  }
  return extents;
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
Value mlir::rock::loadTile(OpBuilder &b, Location loc, Value in, Value kIter,
                           StringRef dName,
                           rock::layout::GridCoordinates gridCoords,
                           int64_t kPerBlock, int64_t dPerBlock, bool isKFirst,
                           SmallVector<int64_t, 3> &bidGridLengths,
                           rock::CacheModifier cache) {
  FailureOr<ArrayAttr> maybeBufferViews = getLoadRegsAsTileViews(
      b, loc, in, dName, bidGridLengths, kPerBlock, dPerBlock, isKFirst);
  assert(succeeded(maybeBufferViews));
  ArrayAttr bufferViews = maybeBufferViews.value();

  // Compute the tile result type by applying the tiling transforms to
  // determine the output shape, then taking the last two dimensions.
  Value wrappedSource = transform(b, in, bufferViews);
  auto sourceType = cast<RankedTensorType>(wrappedSource.getType());
  auto sourceShape = sourceType.getShape();
  auto resultType = RankedTensorType::get(sourceShape.take_back(2),
                                          sourceType.getElementType());

  // Create a LoadMarkerOp placeholder. LowerLoads will later convert this
  // into an actual BlockwiseLoadOp by tracing back through the source chain.
  // We pass the original (un-transformed) input as source and carry the
  // tiling transforms as metadata in extraViews.
  auto markerOp =
      LoadMarkerOp::create(b, loc, resultType, in, bufferViews,
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
