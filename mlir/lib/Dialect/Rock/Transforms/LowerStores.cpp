//===- LowerStores.cpp - Lower rock.store ops to blockwise_store -----===//
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
// =============================================================================
//
// This pass runs AFTER RockLowerLoads and converts rock.store to
// rock.blockwise_store.
//
// At this point, the IR has:
// - StoreMarkerOp mapping tiles to full tensors for fusion ops
// - Fusion ops (arith.addf, etc.) operating on full tensor types
// - Explicit rock.reduce ops preserving logical reduction semantics
// - rock.store storing the fused result
//
// This pass:
// 1. Traces back from rock.store through fusion ops to find StoreMarkerOp
// 2. Gets the tile values and transforms from StoreMarkerOp
// 3. Clones fusion ops to operate on tiles
// 4. Lowers explicit reductions to blockwise partial reductions and atomics
// 5. Combines StoreMarkerOp transforms with store destination transforms
// 6. Creates BlockwiseStoreOp with the combined transform chain
//
// Example:
//   Before:
//     %gemm_tile = scf.for ... -> tensor<16x16xf32>
//     %gemm_full = rock.store_marker %gemm_tile views [<transforms>] [%g, %m,
//     %n]
//                    : tensor<16x16xf32> -> tensor<full>
//     %fused = arith.addf %gemm_full, %bias_full : tensor<full>
//     %dest_tr = rock.transform %arg2 by <dest_transforms>
//     %out = rock.store %fused to %dest_tr by set
//
//   After:
//     %gemm_tile = scf.for ... -> tensor<16x16xf32>
//     %fused_tile = arith.addf %gemm_tile, %bias_tile : tensor<16x16xf32>
//     %combined_dest = rock.transform %arg2 by <dest_transforms +
//     tile_transforms> %out = rock.blockwise_store %fused_tile ->
//     %combined_dest[%g, %m, %n] by set
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKLOWERSTORESPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-lower-stores"

using namespace mlir;
using namespace mlir::rock;

namespace {

/// Recursively search for a StoreMarkerOp through fusion ops.
/// Returns the StoreMarkerOp if found, or nullptr otherwise.
/// Note: UntileOp is only used for extra fusion inputs (from
/// InsertOutputFusionLoads) and is never on the path to StoreMarkerOp.
static FailureOr<StoreMarkerOp> findStoreMarkerOp(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp)
    return failure();

  // Check if this is a StoreMarkerOp
  if (auto storeMarkerOp = dyn_cast<StoreMarkerOp>(defOp))
    return storeMarkerOp;

  if (auto transformOp = dyn_cast<TransformOp>(defOp))
    return findStoreMarkerOp(transformOp.getInput());

  if (auto reduceOp = dyn_cast<ReduceOp>(defOp))
    return findStoreMarkerOp(reduceOp.getIn());

  // Fusion op: recursively search operands.
  if (rock::isFusionOp(defOp)) {
    for (Value operand : defOp->getOperands()) {
      if (StoreMarkerOp found = findStoreMarkerOp(operand).value_or(nullptr))
        return found;
    }
  }

  return failure();
}

/// Returns true when every root-buffer coordinate is unchanged while the
/// selected upper dimension varies over its complete statically-known extent.
static bool addressIsInvariantAlongDim(OpBuilder &builder, Value dest,
                                       int64_t dim) {
  SmallVector<TransformMapAttr> transforms;
  std::tie(std::ignore, std::ignore) = untransform(dest, transforms);
  if (transforms.empty())
    return false;

  ArrayRef<int64_t> upperShape = transforms.front().getUpperBounds();
  if (dim < 0 || static_cast<size_t>(dim) >= upperShape.size())
    return false;
  unsigned upperDim = static_cast<unsigned>(dim);
  ArrayRef<unsigned> dims(&upperDim, 1);
  if (validityDependsOnAnyDim(transforms, dims))
    return false;
  if (!transformChainDependsOnAnyDim(transforms, dims))
    return true;

  // Affine dependency is intentionally conservative for aligned floor-div
  // expressions. Prove those cases with static sub-dimension ranges.
  FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>>> subDims =
      getLowerSubDimensions(
          builder,
          builder.getArrayAttr(llvm::to_vector_of<Attribute>(transforms)), dim);
  if (failed(subDims))
    return false;
  return llvm::all_of(*subDims, [](const auto &entry) {
    return llvm::all_of(entry.second,
                        [](const SubDimInfo &info) { return info.size <= 1; });
  });
}

static Value pinDimToZero(OpBuilder &builder, Location loc, Value dest,
                          int64_t removedDim) {
  ArrayRef<int64_t> destShape = cast<ShapedType>(dest.getType()).getShape();
  SmallVector<SmallString<8>> nameStorage;
  SmallVector<StringRef> keptNames;
  SmallVector<uint32_t> keptDims;
  SmallVector<int64_t> keptShape;
  nameStorage.reserve(destShape.size());
  for (size_t i = 0; i < destShape.size(); ++i) {
    SmallString<8> name;
    ("dim" + Twine(i)).toVector(name);
    nameStorage.push_back(name);
  }
  for (auto [i, size] : llvm::enumerate(destShape)) {
    if (static_cast<int64_t>(i) == removedDim)
      continue;
    keptNames.push_back(nameStorage[i]);
    keptDims.push_back(static_cast<uint32_t>(i));
    keptShape.push_back(size);
  }

  TopDownTMBuilder view(builder, keptNames, keptShape, loc);
  if (!keptNames.empty())
    view.passThrough(keptNames, keptDims, keptNames);
  view.constDim(nameStorage[removedDim], static_cast<uint32_t>(removedDim),
                /*constantVal=*/0, /*lowerSize=*/destShape[removedDim]);
  return TransformOp::create(builder, loc, dest, view.get());
}

/// Recursively convert a full-tensor value to its tile equivalent.
/// Handles UntileOp, StoreMarkerOp and fusion ops.
/// Also collects StoreMarkerOp info if found.
static FailureOr<Value> convertToTile(OpBuilder &builder, Location loc,
                                      Value fullVal, Type tileType,
                                      IRMapping &fullToTileMapping) {
  // Check if already converted
  if (fullToTileMapping.contains(fullVal))
    return fullToTileMapping.lookup(fullVal);

  Operation *defOp = fullVal.getDefiningOp();
  if (!defOp) {
    // Block argument - shouldn't happen
    return failure();
  }

  // UntileOp: used for extra fusion inputs (from InsertOutputFusionLoads).
  // Its source is already a tile (from BlockwiseLoadOp), just return it.
  if (auto untileOp = dyn_cast<UntileOp>(defOp)) {
    Value tile = untileOp.getSource();
    fullToTileMapping.map(fullVal, tile);
    return tile;
  }

  // StoreMarkerOp: extract info and return the tile source
  if (auto storeMarkerOp = dyn_cast<StoreMarkerOp>(defOp)) {
    Value tile = storeMarkerOp.getSource();
    fullToTileMapping.map(fullVal, tile);
    return tile;
  }

  // Fusion op: recursively convert operands and clone with tile types.
  // For type-changing ops like arith.extf (f16->f32), each operand's tile type
  // must use the operand's element type, not the result's.
  if (rock::isFusionOp(defOp)) {
    auto tileShape = cast<RankedTensorType>(tileType).getShape();
    IRMapping fusionMapping;
    for (Value operand : defOp->getOperands()) {
      auto operandElemType =
          cast<RankedTensorType>(operand.getType()).getElementType();
      auto operandTileType = RankedTensorType::get(tileShape, operandElemType);
      FailureOr<Value> maybeTileOperand = convertToTile(
          builder, loc, operand, operandTileType, fullToTileMapping);
      if (failed(maybeTileOperand))
        return failure();
      fusionMapping.map(operand, maybeTileOperand.value());
    }

    // Clone the fusion op with tile operands
    Operation *clonedOp = builder.clone(*defOp, fusionMapping);
    clonedOp->getResult(0).setType(tileType);

    fullToTileMapping.map(fullVal, clonedOp->getResult(0));
    return clonedOp->getResult(0);
  }

  // Other ops: shouldn't happen for well-formed IR
  return failure();
}

static LogicalResult lowerStoreImpl(StoreOp storeOp,
                                    ReductionStorePath *reduction) {
  OpBuilder builder(storeOp);
  Location loc = storeOp.getLoc();
  Value storeSource = reduction ? reduction->tileSource : storeOp.getSource();
  Value storeDest = storeOp.getDest();

  FailureOr<StoreMarkerOp> maybeStoreMarkerOp = findStoreMarkerOp(storeSource);
  if (failed(maybeStoreMarkerOp))
    return storeOp.emitError("No StoreMarkerOp found for store");
  StoreMarkerOp storeMarkerOp = *maybeStoreMarkerOp;

  auto markerType = cast<RankedTensorType>(storeMarkerOp.getSource().getType());
  auto outputType = cast<RankedTensorType>(storeSource.getType());
  auto tileType =
      RankedTensorType::get(markerType.getShape(), outputType.getElementType());

  IRMapping fullToTileMapping;
  FailureOr<Value> maybeFusedTile =
      convertToTile(builder, loc, storeSource, tileType, fullToTileMapping);
  if (failed(maybeFusedTile))
    return storeOp.emitError("Failed to convert to tile");
  Value fusedTile = *maybeFusedTile;

  if (reduction) {
    if (!reduction->postReduceTransforms.empty()) {
      ArrayAttr inverse =
          invertTransformChain(builder, loc, reduction->postReduceTransforms);
      if (!inverse)
        return reduction->reduceOp.emitError(
            "cannot invert post-reduction destination transforms");
      storeDest = rock::transform(builder, storeDest, inverse);
    }

    auto reduceInputType =
        cast<RankedTensorType>(reduction->reduceOp.getIn().getType());
    storeDest =
        insertBroadcast(builder, loc, storeDest, reduceInputType.getShape());

    if (!reduction->preReduceTransforms.empty()) {
      ArrayAttr inverse =
          invertTransformChain(builder, loc, reduction->preReduceTransforms);
      if (!inverse)
        return reduction->reduceOp.emitError(
            "cannot invert pre-reduction input transforms");
      storeDest = rock::transform(builder, storeDest, inverse);
    }

    if (failed(
            setStoreMethodAndPrefill(builder, storeOp, StoreMethod::AtomicAdd)))
      return failure();
  }

  SmallVector<TransformMapAttr> destTransforms;
  Value destRoot;
  std::tie(destRoot, std::ignore) =
      rock::untransform(storeDest, destTransforms);

  SmallVector<Attribute> combinedTransforms;
  llvm::append_range(combinedTransforms, storeMarkerOp.getExtraViews());
  llvm::append_range(combinedTransforms, destTransforms);

  Value combinedDest = destRoot;
  if (!combinedTransforms.empty())
    combinedDest = rock::transform(builder, destRoot,
                                   builder.getArrayAttr(combinedTransforms));

  if (reduction) {
    ArrayRef<int64_t> tileShape =
        cast<RankedTensorType>(fusedTile.getType()).getShape();
    int64_t numExtra = storeMarkerOp.getExtraIndices().size();
    int64_t reductionAxis = -1;
    int64_t reductionExtent = 1;
    for (auto [axis, extent] : llvm::enumerate(tileShape)) {
      if (extent <= reductionExtent)
        continue;
      int64_t destDim = numExtra + static_cast<int64_t>(axis);
      if (!addressIsInvariantAlongDim(builder, combinedDest, destDim))
        continue;
      reductionAxis = static_cast<int64_t>(axis);
      reductionExtent = extent;
    }

    if (reductionAxis >= 0) {
      fusedTile = BlockwiseReduceOp::create(
          builder, loc, fusedTile, builder.getIndexAttr(reductionAxis),
          builder.getAttr<ReduceMethodAttr>(ReduceMethod::Sum));
      combinedDest =
          pinDimToZero(builder, loc, combinedDest, numExtra + reductionAxis);
    }
    // No invariant axis is still valid: retain the atomic store for every
    // tile element, which is the legacy reduction behavior.
  }

  auto blockwiseStore = BlockwiseStoreOp::create(
      builder, loc, storeOp.getResult().getType(), fusedTile, combinedDest,
      storeOp.getResultAlias(), storeMarkerOp.getExtraIndices(),
      storeOp.getStoreMethod());
  LLVM_DEBUG(llvm::dbgs() << "Created BlockwiseStoreOp: " << blockwiseStore
                          << "\n");
  storeOp.getResult().replaceAllUsesWith(blockwiseStore.getResult());
  storeOp.erase();
  return success();
}

static LogicalResult lowerOrdinaryStore(StoreOp storeOp) {
  return lowerStoreImpl(storeOp, nullptr);
}

static LogicalResult lowerReductionStore(StoreOp storeOp,
                                         ReductionStorePath &path) {
  return lowerStoreImpl(storeOp, &path);
}

struct RockLowerStoresPass
    : public rock::impl::RockLowerStoresPassBase<RockLowerStoresPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockLowerStoresPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic())) {
    return;
  }

  // Collect all rock.store ops
  SmallVector<StoreOp> storeOps;
  funcOp.walk([&](StoreOp storeOp) { storeOps.push_back(storeOp); });

  LLVM_DEBUG(llvm::dbgs() << "Found " << storeOps.size()
                          << " rock.store ops to process\n");

  for (StoreOp storeOp : storeOps) {
    FailureOr<std::optional<ReductionStorePath>> maybePath =
        getReductionStorePath(storeOp);
    if (failed(maybePath)) {
      storeOp.emitError("malformed blockwise reduction store path");
      return signalPassFailure();
    }

    LogicalResult lowered = success();
    if (*maybePath) {
      ReductionStorePath &path = **maybePath;
      if (!path.reduceOp.getBlockwise()) {
        path.reduceOp.emitError(
            "unselected reduction reached rock-lower-stores");
        return signalPassFailure();
      }
      lowered = lowerReductionStore(storeOp, path);
    } else {
      lowered = lowerOrdinaryStore(storeOp);
    }
    if (failed(lowered))
      return signalPassFailure();
  }
}
