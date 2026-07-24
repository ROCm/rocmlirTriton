//===- RegularizeInput.cpp - Push fusions past load_markers ---------------===//
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
// This pass runs AFTER InsertOutputFusionLoads and BEFORE RockLowerLoads.
// It ensures that every rock.load_marker's source is a pure transform chain
// (no fusion ops in between), by distributing load_markers past any fusions.
//
// For each load_marker whose source traces through fusion ops:
// 1. Walk backwards collecting transforms between the marker and fusions
// 2. For each leaf of the fusion DAG (transform chain to block arg):
//    apply the collected post-fusion transforms, isolate the chain,
//    create a new load_marker
// 3. For splat constants: create a tile-shaped constant (no load_marker)
// 4. Clone fusion ops after the load_markers, operating on tile types
// 5. Replace the original load_marker with the cloned fusion result
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREGULARIZEINPUTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-regularize-input"

using namespace mlir;
using namespace mlir::rock;

namespace {

using LeafKey = std::pair<Value, ArrayAttr>;

struct AnalyzedInputPath {
  LeafKey key;
  ArrayAttr combinedViews;
  SmallVector<TransformMapAttr> transforms;
};

struct NarrowLoadPlan {
  ArrayAttr combinedViews;
  RankedTensorType resultType;
  SmallVector<Value> extraIndices;
  unsigned removedTileAxis;
};

struct RemovableTileAxis {
  unsigned tileAxis;
  SmallVector<unsigned> partition;
};

/// Apply a list of TransformMapAttrs on top of `source`, creating
/// rock.transform ops.
static Value applyTransforms(OpBuilder &builder, Value source,
                             ArrayRef<TransformMapAttr> transforms) {
  SmallVector<Attribute> attrs(transforms.begin(), transforms.end());
  ArrayAttr transformsAttr = builder.getArrayAttr(attrs);
  return rock::transform(builder, source, transformsAttr);
}

/// Return the number of leading transform maps shared by every input path.
/// Each path is ordered from the load marker's upper coordinate space toward
/// its underlying input, so maps after this prefix are specific to that input.
/// Requires at least one path.
static size_t getCommonViewPrefix(ArrayRef<AnalyzedInputPath> paths) {
  // A common prefix cannot be longer than the shortest transform chain.
  size_t commonSize = paths.front().transforms.size();
  for (const AnalyzedInputPath &path : paths)
    commonSize = std::min(commonSize, path.transforms.size());

  // Stop at the first position whose transform differs on any input path.
  size_t prefixSize = 0;
  for (; prefixSize < commonSize; ++prefixSize) {
    TransformMapAttr expected = paths.front().transforms[prefixSize];
    if (llvm::any_of(paths.drop_front(), [&](const AnalyzedInputPath &path) {
          return path.transforms[prefixSize] != expected;
        }))
      break;
  }
  return prefixSize;
}

/// Return all top-level dimensions that form the same logical partition as
/// `tileDim`. Tiled dimensions are represented by an Unmerge containing a
/// block dimension and an iteration dimension. Both must be removed when the
/// underlying input is broadcast along that logical dimension.
static SmallVector<unsigned>
getCompleteTilePartition(ArrayRef<TransformMapAttr> transforms,
                         unsigned tileDim, unsigned numExtraDims) {
  if (transforms.empty())
    return {};

  for (TransformAttr transform : transforms.front().getOps()) {
    ArrayRef<uint32_t> upperDims = transform.getUpperDims();
    if (!llvm::is_contained(upperDims, tileDim))
      continue;

    if (transform.getType() != TransformType::Unmerge)
      return {tileDim};

    SmallVector<unsigned> partition(upperDims.begin(), upperDims.end());
    if (llvm::any_of(partition, [&](unsigned dim) {
          return dim >= numExtraDims && dim != tileDim;
        }))
      return {};
    return partition;
  }

  return {};
}

/// Combine the load marker's views with each path's input-specific transforms.
/// Constants and block arguments from nested regions are not loadable function
/// inputs, so only entry-block arguments are retained.
static FailureOr<SmallVector<AnalyzedInputPath>>
analyzeInputPaths(OpBuilder &builder, Value markerSource, ArrayAttr extraViews,
                  func::FuncOp funcOp) {
  FailureOr<SmallVector<InputFusionPath>> inputPaths =
      collectInputFusionPaths(markerSource);
  if (failed(inputPaths))
    return failure();

  SmallVector<AnalyzedInputPath> analyzedPaths;
  analyzedPaths.reserve(inputPaths->size());
  for (const InputFusionPath &path : *inputPaths) {
    auto blockArg = dyn_cast<BlockArgument>(path.leaf);
    if (!blockArg || blockArg.getOwner() != &funcOp.front())
      continue;

    ArrayAttr postTransformsAttr = builder.getArrayAttr(
        SmallVector<Attribute>(path.transforms.begin(), path.transforms.end()));
    ArrayAttr combined =
        prependUpperViews(builder, extraViews, postTransformsAttr);
    analyzedPaths.push_back(
        {{path.leaf, postTransformsAttr},
         combined,
         llvm::to_vector(combined.getAsRange<TransformMapAttr>())});
  }
  return analyzedPaths;
}

/// Return true if every dimension in `partition` can be removed from `path`.
/// A removable dimension must satisfy three conditions:
/// 1. This path's address is independent of it, proving the input is broadcast.
/// 2. Another input retains it, providing the full-width fused result.
/// 3. No path-specific validity check uses it, since such a mask could not be
///    reconstructed from that full-width sibling.
static bool canRemoveTilePartition(const AnalyzedInputPath &path,
                                   ArrayRef<AnalyzedInputPath> paths,
                                   ArrayRef<unsigned> partition,
                                   size_t commonPrefixSize) {
  return llvm::all_of(partition, [&](unsigned partitionDim) {
    ArrayRef<unsigned> partitionDims(&partitionDim, 1);
    if (transformChainDependsOnAnyDim(path.transforms, partitionDims))
      return false;

    bool hasFullSibling =
        llvm::any_of(paths, [&](const AnalyzedInputPath &sibling) {
          return sibling.key != path.key &&
                 transformChainDependsOnAnyDim(sibling.transforms,
                                               partitionDims);
        });
    if (!hasFullSibling)
      return false;

    return !validityDependsOnAnyDim(path.transforms, partitionDims,
                                    commonPrefixSize);
  });
}

/// Find tile axes whose complete logical partition can be removed from `path`.
/// Unit axes are already narrow, and incomplete Unmerge partitions are rejected
/// by `getCompleteTilePartition`.
static SmallVector<RemovableTileAxis>
findRemovableTileAxes(const AnalyzedInputPath &path,
                      ArrayRef<AnalyzedInputPath> paths,
                      ArrayRef<int64_t> fullShape, unsigned numExtraDims,
                      size_t commonPrefixSize) {
  SmallVector<RemovableTileAxis> removableAxes;
  for (unsigned tileAxis = 0; tileAxis < fullShape.size(); ++tileAxis) {
    if (fullShape[tileAxis] <= 1)
      continue;

    unsigned topDim = numExtraDims + tileAxis;
    SmallVector<unsigned> partition =
        getCompleteTilePartition(path.transforms, topDim, numExtraDims);
    if (partition.empty() ||
        !canRemoveTilePartition(path, paths, partition, commonPrefixSize))
      continue;

    removableAxes.push_back({tileAxis, std::move(partition)});
  }
  return removableAxes;
}

/// Build a plan for one path and one previously validated removable axis.
/// This removes the axis's complete logical partition from the views and extra
/// indices. The shape checks ensure that the resulting rank-1 tile still maps
/// exactly onto the original input tensor.
static FailureOr<NarrowLoadPlan>
createPlanForRemovableAxis(OpBuilder &builder, const AnalyzedInputPath &path,
                           const RemovableTileAxis &removableAxis,
                           ValueRange extraIndices,
                           RankedTensorType fullTileType) {
  unsigned removedTileAxis = removableAxis.tileAxis;
  llvm::SetVector<int64_t> removeDims;
  for (unsigned dim : removableAxis.partition)
    removeDims.insert(dim);

  SmallVector<Value> narrowExtraIndices;
  for (auto [index, value] : llvm::enumerate(extraIndices))
    if (!removeDims.contains(index))
      narrowExtraIndices.push_back(value);

  FailureOr<ArrayAttr> narrowedViews =
      removeUpperDims(builder, path.combinedViews, removeDims);
  if (failed(narrowedViews) || narrowedViews->empty())
    return failure();

  auto topMap = cast<TransformMapAttr>((*narrowedViews)[0]);
  ArrayRef<int64_t> upperBounds = topMap.getUpperBounds();
  constexpr int64_t narrowTileRank = 1;
  if (upperBounds.size() != narrowExtraIndices.size() + narrowTileRank)
    return failure();

  SmallVector<int64_t> expectedShape(fullTileType.getShape());
  expectedShape.erase(expectedShape.begin() + removedTileAxis);
  ArrayRef<int64_t> narrowShape = upperBounds.take_back(narrowTileRank);
  if (narrowShape != ArrayRef<int64_t>(expectedShape))
    return failure();

  auto inputType = cast<RankedTensorType>(path.key.first.getType());
  if (getLowerShape(*narrowedViews) != inputType.getShape())
    return failure();

  auto resultType = RankedTensorType::get(
      narrowShape, fullTileType.getElementType(), fullTileType.getEncoding());
  return NarrowLoadPlan{*narrowedViews, resultType,
                        std::move(narrowExtraIndices), removedTileAxis};
}

/// Analyze the load marker's input-fusion DAG for rank-reducible broadcasts.
/// This optimization currently narrows rank-2 tiles to rank 1, so each accepted
/// path must have exactly one safely removable tile axis. Paths with no
/// removable axis, multiple possible axes, or an invalid rewritten view are
/// left unchanged. The result is keyed by the input leaf and its path-specific
/// transform sequence for use while distributing the load marker.
static DenseMap<LeafKey, NarrowLoadPlan>
analyzeNarrowBroadcastLoads(OpBuilder &builder, Value markerSource,
                            ArrayAttr extraViews, ValueRange extraIndices,
                            RankedTensorType fullTileType,
                            func::FuncOp funcOp) {
  DenseMap<LeafKey, NarrowLoadPlan> plans;
  if (fullTileType.getRank() != 2 || extraViews.empty())
    return plans;

  FailureOr<SmallVector<AnalyzedInputPath>> analyzedPaths =
      analyzeInputPaths(builder, markerSource, extraViews, funcOp);
  if (failed(analyzedPaths) || analyzedPaths->size() < 2)
    return plans;

  size_t commonPrefixSize = getCommonViewPrefix(*analyzedPaths);
  ArrayRef<int64_t> fullShape = fullTileType.getShape();
  unsigned numExtraDims = extraIndices.size();

  for (const AnalyzedInputPath &path : *analyzedPaths) {
    SmallVector<RemovableTileAxis> removableAxes = findRemovableTileAxes(
        path, *analyzedPaths, fullShape, numExtraDims, commonPrefixSize);
    if (removableAxes.size() != 1)
      continue;

    FailureOr<NarrowLoadPlan> plan = createPlanForRemovableAxis(
        builder, path, removableAxes.front(), extraIndices, fullTileType);
    if (succeeded(plan))
      plans.insert({path.key, std::move(*plan)});
  }

  return plans;
}

/// Recursively resolve a value in the fusion DAG, returning a tile-typed
/// replacement. Each leaf (block arg or constant) gets its own load_marker;
/// fusion ops are cloned to operate on tile types. Transforms are accumulated
/// during the walk and applied at the leaves.
static FailureOr<Value> distributeLoadMarker(
    OpBuilder &builder, Location loc, Value originalVal,
    ArrayRef<TransformMapAttr> postTransforms, ArrayAttr extraViews,
    ValueRange extraIndices, Type tileType, CacheModifier cache,
    llvm::DenseMap<std::pair<Value, ArrayAttr>, Value> &valueMapping,
    const DenseMap<LeafKey, NarrowLoadPlan> &narrowLoadPlans,
    func::FuncOp funcOp) {
  ArrayAttr postTransformsAttr = builder.getArrayAttr(
      SmallVector<Attribute>(postTransforms.begin(), postTransforms.end()));
  std::pair<Value, ArrayAttr> cacheKey{originalVal, postTransformsAttr};
  auto cached = valueMapping.find(cacheKey);
  if (cached != valueMapping.end())
    return cached->second;

  Operation *defOp = originalVal.getDefiningOp();

  // Leaf: block argument — apply postTransforms, create new load_marker.
  if (!defOp) {
    auto blockArg = cast<BlockArgument>(originalVal);
    if (blockArg.getOwner() != &funcOp.front())
      return failure();

    auto narrowPlan = narrowLoadPlans.find(cacheKey);
    if (narrowPlan != narrowLoadPlans.end()) {
      const NarrowLoadPlan &plan = narrowPlan->second;
      auto newMarker =
          LoadMarkerOp::create(builder, loc, plan.resultType, originalVal,
                               plan.combinedViews, plan.extraIndices, cache);
      Value fullTile = expandDimAndBroadcast(
          builder, loc, newMarker.getResult(), plan.removedTileAxis,
          cast<RankedTensorType>(tileType));
      valueMapping.insert({cacheKey, fullTile});
      return fullTile;
    }

    Value source = applyTransforms(builder, originalVal, postTransforms);
    auto newMarker = LoadMarkerOp::create(builder, loc, tileType, source,
                                          extraViews, extraIndices, cache);
    valueMapping.insert({cacheKey, newMarker.getResult()});
    return newMarker.getResult();
  }

  // rock.transform — accumulate and recurse.
  if (auto transformOp = dyn_cast<TransformOp>(defOp)) {
    SmallVector<TransformMapAttr> newPostTransforms;
    newPostTransforms.append(postTransforms.begin(), postTransforms.end());
    newPostTransforms.push_back(transformOp.getTransform());

    FailureOr<Value> result = distributeLoadMarker(
        builder, loc, transformOp.getInput(), newPostTransforms, extraViews,
        extraIndices, tileType, cache, valueMapping, narrowLoadPlans, funcOp);
    if (succeeded(result))
      valueMapping.insert({cacheKey, result.value()});
    return result;
  }

  // Fusion op — recurse on each operand, clone with tile types.
  if (rock::isFusionOp(defOp)) {
    auto tileShape = cast<RankedTensorType>(tileType).getShape();
    IRMapping fusionMapping;
    for (Value operand : defOp->getOperands()) {
      auto operandElemType =
          cast<RankedTensorType>(operand.getType()).getElementType();
      auto operandTileType = RankedTensorType::get(tileShape, operandElemType);
      FailureOr<Value> resolved = distributeLoadMarker(
          builder, loc, operand, postTransforms, extraViews, extraIndices,
          operandTileType, cache, valueMapping, narrowLoadPlans, funcOp);
      if (failed(resolved))
        return failure();
      fusionMapping.map(operand, resolved.value());
    }

    Operation *cloned = builder.clone(*defOp, fusionMapping);
    cloned->getResult(0).setType(tileType);
    valueMapping.insert({cacheKey, cloned->getResult(0)});
    return cloned->getResult(0);
  }

  // Leaf: splat arith.constant — reshape to tile type.
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!attr || !attr.isSplat()) {
      defOp->emitOpError("non-splat constant in fusion chain not supported");
      return failure();
    }
    auto tileRankedType = cast<RankedTensorType>(tileType);
    auto newAttr =
        DenseElementsAttr::get(tileRankedType, attr.getSplatValue<Attribute>());
    auto newConst =
        arith::ConstantOp::create(builder, loc, tileRankedType, newAttr);
    valueMapping.insert({cacheKey, newConst.getResult()});
    return newConst.getResult();
  }

  return failure();
}

struct RockRegularizeInputPass
    : public rock::impl::RockRegularizeInputPassBase<RockRegularizeInputPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void RockRegularizeInputPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (!funcOp->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  SmallVector<LoadMarkerOp> loadMarkersToProcess;
  funcOp.walk(
      [&](LoadMarkerOp markerOp) { loadMarkersToProcess.push_back(markerOp); });

  LLVM_DEBUG(llvm::dbgs() << "RegularizeInput: found "
                          << loadMarkersToProcess.size()
                          << " load_marker ops\n");

  for (LoadMarkerOp markerOp : loadMarkersToProcess) {
    LLVM_DEBUG(llvm::dbgs() << "  Processing: " << markerOp << "\n");

    OpBuilder builder(markerOp);
    Location loc = markerOp.getLoc();
    auto tileType = cast<RankedTensorType>(markerOp.getResult().getType());

    llvm::DenseMap<std::pair<Value, ArrayAttr>, Value> valueMapping;
    DenseMap<LeafKey, NarrowLoadPlan> narrowLoadPlans =
        analyzeNarrowBroadcastLoads(
            builder, markerOp.getSource(), markerOp.getExtraViews(),
            markerOp.getExtraIndices(), tileType, funcOp);
    FailureOr<Value> replacement = distributeLoadMarker(
        builder, loc, markerOp.getSource(), /*postTransforms=*/{},
        markerOp.getExtraViews(), markerOp.getExtraIndices(), tileType,
        markerOp.getCacheModifier(), valueMapping, narrowLoadPlans, funcOp);

    if (failed(replacement)) {
      markerOp->emitError("Failed to distribute load_marker past fusions");
      return signalPassFailure();
    }

    markerOp.getResult().replaceAllUsesWith(replacement.value());
    markerOp.erase();
  }
}
