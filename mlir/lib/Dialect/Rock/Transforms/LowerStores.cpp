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
// - rock.store storing the fused result
//
// This pass:
// 1. Traces back from rock.store through fusion ops to find StoreMarkerOp
// 2. Gets the tile values and transforms from StoreMarkerOp
// 3. Clones fusion ops to operate on tiles
// 4. Combines StoreMarkerOp transforms with store destination transforms
// 5. Creates BlockwiseStoreOp with the combined transform chain
// 6. Reduces a whole tile axis in registers when the destination maps all of it
//    to one address, so one atomic replaces a tile axis worth of them
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
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <tuple>

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

  // Fusion op: recursively search operands
  if (rock::isFusionOp(defOp)) {
    for (Value operand : defOp->getOperands()) {
      if (StoreMarkerOp found = findStoreMarkerOp(operand).value_or(nullptr))
        return found;
    }
  }

  return failure();
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

/// Returns true if every coordinate of the buffer underlying `transforms` is
/// unchanged when upper dimension `dim` varies over its whole extent.
static bool addressIsInvariantAlongDim(OpBuilder &b, ArrayAttr transforms,
                                       int64_t dim) {
  FailureOr<llvm::SmallDenseMap<int64_t, SmallVector<SubDimInfo>>> subDims =
      getLowerSubDimensions(b, transforms, dim);
  if (failed(subDims)) {
    LLVM_DEBUG(llvm::dbgs()
               << "dim " << dim << ": sub-dimension analysis failed\n");
    return false;
  }
  for (const auto &[lowerDim, infos] : *subDims) {
    for (const SubDimInfo &info : infos) {
      if (info.size > 1) {
        LLVM_DEBUG(llvm::dbgs() << "dim " << dim << ": varies over lower dim "
                                << lowerDim << " as " << info << "\n");
        return false;
      }
    }
  }
  return true;
}

/// Returns a view of `dest` with upper dimension `destDim` dropped and pinned
/// to zero. Callers must have proven that neither the address nor the validity
/// mask depends on `destDim`, which makes every coordinate along it equivalent
/// to coordinate zero.
static Value pinDimToZero(OpBuilder &b, Location loc, Value dest,
                          int64_t destDim) {
  ArrayRef<int64_t> destShape = cast<ShapedType>(dest.getType()).getShape();
  assert(destDim >= 0 && destDim < static_cast<int64_t>(destShape.size()) &&
         "destDim must index the destination shape");

  // Fill the name storage before taking any StringRef into it, so that growing
  // the vector cannot invalidate the names handed to the builder.
  SmallVector<SmallString<8>> nameStorage;
  for (size_t i = 0, e = destShape.size(); i < e; ++i) {
    SmallString<8> name;
    ("dim" + Twine(i)).toVector(name);
    nameStorage.push_back(name);
  }

  SmallVector<StringRef> keptNames;
  SmallVector<uint32_t> keptDims;
  SmallVector<int64_t> keptShape;
  for (auto [i, size] : llvm::enumerate(destShape)) {
    if (static_cast<int64_t>(i) == destDim)
      continue;
    keptNames.push_back(nameStorage[i]);
    keptDims.push_back(static_cast<uint32_t>(i));
    keptShape.push_back(size);
  }

  TopDownTMBuilder view(b, keptNames, keptShape, loc);
  if (!keptNames.empty())
    view.passThrough(keptNames, keptDims, keptNames);
  view.constDim(nameStorage[destDim], static_cast<uint32_t>(destDim),
                /*constantVal=*/0, /*lowerSize=*/destShape[destDim]);
  return TransformOp::create(b, loc, dest, view.get());
}

/// Rewrites `store` to reduce one tile axis in registers before storing, when
/// that is provably equivalent to the per-element atomics it replaces.
static void tryReduceStore(OpBuilder &b, BlockwiseStoreOp store) {
  StoreMethod storeMethod = store.getStoreMethod();
  if (storeMethod != StoreMethod::AtomicAdd)
    return;

  ReduceMethod reduceMethod = storeMethod == StoreMethod::AtomicAdd
                                  ? ReduceMethod::Sum
                                  : ReduceMethod::Max;
  assert(reduceMethod != ReduceMethod::Max &&
         "atomic_max pre-reduction is not yet supported");

  auto srcType = cast<RankedTensorType>(store.getSource().getType());
  auto destType = cast<RankedTensorType>(store.getDest().getType());
  if (!srcType.hasStaticShape() || !destType.hasStaticShape())
    return;

  b.setInsertionPoint(store);
  ArrayAttr transformAttrs;
  std::tie(std::ignore, transformAttrs, std::ignore) =
      untransform(b, store.getDest());
  SmallVector<TransformMapAttr> transforms =
      llvm::to_vector(transformAttrs.getAsRange<TransformMapAttr>());
  if (transforms.empty())
    return;

  // If any destination transform affects validity, some lanes may be masked
  // off. Since this rewrite does not track which lanes contribute to the
  // store, reducing them together could include values that never reach memory.
  if (llvm::any_of(transforms, mapImpactsValidity))
    return;

  // A BlockwiseStoreOp's leading destination dimensions are addressed by
  // extraIndices, so source axis `a` is destination upper dimension
  // `extraIndices.size() + a`.
  int64_t numExtra = store.getExtraIndices().size();
  int64_t largestAddressInvariantAxis = -1;
  int64_t largestAddressInvariantExtent = 1;
  for (auto [axis, extent] : llvm::enumerate(srcType.getShape())) {
    if (extent <= largestAddressInvariantExtent)
      continue;
    int64_t destDim = numExtra + static_cast<int64_t>(axis);
    if (!addressIsInvariantAlongDim(b, transformAttrs, destDim))
      continue;
    largestAddressInvariantAxis = static_cast<int64_t>(axis);
    largestAddressInvariantExtent = extent;
  }
  if (largestAddressInvariantAxis < 0)
    return;

  LLVM_DEBUG(llvm::dbgs() << "reducing axis " << largestAddressInvariantAxis
                          << " of " << srcType << " into destination dim "
                          << (numExtra + largestAddressInvariantAxis) << "\n");
  Location loc = store.getLoc();
  Value reduced = BlockwiseReduceOp::create(
      b, loc, store.getSource(), b.getIndexAttr(largestAddressInvariantAxis),
      b.getAttr<ReduceMethodAttr>(reduceMethod));
  Value pinnedDest = pinDimToZero(b, loc, store.getDest(),
                                  numExtra + largestAddressInvariantAxis);
  store.getSourceMutable().assign(reduced);
  store.getDestMutable().assign(pinnedDest);
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
  funcOp.walk([&](StoreOp storeOp) {
    storeOps.push_back(storeOp);
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << storeOps.size()
                          << " rock.store ops to process\n");

  for (StoreOp storeOp : storeOps) {
    OpBuilder builder(storeOp);
    Location loc = storeOp.getLoc();
    
    Value storeSource = storeOp.getSource();  // The fused result (full tensor)
    Value storeDest = storeOp.getDest();      // The destination (transformed arg)

    // Find the StoreMarkerOp in the store chain to get transforms and indices
    FailureOr<StoreMarkerOp> maybeStoreMarkerOp =
        findStoreMarkerOp(storeSource);
    if (failed(maybeStoreMarkerOp)) {
      storeOp->emitError("No StoreMarkerOp found for store");
      return signalPassFailure();
    }
    auto storeMarkerOp = maybeStoreMarkerOp.value();

    auto sourceType =
        cast<RankedTensorType>(storeMarkerOp.getSource().getType());
    ArrayRef<int64_t> storeMarkerShape = sourceType.getShape();

    // Get the output type for determining tile type
    auto outputType = cast<RankedTensorType>(storeSource.getType());

    // Determine tile type
    auto tileType =
        RankedTensorType::get(storeMarkerShape, outputType.getElementType());

    // Convert the store source from full tensor to tile, collecting
    // StoreMarkerOp info
    IRMapping fullToTileMapping;
    FailureOr<Value> maybeFusedTile =
        convertToTile(builder, loc, storeSource, tileType, fullToTileMapping);
    if (failed(maybeFusedTile)) {
      storeOp->emitError("Failed to convert to tile");
      return signalPassFailure();
    }
    Value fusedTile = maybeFusedTile.value();

    LLVM_DEBUG(llvm::dbgs() << "Converted store source to tile: " << fusedTile
                            << "\n");

    // Combine transforms: StoreMarkerOp's extraViews are applied on top of
    // any existing transforms on storeDest.
    // Use untransform to get existing transforms, then combine with
    // StoreMarkerOp's
    SmallVector<TransformMapAttr> destTransforms;
    auto [destRoot, _] = rock::untransform(storeDest, destTransforms);

    // Build combined transforms: extraViews (from StoreMarkerOp) +
    // destTransforms
    SmallVector<Attribute> combinedTransforms;
    combinedTransforms.append(storeMarkerOp.getExtraViews().begin(),
                              storeMarkerOp.getExtraViews().end());
    combinedTransforms.append(destTransforms.begin(), destTransforms.end());

    // Apply combined transforms to the root destination
    Value combinedDest = destRoot;
    if (!combinedTransforms.empty()) {
      ArrayAttr transformsAttr = builder.getArrayAttr(combinedTransforms);
      combinedDest = rock::transform(builder, destRoot, transformsAttr);
    }

    // Create BlockwiseStoreOp with combined transforms
    auto bstOp = BlockwiseStoreOp::create(
        builder, loc, storeOp.getResult().getType(), fusedTile, combinedDest,
        storeOp.getResultAlias(), storeMarkerOp.getExtraIndices(),
        storeOp.getStoreMethod());

    LLVM_DEBUG(llvm::dbgs() << "Created BlockwiseStoreOp: " << bstOp << "\n");

    // Must run before the erase below, which leaves `builder` anchored to a
    // deleted op.
    tryReduceStore(builder, bstOp);

    // Replace rock.store with BlockwiseStoreOp result
    storeOp.getResult().replaceAllUsesWith(bstOp.getResult());
    storeOp.erase();
  }
}
