//===--------------------- ReduceBlockwiseAtomics.cpp ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// A reduction fused into a GEMM is represented by broadcasting the (smaller)
// destination back to the tile shape and turning the store into an atomic add.
// Every tile element then contributes its own atomic to an address that many
// of its neighbours share, which serializes on the memory system. When a whole
// tile axis maps to one address we can add those contributions up in registers
// first and issue a single atomic per remaining coordinate.
//
// Before:
//   rock.blockwise_store %tile -> %dest[...] by atomic_add
//     : tensor<64x256xf32> -> tensor<1x5x128x64x256xf32> -> tensor<64xf32>
//
// After:
//   %red = rock.blockwise_reduce sum %tile {axis = 1}
//     : tensor<64x256xf32> -> tensor<64xf32>
//   %pinned = rock.transform %dest by <ConstDim{0, 256} on dim 4>
//     : tensor<1x5x128x64x256xf32> to tensor<1x5x128x64xf32>
//   rock.blockwise_store %red -> %pinned[...] by atomic_add
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <tuple>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREDUCEBLOCKWISEATOMICSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-reduce-blockwise-atomics"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockReduceBlockwiseAtomicsPass
    : public rock::impl::RockReduceBlockwiseAtomicsPassBase<
          RockReduceBlockwiseAtomicsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

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

/// Rewrites `store` to sum one tile axis in registers before storing, when
/// that is provably equivalent to the per-element atomics it replaces.
static void tryReduceStore(OpBuilder &b, BlockwiseStoreOp store) {
  // Only atomic_add accumulates, so only it can absorb a partial sum. A `set`
  // store keeps the last writer, and atomic_max is left for later because the
  // NaN and signedness semantics of arith.maximumf and the hardware atomic
  // still need to be reconciled.
  if (store.getStoreMethod() != StoreMethod::AtomicAdd)
    return;

  auto srcType = cast<RankedTensorType>(store.getSource().getType());
  auto destType = cast<RankedTensorType>(store.getDest().getType());
  if (!srcType.hasStaticShape() || !destType.hasStaticShape())
    return;

  // A rank-1 source would reduce to a rank-0 tensor, which
  // rock.blockwise_reduce can express but the tt.reduce it lowers to cannot:
  // reducing a rank-1 input yields a bare scalar.
  if (srcType.getRank() < 2)
    return;

  b.setInsertionPoint(store);
  ArrayAttr transformAttrs;
  std::tie(std::ignore, transformAttrs, std::ignore) =
      untransform(b, store.getDest());
  SmallVector<TransformMapAttr> transforms =
      llvm::to_vector(transformAttrs.getAsRange<TransformMapAttr>());
  if (transforms.empty())
    return;

  // A lane whose store mask is false contributes nothing, so whatever it holds
  // is irrelevant; once its value is folded into a sum that other lanes do
  // store, it would corrupt the result. Instead of asking which axis a mask
  // depends on, reject any chain that can produce a mask at all: then every
  // lane stores and no masked-off value can reach memory. This gives up nothing
  // in practice, because the only transforms that produce a mask are
  // non-trivial Pad and invalidatable Embed, and the sub-dimension analysis
  // below already refuses to look at a chain containing either.
  if (llvm::any_of(transforms, mapImpactsValidity))
    return;

  // A BlockwiseStoreOp's leading destination dimensions are addressed by
  // extraIndices, so source axis `a` is destination upper dimension
  // `extraIndices.size() + a`.
  int64_t numExtra = store.getExtraIndices().size();
  int64_t bestAxis = -1;
  int64_t bestExtent = 1;
  for (auto [axis, extent] : llvm::enumerate(srcType.getShape())) {
    if (extent <= bestExtent)
      continue;
    int64_t destDim = numExtra + static_cast<int64_t>(axis);
    if (!addressIsInvariantAlongDim(b, transformAttrs, destDim))
      continue;
    bestAxis = static_cast<int64_t>(axis);
    bestExtent = extent;
  }
  if (bestAxis < 0)
    return;

  LLVM_DEBUG(llvm::dbgs() << "summing axis " << bestAxis << " of " << srcType
                          << " into destination dim " << (numExtra + bestAxis)
                          << "\n");
  Location loc = store.getLoc();
  Value reduced = BlockwiseReduceOp::create(
      b, loc, store.getSource(), b.getIndexAttr(bestAxis),
      b.getAttr<ReduceMethodAttr>(ReduceMethod::Sum));
  Value pinnedDest = pinDimToZero(b, loc, store.getDest(), numExtra + bestAxis);
  store.getSourceMutable().assign(reduced);
  store.getDestMutable().assign(pinnedDest);
}

void RockReduceBlockwiseAtomicsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!func->hasAttr(rock::KernelAttr::getMnemonic()))
    return;

  OpBuilder b(func.getContext());
  func.walk([&](BlockwiseStoreOp store) { tryReduceStore(b, store); });
}
