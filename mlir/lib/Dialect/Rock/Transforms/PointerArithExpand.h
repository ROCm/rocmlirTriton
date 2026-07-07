//===- PointerArithExpand.h - shared transform->arith helpers ------------===//
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
//===----------------------------------------------------------------------===//
//
// Small shared helpers for building arith/triton coordinate arithmetic out of
// rock transform maps. These are the primitives shared by
// TransformsToPointerArith and TransformsInvariantCodeMotion passes.
//
//===----------------------------------------------------------------------===//

#ifndef ROCK_TRANSFORMS_POINTERARITHEXPAND_H
#define ROCK_TRANSFORMS_POINTERARITHEXPAND_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
class AffineMap;
class Location;
class OpBuilder;

namespace rock {

/// Create a tensor range using tt.make_range. The result has `numDims`
/// dimensions; all are unit except `nonUnitDim`, which holds the values
/// `[start, end)`.
Value makeRange(OpBuilder &b, Location loc, int32_t start, int32_t end,
                int64_t numDims, int64_t nonUnitDim);

/// Broadcast two (scalar or tensor) values to a common shape, inserting the
/// needed triton.splat / triton.broadcast ops, and return the rewired pair.
std::pair<Value, Value> ensureCompatible(OpBuilder &b, Location loc, Value lhs,
                                         Value rhs);

/// Splat/broadcast `v` (scalar or tensor) to a tensor of shape `shape`.
Value broadcastToShape(OpBuilder &b, Location loc, Value v,
                       ArrayRef<int64_t> shape);

/// Expand `affineMap` applied to `operands` (which may be scalars or tensors)
/// into arith ops, returning one Value per result expression. Mirrors
/// AffineApplyOp lowering but tolerates tensor operands via broadcasting.
FailureOr<SmallVector<Value>> expandAffineMap(OpBuilder &b, Location loc,
                                              AffineMap affineMap,
                                              ValueRange operands);

/// Emit the validity (bounds) checks contributed by a single validity-impacting
/// `map`, given the just-computed lower coordinates `outputs`. Returns an i1
/// value (scalar or tensor) that is true where every check passes.
Value updateValidityAfter(OpBuilder &b, Location loc, TransformMapAttr map,
                          ValueRange outputs);

/// The linearized buffer offset and validity mask produced by a transform
/// chain. `offset` is an i32 value (scalar or tensor); `mask` is an i1 value.
struct OffsetAndMask {
  Value offset;
  Value mask;
};

/// Run the transform chain `transforms` (ordered view->buffer) starting from
/// `startCoords` (the upper-space coordinates of `transforms.front()`), and
/// return the resulting linearized offset and validity mask, both broadcast to
/// `outShape`. The TransformsToPtrOp lowering seeds `startCoords` with extra
/// indices + per-tile ranges.
FailureOr<OffsetAndMask>
expandCoordsToOffsetAndMask(OpBuilder &b, Location loc,
                            ArrayRef<TransformMapAttr> transforms,
                            ValueRange startCoords, ArrayRef<int64_t> outShape);

/// Like `expandCoordsToOffsetAndMask`, but computes *only* the validity mask
/// (broadcast to `outShape`) and skips the final offset linearization. Used by
/// the carry-based LICM path when the offset is maintained by an incremental
/// pointer recurrence: the only per-iteration coordinate work left is the
/// validity check, so re-linearizing the offset would be wasted arithmetic.
/// Coordinates are still expanded through the validity-impacting maps (the
/// padding halo moves with the iv), but the trailing offset-only segment is
/// dropped. Returns an all-true mask when no map impacts validity.
FailureOr<Value> expandCoordsToMask(OpBuilder &b, Location loc,
                                    ArrayRef<TransformMapAttr> transforms,
                                    ValueRange startCoords,
                                    ArrayRef<int64_t> outShape);

} // namespace rock
} // namespace mlir
#endif // ROCK_TRANSFORMS_POINTERARITHEXPAND_H
