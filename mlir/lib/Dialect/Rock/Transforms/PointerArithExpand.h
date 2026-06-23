//===- PointerArithExpand.h - shared transform->arith expansion ----------===//
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
// Shared machinery for expanding rock transform-map chains into arith/triton
// operations that compute a linearized buffer offset and a validity mask.
//
// This is the common core used by both:
//   * TransformsToPointerArith.cpp - the TransformsToPtrOp lowering, which
//     starts from the op's extra indices plus per-tile ranges, and
//   * TransformsInvariantCodeMotion.cpp - the carry-based LICM path, which
//     starts from loop-carried decomposed coordinate tensors instead.
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
/// `outShape`. This is the shared core of the TransformsToPtrOp lowering: the
/// lowering seeds `startCoords` with extra indices + tile ranges, while the
/// LICM carry path seeds it with loop-carried decomposed coordinate tensors.
FailureOr<OffsetAndMask>
expandCoordsToOffsetAndMask(OpBuilder &b, Location loc,
                            ArrayRef<TransformMapAttr> transforms,
                            ValueRange startCoords, ArrayRef<int64_t> outShape);

} // namespace rock
} // namespace mlir
#endif // ROCK_TRANSFORMS_POINTERARITHEXPAND_H
