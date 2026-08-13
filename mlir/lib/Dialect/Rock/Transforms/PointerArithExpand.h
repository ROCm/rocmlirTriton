//===- PointerArithExpand.h - shared transform->arith helpers ------------===//
//
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
class Type;

namespace rock {

/// Create a tensor range using tt.make_range. The result has `numDims`
/// dimensions; all are unit except `nonUnitDim`, which holds the values
/// `[start, end)`. The range is produced in i32 (tt.make_range requires i32)
/// and sign-extended to `elemType` when a wider (i64) index domain is needed.
Value makeRange(OpBuilder &b, Location loc, int32_t start, int32_t end,
                int64_t numDims, int64_t nonUnitDim, Type elemType);

/// Splat/broadcast `v` (scalar or tensor) to a tensor of shape `shape`.
/// A scalar is always splat. A tensor is broadcast only when its shape has the
/// same rank as `shape` and every dim either matches or is a unit dim; when it
/// is not broadcastable a null Value is returned (callers must check).
Value broadcastToShape(OpBuilder &b, Location loc, Value v,
                       ArrayRef<int64_t> shape);

/// Expand `affineMap` applied to `operands` (which may be scalars or tensors)
/// into arith ops, returning one Value per result expression. Mirrors
/// AffineApplyOp lowering but tolerates tensor operands via broadcasting.
/// `indexType` (i32 or i64) is the integer width used for any constants
/// introduced while expanding, and must match the width of `operands`.
FailureOr<SmallVector<Value>> expandAffineMap(OpBuilder &b, Location loc,
                                              AffineMap affineMap,
                                              ValueRange operands,
                                              Type indexType);

/// The linearized buffer offset and validity mask produced by a transform
/// chain. `mask` is an i1 value; `offset` is an integer value (i32 or i64,
/// scalar or tensor), or null when `computeOffset` is false.
struct OffsetAndMask {
  Value offset;
  Value mask;
};

/// Run the transform chain `transforms` (ordered view->buffer) starting from
/// `startCoords` (the upper-space coordinates of `transforms.front()`), and
/// return the resulting linearized offset and validity mask, both broadcast to
/// `outShape`. The TransformsToPtrOp lowering seeds `startCoords` with extra
/// indices + per-tile ranges.
///
/// When `computeOffset` is false, only the validity mask is produced: the
/// trailing offset-only maps are skipped and `offset` is left null. The
/// carry-based LICM path uses this because it maintains the offset via an
/// incremental pointer recurrence, so re-linearizing it would be wasted
/// arithmetic. An all-true mask is returned when no map impacts validity.
///
/// `indexType` (i32 or i64) is the integer width of the coordinate arithmetic
/// and of the resulting offset; it must match the width of `startCoords`.
FailureOr<OffsetAndMask>
expandCoordsToOffsetAndMask(OpBuilder &b, Location loc,
                            ArrayRef<TransformMapAttr> transforms,
                            ValueRange startCoords, ArrayRef<int64_t> outShape,
                            Type indexType, bool computeOffset = true);

} // namespace rock
} // namespace mlir
#endif // ROCK_TRANSFORMS_POINTERARITHEXPAND_H
