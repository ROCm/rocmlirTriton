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

} // namespace rock
} // namespace mlir
#endif // ROCK_TRANSFORMS_POINTERARITHEXPAND_H
