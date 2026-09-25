//===- dynamicDimUtils.h - Dynamic dimensions of rock kernels ---*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for the dynamic dimensions of rock kernel arguments: materializing
// them (and expressions over them) as i32 values, and the equalities between
// them that producers record with llvm.intr.assume.
//
// The canonical form of a dynamic dimension in kernel code is
//   %d = tensor.dim %argJ, %cI
//   %di = arith.index_cast %d : index to i32
// and an equality between two such dimensions is
//   %eq = arith.cmpi eq, %di, %dk : i32
//   llvm.intr.assume %eq : i1
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_UTILITY_DYNAMICDIMUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_DYNAMICDIMUTILS_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

#include <optional>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace rock {

/// Whether any shaped argument of `func` has a dynamic dimension.
bool isDynamicKernel(func::FuncOp func);

/// The dynamic dimensions of the shaped arguments of `func`, in (argument,
/// dimension) order.
SmallVector<ArgDimAttr> getDynamicArgDims(func::FuncOp func);

/// Dimension `argDim.getDim()` of argument `argDim.getArg()` of `func` as an
/// i32. The dim and cast are created in the entry block prologue and reused
/// by later calls.
Value getArgDimI32(OpBuilder &b, func::FuncOp func, ArgDimAttr argDim);

/// The argument dimension `value` reads, when it is an index-typed
/// `tensor.dim`/`memref.dim` of an entry-block argument with a constant index,
/// or an `arith.index_cast` of one.
std::optional<ArgDimAttr> matchArgDim(Value value);

/// `value`, a non-negative integer or index scalar, converted to the integer
/// or index `type`.
Value castIndexScalar(OpBuilder &b, Location loc, Value value, Type type);

/// Expands `expr`, whose symbol k stands for `symbols[k]`, into arith ops on
/// values of integer type `type`, getting symbol values from `valueFn`.
/// Divisions and remainders are unsigned, so every subexpression must be
/// non-negative, which holds for size expressions.
Value materializeArgExpr(OpBuilder &b, Location loc, AffineExpr expr,
                         ArrayRef<ArgDimAttr> symbols,
                         function_ref<Value(ArgDimAttr)> valueFn, Type type);

/// materializeArgExpr() in kernel code: an i32 value computed from the
/// dimensions of the arguments of `func`.
Value materializeArgExpr(OpBuilder &b, Location loc, func::FuncOp func,
                         AffineExpr expr, ArrayRef<ArgDimAttr> symbols);

/// The equivalence classes of argument dimensions implied by the
/// `llvm.intr.assume(arith.cmpi eq, a, b)` ops of a function.
class ArgDimEqualities {
public:
  ArgDimEqualities() = default;
  explicit ArgDimEqualities(func::FuncOp func);

  /// The representative of the class of `argDim`: its smallest member in
  /// (arg, dim) order.
  ArgDimAttr canonical(ArgDimAttr argDim) const;
  bool equivalent(ArgDimAttr a, ArgDimAttr b) const {
    return canonical(a) == canonical(b);
  }
  void unite(ArgDimAttr a, ArgDimAttr b);

private:
  mutable llvm::DenseMap<ArgDimAttr, ArgDimAttr> parent;
};

/// Renumbers the argument dimensions that attributes in `func` refer to
/// after its arguments were erased or reordered: arg(j, i) becomes
/// arg(oldToNew[j], i). Every referenced argument must still exist.
void remapArgDims(func::FuncOp func, ArrayRef<unsigned> oldToNew);

/// A dimension of a shaped value.
using DimRef = std::pair<Value, uint32_t>;

/// For each pair whose two sides trace (via getDimExpr) to a single dynamic
/// argument dimension, emits the assume that they are equal, unless an
/// existing assume of `func` already implies it.
void emitDimEqualities(OpBuilder &b, func::FuncOp func,
                       ArrayRef<std::pair<DimRef, DimRef>> pairs);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_DYNAMICDIMUTILS_H
