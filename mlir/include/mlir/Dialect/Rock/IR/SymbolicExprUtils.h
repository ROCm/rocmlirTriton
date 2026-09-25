//===- SymbolicExprUtils.h - Symbolic sizes in Rock attributes -*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers for the affine expressions that describe dynamic sizes in
// #rock.transform_map / #rock.transform / #rock.arg_expr. Their symbols
// stand for function-argument dimensions (#rock.arg_dim).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_SYMBOLICEXPRUTILS_H
#define MLIR_DIALECT_ROCK_IR_SYMBOLICEXPRUTILS_H

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace rock {
class ArgDimAttr;

/// The largest number of symbols a Rock attribute can refer to.
constexpr unsigned kMaxRockSymbols = 64;

/// `s0` .. `s{kMaxRockSymbols-1}`, for AsmParser::parseAffineExpr.
void getRockSymbolSet(MLIRContext *ctx,
                      SmallVectorImpl<std::pair<StringRef, AffineExpr>> &set);

/// Parses / prints `arg(j, i)`.
ParseResult parseArgDim(AsmParser &parser, ArgDimAttr &result);
void printArgDim(AsmPrinter &printer, ArgDimAttr attr);

/// Parses an integer or affine expression over `s<k>` symbols.
ParseResult parseSymbolicExpr(AsmParser &parser, AffineExpr &result);

/// The constant value of `expr`, or ShapedType::kDynamic if not constant.
int64_t getStaticOrDynamic(AffineExpr expr);

/// `exprs` if non-empty, otherwise `values` as constant expressions.
SmallVector<AffineExpr> getExprsOrConstants(MLIRContext *ctx,
                                            ArrayRef<int64_t> values,
                                            ArrayRef<AffineExpr> exprs);

/// Evaluates `expr` with the given dimension and symbol values, using the
/// affine semantics of floordiv/ceildiv/mod. Returns nullopt on a division by
/// a non-positive value or an out-of-range position.
std::optional<int64_t> evaluateAffineExpr(AffineExpr expr,
                                          ArrayRef<int64_t> dimValues,
                                          ArrayRef<int64_t> symbolValues);

/// Whether `a` and `b` are equal for all positive dimension sizes. True when
/// their difference simplifies to zero; otherwise the two are compared at a
/// fixed set of sample points, and are considered equal if they agree on all
/// of them. A `false` result is therefore always a proven inequality.
bool symbolicEqual(AffineExpr a, AffineExpr b, unsigned numDims = 0,
                   unsigned numSymbols = kMaxRockSymbols);

/// True if `expr` is the constant 0.
inline bool isZeroExpr(AffineExpr expr) {
  auto c = dyn_cast<AffineConstantExpr>(expr);
  return c && c.getValue() == 0;
}

/// Returns the largest symbol position used by any of `exprs` plus one.
unsigned getNumUsedSymbols(ArrayRef<AffineExpr> exprs);

/// Marks each symbol position used by `expr` in `used`.
void collectUsedSymbols(AffineExpr expr, SmallVectorImpl<bool> &used);

} // namespace rock

/// Lets affine expressions appear in diagnostics. This lives in namespace mlir
/// so that argument-dependent lookup finds it from Diagnostic::append.
inline Diagnostic &operator<<(Diagnostic &diag, AffineExpr expr) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << expr;
  return diag << os.str();
}
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_IR_SYMBOLICEXPRUTILS_H
