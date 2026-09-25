//===- SymbolicExprUtils.cpp - Symbolic sizes in Rock attributes ----------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/SymbolicExprUtils.h"
#include "mlir/Dialect/Rock/IR/Rock.h"

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <string>

using namespace mlir;
using namespace mlir::rock;

void mlir::rock::getRockSymbolSet(
    MLIRContext *ctx, SmallVectorImpl<std::pair<StringRef, AffineExpr>> &set) {
  static const std::array<std::string, kMaxRockSymbols> names = [] {
    std::array<std::string, kMaxRockSymbols> ret;
    for (unsigned i = 0; i < kMaxRockSymbols; ++i)
      ret[i] = "s" + std::to_string(i);
    return ret;
  }();
  set.reserve(kMaxRockSymbols);
  for (unsigned i = 0; i < kMaxRockSymbols; ++i)
    set.emplace_back(names[i], getAffineSymbolExpr(i, ctx));
}

ParseResult mlir::rock::parseArgDim(AsmParser &parser, ArgDimAttr &result) {
  uint32_t arg = 0, dim = 0;
  if (parser.parseKeyword("arg") || parser.parseLParen() ||
      parser.parseInteger(arg) || parser.parseComma() ||
      parser.parseInteger(dim) || parser.parseRParen())
    return failure();
  result = ArgDimAttr::get(parser.getContext(), arg, dim);
  return success();
}

void mlir::rock::printArgDim(AsmPrinter &printer, ArgDimAttr attr) {
  printer << "arg(" << attr.getArg() << ", " << attr.getDim() << ")";
}

ParseResult mlir::rock::parseSymbolicExpr(AsmParser &parser,
                                          AffineExpr &result) {
  SmallVector<std::pair<StringRef, AffineExpr>> symbolSet;
  getRockSymbolSet(parser.getContext(), symbolSet);
  return parser.parseAffineExpr(symbolSet, result);
}

int64_t mlir::rock::getStaticOrDynamic(AffineExpr expr) {
  if (auto c = dyn_cast<AffineConstantExpr>(expr))
    return c.getValue();
  return ShapedType::kDynamic;
}

SmallVector<AffineExpr>
mlir::rock::getExprsOrConstants(MLIRContext *ctx, ArrayRef<int64_t> values,
                                ArrayRef<AffineExpr> exprs) {
  if (!exprs.empty())
    return SmallVector<AffineExpr>(exprs);
  return llvm::map_to_vector(values, [&](int64_t v) -> AffineExpr {
    return getAffineConstantExpr(v, ctx);
  });
}

std::optional<int64_t>
mlir::rock::evaluateAffineExpr(AffineExpr expr, ArrayRef<int64_t> dimValues,
                               ArrayRef<int64_t> symbolValues) {
  switch (expr.getKind()) {
  case AffineExprKind::Constant:
    return cast<AffineConstantExpr>(expr).getValue();
  case AffineExprKind::DimId: {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    if (pos >= dimValues.size())
      return std::nullopt;
    return dimValues[pos];
  }
  case AffineExprKind::SymbolId: {
    unsigned pos = cast<AffineSymbolExpr>(expr).getPosition();
    if (pos >= symbolValues.size())
      return std::nullopt;
    return symbolValues[pos];
  }
  default:
    break;
  }
  auto bin = cast<AffineBinaryOpExpr>(expr);
  std::optional<int64_t> lhs =
      evaluateAffineExpr(bin.getLHS(), dimValues, symbolValues);
  std::optional<int64_t> rhs =
      evaluateAffineExpr(bin.getRHS(), dimValues, symbolValues);
  if (!lhs || !rhs)
    return std::nullopt;
  switch (expr.getKind()) {
  case AffineExprKind::Add:
    return *lhs + *rhs;
  case AffineExprKind::Mul:
    return *lhs * *rhs;
  case AffineExprKind::FloorDiv:
    if (*rhs <= 0)
      return std::nullopt;
    return llvm::divideFloorSigned(*lhs, *rhs);
  case AffineExprKind::CeilDiv:
    if (*rhs <= 0)
      return std::nullopt;
    return llvm::divideCeilSigned(*lhs, *rhs);
  case AffineExprKind::Mod:
    if (*rhs <= 0)
      return std::nullopt;
    return llvm::mod(*lhs, *rhs);
  default:
    return std::nullopt;
  }
}

bool mlir::rock::symbolicEqual(AffineExpr a, AffineExpr b, unsigned numDims,
                               unsigned numSymbols) {
  if (a == b)
    return true;
  AffineExpr diff = simplifyAffineExpr(a - b, numDims, numSymbols);
  if (isZeroExpr(diff))
    return true;
  if (isa<AffineConstantExpr>(diff))
    return false;

  // Positive sample values, including small, power-of-two, odd and large ones,
  // so that divisibility-dependent expressions (ceildiv, mod) are exercised.
  static constexpr int64_t samples[] = {1,  2,   3,   5,    7,    16,  17,
                                        31, 64,  100, 127,  128,  255, 1000,
                                        1024, 4097, 12345};
  constexpr unsigned numSamples = std::size(samples);
  SmallVector<int64_t> dims(numDims), syms(numSymbols);
  for (unsigned trial = 0; trial < 2 * numSamples; ++trial) {
    for (unsigned i = 0; i < numDims; ++i)
      dims[i] = samples[(trial + 5 * i) % numSamples] - 1;
    for (unsigned i = 0; i < numSymbols; ++i)
      syms[i] = samples[(trial * 7 + 3 * i) % numSamples];
    std::optional<int64_t> va = evaluateAffineExpr(a, dims, syms);
    std::optional<int64_t> vb = evaluateAffineExpr(b, dims, syms);
    if (va && vb && *va != *vb)
      return false;
  }
  return true;
}

void mlir::rock::collectUsedSymbols(AffineExpr expr,
                                    SmallVectorImpl<bool> &used) {
  expr.walk([&](AffineExpr e) {
    if (auto sym = dyn_cast<AffineSymbolExpr>(e)) {
      unsigned pos = sym.getPosition();
      if (pos >= used.size())
        used.resize(pos + 1, false);
      used[pos] = true;
    }
  });
}

unsigned mlir::rock::getNumUsedSymbols(ArrayRef<AffineExpr> exprs) {
  SmallVector<bool> used;
  for (AffineExpr e : exprs)
    collectUsedSymbols(e, used);
  return used.size();
}
