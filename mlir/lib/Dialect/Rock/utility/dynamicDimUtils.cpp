//===- dynamicDimUtils.cpp - Dynamic dimensions of rock kernels -----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/dynamicDimUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::rock;

SmallVector<ArgDimAttr> mlir::rock::getDynamicArgDims(func::FuncOp func) {
  SmallVector<ArgDimAttr> result;
  for (auto [j, type] : llvm::enumerate(func.getArgumentTypes())) {
    auto shaped = dyn_cast<ShapedType>(type);
    if (!shaped || !shaped.hasRank())
      continue;
    for (int64_t i = 0, e = shaped.getRank(); i < e; ++i)
      if (shaped.isDynamicDim(i))
        result.push_back(ArgDimAttr::get(func.getContext(), j, i));
  }
  return result;
}

bool mlir::rock::isDynamicKernel(func::FuncOp func) {
  return llvm::any_of(func.getArgumentTypes(), [](Type type) {
    auto shaped = dyn_cast<ShapedType>(type);
    return shaped && !shaped.hasStaticShape();
  });
}

static std::optional<ArgDimAttr> matchDimOf(Value source, Value index) {
  auto arg = dyn_cast<BlockArgument>(source);
  if (!arg || !arg.getOwner()->isEntryBlock() ||
      !isa<func::FuncOp>(arg.getOwner()->getParentOp()))
    return std::nullopt;
  APInt dim;
  if (!matchPattern(index, m_ConstantInt(&dim)))
    return std::nullopt;
  return ArgDimAttr::get(source.getContext(), arg.getArgNumber(),
                         dim.getZExtValue());
}

std::optional<ArgDimAttr> mlir::rock::matchArgDim(Value value) {
  if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
    value = cast.getIn();
  if (auto dim = value.getDefiningOp<tensor::DimOp>())
    return matchDimOf(dim.getSource(), dim.getIndex());
  if (auto dim = value.getDefiningOp<memref::DimOp>())
    return matchDimOf(dim.getSource(), dim.getIndex());
  return std::nullopt;
}

/// Whether `op` belongs to the prologue of dimension reads and assumes at the
/// start of a kernel.
static bool isDimPrologueOp(Operation *op) {
  if (auto cst = dyn_cast<arith::ConstantOp>(op))
    return cst.getType().isIndex();
  if (isa<tensor::DimOp, memref::DimOp>(op))
    return matchArgDim(op->getResult(0)).has_value();
  if (auto cast = dyn_cast<arith::IndexCastOp>(op))
    return matchArgDim(cast.getIn()).has_value();
  if (auto cmp = dyn_cast<arith::CmpIOp>(op))
    return cmp.getPredicate() == arith::CmpIPredicate::eq &&
           matchArgDim(cmp.getLhs()) && matchArgDim(cmp.getRhs());
  return isa<LLVM::AssumeOp>(op);
}

static Block::iterator getPrologueEnd(Block &entry) {
  Block::iterator it = entry.begin();
  while (it != entry.end() && isDimPrologueOp(&*it))
    ++it;
  return it;
}

Value mlir::rock::getArgDimI32(OpBuilder &b, func::FuncOp func,
                               ArgDimAttr argDim) {
  Block &entry = func.getBody().front();
  Type i32 = b.getI32Type();
  for (Operation &op : entry) {
    auto cast = dyn_cast<arith::IndexCastOp>(op);
    if (cast && cast.getType() == i32 && matchArgDim(cast.getIn()) == argDim)
      return cast;
  }

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(&entry, getPrologueEnd(entry));
  Value arg = func.getArgument(argDim.getArg());
  Location loc = arg.getLoc();
  Value idx = arith::ConstantIndexOp::create(b, loc, argDim.getDim());
  Value dim;
  if (isa<MemRefType>(arg.getType()))
    dim = memref::DimOp::create(b, loc, arg, idx);
  else
    dim = tensor::DimOp::create(b, loc, arg, idx);
  return arith::IndexCastOp::create(b, loc, i32, dim);
}

Value mlir::rock::castIndexScalar(OpBuilder &b, Location loc, Value value,
                                  Type type) {
  Type from = value.getType();
  if (from == type)
    return value;
  if (from.isIndex() || type.isIndex())
    return b.createOrFold<arith::IndexCastUIOp>(loc, type, value);
  // Sizes are non-negative, so widening zero-extends.
  if (from.getIntOrFloatBitWidth() < type.getIntOrFloatBitWidth())
    return b.createOrFold<arith::ExtUIOp>(loc, type, value);
  return b.createOrFold<arith::TruncIOp>(loc, type, value);
}

namespace {
class ArgExprExpander : public AffineExprVisitor<ArgExprExpander, Value> {
public:
  ArgExprExpander(OpBuilder &b, Location loc, ArrayRef<ArgDimAttr> symbols,
                  function_ref<Value(ArgDimAttr)> valueFn, Type type)
      : b(b), loc(loc), symbols(symbols), valueFn(valueFn), type(type) {}

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return b.createOrFold<arith::AddIOp>(loc, visit(expr.getLHS()),
                                         visit(expr.getRHS()));
  }
  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return b.createOrFold<arith::MulIOp>(loc, visit(expr.getLHS()),
                                         visit(expr.getRHS()));
  }
  Value visitModExpr(AffineBinaryOpExpr expr) {
    return b.createOrFold<arith::RemUIOp>(loc, visit(expr.getLHS()),
                                          visit(expr.getRHS()));
  }
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
    return b.createOrFold<arith::DivUIOp>(loc, visit(expr.getLHS()),
                                          visit(expr.getRHS()));
  }
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
    Value lhs = visit(expr.getLHS());
    Value rhs = visit(expr.getRHS());
    Value one = constant(1);
    Value rhsMinusOne = b.createOrFold<arith::SubIOp>(loc, rhs, one);
    Value numerator = b.createOrFold<arith::AddIOp>(loc, lhs, rhsMinusOne);
    return b.createOrFold<arith::DivUIOp>(loc, numerator, rhs);
  }
  Value visitConstantExpr(AffineConstantExpr expr) {
    return constant(expr.getValue());
  }
  Value visitDimExpr(AffineDimExpr) {
    llvm_unreachable("argument-dimension expressions have no dimensions");
  }
  Value visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbols.size() && "unbound symbol");
    return castIndexScalar(b, loc, valueFn(symbols[expr.getPosition()]),
                           type);
  }

private:
  Value constant(int64_t value) {
    return arith::ConstantOp::create(b, loc, b.getIntegerAttr(type, value));
  }

  OpBuilder &b;
  Location loc;
  ArrayRef<ArgDimAttr> symbols;
  function_ref<Value(ArgDimAttr)> valueFn;
  Type type;
};
} // namespace

Value mlir::rock::materializeArgExpr(OpBuilder &b, Location loc,
                                     AffineExpr expr,
                                     ArrayRef<ArgDimAttr> symbols,
                                     function_ref<Value(ArgDimAttr)> valueFn,
                                     Type type) {
  return ArgExprExpander(b, loc, symbols, valueFn, type).visit(expr);
}

Value mlir::rock::materializeArgExpr(OpBuilder &b, Location loc,
                                     func::FuncOp func, AffineExpr expr,
                                     ArrayRef<ArgDimAttr> symbols) {
  return materializeArgExpr(
      b, loc, expr, symbols,
      [&](ArgDimAttr argDim) { return getArgDimI32(b, func, argDim); },
      b.getI32Type());
}

//===----------------------------------------------------------------------===//
// ArgDimEqualities
//===----------------------------------------------------------------------===//

static bool argDimLess(ArgDimAttr a, ArgDimAttr b) {
  return std::make_pair(a.getArg(), a.getDim()) <
         std::make_pair(b.getArg(), b.getDim());
}

ArgDimEqualities::ArgDimEqualities(func::FuncOp func) {
  func.walk([&](LLVM::AssumeOp assume) {
    auto cmp = assume.getCond().getDefiningOp<arith::CmpIOp>();
    if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
      return;
    std::optional<ArgDimAttr> lhs = matchArgDim(cmp.getLhs());
    std::optional<ArgDimAttr> rhs = matchArgDim(cmp.getRhs());
    if (lhs && rhs)
      unite(*lhs, *rhs);
  });
}

ArgDimAttr ArgDimEqualities::canonical(ArgDimAttr argDim) const {
  auto it = parent.find(argDim);
  if (it == parent.end() || it->second == argDim)
    return argDim;
  ArgDimAttr root = canonical(it->second);
  parent[argDim] = root;
  return root;
}

void ArgDimEqualities::unite(ArgDimAttr a, ArgDimAttr b) {
  ArgDimAttr ra = canonical(a), rb = canonical(b);
  if (ra == rb)
    return;
  if (argDimLess(rb, ra))
    std::swap(ra, rb);
  parent[ra] = ra;
  parent[rb] = ra;
}

AffineExpr ArgDimEqualities::canonicalize(
    AffineExpr expr, ArrayRef<ArgDimAttr> from,
    SmallVectorImpl<ArgDimAttr> &to) const {
  SmallVector<ArgDimAttr> canonicalFrom =
      llvm::map_to_vector(from, [&](ArgDimAttr a) { return canonical(a); });
  return rebindSymbols(expr, canonicalFrom, to);
}

//===----------------------------------------------------------------------===//
// emitDimEqualities
//===----------------------------------------------------------------------===//

static std::optional<ArgDimAttr> resolveToArgDim(DimRef ref) {
  SmallVector<ArgDimAttr> symbols;
  FailureOr<AffineExpr> expr = getDimExpr(ref.first, ref.second, symbols);
  if (failed(expr))
    return std::nullopt;
  auto sym = dyn_cast<AffineSymbolExpr>(*expr);
  if (!sym)
    return std::nullopt;
  return symbols[sym.getPosition()];
}

void mlir::rock::remapArgDims(func::FuncOp func, ArrayRef<unsigned> oldToNew) {
  MLIRContext *ctx = func.getContext();
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](ArgDimAttr argDim) -> Attribute {
    assert(argDim.getArg() < oldToNew.size() &&
           "argument dimension of an unknown argument");
    return ArgDimAttr::get(ctx, oldToNew[argDim.getArg()], argDim.getDim());
  });
  replacer.recursivelyReplaceElementsIn(func, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/false);
}

void mlir::rock::emitDimEqualities(OpBuilder &b, func::FuncOp func,
                                   ArrayRef<std::pair<DimRef, DimRef>> pairs) {
  ArgDimEqualities equalities(func);
  Block &entry = func.getBody().front();
  for (auto [lhsRef, rhsRef] : pairs) {
    std::optional<ArgDimAttr> lhs = resolveToArgDim(lhsRef);
    std::optional<ArgDimAttr> rhs = resolveToArgDim(rhsRef);
    if (!lhs || !rhs || equalities.equivalent(*lhs, *rhs))
      continue;
    Value lhsVal = getArgDimI32(b, func, *lhs);
    Value rhsVal = getArgDimI32(b, func, *rhs);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(&entry, getPrologueEnd(entry));
    Location loc = func.getLoc();
    Value eq = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lhsVal,
                                     rhsVal);
    LLVM::AssumeOp::create(b, loc, eq);
    equalities.unite(*lhs, *rhs);
  }
}
