//===- TransformMapBuilder.cpp - Rock MLIR Operations
//-----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/SymbolicExprUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::rock;

AffineMapAttr mlir::rock::assembleMapFor(Builder &b,
                                         ArrayRef<TransformAttr> transforms,
                                         ArrayRef<int64_t> upperBounds,
                                         ArrayRef<int64_t> lowerBounds,
                                         unsigned numSymbols) {
  auto simplify = [&](AffineExpr e) {
    return simplifyAffineExpr(e, 0, numSymbols);
  };
  llvm::SmallMapVector<int64_t, AffineExpr, 8> affExprsMap;
  for (const TransformAttr transform : transforms) {
    TransformType type = transform.getType();
    ArrayRef<uint32_t> upperDims = transform.getUpperDims();
    ArrayRef<uint32_t> lowerDims = transform.getLowerDims();
    SmallVector<AffineExpr> params = transform.getParamExprs();
    if (type == TransformType::PassThrough) {
      for (auto pair : llvm::zip(upperDims, lowerDims)) {
        uint32_t upper, lower;
        std::tie(upper, lower) = pair;
        affExprsMap.insert({lower, b.getAffineDimExpr(upper)});
      }
    } else if (type == TransformType::Pad) {
      for (unsigned i = 0, e = upperDims.size(); i < e; ++i) {
        // example of h and w pad parameters [0, 2, 3, 1] :
        // leftpadH = 0 rightPadH = 2 leftpadW = 3 rightPadW = 1
        // first run leftPad = 0 rightPad = 2
        // second run leftPad = 3 rightPad = 1
        // if your pad is one dim , example of pad parameters [1,2]
        // leftPad = 1 rightPad = 2
        AffineExpr leftPad = params[i * 2];
        uint32_t upperDim = upperDims[i];
        uint32_t lowerDim = lowerDims[i];
        AffineExpr expr = b.getAffineDimExpr(upperDim) - leftPad;
        affExprsMap.insert({lowerDim, expr});
      }
    } else if (type == TransformType::Slice) {
      // The params for slice are begin1 end1 begin2 end2... just line on pad
      for (uint32_t i = 0, e = upperDims.size(); i < e; ++i) {
        uint32_t upperDim = upperDims[i];
        uint32_t lowerDim = lowerDims[i];
        AffineExpr begin = params[i * 2];
        AffineExpr expr = b.getAffineDimExpr(upperDim) + begin;
        affExprsMap.insert({lowerDim, expr});
      }
    } else if (type == TransformType::Embed) {
      ArrayRef<AffineExpr> coefficients = params;
      uint32_t lowerDim = lowerDims[0];
      AffineExpr expr = b.getAffineConstantExpr(0);
      for (auto pair : llvm::zip(upperDims, coefficients)) {
        uint32_t upperDim;
        AffineExpr coefficient;
        std::tie(upperDim, coefficient) = pair;
        expr = expr + (b.getAffineDimExpr(upperDim) * coefficient);
      }
      affExprsMap.insert({lowerDim, expr});
    } else if (type == TransformType::Unmerge) {
      ArrayRef<AffineExpr> lengths = params;
      AffineExpr expr = b.getAffineDimExpr(upperDims[0]);
      for (auto pair : llvm::zip(upperDims.slice(1), lengths.slice(1))) {
        uint32_t upperDim;
        AffineExpr length;
        std::tie(upperDim, length) = pair;
        expr = expr * length + b.getAffineDimExpr(upperDim);
      }
      affExprsMap.insert({lowerDims[0], expr});
    } else if (type == TransformType::Merge) {
      // Compute lower dimension strides.
      llvm::SmallVector<AffineExpr, 4> lowerDimStrides;
      AffineExpr totalStride = b.getAffineConstantExpr(1);
      lowerDimStrides.push_back(totalStride);
      for (unsigned i = params.size() - 1; i > 0; --i) {
        totalStride = simplify(totalStride * params[i]);
        lowerDimStrides.push_back(totalStride);
      }
      totalStride = simplify(totalStride * params[0]);
      std::reverse(lowerDimStrides.begin(), lowerDimStrides.end());

      // The Merge can be decomposed in two equivalent ways. Writing
      // S_i for the stride of lower dimension i and P_i for its size:
      //
      //   1. peeled: (x mod S_{i-1}) floordiv S_i  -- sequential remainders
      //   2. nested: (x floordiv S_i) mod P_i      -- independent per dim
      //
      // Choosing one or the other may affect how Triton's AxisInfo analyzes and
      // vectorizes the code, so choosing wisely has performance implications.
      //
      // The nested form (2.) tends to give better contiguity than the peeled
      // form (1.), but it is not always faster. The extra contiguity only pays
      // off if the hardware can fold those elements into a single wider access,
      // and when nothing widens we have changed the addressing for no gain. For
      // example, on 2x2 convs, that alone makes threads collide in LDS and
      // makes the kernel run significantly slower.
      //
      // So we take the nested form only when the innermost size gives us enough
      // contiguity to be worth it. The constant is based on benchmark results:
      // the severe regressions we measured are 2x2 convs, whose innermost size
      // is two, and asking for a multiple of four excludes them.
      //
      // Symbolic merges always use the peeled form.
      //
      // TODO(AIROCMLIR-1238): This is a heuristic. We should fix AxisInfo
      // analysis to properly analyze our IR, instead of making our IR pretty so
      // that AxisInfo can analyze it.
      bool isStaticMerge = transform.isStatic();
      bool useNestedForm =
          isStaticMerge && transform.getParams().back() % 4 == 0;

      // Build affine transformation expressions.
      AffineExpr remainder = b.getAffineDimExpr(upperDims[0]);
      for (uint32_t i = 0, e = lowerDims.size(); i < e; ++i) {
        // If the constant we're about to divide by is the same as the total
        // stride in the input dimension, output 0, as, if you're above
        // said stride, the result of this merge is undefined behavior.
        // (For symbolic strides the general formula below also yields 0 for
        // in-range coordinates.)
        if (lowerDimStrides[i] == totalStride) {
          AffineExpr thisDim = b.getAffineConstantExpr(0);
          affExprsMap.insert({lowerDims[i], thisDim});
          continue;
        }
        // While, in general, (x mod N) floordiv N is not 0, because
        // for x < 0 it is instead -1, in our context, negative coordinates
        // are never produced within an indexing process, so we can make
        // that simplification.
        if (i > 0 && lowerDimStrides[i] == lowerDimStrides[i - 1]) {
          AffineExpr thisDim = b.getAffineConstantExpr(0);
          affExprsMap.insert({lowerDims[i], thisDim});
          continue;
        }
        AffineExpr stride = lowerDimStrides[i];
        AffineExpr thisDim;
        if (useNestedForm) {
          thisDim = b.getAffineDimExpr(upperDims[0]).floorDiv(stride);
          // Only mod when needed. The coordinate is below totalStride, so the
          // quotient stays under params[i] on its own when stride * params[i]
          // spans the whole merge.
          ArrayRef<int64_t> staticParams = transform.getParams();
          int64_t staticStride = cast<AffineConstantExpr>(stride).getValue();
          int64_t staticTotal = cast<AffineConstantExpr>(totalStride).getValue();
          if (staticStride * staticParams[i] < staticTotal)
            thisDim = thisDim % params[i];
        } else {
          thisDim = remainder.floorDiv(stride);
          remainder = remainder % stride;
        }
        affExprsMap.insert({lowerDims[i], thisDim});
      }
    } else if (type == TransformType::AddDim) {
      assert(upperDims.size() == 1 && lowerDims.size() == 0 &&
             "Invalid AddDim");
      // dimension is ignored, do nothing
    } else if (type == TransformType::Broadcast) {
      // Compute lower dimension strides.
      for (auto tuple : llvm::zip(params, lowerDims, upperDims)) {
        AffineExpr param = std::get<0>(tuple);
        uint32_t lowerDim = std::get<1>(tuple);
        uint32_t upperDim = std::get<2>(tuple);
        AffineExpr expr = b.getAffineDimExpr(upperDim) % param;
        affExprsMap.insert({lowerDim, expr});
      }
    } else if (type == TransformType::ConstDim) {
      for (unsigned i = 0, e = lowerDims.size(); i < e; ++i) {
        uint32_t lowerDim = lowerDims[i];
        AffineExpr expr = params[2 * i];
        affExprsMap.insert({lowerDim, expr});
      }
    } else {
      llvm_unreachable("Handled all the cases in affine map building");
    }
  }

  llvm::SmallVector<AffineExpr, 8> affExprsVec;
  affExprsVec.reserve(affExprsMap.size());
  for (uint32_t i = 0, e = lowerBounds.size(); i < e; ++i) {
    assert(affExprsMap.count(i) == 1 &&
           "Lower dimension must have associated output expression");
    affExprsVec.push_back(affExprsMap[i]);
  }
  AffineMap ret = AffineMap::get(upperBounds.size(), numSymbols, affExprsVec,
                                 b.getContext());
  return AffineMapAttr::get(ret);
}

/// Drops symbols that no expression uses and renumbers the rest, then returns
/// the static or symbolic form of the map as appropriate.
static TransformMapAttr
buildTransformMap(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Builder &b, ArrayRef<TransformAttr> transforms,
                  ArrayRef<AffineExpr> upperBounds,
                  ArrayRef<AffineExpr> lowerBounds,
                  ArrayRef<ArgDimAttr> symbols) {
  MLIRContext *ctx = b.getContext();
  SmallVector<bool> used;
  for (AffineExpr e : llvm::concat<const AffineExpr>(upperBounds, lowerBounds))
    collectUsedSymbols(e, used);
  for (TransformAttr t : transforms)
    for (AffineExpr e : t.getSymParams())
      collectUsedSymbols(e, used);
  used.resize(std::max<size_t>(used.size(), symbols.size()), false);

  SmallVector<ArgDimAttr> newSymbols;
  SmallVector<AffineExpr> replacements;
  for (auto [i, s] : llvm::enumerate(symbols)) {
    if (used[i]) {
      replacements.push_back(getAffineSymbolExpr(newSymbols.size(), ctx));
      newSymbols.push_back(s);
    } else {
      replacements.push_back(getAffineConstantExpr(0, ctx));
    }
  }
  unsigned numSymbols = newSymbols.size();
  auto remap = [&](AffineExpr e) {
    return simplifyAffineExpr(e.replaceSymbols(replacements), 0, numSymbols);
  };

  SmallVector<TransformAttr> newTransforms;
  newTransforms.reserve(transforms.size());
  for (TransformAttr t : transforms) {
    if (t.isStatic()) {
      newTransforms.push_back(t);
      continue;
    }
    SmallVector<AffineExpr> params =
        llvm::map_to_vector(t.getSymParams(), remap);
    newTransforms.push_back(getTransformAttrChecked(
        emitError, ctx, t.getType(), params, t.getUpperNames(),
        t.getUpperDims(), t.getLowerNames(), t.getLowerDims()));
    if (!newTransforms.back())
      return {};
  }

  SmallVector<AffineExpr> ub = llvm::map_to_vector(upperBounds, remap);
  SmallVector<AffineExpr> lb = llvm::map_to_vector(lowerBounds, remap);
  SmallVector<int64_t> ubInts = llvm::map_to_vector(ub, getStaticOrDynamic);
  SmallVector<int64_t> lbInts = llvm::map_to_vector(lb, getStaticOrDynamic);
  if (numSymbols == 0) {
    ub.clear();
    lb.clear();
  }
  AffineMapAttr map =
      assembleMapFor(b, newTransforms, ubInts, lbInts, numSymbols);
  return getTransformMapAttrChecked(emitError, ctx, newTransforms, map,
                                    b.getDenseI64ArrayAttr(ubInts),
                                    b.getDenseI64ArrayAttr(lbInts), newSymbols,
                                    ub, lb);
}

/// Builder for when we know what we're doing.
TransformMapAttr TransformMapAttr::get(ArrayRef<TransformAttr> transforms,
                                       ArrayRef<int64_t> upperBounds,
                                       ArrayRef<int64_t> lowerBounds) {
  assert(!transforms.empty() && "This builder does not support the empty map");
  assert(llvm::all_of(transforms,
                      [](TransformAttr t) { return t.isStatic(); }) &&
         "Use the AffineExpr builder for symbolic transforms");
  Builder b(transforms.front().getContext());
  AffineMapAttr map = assembleMapFor(b, transforms, upperBounds, lowerBounds);
  return TransformMapAttr::get(map.getContext(), transforms, map,
                               b.getDenseI64ArrayAttr(upperBounds),
                               b.getDenseI64ArrayAttr(lowerBounds));
}

TransformMapAttr TransformMapAttr::get(ArrayRef<TransformAttr> transforms,
                                       ArrayRef<AffineExpr> upperBounds,
                                       ArrayRef<AffineExpr> lowerBounds,
                                       ArrayRef<ArgDimAttr> symbols) {
  assert(!transforms.empty() && "This builder does not support the empty map");
  Builder b(transforms.front().getContext());
  auto emitError = [&]() {
    return mlir::emitError(UnknownLoc::get(b.getContext()),
                           "invalid symbolic transform map: ");
  };
  return buildTransformMap(emitError, b, transforms, upperBounds, lowerBounds,
                           symbols);
}

/// Symbol binding helpers

AffineExpr mlir::rock::bindArgDim(ArgDimAttr argDim,
                                  SmallVectorImpl<ArgDimAttr> &symbols) {
  auto it = llvm::find(symbols, argDim);
  unsigned pos = std::distance(symbols.begin(), it);
  if (it == symbols.end())
    symbols.push_back(argDim);
  return getAffineSymbolExpr(pos, argDim.getContext());
}

AffineExpr mlir::rock::rebindSymbols(AffineExpr expr, ArrayRef<ArgDimAttr> from,
                                     SmallVectorImpl<ArgDimAttr> &to) {
  if (from.empty())
    return expr;
  SmallVector<AffineExpr> replacements = llvm::map_to_vector(
      from, [&](ArgDimAttr a) { return bindArgDim(a, to); });
  return simplifyAffineExpr(expr.replaceSymbols(replacements), 0, to.size());
}

FailureOr<AffineExpr>
mlir::rock::getDimExpr(Value value, uint32_t dim,
                       SmallVectorImpl<ArgDimAttr> &symbols) {
  auto type = dyn_cast<ShapedType>(value.getType());
  if (!type || !type.hasRank() || dim >= type.getRank())
    return failure();
  MLIRContext *ctx = value.getContext();
  if (!type.isDynamicDim(dim))
    return getAffineConstantExpr(type.getDimSize(dim), ctx);

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    Block *owner = arg.getOwner();
    if (!owner->isEntryBlock() || !isa<func::FuncOp>(owner->getParentOp()))
      return failure();
    return bindArgDim(ArgDimAttr::get(ctx, arg.getArgNumber(), dim), symbols);
  }
  Operation *def = value.getDefiningOp();
  if (auto transform = dyn_cast<TransformOp>(def)) {
    TransformMapAttr map = transform.getTransform();
    return rebindSymbols(map.getUpperBoundExprs()[dim], map.getSymbols(),
                         symbols);
  }
  if (auto cast = dyn_cast<tensor::CastOp>(def))
    return getDimExpr(cast.getSource(), dim, symbols);
  // Ops whose result has the shape of their destination / first operand.
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(def)) {
    auto res = cast<OpResult>(value);
    if (res.getResultNumber() < dps.getNumDpsInits())
      return getDimExpr(dps.getDpsInits()[res.getResultNumber()], dim,
                        symbols);
  }
  return failure();
}

FailureOr<SmallVector<AffineExpr>>
mlir::rock::getShapeExprs(Value value, SmallVectorImpl<ArgDimAttr> &symbols) {
  auto type = dyn_cast<ShapedType>(value.getType());
  if (!type || !type.hasRank())
    return failure();
  SmallVector<AffineExpr> ret;
  for (int64_t i = 0, e = type.getRank(); i < e; ++i) {
    FailureOr<AffineExpr> expr = getDimExpr(value, i, symbols);
    if (failed(expr))
      return failure();
    ret.push_back(*expr);
  }
  return ret;
}

/// Accessors and common infrastructure

static void assertStaticShape(ArrayRef<int64_t> shape) {
  if (llvm::any_of(shape, ShapedType::isDynamic))
    llvm::report_fatal_error("Transform map builders need the AffineExpr "
                             "constructor to describe dynamic shapes");
}

TransformMapBuilder::TransformMapBuilder(mlir::Builder &builder,
                                         ArrayRef<StringRef> startNamesArg,
                                         ArrayRef<int64_t> startShapeArg,
                                         mlir::Location loc)
    : b(builder), result(), loc(loc), startIndices(), startNames(),
      startShape(), endIndices(), endNames(), endShape() {
  assert(startNamesArg.size() == startShapeArg.size() &&
         "Start names and shape must have the same size");
  assertStaticShape(startShapeArg);
  for (auto pair : llvm::enumerate(startNamesArg)) {
    uint32_t index = pair.index();
    StringRef value = pair.value();

    startNames.push_back(value);
    startIndices.insert_or_assign(value, index);
    startShape.push_back(b.getAffineConstantExpr(startShapeArg[index]));
  }
}

TransformMapBuilder::TransformMapBuilder(mlir::Builder &builder,
                                         ArrayRef<StringRef> startNamesArg,
                                         ArrayRef<AffineExpr> startShapeArg,
                                         ArrayRef<ArgDimAttr> startSymbols,
                                         mlir::Location loc)
    : b(builder), result(), loc(loc), startIndices(), startNames(),
      startShape(), endIndices(), endNames(), endShape() {
  initStart(startNamesArg, startShapeArg, startSymbols);
}

TransformMapBuilder::TransformMapBuilder(mlir::Builder &builder,
                                         ArrayRef<StringRef> startNamesArg,
                                         Value shaped, mlir::Location loc)
    : b(builder), result(), loc(loc), startIndices(), startNames(),
      startShape(), endIndices(), endNames(), endShape() {
  SmallVector<ArgDimAttr> shapeSymbols;
  FailureOr<SmallVector<AffineExpr>> shape =
      getShapeExprs(shaped, shapeSymbols);
  if (failed(shape))
    llvm::report_fatal_error("cannot express the dynamic shape of a value in "
                             "terms of function-argument dimensions");
  initStart(startNamesArg, *shape, shapeSymbols);
}

void TransformMapBuilder::initStart(ArrayRef<StringRef> startNamesArg,
                                    ArrayRef<AffineExpr> startShapeArg,
                                    ArrayRef<ArgDimAttr> startSymbols) {
  assert(startNamesArg.size() == startShapeArg.size() &&
         "Start names and shape must have the same size");
  for (auto pair : llvm::enumerate(startNamesArg)) {
    uint32_t index = pair.index();
    StringRef value = pair.value();

    startNames.push_back(value);
    startIndices.insert_or_assign(value, index);
    startShape.push_back(rebind(startShapeArg[index], startSymbols));
  }
}

TransformMapBuilder::TransformMapBuilder(mlir::Builder &builder,
                                         ArrayRef<int64_t> startShapeArg,
                                         mlir::Location loc)
    : b(builder), result(), loc(loc), startIndices(), startNames(),
      startShape(), endIndices(), endNames(), endShape() {
  assertStaticShape(startShapeArg);
  for (auto pair : llvm::enumerate(startShapeArg)) {
    uint32_t index = pair.index();
    int64_t value = pair.value();

    SmallString<8> name;
    ("dim" + Twine(index)).toVector(name);

    startNames.push_back(name);
    startIndices.insert_or_assign(startNames.back(), index);

    startShape.push_back(b.getAffineConstantExpr(value));
  }
}

AffineExpr TransformMapBuilder::bindArg(uint32_t arg, uint32_t dim) {
  return bindArg(ArgDimAttr::get(b.getContext(), arg, dim));
}

AffineExpr TransformMapBuilder::bindArg(ArgDimAttr argDim) {
  return bindArgDim(argDim, symbols);
}

AffineExpr TransformMapBuilder::rebind(AffineExpr expr,
                                       ArrayRef<ArgDimAttr> exprSymbols) {
  return rebindSymbols(expr, exprSymbols, symbols);
}

AffineExpr TransformMapBuilder::simplify(AffineExpr expr) {
  return simplifyAffineExpr(expr, 0, symbols.size());
}

SmallVector<AffineExpr> TransformMapBuilder::toExprs(ArrayRef<int64_t> values) {
  return llvm::map_to_vector(
      values, [&](int64_t v) { return b.getAffineConstantExpr(v); });
}

TransformMapAttr TransformMapBuilder::get() {
  SmallVector<AffineExpr, 8> upperBounds, lowerBounds;
  extractBounds(upperBounds, lowerBounds);
  auto errorEmitter = [&]() -> InFlightDiagnostic {
    InFlightDiagnostic err =
        mlir::emitError(loc, "Error assembling transform map: ");
    if (b.getContext()->shouldPrintOpOnDiagnostic()) {
      err.attachNote(loc).append("The transforms were").appendRange(result);
    }
    return err;
  };
  frozen = true;
  return buildTransformMap(errorEmitter, b, result, upperBounds, lowerBounds,
                           symbols);
}

void TransformMapBuilder::getEndNames(SmallVectorImpl<StringRef> &names) {
  uint32_t e = nEndDims();
  names.reserve(e);
  for (uint32_t i = 0; i < e; ++i) {
    names.emplace_back(endNames[i]);
  }
}

void TransformMapBuilder::getStartNames(SmallVectorImpl<StringRef> &names) {
  names.reserve(startNames.size());
  for (const auto &name : startNames) {
    names.emplace_back(name);
  }
}

StringRef TransformMapBuilder::startName(uint32_t dim) {
  return startNames[dim];
}

StringRef TransformMapBuilder::endName(uint32_t dim) {
  assert(endNames.count(dim) == 1 &&
         "Dimension not defined in ending dimension space");
  return endNames[dim];
}

uint32_t TransformMapBuilder::startIndex(StringRef name) {
  if (startIndices.count(name) != 1) {
    llvm::report_fatal_error(Twine("Key not in starting set of names: ") +
                             name);
  }
  return startIndices[name];
}

uint32_t TransformMapBuilder::endIndex(StringRef name) {
  assert(endIndices.count(name) == 1 &&
         "Key has not yet been defined in the ending set of names");
  return endIndices[name];
}

int64_t TransformMapBuilder::startSize(StringRef name) {
  return getStaticOrDynamic(startSizeExpr(name));
}

int64_t TransformMapBuilder::startSize(uint32_t dim) {
  return getStaticOrDynamic(startSizeExpr(dim));
}

int64_t TransformMapBuilder::endSize(StringRef name) {
  return getStaticOrDynamic(endSizeExpr(name));
}

int64_t TransformMapBuilder::endSize(uint32_t dim) {
  return getStaticOrDynamic(endSizeExpr(dim));
}

AffineExpr TransformMapBuilder::startSizeExpr(StringRef name) {
  return startShape[startIndices[name]];
}

AffineExpr TransformMapBuilder::startSizeExpr(uint32_t dim) {
  return startShape[dim];
}

AffineExpr TransformMapBuilder::endSizeExpr(StringRef name) {
  return endShape[endIndices[name]];
}

AffineExpr TransformMapBuilder::endSizeExpr(uint32_t dim) {
  return endShape[dim];
}

uint32_t TransformMapBuilder::nStartDims() { return startShape.size(); }

uint32_t TransformMapBuilder::nEndDims() { return endShape.size(); }

void TransformMapBuilder::defineDim(StringRef name, uint32_t dim,
                                    AffineExpr size) {
  assert(!frozen && "It's a bug to add to a coordinate transform after "
                    "fetching the attribute");
  [[maybe_unused]] bool nameInsertResult =
      endIndices.insert({name, dim}).second;
  assert(nameInsertResult &&
         "Trying to redefine a result name in a coordinate transform");
  SmallString<8> nameCopy = name;
  [[maybe_unused]] bool dimInsertResult =
      endNames.insert({dim, nameCopy}).second;
  assert(dimInsertResult &&
         "Trying to redefine a result dimension in a coordinate transform");
  for (uint32_t e = endShape.size(); e <= dim; ++e) {
    endShape.push_back(b.getAffineConstantExpr(0));
  }
  endShape[dim] = simplify(size);
}

void TransformMapBuilder::addTransform(TransformType type,
                                       ArrayRef<int64_t> params,
                                       ArrayRef<StringRef> fromNames,
                                       ArrayRef<uint32_t> fromDims,
                                       ArrayRef<StringRef> toNames,
                                       ArrayRef<uint32_t> toDims) {
  addTransformImpl(type, toExprs(params), fromNames, fromDims, toNames,
                   toDims);
}

void TransformMapBuilder::addTransform(TransformType type,
                                       ArrayRef<AffineExpr> params,
                                       ArrayRef<StringRef> fromNames,
                                       ArrayRef<uint32_t> fromDims,
                                       ArrayRef<StringRef> toNames,
                                       ArrayRef<uint32_t> toDims) {
  SmallVector<AffineExpr> simplified =
      llvm::map_to_vector(params, [&](AffineExpr e) { return simplify(e); });
  addTransformImpl(type, simplified, fromNames, fromDims, toNames, toDims);
}

/// Transformations that work basically the same in either direction
void TransformMapBuilder::passThrough(StringRef name) {
  uint32_t dim = startIndex(name);
  AffineExpr size = startSizeExpr(dim);
  defineDim(name, dim, size);
  addTransform(TransformType::PassThrough, ArrayRef<int64_t>{}, {name}, {dim},
               {name}, {dim});
}

void TransformMapBuilder::passThrough(StringRef outName, StringRef inName) {
  uint32_t dim = startIndex(inName);
  AffineExpr size = startSizeExpr(dim);
  defineDim(outName, dim, size);
  addTransform(TransformType::PassThrough, ArrayRef<int64_t>{}, {inName},
               {dim}, {outName}, {dim});
}

void TransformMapBuilder::passThrough(ArrayRef<StringRef> names) {
  llvm::SmallVector<uint32_t> dims;
  llvm::SmallVector<AffineExpr> sizes;
  dims.reserve(names.size());
  sizes.reserve(names.size());
  for (const auto name : names) {
    uint32_t dim = startIndex(name);
    dims.push_back(dim);
    sizes.push_back(startSizeExpr(dim));
  }
  for (uint32_t i = 0, e = names.size(); i < e; ++i) {
    defineDim(names[i], dims[i], sizes[i]);
  }
  addTransform(TransformType::PassThrough, ArrayRef<int64_t>{}, names, dims,
               names, dims);
}

void TransformMapBuilder::passThrough(ArrayRef<StringRef> outNames,
                                      ArrayRef<uint32_t> outDims,
                                      ArrayRef<StringRef> inNames) {
  assert(outNames.size() == inNames.size() && "One output per input");
  assert(outNames.size() == outDims.size() && "One location per output");

  llvm::SmallVector<uint32_t> inDims;
  llvm::SmallVector<AffineExpr> inSizes;
  inDims.reserve(inNames.size());
  inSizes.reserve(inNames.size());
  for (const auto name : inNames) {
    uint32_t dim = startIndex(name);
    inDims.push_back(dim);
    inSizes.push_back(startSizeExpr(dim));
  }
  for (uint32_t i = 0, e = outNames.size(); i < e; ++i) {
    defineDim(outNames[i], outDims[i], inSizes[i]);
  }
  addTransform(TransformType::PassThrough, ArrayRef<int64_t>{}, inNames, inDims,
               outNames, outDims);
}

void TransformMapBuilder::passThrough(ArrayRef<uint32_t> endIdxs,
                                      ArrayRef<uint32_t> startIdxs) {
  assert(endIdxs.size() == startIdxs.size() && "One output per input");

  llvm::SmallVector<StringRef> names;
  names.reserve(endIdxs.size());
  for (auto tuple : llvm::zip(endIdxs, startIdxs)) {
    uint32_t index = std::get<1>(tuple);
    StringRef name = startNames[index];
    names.push_back(name);
    defineDim(name, std::get<0>(tuple), startSizeExpr(index));
  }
  addTransform(TransformType::PassThrough, ArrayRef<int64_t>{}, names,
               startIdxs, names, endIdxs);
}

void TransformMapBuilder::pad(ArrayRef<StringRef> names,
                              ArrayRef<int64_t> params) {
  pad(names, ArrayRef<AffineExpr>(toExprs(params)));
}

void TransformMapBuilder::pad(ArrayRef<StringRef> names,
                              ArrayRef<AffineExpr> params) {
  llvm::SmallVector<uint32_t, 8> dims;
  dims.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(dims),
                 [&](StringRef s) -> uint32_t { return startIndex(s); });
  pad(names, dims, names, params);
}

void TransformMapBuilder::pad(StringRef outName, StringRef inName, int64_t left,
                              int64_t right) {
  pad(outName, inName, cst(left), cst(right));
}

void TransformMapBuilder::pad(StringRef outName, StringRef inName,
                              AffineExpr left, AffineExpr right) {
  uint32_t dim = startIndex(inName);
  SmallVector<AffineExpr, 2> params = {left, right};
  pad({outName}, {dim}, {inName}, params);
}

void TransformMapBuilder::pad(ArrayRef<StringRef> outNames,
                              ArrayRef<uint32_t> outDims,
                              ArrayRef<StringRef> inNames,
                              ArrayRef<int64_t> params) {
  pad(outNames, outDims, inNames, ArrayRef<AffineExpr>(toExprs(params)));
}

void TransformMapBuilder::pad(ArrayRef<StringRef> outNames,
                              ArrayRef<uint32_t> outDims,
                              ArrayRef<StringRef> inNames,
                              ArrayRef<AffineExpr> params) {
  assert(outNames.size() == outDims.size() &&
         "One name needed per dimension in padding");
  assert(outNames.size() == inNames.size() &&
         "Same number of output and input dimensions");
  assert(params.size() == 2 * outNames.size() &&
         "Two padding parameters given per dimension");
  llvm::SmallVector<uint32_t, 8> inDims;
  inDims.reserve(inNames.size());
  std::transform(inNames.begin(), inNames.end(), std::back_inserter(inDims),
                 [&](StringRef s) { return startIndex(s); });
  int64_t padSign = paddingSign();
  for (uint32_t i = 0, e = outNames.size(); i < e; ++i) {
    AffineExpr leftPad = params[i * 2];
    AffineExpr rightPad = params[i * 2 + 1];
    AffineExpr outSize =
        startSizeExpr(inDims[i]) + (leftPad * padSign) + (rightPad * padSign);
    defineDim(outNames[i], outDims[i], outSize);
  }
  addTransform(TransformType::Pad, params, inNames, inDims, outNames, outDims);
}

TransformMapBuilder &
TransformMapBuilder::operator=(const TransformMapBuilder &other) {
  if (this != &other) {
    b = other.b;
    result = other.result;
    loc = other.loc;

    startNames = other.startNames;
    startShape = other.startShape;
    endNames = other.endNames;
    endShape = other.endShape;
    symbols = other.symbols;
    frozen = other.frozen;

    startIndices.clear();
    for (uint32_t i = 0, e = startNames.size(); i < e; ++i)
      startIndices.insert({StringRef(startNames[i]), i});

    endIndices.clear();
    for (const auto &pair : endNames)
      endIndices.insert({StringRef(pair.second), pair.first});
  }
  return *this;
}

TransformMapBuilder::TransformMapBuilder(const TransformMapBuilder &other)
    : b(other.b), result(other.result), loc(other.loc), startIndices(),
      startNames(other.startNames), startShape(other.startShape), endIndices(),
      endNames(other.endNames), endShape(other.endShape),
      symbols(other.symbols), frozen(other.frozen) {
  for (uint32_t i = 0, e = startNames.size(); i < e; ++i)
    startIndices.insert({StringRef(startNames[i]), i});
  for (const auto &pair : endNames)
    endIndices.insert({StringRef(pair.second), pair.first});
}

static void reportTransformError(Location loc, TransformType type,
                                 ArrayRef<StringRef> upperNames,
                                 ArrayRef<uint32_t> upperDims,
                                 ArrayRef<StringRef> lowerNames,
                                 ArrayRef<uint32_t> lowerDims,
                                 ArrayRef<AffineExpr> params,
                                 InFlightDiagnostic &err) {
  err.attachNote(loc)
      .append("The operation type was ")
      .append(getNameForTransformType(type))
      .append("\n  Upper dimensions =")
      .appendRange(upperNames)
      .append(" at ")
      .appendRange(upperDims)
      .append("\n  Lower dimensions = ")
      .appendRange(lowerNames)
      .append(" at ")
      .appendRange(lowerDims)
      .append("\n  Parameters = ");
  for (AffineExpr p : params) {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << p;
    err.append(" ").append(str);
  }
}

/// Building from a defined set of upper dimensions
void TopDownTMBuilder::addTransformImpl(TransformType type,
                                        ArrayRef<AffineExpr> params,
                                        ArrayRef<StringRef> startNames,
                                        ArrayRef<uint32_t> startDims,
                                        ArrayRef<StringRef> endNames,
                                        ArrayRef<uint32_t> endDims) {
  auto emitError = [&]() -> InFlightDiagnostic {
    InFlightDiagnostic err =
        mlir::emitError(loc, "Error constructing coordinate transformation: ");
    reportTransformError(loc, type, startNames, startDims, endNames, endDims,
                         params, err);
    return err;
  };
  TransformAttr attr =
      getTransformAttrChecked(emitError, b.getContext(), type, params,
                              startNames, startDims, endNames, endDims);
  if (!attr) {
    emitError().report();
    llvm::report_fatal_error(Twine("Failed to add transform of type ") +
                             getNameForTransformType(type));
  }
  result.push_back(attr);
}

void TopDownTMBuilder::extractBounds(SmallVectorImpl<AffineExpr> &upperBounds,
                                     SmallVectorImpl<AffineExpr> &lowerBounds) {
  uint32_t nStart = nStartDims(), nEnd = nEndDims();
  upperBounds.reserve(nStart);
  lowerBounds.reserve(nEnd);
  for (uint32_t i = 0; i < nStart; ++i) {
    upperBounds.push_back(startSizeExpr(i));
  }
  for (uint32_t i = 0; i < nEnd; ++i) {
    lowerBounds.push_back(endSizeExpr(i));
  }
}

int64_t TopDownTMBuilder::paddingSign() const {
  // When building top-down, the output size (lower dimension) is the input size
  // (upper dimension) minus padding
  return -1;
}

void TopDownTMBuilder::slice(ArrayRef<StringRef> lowerNames,
                             ArrayRef<uint32_t> lowerDims,
                             ArrayRef<StringRef> upperNames,
                             ArrayRef<int64_t> begins,
                             ArrayRef<int64_t> fullLowerSizes) {
  slice(lowerNames, lowerDims, upperNames, ArrayRef<AffineExpr>(toExprs(begins)),
        ArrayRef<AffineExpr>(toExprs(fullLowerSizes)));
}

void TopDownTMBuilder::slice(ArrayRef<StringRef> lowerNames,
                             ArrayRef<uint32_t> lowerDims,
                             ArrayRef<StringRef> upperNames,
                             ArrayRef<AffineExpr> begins,
                             ArrayRef<AffineExpr> fullLowerSizes) {
  assert(upperNames.size() == lowerNames.size() &&
         "Need same number of upper and lower dimensions in slice");
  assert(upperNames.size() == begins.size() &&
         "Need beginning of slice for each dimension");
  assert(upperNames.size() == fullLowerSizes.size() &&
         "Need full lower size for each dimension");

  uint32_t n = upperNames.size();
  SmallVector<uint32_t, 4> upperDims;
  SmallVector<AffineExpr, 8> params;
  upperDims.reserve(n);
  params.reserve(2 * n);

  for (uint32_t i = 0; i < n; ++i) {
    uint32_t dim = startIndex(upperNames[i]);
    upperDims.push_back(dim);
    AffineExpr upperSize = startSizeExpr(dim);
    AffineExpr begin = begins[i];
    AffineExpr end = begin + upperSize;
    defineDim(lowerNames[i], lowerDims[i], fullLowerSizes[i]);
    params.push_back(begin);
    params.push_back(end);
  }
  addTransform(TransformType::Slice, params, upperNames, upperDims, lowerNames,
               lowerDims);
}

void TopDownTMBuilder::ignore(StringRef name) {
  uint32_t dim = startIndex(name);
  AffineExpr size = startSizeExpr(dim);
  addTransform(TransformType::AddDim, ArrayRef<AffineExpr>{size}, {name}, {dim},
               {}, {});
}

void TopDownTMBuilder::constDim(StringRef lowerName, uint32_t lowerDim,
                                int64_t constantVal, int64_t lowerSize) {
  constDim(lowerName, lowerDim, constantVal, cst(lowerSize));
}

void TopDownTMBuilder::constDim(StringRef lowerName, uint32_t lowerDim,
                                int64_t constantVal, AffineExpr lowerSize) {
  defineDim(lowerName, lowerDim, lowerSize);
  SmallVector<AffineExpr> params = {cst(constantVal), lowerSize};
  addTransform(TransformType::ConstDim, params, {}, {}, {lowerName},
               {lowerDim});
}

void TopDownTMBuilder::constDim(ArrayRef<StringRef> lowerNames,
                                ArrayRef<uint32_t> lowerDims,
                                ArrayRef<int64_t> constantVals,
                                ArrayRef<int64_t> lowerSizes) {
  constDim(lowerNames, lowerDims, constantVals,
           ArrayRef<AffineExpr>(toExprs(lowerSizes)));
}

void TopDownTMBuilder::constDim(ArrayRef<StringRef> lowerNames,
                                ArrayRef<uint32_t> lowerDims,
                                ArrayRef<int64_t> constantVals,
                                ArrayRef<AffineExpr> lowerSizes) {
  assert(constantVals.size() == lowerSizes.size() &&
         "must have equal number of constant values and dimension lengths");
  SmallVector<AffineExpr> params;
  params.reserve(2 * constantVals.size());
  for (const auto &[name, dim, val, size] :
       llvm::zip(lowerNames, lowerDims, constantVals, lowerSizes)) {
    params.emplace_back(cst(val));
    params.emplace_back(size);
    defineDim(name, dim, size);
  }
  addTransform(TransformType::ConstDim, params, {}, {}, lowerNames, lowerDims);
}

void TopDownTMBuilder::embed(StringRef lowerName, uint32_t lowerDim,
                             int64_t lowerSize, ArrayRef<StringRef> upperNames,
                             ArrayRef<int64_t> coefficients) {
  embed(lowerName, lowerDim, cst(lowerSize), upperNames,
        ArrayRef<AffineExpr>(toExprs(coefficients)));
}

void TopDownTMBuilder::embed(StringRef lowerName, uint32_t lowerDim,
                             AffineExpr lowerSize,
                             ArrayRef<StringRef> upperNames,
                             ArrayRef<AffineExpr> coefficients) {
  assert(upperNames.size() == coefficients.size() &&
         "Must provide a coefficient for each dimension");
  SmallVector<uint32_t, 8> upperDims;
  upperDims.reserve(upperNames.size());
  for (const StringRef name : upperNames) {
    upperDims.push_back(startIndex(name));
  }

  defineDim(lowerName, lowerDim, lowerSize);
  addTransform(TransformType::Embed, coefficients, upperNames, upperDims,
               {lowerName}, {lowerDim});
}

void TopDownTMBuilder::unmerge(StringRef lowerName, uint32_t lowerDim,
                               ArrayRef<StringRef> upperNames,
                               ArrayRef<int64_t> lengths) {
  unmerge(lowerName, lowerDim, upperNames,
          ArrayRef<AffineExpr>(toExprs(lengths)));
}

void TopDownTMBuilder::unmerge(StringRef lowerName, uint32_t lowerDim,
                               ArrayRef<StringRef> upperNames,
                               ArrayRef<AffineExpr> lengths) {
  assert(upperNames.size() == lengths.size() &&
         "Must provide a length for each dimension");
  SmallVector<uint32_t, 8> upperDims;
  upperDims.reserve(upperNames.size());
  for (const StringRef name : upperNames) {
    upperDims.push_back(startIndex(name));
  }
  AffineExpr size = cst(1);
  for (AffineExpr length : lengths) {
    size = simplify(size * length);
  }
  defineDim(lowerName, lowerDim, size);
  addTransform(TransformType::Unmerge, lengths, upperNames, upperDims,
               {lowerName}, {lowerDim});
}

void TopDownTMBuilder::merge(ArrayRef<StringRef> lowerNames,
                             ArrayRef<uint32_t> lowerDims, StringRef upperName,
                             ArrayRef<int64_t> sizes) {
  merge(lowerNames, lowerDims, upperName, ArrayRef<AffineExpr>(toExprs(sizes)));
}

void TopDownTMBuilder::merge(ArrayRef<StringRef> lowerNames,
                             ArrayRef<uint32_t> lowerDims, StringRef upperName,
                             ArrayRef<AffineExpr> sizes) {
  assert(lowerNames.size() == lowerDims.size() &&
         "One name per dimension required in merge");
  assert(lowerDims.size() == sizes.size() &&
         "One size per output dimension required in merge");

  uint32_t upperDim = startIndex(upperName);
  [[maybe_unused]] AffineExpr upperSize = startSizeExpr(upperDim);

  [[maybe_unused]] AffineExpr totalLowerSize = cst(1);
  for (AffineExpr s : sizes) {
    totalLowerSize = simplify(totalLowerSize * s);
  }
  assert(symbolicEqual(upperSize, totalLowerSize, 0, getSymbols().size()) &&
         "Upper dimension to merge must have same size as combined lower "
         "dimensions");
  for (auto triple : llvm::zip(lowerNames, lowerDims, sizes)) {
    defineDim(std::get<0>(triple), std::get<1>(triple), std::get<2>(triple));
  }
  addTransform(TransformType::Merge, sizes, {upperName}, {upperDim}, lowerNames,
               lowerDims);
}

void TopDownTMBuilder::takeRemainder(StringRef name, int64_t length) {
  assert(length > 0 && "Remainder can't be zero");
  takeRemainder(name, cst(length));
}

void TopDownTMBuilder::takeRemainder(StringRef name, AffineExpr length) {
  uint32_t dim = startIndex(name);
  // WE're not recording this, but we should be.
  // int64_t size = startSize(dim);
  defineDim(name, dim, length);
  // The semantics of Broadcast are x -> x % l so we might as well use it.
  addTransform(TransformType::Broadcast, ArrayRef<AffineExpr>{length}, {name},
               {dim}, {name}, {dim});
}

llvm::SmallVector<uint32_t>
TopDownTMBottomDimsWrapper::toBottomDims(ArrayRef<StringRef> names) {
  llvm::SmallVector<uint32_t> ret;
  ret.reserve(names.size());
  for (auto name : names) {
    ret.push_back(bottomDims[name]);
  }
  return ret;
}

void TopDownTMBottomDimsWrapper::passThrough(StringRef name) {
  b.passThrough(name, bottomDims[name], name);
}

void TopDownTMBottomDimsWrapper::passThrough(ArrayRef<StringRef> names) {
  b.passThrough(names, toBottomDims(names), names);
}

void TopDownTMBottomDimsWrapper::passThrough(StringRef outName,
                                             StringRef inName) {
  b.passThrough(outName, toBottomDims(outName), inName);
}

void TopDownTMBottomDimsWrapper::pad(ArrayRef<StringRef> outNames,
                                     ArrayRef<StringRef> inNames,
                                     ArrayRef<int64_t> params) {
  b.pad(outNames, toBottomDims(outNames), inNames, params);
}

void TopDownTMBottomDimsWrapper::pad(ArrayRef<StringRef> outNames,
                                     ArrayRef<StringRef> inNames,
                                     ArrayRef<AffineExpr> params) {
  b.pad(outNames, toBottomDims(outNames), inNames, params);
}

void TopDownTMBottomDimsWrapper::constDim(StringRef lowerName,
                                          int64_t constantVal,
                                          int64_t lowerSize) {
  b.constDim(lowerName, bottomDims[lowerName], constantVal, lowerSize);
}

void TopDownTMBottomDimsWrapper::constDim(StringRef lowerName,
                                          int64_t constantVal,
                                          AffineExpr lowerSize) {
  b.constDim(lowerName, bottomDims[lowerName], constantVal, lowerSize);
}

void TopDownTMBottomDimsWrapper::constDim(ArrayRef<StringRef> lowerNames,
                                          ArrayRef<int64_t> constantVals,
                                          ArrayRef<int64_t> lowerSizes) {
  b.constDim(lowerNames, toBottomDims(lowerNames), constantVals, lowerSizes);
}

void TopDownTMBottomDimsWrapper::embed(StringRef lowerName, int64_t lowerSize,
                                       ArrayRef<StringRef> upperNames,
                                       ArrayRef<int64_t> coefficients) {
  b.embed(lowerName, bottomDims[lowerName], lowerSize, upperNames,
          coefficients);
}

void TopDownTMBottomDimsWrapper::embed(StringRef lowerName,
                                       AffineExpr lowerSize,
                                       ArrayRef<StringRef> upperNames,
                                       ArrayRef<AffineExpr> coefficients) {
  b.embed(lowerName, bottomDims[lowerName], lowerSize, upperNames,
          coefficients);
}

void TopDownTMBottomDimsWrapper::unmerge(StringRef lowerName,
                                         ArrayRef<StringRef> upperNames,
                                         ArrayRef<int64_t> lengths) {
  b.unmerge(lowerName, bottomDims[lowerName], upperNames, lengths);
}

void TopDownTMBottomDimsWrapper::unmerge(StringRef lowerName,
                                         ArrayRef<StringRef> upperNames,
                                         ArrayRef<AffineExpr> lengths) {
  b.unmerge(lowerName, bottomDims[lowerName], upperNames, lengths);
}

void TopDownTMBottomDimsWrapper::merge(ArrayRef<StringRef> lowerNames,
                                       StringRef upperName,
                                       ArrayRef<int64_t> sizes) {
  b.merge(lowerNames, toBottomDims(lowerNames), upperName, sizes);
}

void TopDownTMBottomDimsWrapper::merge(ArrayRef<StringRef> lowerNames,
                                       StringRef upperName,
                                       ArrayRef<AffineExpr> sizes) {
  b.merge(lowerNames, toBottomDims(lowerNames), upperName, sizes);
}

/// Building from a defined set of lower dimensions
void BottomUpTMBuilder::addTransformImpl(TransformType type,
                                         ArrayRef<AffineExpr> params,
                                         ArrayRef<StringRef> startNames,
                                         ArrayRef<uint32_t> startDims,
                                         ArrayRef<StringRef> endNames,
                                         ArrayRef<uint32_t> endDims) {
  auto emitError = [&]() -> InFlightDiagnostic {
    InFlightDiagnostic err =
        mlir::emitError(loc, "Error constructing coordinate transformation: ");
    reportTransformError(loc, type, endNames, endDims, startNames, startDims,
                         params, err);
    return err;
  };
  TransformAttr attr =
      getTransformAttrChecked(emitError, b.getContext(), type, params, endNames,
                              endDims, startNames, startDims);
  if (!attr) {
    emitError().report();
    llvm::report_fatal_error(Twine("Failed to add transform of type ") +
                             getNameForTransformType(type));
  }
  result.push_back(attr);
}

void BottomUpTMBuilder::extractBounds(SmallVectorImpl<AffineExpr> &upperBounds,
                                      SmallVectorImpl<AffineExpr> &lowerBounds) {
  uint32_t nStart = nStartDims(), nEnd = nEndDims();
  upperBounds.reserve(nEnd);
  lowerBounds.reserve(nStart);
  for (uint32_t i = 0; i < nEnd; ++i) {
    upperBounds.push_back(endSizeExpr(i));
  }
  for (uint32_t i = 0; i < nStart; ++i) {
    lowerBounds.push_back(startSizeExpr(i));
  }
}

int64_t BottomUpTMBuilder::paddingSign() const {
  // When building bottom-up, the output size (upper dimension) is the input
  // size (bottom dimension) plus padding
  return 1;
}

void BottomUpTMBuilder::addDim(StringRef name, uint32_t dim, int64_t size) {
  addDim(name, dim, cst(size));
}

void BottomUpTMBuilder::addDim(StringRef name, uint32_t dim, AffineExpr size) {
  defineDim(name, dim, size);
  addTransform(TransformType::AddDim, ArrayRef<AffineExpr>{size}, {}, {},
               {name}, {dim});
}

void BottomUpTMBuilder::dropDimAtIndex(StringRef lowerName,
                                       int64_t constantVal) {
  uint32_t dim = startIndex(lowerName);
  AffineExpr size = startSizeExpr(dim);
  assert(constantVal >= 0 &&
         (!isa<AffineConstantExpr>(size) ||
          constantVal < cast<AffineConstantExpr>(size).getValue()) &&
         "constant value must be in range [0, size)");
  SmallVector<AffineExpr> params = {cst(constantVal), size};
  addTransform(TransformType::ConstDim, params, {lowerName}, {dim}, {}, {});
}

void BottomUpTMBuilder::dropDimsAtIndices(ArrayRef<StringRef> lowerNames,
                                          ArrayRef<int64_t> constantVals) {
  assert(lowerNames.size() == constantVals.size() &&
         "One constant value needed per lower dimension");
  for (auto pair : llvm::zip(lowerNames, constantVals)) {
    dropDimAtIndex(std::get<0>(pair), std::get<1>(pair));
  }
}

void BottomUpTMBuilder::broadcast(ArrayRef<uint32_t> endDims,
                                  ArrayRef<int64_t> endSizes) {
  broadcast(endDims, ArrayRef<AffineExpr>(toExprs(endSizes)));
}

void BottomUpTMBuilder::broadcast(ArrayRef<uint32_t> endDims,
                                  ArrayRef<AffineExpr> endSizes) {
  SmallVector<AffineExpr, 8> params;
  SmallVector<StringRef, 8> lowerNames;
  SmallVector<StringRef, 8> upperNames;
  for (auto tuple : llvm::zip(endDims, endSizes)) {
    uint32_t dim = std::get<0>(tuple);
    AffineExpr size = std::get<1>(tuple);
    auto name = startName(dim);
    params.push_back(startSizeExpr(dim));
    lowerNames.push_back(name);
    upperNames.push_back(name);
    defineDim(name, dim, size);
  }
  addTransform(TransformType::Broadcast, params, upperNames, endDims,
               lowerNames, endDims);
}

void BottomUpTMBuilder::slice(ArrayRef<StringRef> upperNames,
                              ArrayRef<StringRef> lowerNames,
                              ArrayRef<int64_t> begins,
                              ArrayRef<int64_t> ends) {
  slice(upperNames, lowerNames, ArrayRef<AffineExpr>(toExprs(begins)),
        ArrayRef<AffineExpr>(toExprs(ends)));
}

void BottomUpTMBuilder::slice(ArrayRef<StringRef> upperNames,
                              ArrayRef<StringRef> lowerNames,
                              ArrayRef<AffineExpr> begins,
                              ArrayRef<AffineExpr> ends) {
  assert(upperNames.size() == lowerNames.size() &&
         "Need same number of input and output dimensions in slice");
  assert(upperNames.size() == begins.size() &&
         "Need beginning of slice for each dimension");
  assert(upperNames.size() == ends.size() &&
         "Need end of slice for each dimension");

  uint32_t n = lowerNames.size();
  SmallVector<uint32_t, 4> dims;
  dims.reserve(n);

  SmallVector<AffineExpr, 8> params;
  params.reserve(2 * n);

  for (uint32_t i = 0; i < n; ++i) {
    uint32_t dim = startIndex(lowerNames[i]);
    dims.push_back(dim);
    AffineExpr begin = begins[i];
    AffineExpr end = ends[i];
    defineDim(upperNames[i], dim, end - begin);
    params.push_back(begin);
    params.push_back(end);
  }
  addTransform(TransformType::Slice, params, lowerNames, dims, upperNames,
               dims);
}

void BottomUpTMBuilder::embed(ArrayRef<StringRef> upperNames,
                              ArrayRef<uint32_t> upperDims,
                              ArrayRef<int64_t> upperSizes, StringRef lowerName,
                              ArrayRef<int64_t> coefficients) {
  embed(upperNames, upperDims, ArrayRef<AffineExpr>(toExprs(upperSizes)),
        lowerName, ArrayRef<AffineExpr>(toExprs(coefficients)));
}

void BottomUpTMBuilder::embed(ArrayRef<StringRef> upperNames,
                              ArrayRef<uint32_t> upperDims,
                              ArrayRef<AffineExpr> upperSizes,
                              StringRef lowerName,
                              ArrayRef<AffineExpr> coefficients) {
  assert(upperNames.size() == upperDims.size() &&
         "One name per upper dimension needed in merge");
  assert(upperDims.size() == coefficients.size() &&
         "One coefficient per upper dimension needed in merge");
  assert(upperDims.size() == upperSizes.size() &&
         "One size per upper dimension needed in merge");

  uint32_t lowerDim = startIndex(lowerName);
  for (auto triple : llvm::zip(upperNames, upperDims, upperSizes)) {
    defineDim(std::get<0>(triple), std::get<1>(triple), std::get<2>(triple));
  }
  addTransform(TransformType::Embed, coefficients, {lowerName}, {lowerDim},
               upperNames, upperDims);
}

void BottomUpTMBuilder::unmerge(ArrayRef<StringRef> upperNames,
                                ArrayRef<uint32_t> upperDims,
                                StringRef lowerName,
                                ArrayRef<int64_t> lengths) {
  unmerge(upperNames, upperDims, lowerName,
          ArrayRef<AffineExpr>(toExprs(lengths)));
}

void BottomUpTMBuilder::unmerge(ArrayRef<StringRef> upperNames,
                                ArrayRef<uint32_t> upperDims,
                                StringRef lowerName,
                                ArrayRef<AffineExpr> lengths) {
  assert(upperNames.size() == upperDims.size() &&
         "One name needed per upper dimension in unmerge");
  assert(upperDims.size() == lengths.size() &&
         "One length needed per upper dimension in unmerge");

  uint32_t lowerDim = startIndex(lowerName);

  [[maybe_unused]] AffineExpr totalLength = startSizeExpr(lowerDim);
  [[maybe_unused]] AffineExpr lengthsProd = cst(1);
  for (AffineExpr length : lengths) {
    lengthsProd = simplify(lengthsProd * length);
  }
  assert(symbolicEqual(lengthsProd, totalLength, 0, getSymbols().size()) &&
         "failed to partition unmerge length among upper dimensions");

  for (auto triple : llvm::zip(upperNames, upperDims, lengths)) {
    defineDim(std::get<0>(triple), std::get<1>(triple), std::get<2>(triple));
  }
  addTransform(TransformType::Unmerge, lengths, {lowerName}, {lowerDim},
               upperNames, upperDims);
}

void BottomUpTMBuilder::merge(StringRef upperName, uint32_t upperDim,
                              ArrayRef<StringRef> lowerNames) {
  uint32_t n = lowerNames.size();
  llvm::SmallVector<uint32_t, 4> lowerDims;
  lowerDims.reserve(n);
  llvm::SmallVector<AffineExpr, 4> lowerSizes;
  lowerSizes.reserve(n);

  AffineExpr upperSize = cst(1);
  for (const StringRef name : lowerNames) {
    uint32_t dim = startIndex(name);
    AffineExpr size = startSizeExpr(dim);
    upperSize = simplify(upperSize * size);
    lowerDims.push_back(dim);
    lowerSizes.push_back(size);
  }
  defineDim(upperName, upperDim, upperSize);
  addTransform(TransformType::Merge, lowerSizes, lowerNames, lowerDims,
               {upperName}, {upperDim});
}

void BottomUpTMTopDimsWrapper::passThrough(StringRef name) {
  b.passThrough({name}, {topDims[name]}, {name});
}

void BottomUpTMTopDimsWrapper::passThrough(ArrayRef<StringRef> names) {
  b.passThrough(names, toTopDims(names), names);
}

void BottomUpTMTopDimsWrapper::passThrough(StringRef outName,
                                           StringRef inName) {
  b.passThrough(outName, toTopDims(outName), inName);
}

void BottomUpTMTopDimsWrapper::pad(ArrayRef<StringRef> outNames,
                                   ArrayRef<StringRef> inNames,
                                   ArrayRef<int64_t> params) {
  b.pad(outNames, toTopDims(outNames), inNames, params);
}

void BottomUpTMTopDimsWrapper::pad(ArrayRef<StringRef> outNames,
                                   ArrayRef<StringRef> inNames,
                                   ArrayRef<AffineExpr> params) {
  b.pad(outNames, toTopDims(outNames), inNames, params);
}

void BottomUpTMTopDimsWrapper::addDim(StringRef name, int64_t size) {
  b.addDim(name, topDims[name], size);
}

void BottomUpTMTopDimsWrapper::addDim(StringRef name, AffineExpr size) {
  b.addDim(name, topDims[name], size);
}

void BottomUpTMTopDimsWrapper::dropDimAtIndex(StringRef lowerName,
                                              int64_t constantVal) {
  b.dropDimAtIndex(lowerName, constantVal);
}

void BottomUpTMTopDimsWrapper::dropDimsAtIndices(
    ArrayRef<StringRef> lowerNames, ArrayRef<int64_t> constantVals) {
  b.dropDimsAtIndices(lowerNames, constantVals);
}

void BottomUpTMTopDimsWrapper::embed(ArrayRef<StringRef> upperNames,
                                     ArrayRef<int64_t> upperSizes,
                                     StringRef lowerName,
                                     ArrayRef<int64_t> coefficients) {
  b.embed(upperNames, toTopDims(upperNames), upperSizes, lowerName,
          coefficients);
}

void BottomUpTMTopDimsWrapper::embed(ArrayRef<StringRef> upperNames,
                                     ArrayRef<AffineExpr> upperSizes,
                                     StringRef lowerName,
                                     ArrayRef<AffineExpr> coefficients) {
  b.embed(upperNames, toTopDims(upperNames), upperSizes, lowerName,
          coefficients);
}

void BottomUpTMTopDimsWrapper::unmerge(ArrayRef<StringRef> upperNames,
                                       StringRef lowerName,
                                       ArrayRef<int64_t> lengths) {
  b.unmerge(upperNames, toTopDims(upperNames), lowerName, lengths);
}

void BottomUpTMTopDimsWrapper::unmerge(ArrayRef<StringRef> upperNames,
                                       StringRef lowerName,
                                       ArrayRef<AffineExpr> lengths) {
  b.unmerge(upperNames, toTopDims(upperNames), lowerName, lengths);
}

void BottomUpTMTopDimsWrapper::merge(StringRef upperName,
                                     ArrayRef<StringRef> lowerNames) {
  b.merge(upperName, topDims[upperName], lowerNames);
}

llvm::SmallVector<uint32_t>
BottomUpTMTopDimsWrapper::toTopDims(ArrayRef<StringRef> names) {
  llvm::SmallVector<uint32_t> ret;
  ret.reserve(names.size());
  for (auto name : names) {
    ret.push_back(topDims[name]);
  }
  return ret;
}

/// Utility methods

llvm::StringMap<uint32_t> mlir::rock::expandNamesInPlace(
    ArrayRef<StringRef> original,
    const llvm::StringMap<SmallVector<StringRef, 2>> expansion) {
  uint32_t offset = 0;
  llvm::StringMap<uint32_t> ret;
  for (auto pair : llvm::enumerate(original)) {
    uint32_t origIndex = pair.index();
    StringRef origName = pair.value();
    if (expansion.count(origName) != 0) {
      for (auto newName : (*expansion.find(origName)).getValue()) {
        [[maybe_unused]] bool insertResult =
            ret.insert({newName, origIndex + offset}).second;
        assert(insertResult && "Duplicate dimension in dimension expansion");
        offset++;
      }
      offset--; // Handle extra count and dropping a dimension
    } else {
      [[maybe_unused]] bool insertResult =
          ret.insert({origName, origIndex + offset}).second;
      assert(insertResult && "Dimension already defined by expansion");
    }
  }
  return ret;
}

llvm::StringMap<uint32_t> mlir::rock::expandNamesInPlace(
    TransformMapBuilder &builder,
    const llvm::StringMap<SmallVector<StringRef, 2>> expansion) {
  SmallVector<StringRef, 8> names;
  builder.getEndNames(names);
  return expandNamesInPlace(names, expansion);
}
