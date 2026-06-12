//===- RockOps.cpp - Rock MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/Generator/ConvGenerator.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockConvInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTypes.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/utility/KnobUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::rock;

#include "mlir/Dialect/Rock/IR/RockOpsDialect.cpp.inc"
#include "mlir/Dialect/Rock/IR/RockTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

// This function changes the shape of scaleA/scaleB to match the shape of A/B
static SmallVector<int64_t> normalizeScaleShape(ArrayRef<int64_t> scaleShape,
                                                uint64_t quantBlockSize,
                                                bool isA) {
  SmallVector<int64_t> modifiedScaleShape(scaleShape);
  modifiedScaleShape[scaleShape.size() - 1] *= quantBlockSize;
  if (!isA)
    std::swap(modifiedScaleShape[scaleShape.size() - 2],
              modifiedScaleShape[scaleShape.size() - 1]);

  return modifiedScaleShape;
}

template <int N>
struct rank : rank<N - 1> {};

template <>
struct rank<0> {};

template <typename OpType>
static void
getCommonEffects(OpType &op,
                 SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto *read = MemoryEffects::Read::get();
  auto *write = MemoryEffects::Write::get();
  effects.emplace_back(read, &op.getSourceMutable());
  effects.emplace_back(write, &op.getDestMutable());
}

template <typename OpType>
static LogicalResult verifyScales(OpType op, Value matrix, Value scale,
                                  std::optional<uint64_t> quantBlockSize,
                                  bool isA) {
  if (scale != nullptr) {
    if (!quantBlockSize.has_value())
      return op.emitError("quantBlockSize is not defined");

    ArrayRef<int64_t> matrixShape =
        cast<ShapedType>(matrix.getType()).getShape();
    SmallVector<int64_t> scaleShape =
        normalizeScaleShape(cast<ShapedType>(scale.getType()).getShape(),
                            quantBlockSize.value(), /*isA=*/isA);

    StringRef matrixName = isA ? "A" : "B";
    if (matrixShape != ArrayRef<int64_t>(scaleShape)) {
      return op.emitError(llvm::formatv(
          "Scale{0} shape must match matrix{0} shape.", matrixName));
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RockDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {
struct RockOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (isa<TransformMapAttr>(attr)) {
      os << "transform_map";
      return AliasResult::OverridableAlias;
    }
    if (isa<GemmParamsAttr>(attr)) {
      os << "gemm_params";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

namespace mlir {
namespace rock {

/// Constant Name for Rock Kernel Module
constexpr const ::llvm::StringLiteral RockDialect::kKernelModuleName;

ArrayAttr noTransformsArray(Builder &b, size_t n) {
  llvm::SmallVector<Attribute, 4> ret;
  ret.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    ret.push_back(b.getArrayAttr({}));
  }
  return b.getArrayAttr(ret);
}

//===---------------------------------------------------------
// TransformAttr
//===---------------------------------------------------------
template <typename T>
static ParseResult
parseAndGather(mlir::AsmParser &parser, AsmParser::Delimiter delim,
               SmallVectorImpl<T> &ret,
               llvm::function_ref<ParseResult(T &)> getElement) {
  return parser.parseCommaSeparatedList(delim, [&]() -> ParseResult {
    T out;
    ParseResult res = getElement(out);
    if (res.succeeded()) {
      ret.push_back(out);
    }
    return res;
  });
}

mlir::Attribute TransformAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  llvm::SMLoc startLoc = parser.getCurrentLocation();
  if (parser.parseLess()) {
    return {};
  }

  std::string transformName;
  if (parser.parseKeywordOrString(&transformName)) {
    return {};
  }

  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  std::optional<TransformType> transformType =
      getTransformTypeForName(transformName);
  if (!transformType.has_value()) {
    parser.emitError(typeLoc, "expected a name of a known transform")
            .attachNote()
        << "The transforms are PassThrough, Pad, Slice, Embed, Unmerge, Merge, "
           "AddDim, Broadcast, ConstDim";
    return {};
  }

  llvm::SmallVector<int64_t> params;
  if (parser.parseOptionalLBrace().succeeded()) {
    if (parseAndGather<int64_t>(parser, AsmParser::Delimiter::None, params,
                                [&](int64_t &out) -> ParseResult {
                                  return parser.parseInteger(out);
                                }) ||
        parser.parseRBrace()) {
      return {};
    }
  }

  llvm::SmallVector<std::string> upperNamesStorage;
  llvm::SmallVector<unsigned> upperDims;
  if (parseAndGather<std::string>(parser, AsmParser::Delimiter::Square,
                                  upperNamesStorage,
                                  [&](std::string &out) -> ParseResult {
                                    return parser.parseKeywordOrString(&out);
                                  }) ||
      parser.parseKeyword("at") ||
      parseAndGather<unsigned>(parser, AsmParser::Delimiter::Square, upperDims,
                               [&](unsigned &out) -> ParseResult {
                                 return parser.parseInteger(out);
                               })) {
    return {};
  }

  if (parser.parseArrow()) {
    return {};
  }

  llvm::SmallVector<std::string> lowerNamesStorage;
  llvm::SmallVector<unsigned> lowerDims;
  if (parseAndGather<std::string>(parser, AsmParser::Delimiter::Square,
                                  lowerNamesStorage,
                                  [&](std::string &out) -> ParseResult {
                                    return parser.parseKeywordOrString(&out);
                                  }) ||
      parser.parseKeyword("at") ||
      parseAndGather<unsigned>(parser, AsmParser::Delimiter::Square, lowerDims,
                               [&](unsigned &out) -> ParseResult {
                                 return parser.parseInteger(out);
                               })) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  SmallVector<StringRef> upperNames;
  for (const std::string &nameStr : upperNamesStorage) {
    upperNames.push_back(nameStr);
  }
  SmallVector<StringRef> lowerNames;
  for (const std::string &nameStr : lowerNamesStorage) {
    lowerNames.push_back(nameStr);
  }

  return parser.getChecked<TransformAttr>(
      startLoc, parser.getContext(), transformType.value(), params, upperNames,
      upperDims, lowerNames, lowerDims);
}

void TransformAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  StringRef transformName = getNameForTransformType(getType());
  printer.printKeywordOrString(transformName);
  ArrayRef<int64_t> params = getParams();
  if (params.size() > 0) {
    printer << "{";
    llvm::interleaveComma(params, printer);
    printer << "}";
  }
  printer << " [";
  llvm::interleaveComma(getUpperNames(), printer,
                        [&](StringRef s) { printer << "\"" << s << "\""; });
  printer << "] at [";
  llvm::interleaveComma(getUpperDims(), printer);
  printer << "] -> [";
  llvm::interleaveComma(getLowerNames(), printer,
                        [&](StringRef s) { printer << "\"" << s << "\""; });
  printer << "] at [";
  llvm::interleaveComma(getLowerDims(), printer);
  printer << "]>";
}

LogicalResult
TransformAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      TransformType type, llvm::ArrayRef<int64_t> params,
                      llvm::ArrayRef<llvm::StringRef> upperNames,
                      llvm::ArrayRef<unsigned> upperDims,
                      llvm::ArrayRef<llvm::StringRef> lowerNames,
                      llvm::ArrayRef<unsigned> lowerDims) {
  if (upperNames.size() != upperDims.size()) {
    return emitError() << "Have " << upperNames.size() << " names for "
                       << upperDims.size() << " dimensions";
  }
  if (lowerNames.size() != lowerDims.size()) {
    return emitError() << "Have " << lowerNames.size() << " names for "
                       << lowerDims.size() << " dimensions";
  }
  if (type != TransformType::AddDim && lowerDims.empty()) {
    return emitError() << "The transformation must define outputs";
  }
  if (type != TransformType::ConstDim && upperDims.empty()) {
    return emitError() << "The transformation must have at least one input";
  }

  switch (type) {
  case TransformType::PassThrough: {
    if (upperDims.size() != lowerDims.size()) {
      return emitError()
             << "PassThrough must have the same number of inputs and outputs";
    }
    if (!params.empty()) {
      return emitError() << "PassThrough has no parameters";
    }
    break;
  }
  case TransformType::Pad: {
    if (upperDims.size() != lowerDims.size()) {
      return emitError()
             << "Pad must have the same number of inputs and outputs";
    }
    if (params.size() != 2 * upperDims.size()) {
      return emitError()
             << "Pad must have two parameters (left, right) per dimension";
    }
    for (size_t i = 0, e = params.size(); i < e; i += 2) {
      if (params[i] < 0)
        return emitError() << "Pad: left padding (" << params[i]
                           << ") must be non-negative";
      if (params[i + 1] < 0)
        return emitError() << "Pad: right padding (" << params[i + 1]
                           << ") must be non-negative";
    }
    break;
  }
  case TransformType::Slice: {
    if (upperDims.size() != lowerDims.size()) {
      return emitError()
             << "Slice must have the same number of inputs and outputs";
    }
    if (params.size() != 2 * upperDims.size()) {
      return emitError()
             << "Slice must have two parameters (begin, end) per dimension";
    }
    for (size_t i = 0, e = params.size(); i < e; i += 2) {
      if (params[i] < 0)
        return emitError() << "Slice: begin (" << params[i]
                           << ") must be non-negative";
      if (params[i + 1] <= params[i])
        return emitError() << "Slice: end (" << params[i + 1]
                           << ") must be greater than begin (" << params[i]
                           << ")";
    }
    break;
  }
  case TransformType::Embed: {
    if (lowerDims.size() != 1) {
      return emitError() << "Embed can only have one output argument";
    }
    if (params.size() != upperDims.size()) {
      return emitError()
             << "Embed must specify one coefficient per input dimension";
    }
    break;
  }
  case TransformType::Unmerge: {
    if (lowerDims.size() != 1) {
      return emitError() << "Unmerge can only have one output argument";
    }
    if (params.size() != upperDims.size()) {
      return emitError()
             << "Unmerge must specify one length per input dimension";
    }
    for (int64_t p : params) {
      if (p <= 0)
        return emitError() << "Unmerge dimension length " << p
                           << " must be positive";
    }
    break;
  }
  case TransformType::Merge: {
    if (upperDims.size() != 1) {
      return emitError() << "Merge can only have one input dimension";
    }
    if (params.size() != lowerDims.size()) {
      return emitError()
             << "Merge must have one parameter per output dimension (its size)";
    }
    for (int64_t p : params) {
      if (p <= 0)
        return emitError() << "Merge dimension size " << p
                           << " must be positive";
    }
    break;
  }
  case TransformType::AddDim:
    if (upperDims.size() != 1) {
      return emitError() << "Can only add one dimension at a time";
    }
    if (params.size() != upperDims.size()) {
      return emitError() << "Must supply a size parameter for each dimension";
    }
    if (params[0] <= 0) {
      return emitError() << "AddDim size " << params[0] << " must be positive";
    }
    if (!lowerDims.empty()) {
      return emitError() << "The added dimension cannot be mapped anywhere";
    }
    break;
  case TransformType::Broadcast:
    if (upperDims.size() != lowerDims.size()) {
      return emitError() << "Broadcast must have same rank";
    }
    if (params.size() != lowerDims.size()) {
      return emitError()
             << "Broadcast must specify the output length for each dimension";
    }
    break;
  case TransformType::ConstDim:
    if (!upperDims.empty())
      return emitError() << "ConstDim must not take any inputs";
    if (params.size() != 2 * lowerDims.size())
      return emitError()
             << "ConstDim is parameterized by [value, length] pairs";
    for (size_t i = 0, e = params.size(); i < e; i += 2) {
      if (params[i] < 0)
        return emitError() << "For constant dimension " << lowerDims[i / 2]
                           << " constant value " << params[i]
                           << " must be non-negative";
      if (params[i] >= params[i + 1])
        return emitError() << "For constant dimension " << lowerDims[i / 2]
                           << " constant value " << params[i]
                           << " must be less than dimension "
                              "length "
                           << params[i + 1];
    }
    break;
  }
  return success();
}

TransformAttr getTransformAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MLIRContext *context, TransformType type, ArrayRef<int64_t> params,
    ArrayRef<StringRef> upperNames, ArrayRef<uint32_t> upperDims,
    ArrayRef<StringRef> lowerNames, ArrayRef<uint32_t> lowerDims) {
  return TransformAttr::getChecked(emitError, context, type, params, upperNames,
                                   upperDims, lowerNames, lowerDims);
}

//===---------------------------------------------------------
// TransformMapAttr
//===---------------------------------------------------------

TransformMapAttr getTransformMapAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MLIRContext *context, ArrayRef<TransformAttr> ops, AffineMapAttr map,
    DenseI64ArrayAttr upperBounds, DenseI64ArrayAttr lowerBounds) {
  return TransformMapAttr::getChecked(emitError, context, ops, map, upperBounds,
                                      lowerBounds);
}

LogicalResult TransformMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<::mlir::rock::TransformAttr> ops, AffineMapAttr map,
    DenseI64ArrayAttr upperBounds, DenseI64ArrayAttr lowerBounds) {
  AffineMap rawMap = map.getAffineMap();
  if (rawMap.getNumSymbols() != 0) {
    return emitError() << "Affine map must not have symbol inputs, but has "
                       << rawMap.getNumSymbols();
  }
  if (rawMap.getNumInputs() != upperBounds.size()) {
    return emitError() << "Affine map has " << rawMap.getNumInputs()
                       << " inputs but there are " << upperBounds.size()
                       << " input dimensions";
  }
  if (rawMap.getNumResults() != lowerBounds.size()) {
    return emitError() << "Affine map has " << rawMap.getNumResults()
                       << " outputs but there are " << lowerBounds.size()
                       << " output dimensions";
  }

  for (int64_t v : upperBounds.asArrayRef()) {
    if (v <= 0) {
      return emitError() << "Upper bound/shape component must be positive, got "
                         << v;
    }
  }
  for (int64_t v : lowerBounds.asArrayRef()) {
    if (v <= 0) {
      return emitError() << "Lower bound/shape component must be positive, got "
                         << v;
    }
  }

  ArrayRef<int64_t> ub = upperBounds.asArrayRef();
  ArrayRef<int64_t> lb = lowerBounds.asArrayRef();
  for (TransformAttr t : ops) {
    ArrayRef<uint32_t> uDims = t.getUpperDims();
    ArrayRef<uint32_t> lDims = t.getLowerDims();
    ArrayRef<int64_t> params = t.getParams();

    for (uint32_t d : uDims) {
      if (d >= ub.size())
        return emitError() << "Upper dimension index " << d
                           << " is out of range [0, " << ub.size() << ")";
    }
    for (uint32_t d : lDims) {
      if (d >= lb.size())
        return emitError() << "Lower dimension index " << d
                           << " is out of range [0, " << lb.size() << ")";
    }

    // Structural checks (dimension counts, parameter counts, value validity)
    // are handled by TransformAttr::verify. Here we only check consistency
    // between the transform parameters and the map's upper/lower bounds.
    switch (t.getType()) {
    case TransformType::Unmerge: {
      int64_t product = 1;
      for (auto [dim, param] : llvm::zip(uDims, params)) {
        if (ub[dim] != param) {
          return emitError()
                 << "Unmerge: upper bound " << ub[dim] << " at dimension "
                 << dim << " does not match parameter " << param;
        }
        product *= param;
      }
      if (product != lb[lDims[0]]) {
        return emitError() << "Unmerge: product of parameters (" << product
                           << ") does not match lower bound (" << lb[lDims[0]]
                           << ")";
      }
      break;
    }
    case TransformType::Merge: {
      int64_t product = 1;
      for (auto [dim, param] : llvm::zip(lDims, params)) {
        if (lb[dim] != param) {
          return emitError()
                 << "Merge: lower bound " << lb[dim] << " at dimension " << dim
                 << " does not match parameter " << param;
        }
        product *= param;
      }
      if (product != ub[uDims[0]]) {
        return emitError() << "Merge: product of parameters (" << product
                           << ") does not match upper bound (" << ub[uDims[0]]
                           << ")";
      }
      break;
    }
    case TransformType::PassThrough: {
      for (auto [uDim, lDim] : llvm::zip(uDims, lDims)) {
        if (ub[uDim] != lb[lDim]) {
          return emitError() << "PassThrough: upper bound " << ub[uDim]
                             << " does not match lower bound " << lb[lDim];
        }
      }
      break;
    }
    case TransformType::Pad: {
      for (unsigned i = 0, e = uDims.size(); i < e; ++i) {
        int64_t leftPad = params[i * 2];
        int64_t rightPad = params[i * 2 + 1];
        int64_t expected = lb[lDims[i]] + leftPad + rightPad;
        if (ub[uDims[i]] != expected) {
          return emitError() << "Pad: upper bound " << ub[uDims[i]]
                             << " does not match lower bound " << lb[lDims[i]]
                             << " + leftPad(" << leftPad << ") + rightPad("
                             << rightPad << ") = " << expected;
        }
      }
      break;
    }
    case TransformType::AddDim: {
      if (params[0] != ub[uDims[0]]) {
        return emitError() << "AddDim: parameter " << params[0]
                           << " does not match upper bound " << ub[uDims[0]];
      }
      break;
    }
    case TransformType::Embed:
      break;
    case TransformType::Slice: {
      for (unsigned i = 0, e = uDims.size(); i < e; ++i) {
        int64_t begin = params[i * 2];
        int64_t end = params[i * 2 + 1];
        if (end > lb[lDims[i]])
          return emitError()
                 << "Slice: end (" << end << ") exceeds lower bound ("
                 << lb[lDims[i]] << ")";
        if (ub[uDims[i]] != end - begin) {
          return emitError() << "Slice: upper bound " << ub[uDims[i]]
                             << " does not match end(" << end << ") - begin("
                             << begin << ") = " << (end - begin);
        }
      }
      break;
    }
    case TransformType::Broadcast: {
      for (unsigned i = 0, e = lDims.size(); i < e; ++i) {
        if (params[i] != lb[lDims[i]]) {
          return emitError() << "Broadcast: parameter " << params[i]
                             << " does not match lower bound " << lb[lDims[i]];
        }
      }
      break;
    }
    case TransformType::ConstDim: {
      for (unsigned i = 0, e = lDims.size(); i < e; ++i) {
        int64_t size = params[i * 2 + 1];
        if (size != lb[lDims[i]]) {
          return emitError() << "ConstDim: size parameter " << size
                             << " does not match lower bound " << lb[lDims[i]];
        }
      }
      break;
    }
    }
  }

  // Check that each upper and lower dimension is referenced by exactly one
  // transform (no duplicates, no gaps). AddDim has no lower dims and ConstDim
  // has no upper dims, so those are excluded from their respective sets.
  llvm::SmallDenseSet<uint32_t> seenUpper, seenLower;
  for (TransformAttr t : ops) {
    for (uint32_t d : t.getUpperDims()) {
      if (!seenUpper.insert(d).second)
        return emitError() << "Upper dimension " << d
                           << " is referenced by multiple transforms";
    }
    for (uint32_t d : t.getLowerDims()) {
      if (!seenLower.insert(d).second)
        return emitError() << "Lower dimension " << d
                           << " is referenced by multiple transforms";
    }
  }
  for (uint32_t i = 0, e = ub.size(); i < e; ++i) {
    if (!seenUpper.contains(i))
      return emitError() << "Upper dimension " << i
                         << " is not covered by any transform";
  }
  for (uint32_t i = 0, e = lb.size(); i < e; ++i) {
    if (!seenLower.contains(i))
      return emitError() << "Lower dimension " << i
                         << " is not covered by any transform";
  }

  MLIRContext *ctx = map.getContext();
  Builder b(ctx);
  AffineMapAttr expectedMap = assembleMapFor(b, ops, ub, lb);
  if (expectedMap != map) {
    return emitError() << "Affine map " << map
                       << " does not match map reconstructed from transforms: "
                       << expectedMap;
  }

  return success();
}

} // namespace rock
} // namespace mlir
//===----------------------------------------------------------------------===//
// RockDialect
//===----------------------------------------------------------------------===//

void RockDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Rock/IR/RockAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Rock/IR/RockOps.cpp.inc"
      >();
  addInterfaces<RockOpAsmDialectInterface>();
}

//===----------------------------------------------------------------------===//
// Convolution operations
//===----------------------------------------------------------------------===//
ConvolutionDims ConvolutionDims::fromOp(Operation *op, bool enableOutput) {
  auto filterLayoutAttr = op->getAttrOfType<ArrayAttr>("filter_layout");
  auto inputLayoutAttr = op->getAttrOfType<ArrayAttr>("input_layout");
  ArrayAttr outputLayoutAttr;
  if (enableOutput)
    outputLayoutAttr = op->template getAttrOfType<ArrayAttr>("output_layout");

  // Use the RockConvInterface to get tensor shapes with consistent semantics.
  // Fall back to operand indices for ops that don't implement the interface
  // (e.g. ConvElementwiseGemmOp).
  ShapedType filterType, inputType;
  ShapedType outputType;
  if (auto convIF = dyn_cast<RockConvInterface>(op)) {
    filterType = convIF.getConvFilter().getType();
    inputType = convIF.getConvInput().getType();
    if (enableOutput)
      outputType = convIF.getConvOutput().getType();
  } else {
    filterType = cast<ShapedType>(op->getOperand(0).getType());
    inputType = cast<ShapedType>(op->getOperand(1).getType());
    if (enableOutput)
      outputType = cast<ShapedType>(op->getOperand(2).getType());
  }
  ArrayRef<int64_t> filterShape = filterType.getShape();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  ArrayRef<int64_t> outputShape =
      enableOutput ? outputType.getShape() : ArrayRef<int64_t>{};

  int64_t y, x, z, ho, wo, dout, hi, wi, di, k, c, n, g;
  y = x = z = ho = wo = dout = hi = wi = di = k = c = n = g = 0;

  for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
    auto filterAttr = cast<StringAttr>(filterLayoutAttr.getValue()[i]);
    auto inputAttr = cast<StringAttr>(inputLayoutAttr.getValue()[i]);
    StringAttr outputAttr;
    if (enableOutput)
      outputAttr = cast<StringAttr>(outputLayoutAttr.getValue()[i]);

    if (filterAttr.getValue() == "0" || filterAttr.getValue() == "y") {
      y = filterShape[i];
    } else if (filterAttr.getValue() == "x" || filterAttr.getValue() == "1") {
      x = filterShape[i];
    } else if (filterAttr.getValue() == "2") {
      z = filterShape[i];
    } else if (filterAttr.getValue() == "k") {
      k = filterShape[i];
    } else if (filterAttr.getValue() == "c") {
      c = filterShape[i];
    } else if (filterAttr.getValue() == "g") {
      g = filterShape[i];
    }

    if (inputAttr.getValue() == "hi" || inputAttr.getValue() == "0i") {
      hi = inputShape[i];
    } else if (inputAttr.getValue() == "wi" || inputAttr.getValue() == "1i") {
      wi = inputShape[i];
    } else if (inputAttr.getValue() == "2i") {
      di = inputShape[i];
    } else if (inputAttr.getValue() == "ni") {
      n = inputShape[i];
    }

    if (enableOutput) {
      if (outputAttr.getValue() == "ho" || outputAttr.getValue() == "0o") {
        ho = outputShape[i];
      } else if (outputAttr.getValue() == "wo" ||
                 outputAttr.getValue() == "1o") {
        wo = outputShape[i];
      } else if (outputAttr.getValue() == "2o") {
        dout = outputShape[i];
      }
    }
  }

  SmallVector<int64_t> fil({y, x});
  if (z > 0)
    fil.push_back(z);
  SmallVector<int64_t> out({ho, wo});
  if (dout > 0)
    out.push_back(dout);
  SmallVector<int64_t> in({hi, wi});
  if (di > 0)
    in.push_back(di);
  return ConvolutionDims(fil, out, in, k, c, n, g);
}

ConvOpType mlir::rock::convOpTypeFromKernelType(KernelType kernelType) {
  switch (kernelType) {
  case KernelType::Conv:
  case KernelType::ConvElementwiseGemm:
    return ConvOpType::Fwd;
  case KernelType::ConvBwdData:
    return ConvOpType::BwdData;
  case KernelType::ConvBwdWeight:
    return ConvOpType::BwdWeight;
  case KernelType::Gemm:
    llvm_unreachable(
        "GEMM ops shouldn't be in convolution-specific lowering passes");
  case KernelType::Attention:
    llvm_unreachable(
        "Attention ops shouldn't be in convolution-specific lowering passes");
  case KernelType::GemmElementwiseGemm:
    llvm_unreachable(
        "gemm+gemm ops shouldn't be in convolution-specific lowering passes");
  }
  llvm_unreachable("Unsuppported KernelType");
}

KernelType mlir::rock::kernelTypeFromConvOpType(ConvOpType convOpType) {
  switch (convOpType) {
  case ConvOpType::Fwd:
    return KernelType::Conv;
  case ConvOpType::BwdData:
    return KernelType::ConvBwdData;
  case ConvOpType::BwdWeight:
    return KernelType::ConvBwdWeight;
  }
  llvm_unreachable("Unsupported ConvOpType");
}

GemmSize GemmSize::fromConvolution(ConvOpType type,
                                   const ConvolutionDims &sizes) {
  assert(type != ConvOpType::BwdData &&
         "Backward data convolutions require stride/dilation-dependent "
         "multi-kernel expansion. Use op.getGemmSize() instead");
  int64_t gemmGSize, gemmMSize, gemmKSize, gemmNSize;
  switch (type) {
  case ConvOpType::Fwd:
    gemmGSize = sizes.g;
    gemmMSize = sizes.k;
    // +++pf: should these accumulate sizes across all dimensions?
    gemmKSize = sizes.c * sizes.fil[0] * sizes.fil[1];
    gemmNSize = sizes.n * sizes.out[0] * sizes.out[1];
    break;
  case ConvOpType::BwdWeight:
    gemmGSize = sizes.g;
    gemmMSize = sizes.k;
    gemmKSize = sizes.n * sizes.out[0] * sizes.out[1];
    gemmNSize = sizes.c * sizes.fil[0] * sizes.fil[1];
    break;
  case ConvOpType::BwdData:
    llvm_unreachable("Should've been caught be an assert");
  }
  return GemmSize(gemmGSize, gemmMSize, gemmKSize, gemmNSize);
}

KernelType ConvOp::getKernelType() { return KernelType::Conv; }

KernelType ConvBwdDataOp::getKernelType() { return KernelType::ConvBwdData; }

KernelType ConvBwdWeightOp::getKernelType() {
  return KernelType::ConvBwdWeight;
}

Type ConvOp::getAType() { return getFilter().getType().getElementType(); }

Type ConvBwdDataOp::getAType() {
  return getFilter().getType().getElementType();
}

Type ConvBwdWeightOp::getAType() {
  return getGradient().getType().getElementType();
}

Type ConvOp::getBType() { return getInput().getType().getElementType(); }

Type ConvBwdDataOp::getBType() {
  return getGradient().getType().getElementType();
}

Type ConvBwdWeightOp::getBType() {
  return getInput().getType().getElementType();
}

Type ConvOp::getCType() { return getResult().getType().getElementType(); }

Type ConvBwdDataOp::getCType() {
  return getResult().getType().getElementType();
}

Type ConvBwdWeightOp::getCType() {
  return getResult().getType().getElementType();
}

// RockConvInterface implementations
TypedValue<ShapedType> ConvOp::getConvFilter() { return getFilter(); }
TypedValue<ShapedType> ConvOp::getConvInput() { return getInput(); }
TypedValue<ShapedType> ConvOp::getConvOutput() {
  return cast<TypedValue<ShapedType>>(getResult());
}

TypedValue<ShapedType> ConvBwdDataOp::getConvFilter() { return getFilter(); }
TypedValue<ShapedType> ConvBwdDataOp::getConvInput() {
  return cast<TypedValue<ShapedType>>(getResult());
}
TypedValue<ShapedType> ConvBwdDataOp::getConvOutput() { return getGradient(); }

TypedValue<ShapedType> ConvBwdWeightOp::getConvFilter() {
  return cast<TypedValue<ShapedType>>(getResult());
}
TypedValue<ShapedType> ConvBwdWeightOp::getConvInput() { return getInput(); }
TypedValue<ShapedType> ConvBwdWeightOp::getConvOutput() {
  return getGradient();
}

GemmSize ConvOp::getGemmSize() {
  auto sizes = ConvolutionDims::fromOp(*this);
  return GemmSize::fromConvolution(ConvOpType::Fwd, sizes);
}

/// Compute the GemmSize for a single backward-data kernel ID.
static GemmSize bwdDataGemmSizeForKernelId(const ConvolutionDims &sizes,
                                           ArrayRef<int64_t> padding,
                                           ArrayRef<int64_t> strides,
                                           ArrayRef<int64_t> dilations,
                                           int64_t kernelId) {
  SmallVector<int64_t, 5> gcdStrideDilations;
  assert(strides.size() == dilations.size());
  for (const auto &[stride, dilation] : zip(strides, dilations))
    gcdStrideDilations.push_back(std::gcd(stride, dilation));

  SmallVector<int64_t, 5> filTilda;
  for (const auto &[stride, gcdSD] : zip(strides, gcdStrideDilations))
    filTilda.push_back(stride / gcdSD);

  SmallVector<int64_t, 5> outTilda;
  for (const auto &[out, dilation, fil, stride] :
       zip(sizes.out, dilations, sizes.fil, strides))
    outTilda.push_back(out + llvm::divideCeil(dilation * (fil - 1), stride));

  SmallVector<int64_t, 5> iTildaLeft;
  SmallVector<int64_t, 5> iTildaRight;
  for (const auto &[padindex, dilation, tilda, stride] :
       enumerate(dilations, filTilda, strides))
    iTildaLeft.push_back(
        std::max((int64_t)0, padding[2 * padindex] - dilation * (tilda - 1)) /
        stride);
  for (const auto &[padindex, out, in, stride] :
       enumerate(outTilda, sizes.in, strides))
    iTildaRight.push_back(
        std::min(out, static_cast<int64_t>(llvm::divideCeil(
                          padding[2 * padindex] + in - 1, stride)) +
                          1));

  SmallVector<int64_t, 5> tildaSlice;
  for (const auto &[right, left] : zip(iTildaRight, iTildaLeft))
    tildaSlice.push_back(right - left);

  // Decompose kernelId into per-spatial-dimension iTilda indices.
  int64_t product = 1;
  for (size_t i = 1; i < sizes.fil.size(); i++)
    product *= filTilda[i];
  int64_t divisor = (sizes.fil.size() == 3) ? filTilda[2] : 1;

  SmallVector<int64_t, 3> iTilda(sizes.fil.size());
  switch (sizes.fil.size()) {
  default:
    llvm_unreachable("Only 2-D and 3-D have been implemented.");
    break;
  case 3:
    iTilda[2] = kernelId % divisor;
    [[fallthrough]];
  case 2:
    iTilda[1] = (kernelId % product) / divisor;
    iTilda[0] = kernelId / product;
  }

  int64_t g = sizes.g;
  int64_t m = sizes.c;
  int64_t k = sizes.k;
  for (size_t i = 0; i < sizes.fil.size(); i++)
    k *= llvm::divideCeil(sizes.fil[i] - iTilda[i], filTilda[i]);
  int64_t n = sizes.n;
  for (auto ts : tildaSlice)
    n *= ts;

  return GemmSize(g, m, k, n);
}

GemmSize ConvBwdDataOp::getGemmSize() {
  // A single ConvBwdDataOp is expanded into multiple GEMMs by ConvToGemm,
  // one per valid kernel ID.  The g, m, and n dimensions are identical across
  // all kernel IDs while k varies. We report the GemmSize with the largest k
  // so that tuning, padding checks, and problem-string generation operate on
  // the most demanding individual GEMM.
  auto sizes = ConvolutionDims::fromOp(*this);
  auto padding = extractFromIntegerArrayAttr<int64_t>(this->getPadding());
  auto strides = extractFromIntegerArrayAttr<int64_t>(this->getStrides());
  auto dilations = extractFromIntegerArrayAttr<int64_t>(this->getDilations());

  auto kernelIds = rock::backwardDataKernelIds(strides, dilations, sizes.fil);

  assert(!kernelIds.empty());
  GemmSize biggest =
      bwdDataGemmSizeForKernelId(sizes, padding, strides, dilations,
                                 kernelIds.front());
  for (int64_t kernelId : llvm::drop_begin(kernelIds)) {
    GemmSize single = bwdDataGemmSizeForKernelId(sizes, padding, strides,
                                                 dilations, kernelId);
    assert(single.g == biggest.g && single.m == biggest.m &&
           single.n == biggest.n &&
           "g, m, and n must be identical across all kernel IDs");
    if (single.k > biggest.k)
      biggest = single;
  }
  return biggest;
}

GemmSize ConvBwdWeightOp::getGemmSize() {
  auto sizes = ConvolutionDims::fromOp(*this);
  return GemmSize::fromConvolution(ConvOpType::BwdWeight, sizes);
}

//===-----------------------------------------------------===//
// Conv Op Verification
//===-----------------------------------------------------===//

static LogicalResult verifyConvLikeOp(RockConvInterface op) {
  auto filterType = cast<ShapedType>(op.getConvFilter().getType());
  auto inputType = cast<ShapedType>(op.getConvInput().getType());
  auto outputType = cast<ShapedType>(op.getConvOutput().getType());

  if (filterType.getRank() != inputType.getRank() ||
      inputType.getRank() != outputType.getRank()) {
    return op.emitOpError("filter, input, and output must have the same rank")
           << " (filter rank = " << filterType.getRank()
           << ", input rank = " << inputType.getRank()
           << ", output rank = " << outputType.getRank() << ")";
  }
  int64_t rank = filterType.getRank();

  Type filterElemType = filterType.getElementType();
  Type inputElemType = inputType.getElementType();
  Type outputElemType = outputType.getElementType();

  // we can't enforce the same type (example: f8 * bf8)
  if (isa<FloatType>(filterElemType) != isa<FloatType>(inputElemType))
    return op.emitOpError(
        "filter and input must both be float or both be integer types");
  if (isa<FloatType>(filterElemType) && !isa<FloatType>(outputElemType))
    return op.emitOpError(
        "float-valued inputs must have a floating-point output type");
  if (isa<IntegerType>(filterElemType) && !isa<IntegerType>(outputElemType))
    return op.emitOpError(
        "integer-valued inputs must have an integer output type");

  auto padding = extractFromIntegerArrayAttr<int64_t>(op.getPadding());
  auto strides = extractFromIntegerArrayAttr<int64_t>(op.getStrides());
  auto dilations = extractFromIntegerArrayAttr<int64_t>(op.getDilations());

  if (strides.size() != dilations.size())
    return op.emitOpError("strides and dilations must have the same size");
  if (padding.size() != 2 * strides.size())
    return op.emitOpError(
        "padding must have twice as many elements as strides");

  int64_t numSpatialDims = rank - 3;
  if (static_cast<int64_t>(strides.size()) != numSpatialDims)
    return op.emitOpError(
               "number of strides must match number of spatial dimensions")
           << " (strides = " << strides.size()
           << ", spatial dims = " << numSpatialDims << ")";

  // TODO: verify layouts
  // TODO: verify output shape (with ConvGenerator::outputDim)
  return success();
}

LogicalResult ConvOp::verify() { return verifyConvLikeOp(*this); }

LogicalResult ConvBwdDataOp::verify() { return verifyConvLikeOp(*this); }

LogicalResult ConvBwdWeightOp::verify() {
  if (failed(verifyConvLikeOp(*this)))
    return failure();

  // kBlocks is optional pre-lowering (affix-tuning-params sets it for the
  // atomic backward-weight path). When present, the lowering in ConvToGemm
  // splits the batch dimension N into (kBlocks, N / kBlocks), so kBlocks must
  // satisfy the same structural invariant that `calculateKBlockNum` enforces.
  IntegerAttr kBlocksAttr = getKBlocksAttr();
  if (!kBlocksAttr)
    return success();
  int64_t kBlocks = kBlocksAttr.getInt();

  // Recover N from the input tensor's layout. The layout attributes are not
  // formally part of the op definition, so skip the divisibility check when
  // they're absent or malformed rather than asserting.
  auto inputLayoutAttr = (*this)->getAttrOfType<ArrayAttr>("input_layout");
  ArrayRef<int64_t> inputShape = getInput().getType().getShape();
  if (!inputLayoutAttr ||
      inputLayoutAttr.size() != static_cast<size_t>(inputShape.size()))
    return success();

  int64_t batchSize = -1;
  for (auto [layoutAttr, dimSize] : llvm::zip(inputLayoutAttr, inputShape)) {
    auto nameAttr = dyn_cast<StringAttr>(layoutAttr);
    if (nameAttr && nameAttr.getValue() == "ni") {
      batchSize = dimSize;
      break;
    }
  }
  if (batchSize <= 0)
    return success();

  if (!isValidKBlocks(kBlocks, batchSize))
    return emitOpError("kBlocks (")
           << kBlocks << ") must be positive and evenly divide batch size N ("
           << batchSize << ")";

  return success();
}

//===-----------------------------------------------------===//
// StoreOp
//===-----------------------------------------------------===//

template <typename StoreOpT>
static LogicalResult verifyStoreResultUses(StoreOpT op, Value result) {
  unsigned returnUseCount = 0;
  for (OpOperand &use : result.getUses()) {
    Operation *user = use.getOwner();
    if (isa<ViewLikeOpInterface>(user))
      continue;
    // Store chaining is valid when the previous store result is threaded as
    // the destination tensor for the next same-kind store. The dest operand is
    // operand 1.
    if (isa<StoreOpT>(user) && use.getOperandNumber() == 1)
      continue;
    if (isa<func::ReturnOp>(user)) {
      if (++returnUseCount > 1)
        return op.emitOpError("result may be returned at most once");
      continue;
    }
    return op.emitOpError(
        "result must be used directly by a func.return, view-like op, or "
        "destination operand of another same-kind store");
  }
  return success();
}

LogicalResult StoreOp::verify() {
  auto sourceType = cast<ShapedType>(getSource().getType());
  auto destType = cast<ShapedType>(getDest().getType());

  if (sourceType.getShape() != destType.getShape())
    return emitOpError("source and dest shapes must match")
           << " (source: " << sourceType << ", dest: " << destType << ")";

  return verifyStoreResultUses(*this, getResult());
}

//===-----------------------------------------------------===//
// CastToPtrOp
//===-----------------------------------------------------===//

LogicalResult CastToPtrOp::verify() {
  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (!isa<triton::PointerType>(resultType.getElementType()))
    return emitOpError("result must be a tensor of !tt.ptr");
  return success();
}

//===-----------------------------------------------------===//
// ExtractPtrOp
//===-----------------------------------------------------===//

LogicalResult ExtractPtrOp::verify() {
  if (!isa<BlockArgument>(getSource()))
    return emitOpError("source must be a block argument");
  return success();
}

//===-----------------------------------------------------===//
// TransformOp
//===-----------------------------------------------------===//

LogicalResult TransformOp::verify() {
  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getOutput().getType());

  TransformMapAttr tmap = getTransform();

  ArrayRef<int64_t> lowerBounds = tmap.getLowerBounds();
  ArrayRef<int64_t> upperBounds = tmap.getUpperBounds();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  if (inputShape.size() != lowerBounds.size())
    return emitOpError("input rank must match transform lower bounds rank");
  for (size_t i = 0; i < inputShape.size(); ++i) {
    if (inputShape[i] != lowerBounds[i])
      return emitOpError()
             << "input shape must match transform lower bounds (input shape="
             << inputShape << ", lower bounds=" << lowerBounds << ")";
  }

  ArrayRef<int64_t> outputShape = outputType.getShape();
  if (outputShape.size() != upperBounds.size())
    return emitOpError() << "output rank must match transform upper bounds "
                            "rank (output shape="
                         << outputShape << ", upper bounds=" << upperBounds
                         << ")";
  for (size_t i = 0; i < outputShape.size(); ++i) {
    if (outputShape[i] != upperBounds[i])
      return emitOpError("output shape must match transform upper bounds");
  }

  return success();
}

LogicalResult TransformOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  TransformOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = dyn_cast<ShapedType>(adaptor.getInput().getType());
  if (!inputType)
    return emitOptionalError(location, "input must be a shaped type");
  TransformMapAttr transform = adaptor.getTransform();
  if (!transform)
    return emitOptionalError(location, "transform attribute is required");
  inferredReturnTypes.push_back(
      inputType.clone(transform.getUpperBounds().asArrayRef()));
  return success();
}

//===-----------------------------------------------------===//
// UntileOp
//===-----------------------------------------------------===//

LogicalResult UntileOp::verify() {
  int64_t sourceRank = getSource().getType().getRank();
  int64_t resultRank = getResult().getType().getRank();
  if (sourceRank > resultRank) {
    return emitOpError("source rank is greater than result rank")
           << " (" << sourceRank << " > " << resultRank << ")";
  }
  return success();
}

//===-----------------------------------------------------===//
// LoadMarkerOp / StoreMarkerOp
//===-----------------------------------------------------===//

static LogicalResult verifyMarkerOp(Operation *op, ArrayAttr views,
                                    ArrayRef<int64_t> inputTensorShape,
                                    ArrayRef<int64_t> resultTensorShape,
                                    size_t extraIndicesCount) {
  // The upper bounds rank must match result.rank + extraIndices.size().
  // Whatever happens after the upper view doesn't have to be consistent in
  // terms of rank or overall shapes, because we can merge/unmerge, broadcast
  // slice, etc.
  ArrayRef<int64_t> upperBounds =
      views.empty()
          ? inputTensorShape
          : cast<TransformMapAttr>(views[0]).getUpperBounds().asArrayRef();
  size_t upperRank = upperBounds.size();
  size_t resultTensorRank = resultTensorShape.size();
  size_t expectedRank = resultTensorRank + extraIndicesCount;
  if (upperRank != expectedRank)
    return op->emitOpError("upper bounds must equal tensor rank + "
                           "extraIndices count")
           << " (" << upperRank << " != " << resultTensorRank << " + "
           << extraIndicesCount << ")";

  // We can check the shape of the result tensor (after removing extra indices)
  // is the same as the upper view bounds. Example, rock.load_marker ... bounds
  // = [2, 1, 2, 2, 64, 64] -> [1, 128, 128]>][%arg5, %12, %19, %21] :
  // tensor<1x128x128xf16> -> tensor<64x64xf16> So, [X, X, X, X, 64, 64] and
  // tensor<64x64xf16> must match.
  if (upperBounds.take_back(resultTensorRank) != resultTensorShape) {
    return op->emitOpError(
               "Upper bounds last dimensions must match with result shape ")
           << " (" << upperBounds.take_back(resultTensorRank)
           << " != " << resultTensorShape << ")";
  }

  ArrayRef<int64_t> lowerBounds =
      views.empty() ? inputTensorShape
                    : cast<TransformMapAttr>(views[views.size() - 1])
                          .getLowerBounds()
                          .asArrayRef();
  if (lowerBounds != inputTensorShape) {
    return op->emitOpError("Lower bounds must match with input shape ")
           << " (" << lowerBounds << " != " << inputTensorShape << ")";
  }

  return success();
}

LogicalResult LoadMarkerOp::verify() {
  return verifyMarkerOp(
      *this, getExtraViews(),
      cast<RankedTensorType>(getSource().getType()).getShape(),
      cast<RankedTensorType>(getResult().getType()).getShape(),
      getExtraIndices().size());
}

LogicalResult StoreMarkerOp::verify() {
  // Note that input <-> result are swapped wrt LoadMarkerOp
  return verifyMarkerOp(
      *this, getExtraViews(),
      cast<RankedTensorType>(getResult().getType()).getShape(),
      cast<RankedTensorType>(getSource().getType()).getShape(),
      getExtraIndices().size());
}

//===-----------------------------------------------------===//
// GemmOp
//===-----------------------------------------------------===//

LogicalResult GemmOp::verify() {
  ShapedType typeA = getA().getType(), typeB = getB().getType(),
             typeResult = getResult().getType();

  Type inElems = typeA.getElementType(), outElems = typeResult.getElementType();
  // The integer gemm will produce i32 and then truncate/extend to the requested
  // iN e.g. i8.
  if (isa<FloatType>(inElems) && !isa<FloatType>(outElems))
    return emitOpError(
        "float-valued inputs must have a floating-point output type");

  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape(),
                    dimsResult = typeResult.getShape();
  auto rankCheck = [&](ArrayRef<int64_t> dims,
                       StringRef name) -> LogicalResult {
    if (dims.size() != 2 && dims.size() != 3) {
      return emitOpError()
             << name
             << " must be a rank 2 or rank 3 tensor representing [G,] D, K";
    }
    return success();
  };
  if (failed(rankCheck(dimsA, "Matrix A")) ||
      failed(rankCheck(dimsB, "Matrix B")) ||
      failed(rankCheck(dimsResult, "Result"))) {
    return failure();
  }
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1,
          offsetResult = dimsResult.size() == 2 ? 0 : 1;
  int64_t gA = offsetA ? dimsA[0] : 1, gB = offsetB ? dimsB[0] : 1,
          gResult = offsetResult ? dimsResult[0] : 1;
  int64_t mA = dimsA[offsetA + (getATransposed() ? 1 : 0)],
          kA = dimsA[offsetA + (getATransposed() ? 0 : 1)],
          kB = dimsB[offsetB + (getBTransposed() ? 1 : 0)],
          nB = dimsB[offsetB + (getBTransposed() ? 0 : 1)],
          mResult = dimsResult[offsetResult + (getOTransposed() ? 1 : 0)],
          nResult = dimsResult[offsetResult + (getOTransposed() ? 0 : 1)];
  if (gA != gB || gA != gResult)
    return emitOpError("group dimensions don't match")
           << " g_a = " << gA << " g_b = " << gB << " g_result = " << gResult;
  if (mA != mResult)
    return emitOpError("M dimensions don't match")
           << " m_a = " << mA << " m_result = " << mResult;
  if (nB != nResult)
    return emitOpError("N dimensions don't match")
           << " n_b = " << nB << " n_result = " << nResult;
  if (kA != kB)
    return emitOpError("K dimensions don't match")
           << " k_a = " << kA << " k_b = " << kB;
  bool hasScaleA = getScaleA() != nullptr;
  bool hasScaleB = getScaleB() != nullptr;
  if (hasScaleA ^ hasScaleB) {
    return emitOpError("both scaleA and scaleB must be provided or neither");
  }
  // Unified verification for scaleA / scaleB.
  auto verifyScale = [&](Value scale, bool isA) -> LogicalResult {
    if (!scale)
      return success();
    ShapedType ty = cast<ShapedType>(scale.getType());
    ArrayRef<int64_t> dims = ty.getShape();
    StringRef scaleName = isA ? "scaleA" : "scaleB";
    if (!getQuantBlockSizeAttr())
      return emitOpError("quantBlockSize not defined");

    if (failed(rankCheck(dims, scaleName)))
      return failure();

    int64_t quantBlockSize = getQuantBlockSize().value();

    bool transposed = isA ? getAScaleTransposed() : getBScaleTransposed();
    int64_t offset = dims.size() == 2 ? 0 : 1;
    int64_t g = offset ? dims[0] : 1;
    int64_t first = dims[offset + (transposed ? 1 : 0)];
    int64_t second = dims[offset + (transposed ? 0 : 1)];

    int64_t expectedG = isA ? gA : gB;
    int64_t expectedFirst = isA ? mA : nB; // scaleA: M; scaleB: N
    int64_t expectedSecond =
        llvm::divideCeil(isA ? kA : kB, quantBlockSize); // scaleA: K; scaleB: K

    StringRef dDim = isA ? "M" : "N";
    StringRef dDimLower = isA ? "m" : "n";

    if (second != expectedSecond)
      return emitOpError() << scaleName << "'s "
                           << "K dimension must match matrix "
                           << (isA ? "A" : "B") << "'s "
                           << "K dimension"
                           << " " << scaleName << "_"
                           << "k = " << second << " " << (isA ? "k_a" : "k_b")
                           << " = " << expectedSecond;
    if (first != expectedFirst)
      return emitOpError() << scaleName << "'s " << dDim
                           << " dimension must match matrix "
                           << (isA ? "A" : "B") << "'s " << dDim << " dimension"
                           << " " << scaleName << "_" << dDimLower << " = "
                           << first << " " << (isA ? "m_a" : "n_b") << " = "
                           << expectedFirst;
    if (g != expectedG)
      return emitOpError() << scaleName << "'s G dimension must match matrix "
                           << (isA ? "A" : "B") << "'s G dimension"
                           << " " << scaleName << "_g = " << g << " "
                           << (isA ? "g_a" : "g_b") << " = " << expectedG;
    return success();
  };

  if (failed(verifyScale(getScaleA(), /*isA=*/true)) ||
      failed(verifyScale(getScaleB(), /*isA=*/false)))
    return failure();

  if (getParams().has_value() && getQuantBlockSize().has_value()) {
    int64_t quantBlockSize = getQuantBlockSize().value();
    auto params = cast<GemmParamsAttr>(getParams().value());
    if (params.getKPerBlock() % quantBlockSize != 0) {
      return emitOpError() << "kPerBlock must be divisible by quantBlockSize";
    }
  }

  return success();
}

KernelType GemmOp::getKernelType() { return KernelType::Gemm; }

Type GemmOp::getAType() { return getA().getType().getElementType(); }

Type GemmOp::getBType() { return getB().getType().getElementType(); }

Type GemmOp::getCType() { return getResult().getType().getElementType(); }

GemmSize GemmOp::getGemmSize() {
  ShapedType typeA = getA().getType(), typeB = getB().getType();
  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape();
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1;
  int64_t g = offsetA ? dimsA[0] : 1,
          m = dimsA[offsetA + (getATransposed() ? 1 : 0)],
          k = dimsA[offsetA + (getATransposed() ? 0 : 1)],
          n = dimsB[offsetB + (getBTransposed() ? 0 : 1)];
  return GemmSize(g, m, k, n);
}

//===-----------------------------------------------------===//
//  GridwiseGemm Op
//===-----------------------------------------------------===//
template <typename GridOp>
static LogicalResult verifyGridwiseGemm(GridOp op) {
  RankedTensorType aType = op.getA().getType(), bType = op.getB().getType(),
                   resultType = op.getResult().getType();

  ArrayRef<int64_t> aShape = aType.getShape(), bShape = bType.getShape(),
                    resultShape = resultType.getShape();
  int64_t g = aShape[0], k = aShape[2], m = aShape[1], n = bShape[2];
  if (bShape[0] != g || resultShape[0] != g) {
    return op.emitOpError("Mismatched G dimensions in matrix multiply;")
           << " A[0] = " << g << " b[0] = " << bShape[0]
           << " result[0] = " << resultShape[0];
  }
  if (resultShape[1] != m)
    return op.emitOpError("Mismatched M dimensions in matrix multiply:")
           << " A[2] = " << m << " result[1] = " << resultShape[1];
  if (bShape[1] != k)
    return op.emitOpError("Mismatched K dimensions in matrix multiply:")
           << " A[1] = " << k << " B[1] = " << bShape[1];
  if (resultShape[2] != n)
    return op.emitOpError("Mismatched N dimensions in matrix multiply:")
           << " B[2] = " << n << " result[2] = " << resultShape[2];

  constexpr int64_t intMax = std::numeric_limits<int32_t>::max();
  if (g > intMax)
    return op.emitOpError("G dimmension ")
           << g << " cannot be greater than int32_max " << intMax;
  if (m > intMax)
    return op.emitOpError("M dimmension ")
           << m << " cannot be greater than int32_max " << intMax;
  if (k > intMax)
    return op.emitOpError("K dimmension ")
           << k << " cannot be greater than int32_max " << intMax;
  if (n > intMax)
    return op.emitOpError("N dimmension ")
           << n << " cannot be greater than int32_max " << intMax;

  return success();
}

LogicalResult GridwiseGemmOp::verify() {
  Value scaleA = getScaleA();
  Value scaleB = getScaleB();
  bool hasScaleA = scaleA != nullptr;
  bool hasScaleB = scaleB != nullptr;
  if (hasScaleA ^ hasScaleB) {
    return emitOpError("both scaleA and scaleB must be provided or neither");
  }
  if (failed(verifyScales(*this, getA(), scaleA, getQuantBlockSize(),
                          /*isA=*/true)) ||
      failed(verifyScales(*this, getB(), scaleB, getQuantBlockSize(),
                          /*isA=*/false))) {
    return failure();
  }
  return verifyGridwiseGemm(*this);
}

//===-----------------------------------------------------===//
// BlockwiseLoadOp
//===-----------------------------------------------------===//

LogicalResult BlockwiseLoadOp::verify() {
  auto sourceType = cast<RankedTensorType>(getSource().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());

  size_t idxCount = getSourceIndices().size();
  if (idxCount + resultType.getRank() !=
      static_cast<size_t>(sourceType.getRank()))
    return emitOpError(
               "sourceIndices.size() + result rank must equal source rank")
           << " (" << idxCount << " + " << resultType.getRank()
           << " != " << sourceType.getRank() << ")";

  if (sourceType.getShape().take_back(resultType.getRank()) !=
      resultType.getShape()) {
    return emitOpError("Input last dimensions must match with result shape ")
           << " (" << sourceType.getShape().take_back(resultType.getRank())
           << " != " << resultType.getShape() << ")";
  }

  return success();
}

LogicalResult BlockwiseLoadOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  BlockwiseLoadOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto sourceType = dyn_cast<RankedTensorType>(adaptor.getSource().getType());
  if (!sourceType)
    return emitOptionalError(location, "source must be a ranked tensor");
  size_t numSourceIndices = adaptor.getSourceIndices().size();
  if (numSourceIndices > static_cast<size_t>(sourceType.getRank()))
    return emitOptionalError(location,
                             "number of source indices exceeds source rank");
  auto shape =
      sourceType.getShape().take_back(sourceType.getRank() - numSourceIndices);
  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, sourceType.getElementType()));
  return success();
}

//===-----------------------------------------------------===//
// BlockwiseStoreOp
//===-----------------------------------------------------===//

SmallPtrSet<OpOperand *, 2> BlockwiseStoreOp::getAcceptingViewOperands() {
  auto operands = getOperation()->getOpOperands();
  return {operands.begin() + 1};
}

std::optional<OperandRange>
BlockwiseStoreOp::getExtraIndices(OpOperand &operand) {
  if (!getAcceptingViewOperands().contains(&operand)) {
    return std::nullopt;
  }
  // Only one operand supports view
  return getExtraIndices();
}

Operation *
BlockwiseStoreOp::cloneWithExtraIndices(OpBuilder &builder, OpOperand &operand,
                                        Value view,
                                        ArrayRef<Value> newExtraIndices) {
  if (!getAcceptingViewOperands().contains(&operand)) {
    return getOperation();
  }

  // Only one operand supports view
  auto newOp = BlockwiseStoreOp::create(
      builder, getLoc(), getResult().getType(), getSource(), view,
      newExtraIndices, getStoreMethod());
  return newOp.getOperation();
}

LogicalResult BlockwiseStoreOp::verify() {
  auto sourceType = cast<RankedTensorType>(getSource().getType());
  auto destType = cast<RankedTensorType>(getDest().getType());

  size_t extraIdxCount = getExtraIndices().size();
  if (extraIdxCount + sourceType.getRank() !=
      static_cast<size_t>(destType.getRank()))
    return emitOpError("extraIndices.size() + source rank must equal dest rank")
           << " (" << extraIdxCount << " + " << sourceType.getRank()
           << " != " << destType.getRank() << ")";

  if (destType.getShape().take_back(sourceType.getRank()) !=
      sourceType.getShape()) {
    return emitOpError("Dest last dimensions must match with input shape ")
           << " (" << destType.getShape().take_back(sourceType.getRank())
           << " != " << sourceType.getShape() << ")";
  }

  return verifyStoreResultUses(*this, getResult());
}

//===----------------------------------------------------------------------===//
// BlockwiseGemmOp
//===----------------------------------------------------------------------===//

LogicalResult BlockwiseGemmOp::verify() {
  bool hasScaleA = getMatrixScaleA() != nullptr;
  bool hasScaleB = getMatrixScaleB() != nullptr;
  auto aShape = cast<ShapedType>(getMatrixA().getType()).getShape();
  auto bShape = cast<ShapedType>(getMatrixB().getType()).getShape();

  auto verifyMatrixAndScale = [&](Value scale, ArrayRef<int64_t> matrixShape,
                                  bool isA) -> LogicalResult {
    bool hasScale = scale != nullptr;

    if (matrixShape.size() != 2)
      return emitOpError() << "matrix shape must be 2D";

    StringRef matrixName = isA ? "A" : "B";
    if (hasScale) {
      std::optional<int64_t> quantBlockSize = getQuantBlockSize();
      if (!quantBlockSize.has_value())
        return emitOpError() << "quantBlockSize is not set but we found scale";

      SmallVector<int64_t> scaleShape =
          normalizeScaleShape(cast<ShapedType>(scale.getType()).getShape(),
                              quantBlockSize.value(), isA);

      TypeAttr origElemType =
          isA ? getMatrixAOrigElemTypeAttr() : getMatrixBOrigElemTypeAttr();
      if (origElemType != nullptr) {
        // After sub-byte packing (f4->i8), one matrix dimension was halved.
        // Verify that doubling the packed dimension recovers the scale shape.
        bool kPack = isA ? getMatrixAKPack().value_or(true)
                         : getMatrixBKPack().value_or(true);
        int64_t origLen = origElemType.getValue().getIntOrFloatBitWidth();
        int64_t currLen = isA ? cast<ShapedType>(getMatrixA().getType())
                                    .getElementTypeBitWidth()
                              : cast<ShapedType>(getMatrixB().getType())
                                    .getElementTypeBitWidth();
        if (currLen % origLen != 0)
          return emitOpError(
              "unexpected error: currLen is not divisible by origLen");

        // matrixA is MxK: K is dim 1, M is dim 0.
        // matrixB is KxN: K is dim 0, N is dim 1.
        int64_t packedDim = (isA == kPack) ? 1 : 0;
        SmallVector<int64_t> expectedShape(matrixShape);
        expectedShape[packedDim] *= currLen / origLen;
        if (expectedShape != scaleShape) {
          return emitOpError(llvm::formatv(
              "Packed matrix{0} shape (with dim {1} {2}x) must match "
              "normalized scale{0} shape.",
              matrixName, packedDim, currLen / origLen));
        }
      } else {
        if (matrixShape != ArrayRef<int64_t>(scaleShape)) {
          return emitOpError(
              llvm::formatv("If scale{0} is non-null, its shape must match "
                            "{0}'s shape.",
                            matrixName));
        }
      }
    }

    return success();
  };

  // Verify matrix A and its scales
  if (failed(verifyMatrixAndScale(getMatrixScaleA(), aShape, true)))
    return failure();

  // Verify matrix B and its scales
  if (failed(verifyMatrixAndScale(getMatrixScaleB(), bShape, false)))
    return failure();

  if (hasScaleA ^ hasScaleB)
    return emitOpError(
        "scaleA and scaleB must both be present or both be null.");

  return success();
}

LogicalResult BlockwiseGemmOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  BlockwiseGemmOp::Adaptor adaptor(operands, attributes, properties, regions);
  inferredReturnTypes.push_back(adaptor.getMatrixC().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// GridwiseAttentionOp
//===----------------------------------------------------------------------===//
LogicalResult GridwiseAttentionOp::verify() {
  GemmParamsAttr gemm0TuningParams = getParams0();
  int64_t gemm0kpack = gemm0TuningParams.getKpack();
  int64_t gemm0NPerBlock = gemm0TuningParams.getNPerBlock();
  if (gemm0NPerBlock % gemm0kpack != 0) {
    return emitError("NPerBlock should be divisible by kpack.");
  }

  if (!getEnableSoftmax() && getLse())
    return emitError("LSE only works for attention.");

  if (!getEnableSoftmax() && getSplitKV() != 1)
    return emitError("split-kv is implemented for attention only.");

  if (!getEnableSoftmax() && getSoftmaxType()) {
    return emitError("Setting softmax type only works for attention.");
  }

  if (!getEnableSoftmax() && getCurrentSeqLen())
    return emitError("currentSeqLen only works for attention.");

  if (!getEnableSoftmax() && getPrefixOffset())
    return emitError("prefixOffset only works for attention.");

  if (!getEnableSoftmax() && getCausal())
    return emitError("causal only works for attention.");

  // Validate prefix offset constraints
  // prefixOffset requires causal to be enabled (prefix causal = causal +
  // prefixOffset)
  if (getPrefixOffset() && !getCausal())
    return emitError(
        "prefixOffset requires causal to be enabled. "
        "Prefix causal attention is causal masking with an offset.");

  return success();
}

//===-----------------------------------------------------===//
// TransformsToPtrOp
//===-----------------------------------------------------===//

LogicalResult TransformsToPtrOp::verify() {
  // No need to check source shapes match mask shapes, because
  // AllShapesMatch<["pointers", "mask"]> in RockOps.td
  auto sourceType = cast<RankedTensorType>(getSource().getType());
  auto ptrType = cast<RankedTensorType>(getPointers().getType());

  size_t idxCount = getExtraIndices().size();
  if (idxCount + ptrType.getRank() !=
      static_cast<size_t>(sourceType.getRank()))
    return emitOpError(
               "extraIndices.size() + pointers rank must equal source rank")
           << " (" << idxCount << " + " << ptrType.getRank()
           << " != " << sourceType.getRank() << ")";

  if (sourceType.getShape().take_back(ptrType.getRank()) !=
      ptrType.getShape()) {
    return emitOpError("Source last dimensions must match with pointers shape ")
           << " (" << sourceType.getShape().take_back(ptrType.getRank())
           << " != " << ptrType.getShape() << ")";
  }

  return success();
}

LogicalResult TransformsToPtrOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  TransformsToPtrOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto sourceType = dyn_cast<RankedTensorType>(adaptor.getSource().getType());
  if (!sourceType)
    return emitOptionalError(location, "source must be a ranked tensor");
  size_t numExtraIndices = adaptor.getExtraIndices().size();
  if (numExtraIndices > static_cast<size_t>(sourceType.getRank()))
    return emitOptionalError(location,
                             "number of extra indices exceeds source rank");
  auto shape =
      sourceType.getShape().take_back(sourceType.getRank() - numExtraIndices);
  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, IntegerType::get(context, 32)));
  inferredReturnTypes.push_back(
      RankedTensorType::get(shape, IntegerType::get(context, 1)));
  return success();
}

//===-----------------------------------------------------===//
// ReduceOp
//===-----------------------------------------------------===//

LogicalResult ReduceOp::verify() {
  auto inpShape = cast<ShapedType>(getIn().getType()).getShape();
  auto outShape = cast<ShapedType>(getResult().getType()).getShape();
  int64_t axis = getAxis().getSExtValue();
  if (axis < 0 || axis >= int64_t(inpShape.size()))
    return emitError("Axis is out of range");
  if (inpShape.size() != outShape.size())
    return emitError("Input and output rank is not the same");
  for (const auto &[dim, dimSize] : llvm::enumerate(outShape)) {
    if (int64_t(dim) == axis) {
      if (dimSize != 1)
        return emitError("The size of the reduction dimension should be 1.");
    } else {
      if (dimSize != inpShape[dim])
        return emitError("The size of the non-reduction dimension should "
                         "match the input.");
    }
  }
  return success();
}

LogicalResult ReduceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  ReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getIn().getType());
  if (!inputType)
    return emitOptionalError(location, "input must be a ranked tensor");
  int64_t axis = adaptor.getAxis().getSExtValue();
  if (axis < 0 || axis >= inputType.getRank())
    return emitOptionalError(location, "axis is out of range");
  SmallVector<int64_t> outShape(inputType.getShape());
  outShape[axis] = 1;
  inferredReturnTypes.push_back(
      RankedTensorType::get(outShape, inputType.getElementType()));
  return success();
}

//===-----------------------------------------------------===//
// BlockwiseReduceOp
//===-----------------------------------------------------===//

LogicalResult BlockwiseReduceOp::verify() {
  auto inpShape = cast<ShapedType>(getInput().getType()).getShape();
  auto outShape = cast<ShapedType>(getResult().getType()).getShape();
  int64_t axis = getAxis().getSExtValue();
  if (axis < 0 || axis >= int64_t(inpShape.size()))
    return emitError("axis is out of range");
  if (outShape.size() + 1 != inpShape.size())
    return emitError("output rank must be input rank - 1");
  for (size_t inDim = 0, outDim = 0; inDim < inpShape.size(); ++inDim) {
    if (int64_t(inDim) == axis)
      continue;
    if (outShape[outDim] != inpShape[inDim])
      return emitError(
          "non-reduction dimension size mismatch at output dim ")
             << outDim;
    ++outDim;
  }
  return success();
}

LogicalResult BlockwiseReduceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  BlockwiseReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!inputType)
    return emitOptionalError(location, "input must be a ranked tensor");
  int64_t axis = adaptor.getAxis().getSExtValue();
  if (axis < 0 || axis >= inputType.getRank())
    return emitOptionalError(location, "axis is out of range");
  SmallVector<int64_t> outShape;
  outShape.reserve(inputType.getRank() - 1);
  for (auto [i, dim] : llvm::enumerate(inputType.getShape())) {
    if (static_cast<int64_t>(i) != axis)
      outShape.push_back(dim);
  }
  inferredReturnTypes.push_back(
      RankedTensorType::get(outShape, inputType.getElementType()));
  return success();
}

//===-----------------------------------------------------===//
// GemmElementwiseGemmOp
//===-----------------------------------------------------===//
Type GemmElementwiseGemmOp::getOutType() { return getResult().getType(); }

Type GemmElementwiseGemmOp::getAType() { return getA().getType(); }

Type GemmElementwiseGemmOp::getBType() { return getB().getType(); }

Type GemmElementwiseGemmOp::getCType() { return getC().getType(); }

bool GemmElementwiseGemmOp::getTransposedA() { return getATransposed(); }

bool GemmElementwiseGemmOp::getTransposedB() { return getBTransposed(); }

bool GemmElementwiseGemmOp::getTransposedC() { return getCTransposed(); }

bool GemmElementwiseGemmOp::getTransposedOut() { return getOTransposed(); }

KernelType GemmElementwiseGemmOp::getKernelType() {
  return KernelType::GemmElementwiseGemm;
}

Region &GemmElementwiseGemmOp::getPreSecondGemmRegion() {
  return getPreSecondGemmBody();
}

MutableOperandRange
GemmElementwiseGemmOp::getPreSecondGemmElemwiseInputsMutable() {
  return getElemwiseInputsMutable();
}

GemmGemmSize GemmElementwiseGemmOp::getGemmGemmSize() {
  ShapedType typeA = getA().getType(), typeB = getB().getType(),
             typeC = getC().getType();
  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape(),
                    dimsC = typeC.getShape();
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1,
          offsetC = dimsC.size() == 2 ? 0 : 1;
  int64_t g = offsetA ? dimsA[0] : 1,
          m = dimsA[offsetA + (getATransposed() ? 1 : 0)],
          k = dimsA[offsetA + (getATransposed() ? 0 : 1)],
          n = dimsB[offsetB + (getBTransposed() ? 0 : 1)],
          o = dimsC[offsetC + (getCTransposed() ? 1 : 0)];
  return GemmGemmSize(g, m, k, n, o);
}

static LogicalResult verifyGemmPlusGemmLikeOp(RockGemmGemmWrapperInterface op,
                                              Value currentSeqLen, Value lse,
                                              int32_t numHeadsQ,
                                              int32_t numHeadsKV) {
  // number of heads for Q and K, V
  if (numHeadsQ <= 0) {
    return op.emitError("numHeadsQ must be positive");
  }
  if (numHeadsKV <= 0) {
    return op.emitError("numHeadsKV must be positive");
  }
  if (numHeadsQ % numHeadsKV != 0) {
    return op.emitError("numHeadsQ is not divisible by numHeadsKV");
  }
  int64_t factorGQA = numHeadsQ / numHeadsKV;

  ShapedType qType = cast<ShapedType>(op.getAType());
  int64_t qBatchDim = qType.getShape().size() == 3 ? qType.getShape()[0] : 1;
  ArrayRef<int64_t> qLastDims = qType.getShape().slice(qType.getRank() - 2);
  auto [queryM, queryK] = op.getTransposedA()
                              ? std::tuple{qLastDims[1], qLastDims[0]}
                              : std::tuple{qLastDims[0], qLastDims[1]};

  ShapedType kType = cast<ShapedType>(op.getBType());
  int64_t kBatchDim = kType.getShape().size() == 3 ? kType.getShape()[0] : 1;
  kBatchDim *= factorGQA;
  ArrayRef<int64_t> kLastDims = kType.getShape().slice(kType.getRank() - 2);
  auto [keyK, keyN] = op.getTransposedB()
                          ? std::tuple{kLastDims[1], kLastDims[0]}
                          : std::tuple{kLastDims[0], kLastDims[1]};

  ShapedType vType = cast<ShapedType>(op.getCType());
  int64_t vBatchDim = vType.getShape().size() == 3 ? vType.getShape()[0] : 1;
  vBatchDim *= factorGQA;
  ArrayRef<int64_t> vLastDims = vType.getShape().slice(vType.getRank() - 2);
  auto [valueK, valueN] = op.getTransposedC()
                              ? std::tuple{vLastDims[1], vLastDims[0]}
                              : std::tuple{vLastDims[0], vLastDims[1]};

  if (qBatchDim != kBatchDim || kBatchDim != vBatchDim) {
    return op.emitError("Batch dimensions do not match");
  }
  if (queryK != keyK) {
    return op.emitError("reduction dimensions of first gemm do not match");
  }
  if (keyN != valueK) {
    return op.emitError("reduction dimensions of second gemm do not match");
  }

  // check output type
  ShapedType oType = cast<ShapedType>(op.getOutType());
  int64_t oBatchDim = oType.getShape().size() == 3 ? oType.getShape()[0] : 1;
  int64_t oBatchDimOrig = oBatchDim;
  if (isa<AttentionOp>(op)) {
    int64_t splitKV = cast<AttentionOp>(op).getSplitKV();
    if (oBatchDim % splitKV != 0)
      return op.emitError("Batch size must be divisible by splitKV");

    oBatchDim = oBatchDim / splitKV;
  }

  ArrayRef<int64_t> oLastDims = oType.getShape().slice(oType.getRank() - 2);
  auto [outputSeqLen, outputHeadDim] =
      op.getTransposedOut() ? std::tuple{oLastDims[1], oLastDims[0]}
                            : std::tuple{oLastDims[0], oLastDims[1]};

  if (qType.getShape().size() != oType.getShape().size()) {
    return op.emitError("Number of dimensions do not match (Q and Output)");
  }
  if (qBatchDim != oBatchDim) {
    return op.emitError("Batch dimensions do not match (Q and Output)");
  }
  if (queryM != outputSeqLen) {
    return op.emitError("Sequence length does not match (Q and Output)");
  }
  if (valueN != outputHeadDim) {
    return op.emitError("Head dimensions do not match (V and Output)");
  }

  // check currentSeqLen (KV Cache)
  if (currentSeqLen) {
    ShapedType seqLenType = cast<ShapedType>(currentSeqLen.getType());
    if (seqLenType.getShape().size() != 1) {
      return op.emitError("Number of dimensions is not one (currentSeqLen)");
    }
    if (seqLenType.getShape()[0] != oBatchDim) {
      return op.emitError(
          "Batch dimensions do not match (currentSeqLen and Output)");
    }
  }

  // check LSE (log-sum-exp)
  if (lse) {
    ShapedType lseType = cast<ShapedType>(lse.getType());
    if (lseType.getShape().size() != 2) {
      return op.emitError("Number of dimensions is not two (LSE)");
    }
    if (lseType.getShape()[0] != oBatchDimOrig) {
      return op.emitError("Batch dimensions do not match (LSE and Output)");
    }
    if (lseType.getShape()[1] != queryM) {
      return op.emitError("SeqLenQ dimensions do not match (LSE and Q)");
    }
  }

  // The pre-second-GEMM region is optional in the assembly format, so an
  // empty region is legal. When present, however, downstream passes
  // (e.g. RegularizeInterGemmFusion, GridwiseAttnToBlockwise) assume a
  // single block whose terminator is a `rock.yield` of exactly one value.
  // Enforce that shape here so malformed IR is rejected up front rather than
  // crashing later in a pass.
  Region &body = op.getPreSecondGemmRegion();
  if (!body.empty()) {
    if (!body.hasOneBlock())
      return op.emitOpError(
          "pre-second-GEMM region must contain a single block");
    Block &block = body.front();
    if (block.getNumArguments() == 0)
      return op.emitOpError(
          "pre-second-GEMM body must have at least one block argument");
    auto yieldOp = dyn_cast<rock::YieldOp>(block.getTerminator());
    if (!yieldOp)
      return op.emitOpError(
          "pre-second-GEMM body must be terminated by a rock.yield");
    if (yieldOp.getNumOperands() != 1)
      return op.emitOpError(
          "pre-second-GEMM body must yield exactly one value");
  }

  return success();
}

LogicalResult GemmElementwiseGemmOp::verify() {
  return verifyGemmPlusGemmLikeOp(*this, /*currentSeqLen=*/nullptr,
                                  /*lse=*/nullptr, /*numHeadsQ=*/1,
                                  /*numHeadsKV=*/1);
}

//===-----------------------------------------------------===//
// ConvElementwiseGemmOp
//===-----------------------------------------------------===//
Type ConvElementwiseGemmOp::getOutType() { return getResult().getType(); }

Type ConvElementwiseGemmOp::getAType() {
  auto size = getGemmGemmSize();
  auto elementType = getInput().getType().getElementType();
  int64_t dim1 = getTransposedA() ? size.k : size.n;
  int64_t dim2 = getTransposedA() ? size.n : size.k;
  return RankedTensorType::get({size.g, dim1, dim2}, elementType);
}

Type ConvElementwiseGemmOp::getBType() {
  auto size = getGemmGemmSize();
  auto elementType = getFilter().getType().getElementType();
  int64_t dim1 = getTransposedB() ? size.m : size.k;
  int64_t dim2 = getTransposedB() ? size.k : size.m;
  return RankedTensorType::get({size.g, dim1, dim2}, elementType);
}

Type ConvElementwiseGemmOp::getCType() { return getC().getType(); }

bool ConvElementwiseGemmOp::getTransposedA() {
  // see ConvToGemm pass
  return true;
}

bool ConvElementwiseGemmOp::getTransposedB() {
  // see ConvToGemm pass
  return false;
}

bool ConvElementwiseGemmOp::getTransposedC() { return getCTransposed(); }

bool ConvElementwiseGemmOp::getTransposedOut() { return getOTransposed(); }

KernelType ConvElementwiseGemmOp::getKernelType() {
  return KernelType::ConvElementwiseGemm;
}

Region &ConvElementwiseGemmOp::getPreSecondGemmRegion() {
  return getPreSecondGemmBody();
}

MutableOperandRange
ConvElementwiseGemmOp::getPreSecondGemmElemwiseInputsMutable() {
  return getElemwiseInputsMutable();
}

GemmGemmSize ConvElementwiseGemmOp::getGemmGemmSize() {
  auto strideVal = extractFromIntegerArrayAttr<int64_t>(getStrides());
  auto dilationVal = extractFromIntegerArrayAttr<int64_t>(getDilations());
  auto paddingVal = extractFromIntegerArrayAttr<int64_t>(getPadding());
  auto sizes = ConvolutionDims::fromOp(*this, false);

  // generate sizes.out with ConvGenerator
  sizes.out[0] = rock::ConvGenerator::outputDim(sizes.in[0], sizes.fil[0],
                                                paddingVal[0], paddingVal[1],
                                                strideVal[0], dilationVal[0]);
  sizes.out[1] = rock::ConvGenerator::outputDim(sizes.in[1], sizes.fil[1],
                                                paddingVal[2], paddingVal[3],
                                                strideVal[1], dilationVal[1]);

  rock::GemmSize gemmSize =
      rock::GemmSize::fromConvolution(rock::ConvOpType::Fwd, sizes);
  ArrayRef<int64_t> dimsC = getC().getType().getShape();
  int64_t offsetC = dimsC.size() == 2 ? 0 : 1;
  int64_t g = gemmSize.g, m = gemmSize.m, k = gemmSize.k, n = gemmSize.n,
          o = dimsC[offsetC + (getCTransposed() ? 0 : 1)];
  return GemmGemmSize(g, m, k, n, o);
}

LogicalResult ConvElementwiseGemmOp::verify() {
  return verifyGemmPlusGemmLikeOp(*this, /*currentSeqLen=*/nullptr,
                                  /*lse=*/nullptr, /*numHeadsQ=*/1,
                                  /*numHeadsKV=*/1);
}

//===-----------------------------------------------------===//
// AttentionOp
//===-----------------------------------------------------===//
Type AttentionOp::getOutType() { return getResult().getType(); }

Type AttentionOp::getAType() { return getQueries().getType(); }

Type AttentionOp::getBType() { return getKeys().getType(); }

Type AttentionOp::getCType() { return getValues().getType(); }

bool AttentionOp::getTransposedA() { return getQTransposed(); }

bool AttentionOp::getTransposedB() { return getKTransposed(); }

bool AttentionOp::getTransposedC() { return getVTransposed(); }

bool AttentionOp::getTransposedOut() { return getOTransposed(); }

KernelType AttentionOp::getKernelType() { return KernelType::Attention; }

Region &AttentionOp::getPreSecondGemmRegion() { return getPreSoftmaxBody(); }

MutableOperandRange AttentionOp::getPreSecondGemmElemwiseInputsMutable() {
  return getPreSoftmaxElemWiseInputsMutable();
}

GemmGemmSize AttentionOp::getGemmGemmSize() {
  ShapedType typeA = getQueries().getType(), typeB = getKeys().getType(),
             typeC = getValues().getType();
  ArrayRef<int64_t> dimsA = typeA.getShape(), dimsB = typeB.getShape(),
                    dimsC = typeC.getShape();
  int64_t offsetA = dimsA.size() == 2 ? 0 : 1,
          offsetB = dimsB.size() == 2 ? 0 : 1,
          offsetC = dimsC.size() == 2 ? 0 : 1;
  int64_t g = offsetA ? dimsA[0] : 1,
          m = dimsA[offsetA + (getQTransposed() ? 1 : 0)],
          k = dimsA[offsetA + (getQTransposed() ? 0 : 1)],
          n = dimsB[offsetB + (getKTransposed() ? 0 : 1)],
          o = dimsC[offsetC + (getVTransposed() ? 1 : 0)];
  return GemmGemmSize(g, m, k, n, o);
}

LogicalResult AttentionOp::verify() {
  if (getSplitKV() != 1 && !getLse())
    return emitError("Flash decoding needs LSE output");

  if (getSplitKV() <= 0)
    return emitError("Negative or zero split-kv does not make sense");

  // Validate prefix offset constraints
  // prefixOffset requires causal to be enabled (prefix causal = causal +
  // prefixOffset)
  if (getPrefixOffset() && !getCausal())
    return emitError(
        "prefixOffset requires causal to be enabled. "
        "Prefix causal attention is causal masking with an offset.");

  return verifyGemmPlusGemmLikeOp(*this, getCurrentSeqLen(), getLse(),
                                  getNumHeadsQ(), getNumHeadsKV());
}

//===-----------------------------------------------------===//
// PerfConfigStr parsing
//===-----------------------------------------------------===//

namespace {

constexpr size_t SmallVectorInlineSize = 32;

// Number of trailing knob fields appended in the v2 perfConfig schema.
// The sentinel value (`rock::kKnobDefault` = `-1`) and the `scheduleHint`
// bit constants live in `KnobUtils.h`.
//
// NOTE: If you want to bump the perfConfig to v3, you need to add a new
// `kNumKnobFieldsV3` constant and update the parser to expect the new number
// of fields.
constexpr size_t kNumKnobFieldsV2 = 6;

// Reject invalid knob values, reusing `rock::isValidKnobBoolean` and
// `rock::isValidScheduleHintBitfield`.
LogicalResult validateKnobBlock(StringRef perfConfigStr, int64_t useAsyncCopy,
                                int64_t useBlockPingpong,
                                int64_t useInThreadTranspose,
                                int64_t useBufferOps, int64_t useBufferAtomics,
                                int64_t scheduleHint) {
  const std::pair<StringRef, int64_t> boolKnobs[] = {
      {"useAsyncCopy", useAsyncCopy},
      {"useBlockPingpong", useBlockPingpong},
      {"useInThreadTranspose", useInThreadTranspose},
      {"useBufferOps", useBufferOps},
      {"useBufferAtomics", useBufferAtomics},
  };
  for (auto [name, value] : boolKnobs) {
    if (!isValidKnobBoolean(value)) {
      llvm::errs() << "invalid perfConfig '" << perfConfigStr << "': field `"
                   << name << "` = " << value << "; expected " << kKnobDefault
                   << " (arch default), 0 (off), or 1 (on)\n";
      return failure();
    }
  }
  if (!rock::isValidScheduleHintBitfield(scheduleHint)) {
    llvm::errs() << "invalid perfConfig '" << perfConfigStr
                 << "': field `scheduleHint` = " << scheduleHint << "\n";
    return failure();
  }
  return success();
}

struct PerfConfigParseResult {
  int version;
  SmallVector<int64_t, SmallVectorInlineSize> params;
};

std::optional<PerfConfigParseResult>
parsePerfConfigStr(StringRef configStr, StringRef expectedPrefix = "") {
  StringRef rest = configStr;

  // Handle optional prefix
  if (!expectedPrefix.empty()) {
    StringRef prefix;
    std::tie(prefix, rest) = rest.split(':');
    if (prefix != expectedPrefix)
      return std::nullopt;
  }

  // Parse "vN:"
  StringRef versionStr;
  std::tie(versionStr, rest) = rest.split(':');
  if (!versionStr.consume_front("v"))
    return std::nullopt;

  int version;
  if (!llvm::to_integer(versionStr, version))
    return std::nullopt;

  // Parse comma-separated parameters
  SmallVector<StringRef, SmallVectorInlineSize> tokens;
  rest.split(tokens, ',');

  SmallVector<int64_t, SmallVectorInlineSize> params;
  params.reserve(tokens.size());
  for (StringRef tok : tokens) {
    int64_t val;
    if (!llvm::to_integer(tok.trim(), val))
      return std::nullopt;
    params.push_back(val);
  }

  return PerfConfigParseResult{version, params};
}

} // namespace

//===-----------------------------------------------------===//
// GemmParamsAttr
//===-----------------------------------------------------===//

GemmParamsAttr GemmParamsAttr::get(StringAttr perfConfigStrAttr) {
  auto parsed = parsePerfConfigStr(perfConfigStrAttr.strref(), "gemm");
  if (!parsed) {
    return {};
  }

  int version = parsed->version;
  auto &params = parsed->params;

  // v1: 11 tunable fields. The 6 knob fields default to `kKnobDefault`.
  // v2: 11 tunable fields + 6 knob fields = 17 fields.
  size_t expectedCount = 0;
  if (version == 1)
    expectedCount = 11;
  else if (version == 2)
    expectedCount = 11 + kNumKnobFieldsV2;
  if (expectedCount == 0 || params.size() != expectedCount) {
    return {};
  }

  int64_t idx = 0;
  int64_t mPerBlock = params[idx++];
  int64_t nPerBlock = params[idx++];
  int64_t kPerBlock = params[idx++];
  int64_t kpack = params[idx++];
  int64_t numCTAs = params[idx++];
  int64_t numWaves = params[idx++];
  int64_t matrixInstrNonkdim = params[idx++];
  int64_t splitKFactor = params[idx++];
  int64_t numStages = params[idx++];
  int64_t wavesPerEU = params[idx++];
  int64_t gridGroupSize = params[idx++];
  int64_t useAsyncCopy = kKnobDefault;
  int64_t useBlockPingpong = kKnobDefault;
  int64_t useInThreadTranspose = kKnobDefault;
  int64_t useBufferOps = kKnobDefault;
  int64_t useBufferAtomics = kKnobDefault;
  int64_t scheduleHint = kKnobDefault;
  if (version >= 2) {
    useAsyncCopy = params[idx++];
    useBlockPingpong = params[idx++];
    useInThreadTranspose = params[idx++];
    useBufferOps = params[idx++];
    useBufferAtomics = params[idx++];
    scheduleHint = params[idx++];
    if (failed(validateKnobBlock(perfConfigStrAttr.strref(), useAsyncCopy,
                                 useBlockPingpong, useInThreadTranspose,
                                 useBufferOps, useBufferAtomics,
                                 scheduleHint))) {
      return {};
    }
  }

  return GemmParamsAttr::get(
      perfConfigStrAttr.getContext(), mPerBlock, nPerBlock, kPerBlock, kpack,
      numCTAs, numWaves, matrixInstrNonkdim, splitKFactor, numStages,
      wavesPerEU, gridGroupSize, useAsyncCopy, useBlockPingpong,
      useInThreadTranspose, useBufferOps, useBufferAtomics, scheduleHint);
}

//===-----------------------------------------------------===//
// GemmGemmParamsAttr
//===-----------------------------------------------------===//

GemmGemmParamsAttr GemmGemmParamsAttr::get(StringAttr perfConfigStrAttr) {
  auto parsed = parsePerfConfigStr(perfConfigStrAttr.strref(), "attn");
  if (!parsed) {
    return {};
  }

  int version = parsed->version;
  auto &params = parsed->params;

  // v1: 11 tunable fields. The 6 knob fields default to `kKnobDefault`.
  // v2: 11 tunable fields + 6 knob fields = 17 fields.
  size_t expectedCount = 0;
  if (version == 1)
    expectedCount = 11;
  else if (version == 2)
    expectedCount = 11 + kNumKnobFieldsV2;
  if (expectedCount == 0 || params.size() != expectedCount) {
    return {};
  }

  int idx = 0;
  int64_t mPerBlockG0 = params[idx++];
  int64_t nPerBlockG0 = params[idx++];
  int64_t kPerBlock = params[idx++];
  int64_t kpack = params[idx++];
  int64_t numCTAs = params[idx++];
  int64_t numWaves = params[idx++];
  int64_t matrixInstrNonkdim = params[idx++];
  int64_t splitKFactor = params[idx++];
  int64_t numStages = params[idx++];
  int64_t wavesPerEU = params[idx++];
  int64_t gridGroupSize = params[idx++];
  int64_t useAsyncCopy = kKnobDefault;
  int64_t useBlockPingpong = kKnobDefault;
  int64_t useInThreadTranspose = kKnobDefault;
  int64_t useBufferOps = kKnobDefault;
  int64_t useBufferAtomics = kKnobDefault;
  int64_t scheduleHint = kKnobDefault;
  if (version >= 2) {
    useAsyncCopy = params[idx++];
    useBlockPingpong = params[idx++];
    useInThreadTranspose = params[idx++];
    useBufferOps = params[idx++];
    useBufferAtomics = params[idx++];
    scheduleHint = params[idx++];
    if (failed(validateKnobBlock(perfConfigStrAttr.strref(), useAsyncCopy,
                                 useBlockPingpong, useInThreadTranspose,
                                 useBufferOps, useBufferAtomics,
                                 scheduleHint))) {
      return {};
    }
  }

  return GemmGemmParamsAttr::get(
      perfConfigStrAttr.getContext(), mPerBlockG0, nPerBlockG0, kPerBlock,
      kpack, numCTAs, numWaves, matrixInstrNonkdim, splitKFactor, numStages,
      wavesPerEU, gridGroupSize, useAsyncCopy, useBlockPingpong,
      useInThreadTranspose, useBufferOps, useBufferAtomics, scheduleHint);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Rock/IR/RockAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Rock/IR/RockOps.cpp.inc"
