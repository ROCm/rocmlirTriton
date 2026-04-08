//===- builderUtils.cpp - Rock utility functions ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"

using mlir::arith::ConstantOp;

namespace mlir {
namespace rock {
Value createConstantIntOp(OpBuilder &b, Location loc, Type type,
                          Type elementType, int64_t value) {
  APInt apValue(elementType.getIntOrFloatBitWidth(), value, true);
  auto constValue = b.getIntegerAttr(elementType, apValue);

  Value retValue;
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    retValue = ConstantOp::create(
        b, loc, SplatElementsAttr::get(shapedType, constValue));
  } else {
    retValue = ConstantOp::create(b, loc, type, constValue);
  }

  return retValue;
}

std::pair<APFloat, llvm::detail::opStatus> createAPFloat(Type elemType,
                                                         float value) {
  const llvm::fltSemantics &semantics =
      cast<FloatType>(elemType).getFloatSemantics();
  APFloat apValue(value);
  bool lostInfo = false;
  auto status =
      apValue.convert(semantics, APFloat::rmNearestTiesToEven, &lostInfo);

  return std::make_pair(apValue, status);
}

Value createConstantFloatOp(OpBuilder &b, Location loc, Type type,
                            Type elemType, float value,
                            APFloat::opStatus expectedStatus) {
  std::pair<APFloat, llvm::detail::opStatus> floatRes =
      createAPFloat(elemType, value);
  APFloat apValue = floatRes.first;
  auto status = floatRes.second;
  assert(status == expectedStatus);
  Value retValue;

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    Attribute constValue = b.getFloatAttr(elemType, apValue);
    assert(shapedType.getElementType() == elemType);
    retValue = ConstantOp::create(
        b, loc, SplatElementsAttr::get(shapedType, constValue));
  } else {
    retValue =
        ConstantOp::create(b, loc, type, b.getFloatAttr(elemType, apValue));
  }

  return retValue;
}

Value createZeroConstantOp(OpBuilder &b, Location loc, Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isIntOrIndex()) {
    return createConstantIntOp(b, loc, type, elementType, 0);
  } else {
    return createConstantFloatOp(b, loc, type, elementType, 0.0);
  }
}

//===----------------------------------------------------------------------===//
// Utility function to emit type conversion ops.
//===----------------------------------------------------------------------===//
Value createTypeConversionOp(OpBuilder &b, Location loc, Value source,
                             Type destType) {
  // Convert from sourceType to destType if necessary.
  Type sourceElemType = getElementTypeOrSelf(source.getType());
  Type destElemType = getElementTypeOrSelf(destType);
  unsigned sourceWidth = sourceElemType.getIntOrFloatBitWidth();
  unsigned destWidth = destElemType.getIntOrFloatBitWidth();
  Value result = source;
  if (sourceElemType != destElemType) {
    if (isa<IntegerType>(sourceElemType) && isa<IntegerType>(destElemType)) {
      if (sourceWidth <= destWidth) {
        result = arith::ExtSIOp::create(b, loc, destType, source);
      } else {
        result = arith::TruncIOp::create(b, loc, destType, source);
      }
    } else if (isa<FloatType>(sourceElemType) && isa<FloatType>(destElemType)) {
      if (sourceWidth < destWidth) {
        result = arith::ExtFOp::create(b, loc, destType, source);
      } else {
        result = arith::TruncFOp::create(b, loc, destType, source);
      }
    } else if (isa<FloatType>(sourceElemType) &&
               isa<IntegerType>(destElemType)) {
      result = arith::FPToSIOp::create(b, loc, destType, source);
    } else if (isa<IntegerType>(sourceElemType) &&
               isa<FloatType>(destElemType)) {
      result = arith::SIToFPOp::create(b, loc, destType, source);
    } else {
      llvm_unreachable("Unsupported type conversion");
    }
  }
  return result;
}

Type getFlattenedType(Type type) {
  if (auto st = dyn_cast<ShapedType>(type))
    return st.cloneWith(st.getNumElements(), st.getElementType());
  llvm_unreachable("not a ShapedType");
}

Value getAsTensor(OpBuilder &builder, Location loc, mlir::Value value,
                  bool isWritable) {
  constexpr bool isRestrict{true};
  Value origTensor = bufferization::ToTensorOp::create(
      builder, loc, memref::getTensorTypeFromMemRefType(value.getType()), value,
      isRestrict, isWritable);
  return origTensor;
}

} // namespace rock
} // namespace mlir
