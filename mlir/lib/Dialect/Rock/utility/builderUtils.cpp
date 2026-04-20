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

#include <cmath>

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

// All the complexity here is driven by MIGraphX. MIGraphX's reference
// `convert` op (see MIGraphX `src/include/migraphx/op/convert.hpp` and the
// `migraphx::convert` reference kernel) defines float -> int conversion as
// saturating + truncating:
//   - Out-of-range finite floats clamp to [INT_MIN, INT_MAX] (NOT modular
//     wraparound, NOT C-style undefined behavior).
//   - +inf  -> INT_MAX
//   - -inf  -> INT_MIN  (or 0 for unsigned destinations)
//   - NaN   -> 0
//   - Rounding mode is truncation (round-toward-zero), NOT round-to-nearest.
//
// None of the off-the-shelf MLIR lowerings give us this:
//   - `arith.fptosi`/`arith.fptoui` are LLVM-style: any out-of-range,
//     infinite, or NaN input is *poison*. We must clamp before calling them
//     or risk producing UB at runtime.
//   - Upstream `tosa-to-linalg` lowers `tosa.cast` for fp->int with
//     round-to-nearest-even, which disagrees with MIGraphX's truncation
//     semantics, so we can't just emit a `tosa.cast` either.
//
// That's why MIGraphXToTosa emits dedicated `tosa.custom` ops
// (`unsigned_cast`, `fp_to_int_cast`) for these casts instead of plain
// `tosa.cast`, and why both lowering paths (CPU via
// RocmlirCustomTosaToLinalg, GPU/kernel via RockTosaToElementwise) call
// into this single helper to expand them. The three-case structure below
// (exponent fits / mantissa fits / mixed clamp + overflow fix-up) is what
// it takes to implement the saturating semantics correctly across every
// (source-float, dest-int) pair without ever feeding poison values into
// `arith.fptosi`/`arith.fptoui` on the way through.
//
// If MIGraphX's convert semantics ever change, or if we drop the MIGraphX
// frontend, much of this can collapse back to a plain `tosa.cast` plus the
// upstream tosa-to-linalg lowering.
Value createClampedFPToInt(OpBuilder &b, Location loc, Value input,
                           Type dstIntType, bool isUnsigned) {
  Type srcType = input.getType();
  auto srcShapedTy = dyn_cast<ShapedType>(srcType);
  Type srcElemTy = srcShapedTy ? srcShapedTy.getElementType() : srcType;
  auto srcFloatTy = cast<FloatType>(srcElemTy);
  Type dstType =
      srcShapedTy ? srcShapedTy.cloneWith({}, dstIntType) : Type(dstIntType);
  unsigned dstWidth = cast<IntegerType>(dstIntType).getWidth();
  const auto &fltSemantics = srcFloatTy.getFloatSemantics();

  // The algorithm below relies on being able to materialize 0.0 (for the
  // NaN -> 0 sanitization and the unsigned-zero clamp) and +/-inf (for the
  // Case 1 overflow checks). Some float types lack these
  // special values (e.g. F8E8M0FNU has no zero; F4E2M1FN has no infinity);
  // they should be promoted to a wider type by an earlier pass before
  // reaching this conversion. If one slips through, fail loudly rather
  // than emit IR with unrepresentable constants. (NaN representability is
  // not required: types without NaN simply can't produce a NaN input, so
  // the cmpf UNO becomes dead code.)
  if (!APFloat::semanticsHasZero(fltSemantics) ||
      !APFloat::semanticsHasInf(fltSemantics)) {
    llvm::reportFatalUsageError(
        "rock::createClampedFPToInt: source float type lacks a "
        "representable zero or infinity; promote it to a wider float type "
        "before invoking this conversion");
  }

  auto fpConst = [&](APFloat v) -> Value {
    TypedAttr fa = b.getFloatAttr(srcElemTy, v);
    if (srcShapedTy)
      return ConstantOp::create(b, loc,
                                DenseElementsAttr::get(srcShapedTy, fa));
    return ConstantOp::create(b, loc, fa);
  };
  auto fpConstD = [&](double v) -> Value {
    TypedAttr fa = b.getFloatAttr(srcElemTy, v);
    if (srcShapedTy)
      return ConstantOp::create(b, loc,
                                DenseElementsAttr::get(srcShapedTy, fa));
    return ConstantOp::create(b, loc, fa);
  };
  auto intConst = [&](APInt v) -> Value {
    TypedAttr ia = b.getIntegerAttr(dstIntType, v);
    if (srcShapedTy) {
      auto dstShapedTy = cast<ShapedType>(dstType);
      return ConstantOp::create(b, loc,
                                DenseElementsAttr::get(dstShapedTy, ia));
    }
    return ConstantOp::create(b, loc, ia);
  };
  auto fpToInt = [&](Value v) -> Value {
    if (isUnsigned)
      return arith::FPToUIOp::create(b, loc, dstType, v);
    return arith::FPToSIOp::create(b, loc, dstType, v);
  };

  APInt intMin = isUnsigned ? APInt::getZero(dstWidth)
                            : APInt::getSignedMinValue(dstWidth);
  APInt intMax = isUnsigned ? APInt::getMaxValue(dstWidth)
                            : APInt::getSignedMaxValue(dstWidth);

  // Sanitize NaN -> 0 up front so the case logic below never feeds NaN to
  // arith.fptosi/fptoui (which would be poison) and never has to disambiguate
  // it from +/-inf via UEQ comparisons (UEQ is true for unordered, so without
  // this NaN would erroneously match both the +inf and -inf overflow paths
  // in case 1, and would leak poison through case 2 entirely). NaN -> 0 also
  // matches MIGraphX's reference convert behaviour.
  Value isNaN =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UNO, input, input);
  input = arith::SelectOp::create(b, loc, isNaN, fpConstD(0.0), input);

  // Case 1: int range exceeds the float exponent range, so every finite
  // float maps to a representable integer; only +/-inf need fix-ups.
  //
  // The largest finite value of a float type with max-exponent E is just
  // below 2^(E+1). For all finite floats to fit in the int range:
  //   signed   [-2^(W-1), 2^(W-1)-1] : need 2^(E+1) <= 2^(W-1) -> W-1 > E
  //   unsigned [0,        2^W   -1]  : need 2^(E+1) <= 2^W     -> W   > E
  int maxExp = APFloat::semanticsMaxExponent(fltSemantics);
  int rangeExpThreshold =
      isUnsigned ? static_cast<int>(dstWidth) : static_cast<int>(dstWidth) - 1;
  if (rangeExpThreshold > maxExp) {
    Value posInf = fpConst(APFloat::getInf(fltSemantics));
    Value overflow =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UEQ, input, posInf);
    if (isUnsigned) {
      // Unsigned: clamp negatives (incl. -inf) to 0 before converting,
      // then patch up +inf to intMax.
      Value zeroF = fpConstD(0.0);
      Value clampedPos = arith::MaximumFOp::create(b, loc, input, zeroF);
      Value conv = fpToInt(clampedPos);
      return arith::SelectOp::create(b, loc, overflow, intConst(intMax), conv);
    }
    Value conv = fpToInt(input);
    Value negInf = fpConst(APFloat::getInf(fltSemantics, /*Negative=*/true));
    Value underflow =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UEQ, input, negInf);
    Value maxClamped =
        arith::SelectOp::create(b, loc, overflow, intConst(intMax), conv);
    return arith::SelectOp::create(b, loc, underflow, intConst(intMin),
                                   maxClamped);
  }

  // intMinFP is shared between cases 2 and 3 (intMin is always exactly
  // representable: 0 for unsigned, -2^(W-1) for signed).
  double intMinDouble = isUnsigned ? static_cast<double>(intMin.getZExtValue())
                                   : static_cast<double>(intMin.getSExtValue());
  Value intMinFP = fpConstD(intMinDouble);

  // Case 2: float mantissa is wide enough to represent intMax exactly.
  unsigned requiredMantissa = isUnsigned ? dstWidth : dstWidth - 1;
  if (srcFloatTy.getFPMantissaWidth() >= requiredMantissa) {
    double intMaxDouble = isUnsigned
                              ? static_cast<double>(intMax.getZExtValue())
                              : static_cast<double>(intMax.getSExtValue());
    Value intMaxFP = fpConstD(intMaxDouble);
    Value hi = arith::MinimumFOp::create(b, loc, input, intMaxFP);
    Value clamped = arith::MaximumFOp::create(b, loc, hi, intMinFP);
    return fpToInt(clamped);
  }

  // Case 3: exponent fits but mantissa cannot represent intMax exactly.
  // (intMax + 1) is a power of two and always exactly representable, so
  // compare against it to detect overflow after a lower-bound clamp.
  double intMaxPlusOneDouble =
      isUnsigned ? std::ldexp(1.0, dstWidth)
                 : static_cast<double>(intMax.getSExtValue()) + 1.0;
  Value intMaxPlusOneFP = fpConstD(intMaxPlusOneDouble);

  Value minClampedFP = arith::MaximumFOp::create(b, loc, input, intMinFP);
  Value minClamped = fpToInt(minClampedFP);
  Value overflow = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UGE,
                                         input, intMaxPlusOneFP);
  return arith::SelectOp::create(b, loc, overflow, intConst(intMax),
                                 minClamped);
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
