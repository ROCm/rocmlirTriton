//===- LLVMInferIntRangeInterfaceImpls.cpp - Integer range for LLVM ops ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Integer range inference for the LLVM dialect's integer operations. The
// transfer functions are the ones already shared by the `arith` and `index`
// dialects, so this file is only the wiring; the LLVM ops carry no signedness
// of their own, exactly like their `arith` counterparts.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::intrange;

/// Map an LLVM integer-overflow flag set onto the shared inference helpers'
/// flags, so that `nsw`/`nuw` tighten the result the same way `arith`'s do.
static intrange::OverflowFlags
convertLLVMOverflowFlags(LLVM::IntegerOverflowFlags flags) {
  intrange::OverflowFlags retFlags = intrange::OverflowFlags::None;
  if (bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nsw))
    retFlags |= intrange::OverflowFlags::Nsw;
  if (bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nuw))
    retFlags |= intrange::OverflowFlags::Nuw;
  return retFlags;
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

void ConstantOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                                   SetIntRangeFn setResultRange) {
  auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(getValue());
  if (!intAttr)
    return;
  auto resultType = llvm::dyn_cast<IntegerType>(getType());
  if (!resultType)
    return;

  // The attribute's width need not match the result's: `llvm.mlir.constant`
  // accepts a wider attribute type such as `index` or the implied `i64`, e.g.
  // `llvm.mlir.constant(0 : index) : i32`. Narrow it, but only when the value
  // survives the round trip, since a range of the wrong width would be
  // meaningless to the analysis.
  APInt value = intAttr.getValue();
  unsigned width = resultType.getWidth();
  if (value.getBitWidth() != width) {
    if (value.getBitWidth() < width)
      return;
    APInt narrowed = value.trunc(width);
    if (narrowed.zext(value.getBitWidth()) != value &&
        narrowed.sext(value.getBitWidth()) != value)
      return;
    value = std::move(narrowed);
  }
  setResultRange(getResult(), ConstantIntRanges::constant(value));
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//

void AddOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferAdd(argRanges, convertLLVMOverflowFlags(
                                                      getOverflowFlags())));
}

void SubOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferSub(argRanges, convertLLVMOverflowFlags(
                                                      getOverflowFlags())));
}

void MulOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMul(argRanges, convertLLVMOverflowFlags(
                                                      getOverflowFlags())));
}

void UDivOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivU(argRanges));
}

void SDivOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferDivS(argRanges));
}

void URemOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemU(argRanges));
}

void SRemOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferRemS(argRanges));
}

//===----------------------------------------------------------------------===//
// Bitwise
//===----------------------------------------------------------------------===//

void AndOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferAnd(argRanges));
}

void OrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                             SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferOr(argRanges));
}

void XOrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferXor(argRanges));
}

void ShlOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                              SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShl(argRanges, convertLLVMOverflowFlags(
                                                      getOverflowFlags())));
}

void LShrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrU(argRanges));
}

void AShrOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferShrS(argRanges));
}

//===----------------------------------------------------------------------===//
// Min/max intrinsics
//===----------------------------------------------------------------------===//

void UMinOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinU(argRanges));
}

void UMaxOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxU(argRanges));
}

void SMinOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMinS(argRanges));
}

void SMaxOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  setResultRange(getResult(), inferMaxS(argRanges));
}

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

void ICmpOp::inferResultRanges(ArrayRef<ConstantIntRanges> argRanges,
                               SetIntRangeFn setResultRange) {
  // LLVM's ICmpPredicate and intrange::CmpPredicate are declared in the same
  // order, as are `arith`'s, so the cast is the mapping.
  auto pred = static_cast<intrange::CmpPredicate>(getPredicate());
  const ConstantIntRanges &lhs = argRanges[0], &rhs = argRanges[1];

  APInt min = APInt::getZero(1);
  APInt max = APInt::getAllOnes(1);

  std::optional<bool> truthValue = intrange::evaluatePred(pred, lhs, rhs);
  if (truthValue.has_value() && *truthValue)
    min = max;
  else if (truthValue.has_value() && !(*truthValue))
    max = min;

  setResultRange(getResult(), ConstantIntRanges::fromUnsigned(min, max));
}

//===----------------------------------------------------------------------===//
// Select
//===----------------------------------------------------------------------===//

void SelectOp::inferResultRangesFromOptional(
    ArrayRef<IntegerValueRange> argRanges, SetIntLatticeFn setResultRange) {
  std::optional<APInt> mbCondVal =
      argRanges[0].isUninitialized()
          ? std::nullopt
          : argRanges[0].getValue().getConstantValue();

  const IntegerValueRange &trueCase = argRanges[1];
  const IntegerValueRange &falseCase = argRanges[2];

  if (mbCondVal) {
    if (mbCondVal->isZero())
      setResultRange(getResult(), falseCase);
    else
      setResultRange(getResult(), trueCase);
    return;
  }

  // When one of the ranges is uninitialized, set the whole range to max
  // otherwise the result will ignore the uninitialized range.
  if (trueCase.isUninitialized() || falseCase.isUninitialized())
    setResultRange(getResult(), IntegerValueRange::getMaxRange(getResult()));
  else
    setResultRange(getResult(), IntegerValueRange::join(trueCase, falseCase));
}
