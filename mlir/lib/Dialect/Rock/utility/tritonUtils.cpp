//===- tritonUtils.cpp - Triton-dependent utilities for Rock --------------===//
//
// Portions derived from Triton:
// external/triton/third_party/amd/lib/TritonAMDGPUTransforms/
// AccelerateAMDMatmul.cpp
// external/triton/include/triton/Dialect/Triton/IR/TritonTypes.td
//
// Copyright 2018-2020 Philippe Tillet
// Copyright 2020-2022 OpenAI
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// Rock-specific additions and modifications:
// Copyright Advanced Micro Devices, Inc.
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception AND MIT
//
// Centralizes Triton-dependent helpers and C++ replicas of Triton-internal
// functions that must be kept in sync on every Triton version bump. See
// tritonUtils.h for upstream sources.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/tritonUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton::amdgpu;

namespace mlir {
namespace rock {

// Keep in sync with AccelerateAMDMatmul.cpp::getMfmaVersion()
int getMfmaVersion(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
    return 1;
  case ISAFamily::CDNA2:
    return 2;
  case ISAFamily::CDNA3:
    return 3;
  case ISAFamily::CDNA4:
    return 4;
  default:
    return 0;
  }
}

// Keep in sync with AccelerateAMDMatmul.cpp::getWmmaVersion()
int getWmmaVersion(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::RDNA3:
    return 1;
  // gfx1170 uses the gfx12/RDNA4-style 128-bit WMMA layout
  // (FeatureWMMA128bInsts + FeatureSWMMACGfx1200Insts in LLVM), so it selects
  // WMMA v2 like RDNA4. Keep in sync with AccelerateAMDMatmul.cpp.
  case ISAFamily::GFX1170:
  case ISAFamily::RDNA4:
    return 2;
  case ISAFamily::GFX1250:
    return 3;
  default:
    break;
  }
  return 0;
}

// Keep in sync with TT_Float in TritonTypes.td.
bool isTTFloat(Type t) {
  return isa<Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type,
             Float8E5M2FNUZType, Float16Type, BFloat16Type, Float32Type,
             Float64Type>(t);
}

// Keep in sync with TT_Int in TritonTypes.td.
bool isTTInt(Type t) {
  auto intType = dyn_cast<IntegerType>(t);
  if (!intType || !intType.isSignless())
    return false;
  switch (intType.getWidth()) {
  case 1:
  case 4:
  case 8:
  case 16:
  case 32:
  case 64:
    return true;
  default:
    return false;
  }
}

// Keep in sync with AccelerateAMDMatmul.cpp::mlirTypeToScaledElemType()
// Extended with BF16/FP16 coverage.
FailureOr<triton::ScaleDotElemType> mlirTypeToScaleDotElemType(Type type) {
  return llvm::TypeSwitch<Type, FailureOr<triton::ScaleDotElemType>>(type)
      .Case<Float8E4M3FNType>(
          [](Type) { return triton::ScaleDotElemType::E4M3; })
      .Case<Float8E5M2Type>([](Type) { return triton::ScaleDotElemType::E5M2; })
      .Case<Float6E2M3FNType>(
          [](Type) { return triton::ScaleDotElemType::E2M3; })
      .Case<Float6E3M2FNType>(
          [](Type) { return triton::ScaleDotElemType::E3M2; })
      .Case<Float4E2M1FNType>(
          [](Type) { return triton::ScaleDotElemType::E2M1; })
      .Case<BFloat16Type>([](Type) { return triton::ScaleDotElemType::BF16; })
      .Case<Float16Type>([](Type) { return triton::ScaleDotElemType::FP16; })
      .Default([](Type) { return failure(); });
}

Value expandDimAndBroadcast(OpBuilder &builder, Location loc, Value source,
                            int64_t axis, RankedTensorType resultType) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  assert(axis >= 0 && axis <= sourceType.getRank() &&
         "expanded axis must be within the source rank");
  assert(resultType.getRank() == sourceType.getRank() + 1 &&
         "broadcast result must have one more dimension than the source");
  assert(resultType.getElementType() == sourceType.getElementType() &&
         "broadcast cannot change the element type");

  SmallVector<int64_t> expandedShape(sourceType.getShape());
  expandedShape.insert(expandedShape.begin() + axis, 1);
  auto expandedType = RankedTensorType::get(
      expandedShape, sourceType.getElementType(), sourceType.getEncoding());
  Value expanded =
      triton::ExpandDimsOp::create(builder, loc, expandedType, source, axis);
  return triton::BroadcastOp::create(builder, loc, resultType, expanded);
}

} // namespace rock
} // namespace mlir
