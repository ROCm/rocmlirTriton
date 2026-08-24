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
#ifndef MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H
#define MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H

//===- tritonUtils.h - Triton-dependent utilities for Rock ----------------===//
//
// Centralizes Triton-dependent helpers and C++ replicas of Triton-internal
// functions that must be kept in sync on every Triton version bump. Having
// them in one place makes the bump_triton_version.md checklist easier to
// follow.
//
// Upstream sources:
//   getMfmaVersion / getWmmaVersion
//     ->
//     triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp
//   mlirTypeToScaleDotElemType
//     ->
//     triton/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp
//        (mlirTypeToScaledElemType, extended with BF16/FP16)
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {

namespace triton {
enum class ScaleDotElemType : uint32_t;
namespace amdgpu {
enum class ISAFamily;
} // namespace amdgpu
} // namespace triton

namespace rock {

/// Get MFMA version from ISA family.
/// Mirrors the internal getMfmaVersion() in AccelerateAMDMatmul.cpp.
int getMfmaVersion(triton::amdgpu::ISAFamily isaFamily);

/// Get WMMA version from ISA family.
/// Mirrors the internal getWmmaVersion() in AccelerateAMDMatmul.cpp.
int getWmmaVersion(triton::amdgpu::ISAFamily isaFamily);

/// Return true if `t` is one of the types in Triton's TT_Float set.
/// Mirrors the TT_Float type constraint from TritonTypes.td:
///   {F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ, F16, BF16, F32, F64}
bool isTTFloat(Type t);

/// Return true if `t` is one of the types in Triton's TT_Int set.
/// Mirrors the TT_Int type constraint from TritonTypes.td:
///   {I1, I4, I8, I16, I32, I64}
bool isTTInt(Type t);

/// Map an MLIR element type to the corresponding triton::ScaleDotElemType.
/// Covers F8 (E4M3, E5M2), F6 (E2M3, E3M2), F4 (E2M1), BF16, and FP16.
/// Returns failure() for unsupported types.
///
/// Adapted from mlirTypeToScaledElemType in AccelerateAMDMatmul.cpp with
/// additional BF16/FP16 coverage.
FailureOr<triton::ScaleDotElemType> mlirTypeToScaleDotElemType(Type type);

/// Insert a unit dimension at `axis` with tt.expand_dims, then broadcast the
/// expanded tensor to `resultType`.
Value expandDimAndBroadcast(OpBuilder &builder, Location loc, Value source,
                            int64_t axis, RankedTensorType resultType);

} // namespace rock
} // namespace mlir

#endif // MLIR_DIALECT_ROCK_UTILITY_TRITONUTILS_H
