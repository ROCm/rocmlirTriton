//===-- RocmlirPromoteSoftmaxPrecision.h - Softmax precision --*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pass that promotes f16/bf16 softmax normalization (reduce_sum, reciprocal,
// mul) to f32 in the CPU reference path for improved numerical accuracy.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ROCMLIRPROMOTESOFTMAXPRECISION_H
#define MLIR_CONVERSION_ROCMLIRPROMOTESOFTMAXPRECISION_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL_ROCMLIRPROMOTESOFTMAXPRECISIONPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_ROCMLIRPROMOTESOFTMAXPRECISION_H
