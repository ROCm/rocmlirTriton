//===-- HIPToTosa.h - HIP conversion to Tosa pass declarations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the HIP to Tosa Dialect conversion in
// MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_HIPTOTOSA_H
#define MLIR_CONVERSION_HIPTOTOSA_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "hip/Dialect/IR/HipDialect.h"

namespace mlir {
#define GEN_PASS_DECL_HIPTOTOSAPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"

namespace hip {

/// Populates conversion patterns from the HIP dialect to the TOSA dialect.
void populateHIPToTosaConversionPatterns(RewritePatternSet &patterns);

/// Mark `func` as a Rock kernel targeting `arch`, so that the TosaToRock pass
/// will accept it. See the TODO in HIPToTosa.cpp -- this does not belong in a
/// HIP-to-TOSA conversion.
void annotateAsRockKernel(func::FuncOp func, StringRef arch);

/// Build the HIP-to-TOSA stage of the `hipep` kernel pipeline, targeting
/// `arch`.
void addHIPToTosaPasses(OpPassManager &pm, StringRef arch);

} // namespace hip
} // namespace mlir

#endif // MLIR_CONVERSION_HIPTOTOSA_H
