//===-- RocMLIRPasses.td - Conversion pass definition file -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ROCMLIR_CONVERSION_PASSES
#define ROCMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


//===----------------------------------------------------------------------===//
// RocmlirCustomTosaDecomposePass
//===----------------------------------------------------------------------===//

def RocmlirCustomTosaDecomposePass : Pass<"rocmlir-custom-tosa-decompose", ""> {
  let summary = "Decompose RocMLIR's custom TOSA ops";
  let description = [{
    Pass that decomposes RocMLIR's custom TOSA operators.
  }];
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect",
                           "tensor::TensorDialect", "tosa::TosaDialect"];
}

//===----------------------------------------------------------------------===//
// RocmlirCustomTosaToLinalgPass
//===----------------------------------------------------------------------===//

def RocmlirCustomTosaToLinalgPass : Pass<"rocmlir-custom-tosa-to-linalg", ""> {
  let summary = "Lower RocMLIR's custom TOSA ops to Linalg";
  let description = [{
    Pass that converts RocMLIR's custom TOSA operators to `linalg`.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "tosa::TosaDialect"
  ];
}

//===----------------------------------------------------------------------===//
// RocmlirPromoteSoftmaxPrecisionPass
//===----------------------------------------------------------------------===//

def RocmlirPromoteSoftmaxPrecisionPass
    : Pass<"rocmlir-promote-softmax-precision", ""> {
  let summary =
      "Promote f16/bf16 softmax/LSE numerator-sum and consumers to f32";
  let description = [{
    Matches `tosa.reduce_sum` whose input is `tosa.exp` of an f16/bf16
    tensor (the TOSA softmax/LSE numerator sum) and rewrites the sum in f32,
    compensating for the CPU's single-pass reduction vs the GPU's blockwise
    reduction.

    Opportunistically promotes the two consumer chains to f32 as well so the
    precision win is not immediately truncated:
      * Softmax tail (`reciprocal -> mul(exp, reciprocal)`): rewritten end
        to end in f32 with a single cast back to the original element type
        before downstream consumers.
      * LSE chain (`log` and the `log + add(max)` follow-up): rewritten in
        f32 with the inputs cast up and the result cast back, mirroring the
        GPU's `(log2(l) + m)*ln2` formulation that runs entirely in the
        softmax-acc type.

    Any other consumer of the matched `reduce_sum` falls back to a single
    cast back to the original element type, so semantics are preserved.
  }];
  let dependentDialects = ["tosa::TosaDialect"];
}

//===----------------------------------------------------------------------===//
// TosaToRock
//===----------------------------------------------------------------------===//

def TosaToRockPass : Pass<"tosa-to-rock", "::mlir::func::FuncOp"> {
  let summary = "Lower TOSA to Rock";
  let description = [{
    Pass that converts TOSA operations to Rock operations.
  }];
  let dependentDialects = [
    "func::FuncDialect",
    "rock::RockDialect",
    "tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// MIGraphXToTosa
//===----------------------------------------------------------------------===//

def MIGraphXToTosaPass : Pass<"migraphx-to-tosa", "::mlir::func::FuncOp"> {
  let summary = "Lower MIGraphX to TOSA";
  let description = [{
    Pass that converts MIGraphX operations to TOSA operations.
  }];

  let options = [Option<
                     "disableFastMath", "disable-fast-math", "bool", "false",
                     "Preserve IEEE NaN semantics in min/max/clamp lowerings">,
  ];

  let dependentDialects = ["func::FuncDialect", "tosa::TosaDialect", ];
}

//===----------------------------------------------------------------------===//
// HIPToTosa
//===----------------------------------------------------------------------===//

def HIPToTosaPass : Pass<"hip-to-tosa", "::mlir::func::FuncOp"> {
  let summary = "Lower HIP to TOSA";
  let description = [{
    Pass that converts HIP operations to TOSA operations.
  }];

  let dependentDialects = ["func::FuncDialect", "tosa::TosaDialect", ];
}

//===----------------------------------------------------------------------===//
// EmulateFp8ExtTrunc
//===----------------------------------------------------------------------===//

def EmulateFp8ExtTruncPass : Pass<"emulate-fp8-ext-trunc"> {
  let summary = "Emulate arith.extf/truncf on fp8 types";
  let description = [{
    Pass that emulates arith.extf and arith.truncf on 8-bit float types
    so as to allow testing on non-MI-300 platforms.

    arith.extf is converted into lookups in a table generated at
    code generation-time.

    arith.truncf gets converted to calls to functions which this pass inserts.

    This is a quick implementation that provides the mimimal functionality to
    test 8-bit float kernels.

    This pass must be run at the `builtin.module`/`gpu.module` level
    (the root operation needs to have a symbol table)
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "cf::ControlFlowDialect",
    "func::FuncDialect",
    "memref::MemRefDialect",
    "vector::VectorDialect",
  ];
  let options = [
    Option<"hasFp8ConversionInstrs", "f8-conversion-instrs", "bool", /*default=*/"false",
           "if the chip has fp8 Nanoo to fp32 conversion instructions or not">,
    Option<"hasOcpFp8ConversionInstrs", "ocpf8-conversion-instrs", "bool", /*default=*/"false",
           "if the chip has OCP fp8 to fp32 conversion instructions or not">,
  ];
}

#endif // ROCMLIR_CONVERSION_PASSES
