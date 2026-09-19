// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1200' %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

// A scalarized tensor epilogue can contain many calls to the same nontrivial
// OCML function. The call-count x body-size budget keeps that function out of
// line instead of cloning its body at every call site.
//
// CHECK-DAG: define internal fastcc noundef float @__ocml_erf_f32
// CHECK-DAG: call fastcc float @__ocml_erf_f32
// CHECK-DAG: attributes #{{[0-9]+}} = { {{.*}}noinline

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa"} {
  llvm.func @__ocml_erf_f32(f32) -> f32

  llvm.func amdgpu_kernelcc @kernel(%arg0: !llvm.ptr) {
    %value = llvm.load %arg0 : !llvm.ptr -> f32
    %c00 = llvm.call @__ocml_erf_f32(%value) : (f32) -> f32
    %c01 = llvm.call @__ocml_erf_f32(%c00) : (f32) -> f32
    %c02 = llvm.call @__ocml_erf_f32(%c01) : (f32) -> f32
    %c03 = llvm.call @__ocml_erf_f32(%c02) : (f32) -> f32
    %c04 = llvm.call @__ocml_erf_f32(%c03) : (f32) -> f32
    %c05 = llvm.call @__ocml_erf_f32(%c04) : (f32) -> f32
    %c06 = llvm.call @__ocml_erf_f32(%c05) : (f32) -> f32
    %c07 = llvm.call @__ocml_erf_f32(%c06) : (f32) -> f32
    %c08 = llvm.call @__ocml_erf_f32(%c07) : (f32) -> f32
    %c09 = llvm.call @__ocml_erf_f32(%c08) : (f32) -> f32
    %c10 = llvm.call @__ocml_erf_f32(%c09) : (f32) -> f32
    %c11 = llvm.call @__ocml_erf_f32(%c10) : (f32) -> f32
    %c12 = llvm.call @__ocml_erf_f32(%c11) : (f32) -> f32
    %c13 = llvm.call @__ocml_erf_f32(%c12) : (f32) -> f32
    %c14 = llvm.call @__ocml_erf_f32(%c13) : (f32) -> f32
    %c15 = llvm.call @__ocml_erf_f32(%c14) : (f32) -> f32
    %c16 = llvm.call @__ocml_erf_f32(%c15) : (f32) -> f32
    %c17 = llvm.call @__ocml_erf_f32(%c16) : (f32) -> f32
    %c18 = llvm.call @__ocml_erf_f32(%c17) : (f32) -> f32
    %c19 = llvm.call @__ocml_erf_f32(%c18) : (f32) -> f32
    %c20 = llvm.call @__ocml_erf_f32(%c19) : (f32) -> f32
    %c21 = llvm.call @__ocml_erf_f32(%c20) : (f32) -> f32
    %c22 = llvm.call @__ocml_erf_f32(%c21) : (f32) -> f32
    %c23 = llvm.call @__ocml_erf_f32(%c22) : (f32) -> f32
    %c24 = llvm.call @__ocml_erf_f32(%c23) : (f32) -> f32
    %c25 = llvm.call @__ocml_erf_f32(%c24) : (f32) -> f32
    %c26 = llvm.call @__ocml_erf_f32(%c25) : (f32) -> f32
    %c27 = llvm.call @__ocml_erf_f32(%c26) : (f32) -> f32
    %c28 = llvm.call @__ocml_erf_f32(%c27) : (f32) -> f32
    %c29 = llvm.call @__ocml_erf_f32(%c28) : (f32) -> f32
    %c30 = llvm.call @__ocml_erf_f32(%c29) : (f32) -> f32
    %c31 = llvm.call @__ocml_erf_f32(%c30) : (f32) -> f32
    %c32 = llvm.call @__ocml_erf_f32(%c31) : (f32) -> f32
    %c33 = llvm.call @__ocml_erf_f32(%c32) : (f32) -> f32
    %c34 = llvm.call @__ocml_erf_f32(%c33) : (f32) -> f32
    %c35 = llvm.call @__ocml_erf_f32(%c34) : (f32) -> f32
    %c36 = llvm.call @__ocml_erf_f32(%c35) : (f32) -> f32
    %c37 = llvm.call @__ocml_erf_f32(%c36) : (f32) -> f32
    %c38 = llvm.call @__ocml_erf_f32(%c37) : (f32) -> f32
    %c39 = llvm.call @__ocml_erf_f32(%c38) : (f32) -> f32
    %c40 = llvm.call @__ocml_erf_f32(%c39) : (f32) -> f32
    %c41 = llvm.call @__ocml_erf_f32(%c40) : (f32) -> f32
    %c42 = llvm.call @__ocml_erf_f32(%c41) : (f32) -> f32
    %c43 = llvm.call @__ocml_erf_f32(%c42) : (f32) -> f32
    %c44 = llvm.call @__ocml_erf_f32(%c43) : (f32) -> f32
    %c45 = llvm.call @__ocml_erf_f32(%c44) : (f32) -> f32
    %c46 = llvm.call @__ocml_erf_f32(%c45) : (f32) -> f32
    %c47 = llvm.call @__ocml_erf_f32(%c46) : (f32) -> f32
    %c48 = llvm.call @__ocml_erf_f32(%c47) : (f32) -> f32
    %c49 = llvm.call @__ocml_erf_f32(%c48) : (f32) -> f32
    %c50 = llvm.call @__ocml_erf_f32(%c49) : (f32) -> f32
    %c51 = llvm.call @__ocml_erf_f32(%c50) : (f32) -> f32
    %c52 = llvm.call @__ocml_erf_f32(%c51) : (f32) -> f32
    %c53 = llvm.call @__ocml_erf_f32(%c52) : (f32) -> f32
    %c54 = llvm.call @__ocml_erf_f32(%c53) : (f32) -> f32
    %c55 = llvm.call @__ocml_erf_f32(%c54) : (f32) -> f32
    %c56 = llvm.call @__ocml_erf_f32(%c55) : (f32) -> f32
    %c57 = llvm.call @__ocml_erf_f32(%c56) : (f32) -> f32
    %c58 = llvm.call @__ocml_erf_f32(%c57) : (f32) -> f32
    %c59 = llvm.call @__ocml_erf_f32(%c58) : (f32) -> f32
    %c60 = llvm.call @__ocml_erf_f32(%c59) : (f32) -> f32
    %c61 = llvm.call @__ocml_erf_f32(%c60) : (f32) -> f32
    %c62 = llvm.call @__ocml_erf_f32(%c61) : (f32) -> f32
    %c63 = llvm.call @__ocml_erf_f32(%c62) : (f32) -> f32
    llvm.store %c63, %arg0 : f32, !llvm.ptr
    llvm.return
  }
}
