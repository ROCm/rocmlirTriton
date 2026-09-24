// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1200' %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s
// RUN: env AMDGCN_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1200' %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ASM
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1250' %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ATTR
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1250 allow-flush-denorm=false use-expert-scheduling=0' \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ATTR-OFF

// Keep the 128 calls in the dense block out of line, but still inline the
// single call in the following sparse block.
//
// CHECK-LABEL: define amdgpu_kernel void @kernel
// CHECK: call fastcc float @__ocml_erf_f32({{.*}}) #[[NOINLINE:[0-9]+]]
// CHECK-COUNT-127: call fastcc float @__ocml_erf_f32({{.*}}) #[[NOINLINE]]
// CHECK-NOT: call fastcc float @__ocml_erf_f32
// CHECK: define internal fastcc noundef float @__ocml_erf_f32
// CHECK: attributes #[[NOINLINE]] = { noinline }

// Verify that emitMachineCode()'s always-inliner still honors the call-site
// noinline attributes and that all dense-block calls survive to final ISA.
// ASM-LABEL: __ocml_erf_f32:
// ASM-LABEL: kernel:
// ASM: __ocml_erf_f32@rel32@lo
// ASM-COUNT-128: s_swappc_b64

// Linked functions that survive outlining must use the same denormal and
// expert-scheduling policy as the kernel.
// ATTR-LABEL: define internal fastcc noundef float @__ocml_erf_f32
// ATTR-SAME: #[[OCML_ATTRS:[0-9]+]]
// ATTR: attributes #[[OCML_ATTRS]] = {
// ATTR-SAME: denormal_fpenv(float: preservesign)
// ATTR-SAME: "amdgpu-expert-scheduling-mode"="true"

// ATTR-OFF-LABEL: define internal fastcc noundef float @__ocml_erf_f32
// ATTR-OFF-SAME: #[[OCML_ATTRS_OFF:[0-9]+]]
// ATTR-OFF: attributes #[[OCML_ATTRS_OFF]] = {
// ATTR-OFF-SAME: denormal_fpenv(ieee)
// ATTR-OFF-SAME: "amdgpu-expert-scheduling-mode"="false"

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
    %c64 = llvm.call @__ocml_erf_f32(%c63) : (f32) -> f32
    %c65 = llvm.call @__ocml_erf_f32(%c64) : (f32) -> f32
    %c66 = llvm.call @__ocml_erf_f32(%c65) : (f32) -> f32
    %c67 = llvm.call @__ocml_erf_f32(%c66) : (f32) -> f32
    %c68 = llvm.call @__ocml_erf_f32(%c67) : (f32) -> f32
    %c69 = llvm.call @__ocml_erf_f32(%c68) : (f32) -> f32
    %c70 = llvm.call @__ocml_erf_f32(%c69) : (f32) -> f32
    %c71 = llvm.call @__ocml_erf_f32(%c70) : (f32) -> f32
    %c72 = llvm.call @__ocml_erf_f32(%c71) : (f32) -> f32
    %c73 = llvm.call @__ocml_erf_f32(%c72) : (f32) -> f32
    %c74 = llvm.call @__ocml_erf_f32(%c73) : (f32) -> f32
    %c75 = llvm.call @__ocml_erf_f32(%c74) : (f32) -> f32
    %c76 = llvm.call @__ocml_erf_f32(%c75) : (f32) -> f32
    %c77 = llvm.call @__ocml_erf_f32(%c76) : (f32) -> f32
    %c78 = llvm.call @__ocml_erf_f32(%c77) : (f32) -> f32
    %c79 = llvm.call @__ocml_erf_f32(%c78) : (f32) -> f32
    %c80 = llvm.call @__ocml_erf_f32(%c79) : (f32) -> f32
    %c81 = llvm.call @__ocml_erf_f32(%c80) : (f32) -> f32
    %c82 = llvm.call @__ocml_erf_f32(%c81) : (f32) -> f32
    %c83 = llvm.call @__ocml_erf_f32(%c82) : (f32) -> f32
    %c84 = llvm.call @__ocml_erf_f32(%c83) : (f32) -> f32
    %c85 = llvm.call @__ocml_erf_f32(%c84) : (f32) -> f32
    %c86 = llvm.call @__ocml_erf_f32(%c85) : (f32) -> f32
    %c87 = llvm.call @__ocml_erf_f32(%c86) : (f32) -> f32
    %c88 = llvm.call @__ocml_erf_f32(%c87) : (f32) -> f32
    %c89 = llvm.call @__ocml_erf_f32(%c88) : (f32) -> f32
    %c90 = llvm.call @__ocml_erf_f32(%c89) : (f32) -> f32
    %c91 = llvm.call @__ocml_erf_f32(%c90) : (f32) -> f32
    %c92 = llvm.call @__ocml_erf_f32(%c91) : (f32) -> f32
    %c93 = llvm.call @__ocml_erf_f32(%c92) : (f32) -> f32
    %c94 = llvm.call @__ocml_erf_f32(%c93) : (f32) -> f32
    %c95 = llvm.call @__ocml_erf_f32(%c94) : (f32) -> f32
    %c96 = llvm.call @__ocml_erf_f32(%c95) : (f32) -> f32
    %c97 = llvm.call @__ocml_erf_f32(%c96) : (f32) -> f32
    %c98 = llvm.call @__ocml_erf_f32(%c97) : (f32) -> f32
    %c99 = llvm.call @__ocml_erf_f32(%c98) : (f32) -> f32
    %c100 = llvm.call @__ocml_erf_f32(%c99) : (f32) -> f32
    %c101 = llvm.call @__ocml_erf_f32(%c100) : (f32) -> f32
    %c102 = llvm.call @__ocml_erf_f32(%c101) : (f32) -> f32
    %c103 = llvm.call @__ocml_erf_f32(%c102) : (f32) -> f32
    %c104 = llvm.call @__ocml_erf_f32(%c103) : (f32) -> f32
    %c105 = llvm.call @__ocml_erf_f32(%c104) : (f32) -> f32
    %c106 = llvm.call @__ocml_erf_f32(%c105) : (f32) -> f32
    %c107 = llvm.call @__ocml_erf_f32(%c106) : (f32) -> f32
    %c108 = llvm.call @__ocml_erf_f32(%c107) : (f32) -> f32
    %c109 = llvm.call @__ocml_erf_f32(%c108) : (f32) -> f32
    %c110 = llvm.call @__ocml_erf_f32(%c109) : (f32) -> f32
    %c111 = llvm.call @__ocml_erf_f32(%c110) : (f32) -> f32
    %c112 = llvm.call @__ocml_erf_f32(%c111) : (f32) -> f32
    %c113 = llvm.call @__ocml_erf_f32(%c112) : (f32) -> f32
    %c114 = llvm.call @__ocml_erf_f32(%c113) : (f32) -> f32
    %c115 = llvm.call @__ocml_erf_f32(%c114) : (f32) -> f32
    %c116 = llvm.call @__ocml_erf_f32(%c115) : (f32) -> f32
    %c117 = llvm.call @__ocml_erf_f32(%c116) : (f32) -> f32
    %c118 = llvm.call @__ocml_erf_f32(%c117) : (f32) -> f32
    %c119 = llvm.call @__ocml_erf_f32(%c118) : (f32) -> f32
    %c120 = llvm.call @__ocml_erf_f32(%c119) : (f32) -> f32
    %c121 = llvm.call @__ocml_erf_f32(%c120) : (f32) -> f32
    %c122 = llvm.call @__ocml_erf_f32(%c121) : (f32) -> f32
    %c123 = llvm.call @__ocml_erf_f32(%c122) : (f32) -> f32
    %c124 = llvm.call @__ocml_erf_f32(%c123) : (f32) -> f32
    %c125 = llvm.call @__ocml_erf_f32(%c124) : (f32) -> f32
    %c126 = llvm.call @__ocml_erf_f32(%c125) : (f32) -> f32
    %c127 = llvm.call @__ocml_erf_f32(%c126) : (f32) -> f32
    llvm.store %c127, %arg0 : f32, !llvm.ptr
    llvm.br ^bb1
  ^bb1:
    %sparseValue = llvm.load %arg0 : !llvm.ptr -> f32
    %sparseCall = llvm.call @__ocml_erf_f32(%sparseValue) : (f32) -> f32
    llvm.store %sparseCall, %arg0 : f32, !llvm.ptr
    llvm.return
  }
}
