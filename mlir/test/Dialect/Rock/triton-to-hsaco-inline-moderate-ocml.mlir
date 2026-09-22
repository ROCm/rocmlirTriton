// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: env LLVM_IR_ENABLE_DUMP=1 rocmlir-opt \
// RUN:   -triton-to-hsaco='arch=gfx1200' %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

// The 128 calls are split evenly across two basic blocks. Apply the
// high-duplication threshold to each block independently and keep the normal
// inlining path for both moderate-sized fusion regions.
//
// CHECK: // -----// LLVM IR Dump //----- //
// CHECK-NOT: call fastcc float @__ocml_erf_f32
// CHECK-NOT: define internal {{.*}} @__ocml_erf_f32
// CHECK: attributes #

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
    llvm.br ^bb1
  ^bb1:
    %value2 = llvm.load %arg0 : !llvm.ptr -> f32
    %d00 = llvm.call @__ocml_erf_f32(%value2) : (f32) -> f32
    %d01 = llvm.call @__ocml_erf_f32(%d00) : (f32) -> f32
    %d02 = llvm.call @__ocml_erf_f32(%d01) : (f32) -> f32
    %d03 = llvm.call @__ocml_erf_f32(%d02) : (f32) -> f32
    %d04 = llvm.call @__ocml_erf_f32(%d03) : (f32) -> f32
    %d05 = llvm.call @__ocml_erf_f32(%d04) : (f32) -> f32
    %d06 = llvm.call @__ocml_erf_f32(%d05) : (f32) -> f32
    %d07 = llvm.call @__ocml_erf_f32(%d06) : (f32) -> f32
    %d08 = llvm.call @__ocml_erf_f32(%d07) : (f32) -> f32
    %d09 = llvm.call @__ocml_erf_f32(%d08) : (f32) -> f32
    %d10 = llvm.call @__ocml_erf_f32(%d09) : (f32) -> f32
    %d11 = llvm.call @__ocml_erf_f32(%d10) : (f32) -> f32
    %d12 = llvm.call @__ocml_erf_f32(%d11) : (f32) -> f32
    %d13 = llvm.call @__ocml_erf_f32(%d12) : (f32) -> f32
    %d14 = llvm.call @__ocml_erf_f32(%d13) : (f32) -> f32
    %d15 = llvm.call @__ocml_erf_f32(%d14) : (f32) -> f32
    %d16 = llvm.call @__ocml_erf_f32(%d15) : (f32) -> f32
    %d17 = llvm.call @__ocml_erf_f32(%d16) : (f32) -> f32
    %d18 = llvm.call @__ocml_erf_f32(%d17) : (f32) -> f32
    %d19 = llvm.call @__ocml_erf_f32(%d18) : (f32) -> f32
    %d20 = llvm.call @__ocml_erf_f32(%d19) : (f32) -> f32
    %d21 = llvm.call @__ocml_erf_f32(%d20) : (f32) -> f32
    %d22 = llvm.call @__ocml_erf_f32(%d21) : (f32) -> f32
    %d23 = llvm.call @__ocml_erf_f32(%d22) : (f32) -> f32
    %d24 = llvm.call @__ocml_erf_f32(%d23) : (f32) -> f32
    %d25 = llvm.call @__ocml_erf_f32(%d24) : (f32) -> f32
    %d26 = llvm.call @__ocml_erf_f32(%d25) : (f32) -> f32
    %d27 = llvm.call @__ocml_erf_f32(%d26) : (f32) -> f32
    %d28 = llvm.call @__ocml_erf_f32(%d27) : (f32) -> f32
    %d29 = llvm.call @__ocml_erf_f32(%d28) : (f32) -> f32
    %d30 = llvm.call @__ocml_erf_f32(%d29) : (f32) -> f32
    %d31 = llvm.call @__ocml_erf_f32(%d30) : (f32) -> f32
    %d32 = llvm.call @__ocml_erf_f32(%d31) : (f32) -> f32
    %d33 = llvm.call @__ocml_erf_f32(%d32) : (f32) -> f32
    %d34 = llvm.call @__ocml_erf_f32(%d33) : (f32) -> f32
    %d35 = llvm.call @__ocml_erf_f32(%d34) : (f32) -> f32
    %d36 = llvm.call @__ocml_erf_f32(%d35) : (f32) -> f32
    %d37 = llvm.call @__ocml_erf_f32(%d36) : (f32) -> f32
    %d38 = llvm.call @__ocml_erf_f32(%d37) : (f32) -> f32
    %d39 = llvm.call @__ocml_erf_f32(%d38) : (f32) -> f32
    %d40 = llvm.call @__ocml_erf_f32(%d39) : (f32) -> f32
    %d41 = llvm.call @__ocml_erf_f32(%d40) : (f32) -> f32
    %d42 = llvm.call @__ocml_erf_f32(%d41) : (f32) -> f32
    %d43 = llvm.call @__ocml_erf_f32(%d42) : (f32) -> f32
    %d44 = llvm.call @__ocml_erf_f32(%d43) : (f32) -> f32
    %d45 = llvm.call @__ocml_erf_f32(%d44) : (f32) -> f32
    %d46 = llvm.call @__ocml_erf_f32(%d45) : (f32) -> f32
    %d47 = llvm.call @__ocml_erf_f32(%d46) : (f32) -> f32
    %d48 = llvm.call @__ocml_erf_f32(%d47) : (f32) -> f32
    %d49 = llvm.call @__ocml_erf_f32(%d48) : (f32) -> f32
    %d50 = llvm.call @__ocml_erf_f32(%d49) : (f32) -> f32
    %d51 = llvm.call @__ocml_erf_f32(%d50) : (f32) -> f32
    %d52 = llvm.call @__ocml_erf_f32(%d51) : (f32) -> f32
    %d53 = llvm.call @__ocml_erf_f32(%d52) : (f32) -> f32
    %d54 = llvm.call @__ocml_erf_f32(%d53) : (f32) -> f32
    %d55 = llvm.call @__ocml_erf_f32(%d54) : (f32) -> f32
    %d56 = llvm.call @__ocml_erf_f32(%d55) : (f32) -> f32
    %d57 = llvm.call @__ocml_erf_f32(%d56) : (f32) -> f32
    %d58 = llvm.call @__ocml_erf_f32(%d57) : (f32) -> f32
    %d59 = llvm.call @__ocml_erf_f32(%d58) : (f32) -> f32
    %d60 = llvm.call @__ocml_erf_f32(%d59) : (f32) -> f32
    %d61 = llvm.call @__ocml_erf_f32(%d60) : (f32) -> f32
    %d62 = llvm.call @__ocml_erf_f32(%d61) : (f32) -> f32
    %d63 = llvm.call @__ocml_erf_f32(%d62) : (f32) -> f32
    llvm.store %d63, %arg0 : f32, !llvm.ptr
    llvm.return
  }
}
