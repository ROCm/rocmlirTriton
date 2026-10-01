// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut dot_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_add --verifier clone - | rocmlir-driver -c -arch %arch | rocm-run | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

func.func private @dot_add(%arg0: tensor<1x128x64xf16>, %arg1: tensor<1x64x256xf16>) -> tensor<1x1x256xf16> attributes {rock.kernel} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32} : (tensor<1x128x64xf16>, tensor<1x64x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x128x256xf16>
  %1 = "tosa.reduce_sum"(%0) {axis = 1 : i32} : (tensor<1x128x256xf16>) -> tensor<1x1x256xf16>
  return %1 : tensor<1x1x256xf16>
}
