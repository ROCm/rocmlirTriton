// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut dot_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_add %bf16_reduce_rtol_flag --verifier clone - | rocmlir-driver -c -arch %arch | rocm-run | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

func.func private @dot_add(%arg0: tensor<1x128x64xbf16>, %arg1: tensor<1x64x256xbf16>, %arg2: tensor<1x128x256xbf16>) -> tensor<1x128x1xbf16> attributes {rock.kernel} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xbf16>}> : () -> tensor<1xbf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xbf16>}> : () -> tensor<1xbf16>
  %0 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32} : (tensor<1x128x64xbf16>, tensor<1x64x256xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x128x256xbf16>
  %1 = "tosa.add"(%0, %arg2) : (tensor<1x128x256xbf16>, tensor<1x128x256xbf16>) -> tensor<1x128x256xbf16>
  %2 = "tosa.reduce_sum"(%1) {axis = 2 : i32} : (tensor<1x128x256xbf16>) -> tensor<1x128x1xbf16>
  return %2 : tensor<1x128x1xbf16>
}
