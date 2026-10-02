// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut test_fusion --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel | rocmlir-gen -ph -print-results -fut test_fusion -rand none - | rocmlir-driver -c -arch %arch | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut test_fusion --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel | rocmlir-gen -ph -fut test_fusion -rand 1 -rand_type float --verifier clone - | rocmlir-driver -c -arch %arch | rocm-run | FileCheck %s --check-prefix=CLONE
// CHECK: Unranked Memref base
// CLONE: [1 1 1]
func.func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x3x3x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x30x30x16xf32> attributes {rock.kernel} {
  %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x30x30x16xf32>

  return %0 : tensor<1x30x30x16xf32>
}

// -----
