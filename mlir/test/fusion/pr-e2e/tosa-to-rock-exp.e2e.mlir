// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// exp() amplifies conv rounding errors: exp(x+e)-exp(x) ~ e*exp(x), so the
// default f32 rtol (1.3e-6) is insufficient. 2e-5 covers the amplification
// factor on its own; arches that decompose the f32 dot into three bf16
// products start from a coarser conv result and need 7e-5.
// RUN: rocmlir-gen -fut test_fusion --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel | rocmlir-gen -ph -fut test_fusion -rand 1 -rand_type float %if bf16x3_f32_dot %{-rtol=7e-5%} %else %{-rtol=2e-5%} --verifier clone - | rocmlir-driver -c -arch %arch | rocm-run | FileCheck %s

module {
// CHECK: [1 1 1]
  func.func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>) -> tensor<128x30x30x128xf32> attributes {rock.kernel} {

    %zero = arith.constant dense<0.0> : tensor<128xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %zero, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<128x30x30x128xf32>
    %1 = "tosa.exp"(%0) : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>

    return %1 : tensor<128x30x30x128xf32>
  }

}
