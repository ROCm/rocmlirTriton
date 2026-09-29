// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// MLIR#765: TinyBERT partition 2
// RUN: rocmlir-gen -fut tinybert_part_2 -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut tinybert_part_2 --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @tinybert_part_2(%arg0: tensor<2x2x128x128xf32>, %arg1: tensor<2x2x128x64xf32>) -> (tensor<2x2x128x64xf32>) {
    %_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %_s0 = "tosa.const_shape"() {values = dense<[4, 128, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %_s1 = "tosa.const_shape"() {values = dense<[4, 128, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %_s2 = "tosa.const_shape"() {values = dense<[2, 2, 128, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = "tosa.reshape"(%arg1, %_s0) : (tensor<2x2x128x64xf32>, !tosa.shape<3>) -> tensor<4x128x64xf32>
    %1 = "tosa.reshape"(%arg0, %_s1) : (tensor<2x2x128x128xf32>, !tosa.shape<3>) -> tensor<4x128x128xf32>
    %2 = "tosa.matmul"(%1, %0, %_zp, %_zp) {acc_type = f32} : (tensor<4x128x128xf32>, tensor<4x128x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x128x64xf32>
    %3 = "tosa.reshape"(%2, %_s2) : (tensor<4x128x64xf32>, !tosa.shape<4>) -> tensor<2x2x128x64xf32>
    return %3 : tensor<2x2x128x64xf32>
  }
}
