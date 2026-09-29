// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// MLIR#765: TinyBERT partition 4
// RUN: rocmlir-gen -fut tinybert_part_4 -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut tinybert_part_4 --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK-COUNT-2: [1 1 1]
module {
  func.func @tinybert_part_4(%arg0: tensor<2x128x128xf32>, %arg1: tensor<1x128x512xf32>, %arg2: tensor<1x1x512xf32>) -> (tensor<2x128x512xf32>, tensor<2x128x512xf32>) {
    %_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %_s0 = "tosa.const_shape"() {values = dense<[1, 256, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %_s1 = "tosa.const_shape"() {values = dense<[2, 128, 512]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %0 = "tosa.reshape"(%arg0, %_s0) : (tensor<2x128x128xf32>, !tosa.shape<3>) -> tensor<1x256x128xf32>
    %1 = "tosa.matmul"(%0, %arg1, %_zp, %_zp) {acc_type = f32} : (tensor<1x256x128xf32>, tensor<1x128x512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x512xf32>
    %2 = "tosa.reshape"(%1, %_s1) : (tensor<1x256x512xf32>, !tosa.shape<3>) -> tensor<2x128x512xf32>
    %3 = "tosa.add"(%2, %arg2) : (tensor<2x128x512xf32>, tensor<1x1x512xf32>) -> tensor<2x128x512xf32>
    %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
    %5 = "tosa.sub"(%3, %4) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
    return %3, %5 : tensor<2x128x512xf32>, tensor<2x128x512xf32>
  }
}
