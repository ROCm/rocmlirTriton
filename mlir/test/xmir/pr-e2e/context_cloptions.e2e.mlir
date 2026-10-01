// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-driver -host-pipeline highlevel %s | \
// RUN: rocmlir-opt -empty-tensor-to-alloc-tensor \
// RUN:   --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
// RUN:   -buffer-results-to-out-params="add-result-attr=false hoist-dynamic-allocs=false hoist-static-allocs=false modify-public-functions=true" | \
// RUN: rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut add - | \
// RUN: rocmlir-driver --host-pipeline=backend | \
// RUN: rocm-run | FileCheck %s


module {
// CHECK: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [65536] strides = [1] data =
  func.func @add(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32> {
    %9 = "tosa.add"(%arg0, %arg1)
     : (tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>
    return %9 : tensor<1x32x32x64xf32>
  }
}
