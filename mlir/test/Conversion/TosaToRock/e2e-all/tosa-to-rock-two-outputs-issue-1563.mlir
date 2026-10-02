// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut mlir_dot_transpose_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_transpose_add --verifier clone - | rocmlir-driver -c -arch %arch | rocm-run | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]
module {
  func.func @mlir_dot_transpose_add(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x5xf32, 20x5x1>, %arg2: !migraphx.shaped<1x5x5xf32, 25x5x1>) -> (!migraphx.shaped<1x4x5xf32, 20x5x1>, !migraphx.shaped<1x5x4xf32, 20x4x1>) attributes {} {
    %0 = migraphx.dot %arg1, %arg2 : <1x4x5xf32, 20x5x1>, <1x5x5xf32, 25x5x1> -> <1x4x5xf32, 20x5x1>
    %1 = migraphx.transpose %0 {permutation = [0, 2, 1]} : <1x4x5xf32, 20x5x1> -> <1x5x4xf32, 20x1x5>
    %2 = migraphx.add %1, %arg0 : <1x5x4xf32, 20x1x5>, <1x5x4xf32, 20x4x1> -> <1x5x4xf32, 20x4x1>
    return %0, %2 : !migraphx.shaped<1x4x5xf32, 20x5x1>, !migraphx.shaped<1x5x4xf32, 20x4x1>
  }
}
