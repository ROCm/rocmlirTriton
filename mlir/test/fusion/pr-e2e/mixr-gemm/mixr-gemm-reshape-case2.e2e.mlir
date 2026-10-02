// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut dot_reshape_2 --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut dot_reshape_2 - | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut dot_reshape_2 --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_reshape_2 --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s --check-prefix=CLONE

module {
  // CHECK:  [4, 4, 4, 4, 4,  4, 4, 4, 4, 4,  4, 4, 4, 4, 4]
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  func.func @dot_reshape_2(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x3x5xf32, 15x5x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %2 = migraphx.reshape %0 {dims = [1:i64, 3:i64, 5:i64]} : <1x5x3xf32, 15x3x1> -> <1x3x5xf32, 15x5x1>
    return %2 : !migraphx.shaped<1x3x5xf32, 15x5x1>
  }
}
