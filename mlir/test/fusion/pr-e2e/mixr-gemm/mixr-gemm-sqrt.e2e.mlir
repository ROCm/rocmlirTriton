// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut dot_sqrt --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut dot_sqrt - | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut dot_sqrt --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_sqrt --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s --check-prefix=CLONE

module {
  // dot of all-ones 1x5x4 * 1x4x3 = 4.0, sqrt(4.0) = 2.0
  // CHECK:  [2,     2,     2,  2,     2,     2,  2,     2,     2,  2,     2,     2,  2,     2,     2]

  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base

  func.func @dot_sqrt(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %1 = migraphx.sqrt %0 : <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    return %1 : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }
}
