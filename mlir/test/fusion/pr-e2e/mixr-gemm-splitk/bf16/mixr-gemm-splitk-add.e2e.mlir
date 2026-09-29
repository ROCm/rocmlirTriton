// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut dot_splitk_add_trunc --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut dot_splitk_add_trunc --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut dot_splitk_add_trunc --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_splitk_add_trunc --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s --check-prefix=CLONE
module {
  // CHECK: [1 1 1]
  // CHECK:  [5,     5,     5,  5,     5,     5,  5,     5,     5,  5,     5,     5,  5,     5,     5]

  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base

  func.func @dot_splitk_add_trunc(%arg0: !migraphx.shaped<1x5x4xbf16, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xbf16, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xbf16, 15x3x1>) -> !migraphx.shaped<1x5x3xbf16, 15x3x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 {perf_config="gemm:v1:16,32,4,1,1,4,16,2,1,0,0"} : <1x5x4xbf16, 20x4x1>, <1x4x3xbf16, 12x3x1> -> <1x5x3xbf16, 15x3x1>
    %2 = migraphx.add %0, %arg2 {} : <1x5x3xbf16, 15x3x1>, <1x5x3xbf16, 15x3x1> -> <1x5x3xbf16, 15x3x1>
    return %2 : !migraphx.shaped<1x5x3xbf16, 15x3x1>
  }
}
