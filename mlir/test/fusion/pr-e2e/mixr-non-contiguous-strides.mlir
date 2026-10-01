// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut mlir_dot_sigmoid --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_sigmoid --verifier clone - | rocmlir-driver -c |  rocm-run | FileCheck %s

// Only half of the results will be correct since the non-contiguous strides
// in this example means that half of the memory is uninitialized.
// CHECK: zero_diff: {{230[0-9]}}/4608
// CHECK: [0 0 0]

module {
  func.func @mlir_dot_sigmoid(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x24x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
    %1 = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x24x1>
    return %1 : !migraphx.shaped<4x24x24xf16, 1152x24x1>
  }
}
