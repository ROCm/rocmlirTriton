// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand 3 - | rocmlir-driver -arch %arch -c | rocm-run | FileCheck %s
module {
  // CHECK:  [-1, 1, 1, -1, 1, -1, -1, 1, -1, {{.*}}, -1, -1, 0, -1, -1]
  // NOTE: the missing number is -1.07288e-06 in navi instead of 0
  func.func @dot_add(%arg0: !migraphx.shaped<1x5x4xf16, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf16, 12x3x1>) -> !migraphx.shaped<1x5x3xf16, 15x3x1> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf16, 20x4x1>, <1x4x3xf16, 12x3x1> -> <1x5x3xf16, 15x3x1>
    %2 = migraphx.erf %0 : <1x5x3xf16, 15x3x1> -> <1x5x3xf16, 15x3x1>
    return %2 : !migraphx.shaped<1x5x3xf16, 15x3x1>
  }
}
