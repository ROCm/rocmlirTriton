// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// gfx950's gemm+gemm quick list contains splitKFactor > 1; most other arches
// fall back to an attention list that does not, so we can not use %arch here.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx950 %s | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s
// CHECK: splitKFactor={{([2-9]|[1-9][0-9]+)}}

module {
  func.func private @mlir_gemm_gemm(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>,
                                    %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>)
                                    -> (!migraphx.shaped<1x7x3xf32, 21x3x1>) attributes {rock.kernel, rock.arch = "gfx950", rock.num_cu = 256 : i64} {
    %0 = migraphx.dot %arg0, %arg1 : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %1 = migraphx.dot %0, %arg2 : <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    return %1 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
