// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Relu is not a valid split-K output fusion, so the quick space must not
// contain any splitKFactor > 1. Pinned to gfx950 like
// quick-tuning-space-gemm-gemm-split-k.mlir: on an arch whose gemm+gemm quick
// list has no splitKFactor > 1 to begin with, this check would pass even if
// the filter were dropped.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx950 %s | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s --implicit-check-not='splitKFactor={{([2-9]|[1-9][0-9]+)}}'
// CHECK: splitKFactor=1,

module {
  func.func private @mlir_gemm_gemm_relu(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                         %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>,
                                         %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>)
                                         -> (!migraphx.shaped<1x7x3xf32, 21x3x1>) attributes {rock.kernel, rock.arch = "gfx950", rock.num_cu = 256 : i64} {
    %0 = migraphx.dot %arg0, %arg1 : <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %1 = migraphx.dot %0, %arg2 : <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    %2 = migraphx.relu %1 : <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    return %2 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
