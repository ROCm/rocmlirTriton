// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// gfx950's conv f32 quick list contains splitKFactor > 1; gfx908/gfx90a do not,
// so we can not use %arch here.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch gfx950 %s | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s
// CHECK: splitKFactor={{([2-9]|[1-9][0-9]+)}}

module {
  func.func @conv(%input: !migraphx.shaped<1x3x32x32xf32, 3072x1024x32x1>,
                  %filter: !migraphx.shaped<1x3x5x5xf32, 75x25x5x1>)
      -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
      attributes {rock.arch = "gfx950", rock.kernel = "mixr"} {
    %0 = migraphx.convolution %input, %filter {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x3x32x32xf32, 3072x1024x32x1>, <1x3x5x5xf32, 75x25x5x1> -> <1x1x28x28xf32, 784x784x28x1>
    return %0 : !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
  }
}
