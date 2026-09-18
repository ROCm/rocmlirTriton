// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s --implicit-check-not='splitKFactor={{([2-9]|[1-9][0-9]+)}}'
// CHECK: splitKFactor=1,

module {
  func.func @conv_add_relu(%input: !migraphx.shaped<1x3x32x32xf32, 3072x1024x32x1>,
                           %filter: !migraphx.shaped<1x3x5x5xf32, 75x25x5x1>,
                           %bias: !migraphx.shaped<1x1x1x1xf32, 1x1x1x1>)
      -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
      attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = migraphx.convolution %input, %filter {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x3x32x32xf32, 3072x1024x32x1>, <1x3x5x5xf32, 75x25x5x1> -> <1x1x28x28xf32, 784x784x28x1>
    %1 = migraphx.multibroadcast %bias {out_dyn_dims = [], out_lens = [1, 1, 28, 28]} : <1x1x1x1xf32, 1x1x1x1> -> <1x1x28x28xf32, 1x1x0x0>
    %2 = migraphx.add %0, %1 : <1x1x28x28xf32, 784x784x28x1>, <1x1x28x28xf32, 1x1x0x0> -> <1x1x28x28xf32, 784x784x28x1>
    %3 = migraphx.relu %2 : <1x1x28x28xf32, 784x784x28x1> -> <1x1x28x28xf32, 784x784x28x1>
    return %3 : !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
  }
}
