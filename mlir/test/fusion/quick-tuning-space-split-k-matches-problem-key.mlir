// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Verify that `-supportsSplitK` in the problem key matches the selected quick
// tuning list. gfx1151 has distinct regular and no-split-K conv f32 lists.

// Relu cannot be applied to each split and then summed, so split-K is illegal.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=KEY-NO-SPLIT-K
// KEY-NO-SPLIT-K: -supportsSplitK false
//
// This config exists only in the no-split-K list, proving list selection rather
// than filtering the regular list.
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s --check-prefix=SPACE-NO-SPLIT-K --implicit-check-not='splitKFactor={{([2-9]|[1-9][0-9]+)}}'
// SPACE-NO-SPLIT-K: mPerBlock=256,nPerBlock=9,kPerBlock=2,{{.*}}matrixInstrNonkdim=0,splitKFactor=1,numStages=3

// Dropping the relu leaves a broadcast bias add, which is a legal split-K
// output fusion.
// RUN: sed -e '/migraphx.relu/d' -e 's/return %3 :/return %2 :/' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=KEY-SPLIT-K
// KEY-SPLIT-K: -supportsSplitK true
// RUN: sed -e '/migraphx.relu/d' -e 's/return %3 :/return %2 :/' %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s --check-prefix=SPACE-SPLIT-K
// SPACE-SPLIT-K: splitKFactor={{([2-9]|[1-9][0-9]+)}}

module {
  func.func @conv_add_relu(%input: !migraphx.shaped<1x3x32x32xf32, 3072x1024x32x1>,
                           %filter: !migraphx.shaped<1x3x5x5xf32, 75x25x5x1>,
                           %bias: !migraphx.shaped<1x1x1x1xf32, 1x1x1x1>)
      -> !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
      attributes {rock.arch = "gfx1151", rock.kernel = "mixr"} {
    %0 = migraphx.convolution %input, %filter {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x3x32x32xf32, 3072x1024x32x1>, <1x3x5x5xf32, 75x25x5x1> -> <1x1x28x28xf32, 784x784x28x1>
    %1 = migraphx.multibroadcast %bias {out_dyn_dims = [], out_lens = [1, 1, 28, 28]} : <1x1x1x1xf32, 1x1x1x1> -> <1x1x28x28xf32, 1x1x0x0>
    %2 = migraphx.add %0, %1 : <1x1x28x28xf32, 784x784x28x1>, <1x1x28x28xf32, 1x1x0x0> -> <1x1x28x28xf32, 784x784x28x1>
    %3 = migraphx.relu %2 : <1x1x28x28xf32, 784x784x28x1> -> <1x1x28x28xf32, 784x784x28x1>
    return %3 : !migraphx.shaped<1x1x28x28xf32, 784x784x28x1>
  }
}
