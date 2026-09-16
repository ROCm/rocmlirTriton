// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut conv_gemm_splitk_intergemm_exp --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut conv_gemm_splitk_intergemm_exp --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

// The conv+gemm counterpart of the non-zero-preserving inter-gemm body. The
// convolution's output space is padded up to the tile size before being split,
// and exp turns the zeros in the padded lanes into ones, so those lanes must be
// selected back to zero before the second GEMM accumulates them.
// RUN: rocmlir-gen -fut conv_gemm_splitk_intergemm_exp --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=SPLITK
// SPLITK: splitKFactor = 4
module {
  func.func @conv_gemm_splitk_intergemm_exp(%arg0: !migraphx.shaped<2x8x8x16xf32, 1024x8x1x64>, %arg1: !migraphx.shaped<16x16x3x3xf32, 144x1x48x16>, %arg2: !migraphx.shaped<1x16x32xf32, 0x1x0>, %arg3: !migraphx.shaped<1x128x16xf32, 2048x16x1>) -> !migraphx.shaped<1x128x32xf32, 4096x32x1> attributes {rock.kernel} {
    %transposed = migraphx.transpose %arg0 {permutation = [0, 3, 1, 2]} : <2x8x8x16xf32, 1024x8x1x64> -> <2x16x8x8xf32, 1024x64x8x1>
    %1 = migraphx.convolution %transposed, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x16x8x8xf32, 1024x64x8x1>, <16x16x3x3xf32, 144x1x48x16> -> <2x16x8x8xf32, 2048x1x128x16>
    %2 = migraphx.transpose %1 {permutation = [0, 2, 3, 1]} : <2x16x8x8xf32, 2048x1x128x16> -> <2x8x8x16xf32, 2048x128x16x1>
    %3 = migraphx.reshape %2 {dims = [1, 128, 16]} : <2x8x8x16xf32, 2048x128x16x1> -> <1x128x16xf32, 2048x16x1>
    %scaled = migraphx.mul %3, %arg3 : <1x128x16xf32, 2048x16x1>, <1x128x16xf32, 2048x16x1> -> <1x128x16xf32, 2048x16x1>
    // The convolution accumulates 16*3*3 products, so exp of that overflows
    // f32's usable range and the comparison against the reference drowns in
    // rounding error. Scaling the exponent down keeps the kernel numerically
    // comparable; exp(0) is still 1, which is all this test needs.
    %cst = migraphx.literal(dense<1.562500e-02> : tensor<1x128x16xf32>) : <1x128x16xf32, 0x0x0>
    %tamed = migraphx.mul %scaled, %cst : <1x128x16xf32, 2048x16x1>, <1x128x16xf32, 0x0x0> -> <1x128x16xf32, 2048x16x1>
    %exp = migraphx.exp %tamed : <1x128x16xf32, 2048x16x1> -> <1x128x16xf32, 2048x16x1>
    %4 = migraphx.dot %exp, %arg2 {perf_config="attn:mPerBlockG0=128,nPerBlockG0=64,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : <1x128x16xf32, 2048x16x1>, <1x16x32xf32, 0x1x0> -> <1x128x32xf32, 4096x32x1>
    return %4 : !migraphx.shaped<1x128x32xf32, 4096x32x1>
  }
}
