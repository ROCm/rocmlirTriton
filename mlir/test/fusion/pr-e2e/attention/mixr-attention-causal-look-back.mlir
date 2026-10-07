// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end numerics for a banded causal mask arriving through the MIGraphX
// path. The matcher turns the literal below into causal plus a causalLookBack
// width, so this checks that the band the frontend detects is the band the
// kernel actually computes, not just that the attribute gets attached.

// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s

module {
  // CHECK: [1 1 1]

  // The 8x8 literal keeps max(0, row - 3) <= col <= row, so L = 3. Rows 0 to 3
  // still reach key 0 and are indistinguishable from plain causal; from row 4
  // on the lower edge marches right with the diagonal, which is the part that
  // makes it a band and the part the per-element mask has to get right.
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x2x8x64xf16, 1024x512x64x1>, %arg1: !migraphx.shaped<1x14x8x64xf16, 7168x64x896x1>, %arg2: !migraphx.shaped<1x14x8x64xf16, 7168x512x64x1>) -> !migraphx.shaped<1x14x8x64xf16, 7168x512x64x1> attributes {rock.kernel} {
    %0 = migraphx.literal(dense<[[[[-0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00], [0xFC00, 0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00], [0xFC00, 0xFC00, 0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00], [0xFC00, 0xFC00, 0xFC00, 0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00]]]]> : tensor<1x1x8x8xf16>) : <1x1x8x8xf16, 64x64x8x1>
    %1 = migraphx.literal(dense<3.535160e-01> : tensor<1xf16>) : <1xf16, 0>
    %2 = migraphx.reshape %arg0 {dims = [1, 2, 1, 8, 64]} : <1x2x8x64xf16, 1024x512x64x1> -> <1x2x1x8x64xf16, 1024x512x512x64x1>
    %3 = migraphx.transpose %2 {permutation = [0, 1, 2, 4, 3]} : <1x2x1x8x64xf16, 1024x512x512x64x1> -> <1x2x1x64x8xf16, 1024x512x512x1x64>
    %4 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [1, 2, 7, 64, 8]} : <1x2x1x64x8xf16, 1024x512x512x1x64> -> <1x2x7x64x8xf16, 1024x512x0x1x64>
    %5 = migraphx.reshape %4 {dims = [1, 14, 64, 8]} : <1x2x7x64x8xf16, 1024x512x0x1x64> -> <1x14x64x8xf16, 7168x512x8x1>
    %6 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 14, 64, 8]} : <1xf16, 0> -> <1x14x64x8xf16, 0x0x0x0>
    %7 = migraphx.mul %5, %6 : <1x14x64x8xf16, 7168x512x8x1>, <1x14x64x8xf16, 0x0x0x0> -> <1x14x64x8xf16, 7168x512x8x1>
    %8 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 14, 8, 8]} : <1x1x8x8xf16, 64x64x8x1> -> <1x14x8x8xf16, 64x0x8x1>
    %9 = migraphx.dot %arg1, %7 : <1x14x8x64xf16, 7168x64x896x1>, <1x14x64x8xf16, 7168x512x8x1> -> <1x14x8x8xf16, 896x64x8x1>
    %10 = migraphx.add %9, %8 : <1x14x8x8xf16, 896x64x8x1>, <1x14x8x8xf16, 64x0x8x1> -> <1x14x8x8xf16, 896x64x8x1>
    %11 = migraphx.convert %10 {target_type = 2 : i64} : <1x14x8x8xf16, 896x64x8x1> to <1x14x8x8xf32, 896x64x8x1>
    %12 = migraphx.reshape %11 {dims = [1, 14, 8, 8]} : <1x14x8x8xf32, 896x64x8x1> -> <1x14x8x8xf32, 896x64x8x1>
    %13 = migraphx.reduce_max %12 {axes = [3]} : <1x14x8x8xf32, 896x64x8x1> -> <1x14x8x1xf32, 112x8x1x1>
    %14 = migraphx.reshape %13 {dims = [1, 14, 8, 1]} : <1x14x8x1xf32, 112x8x1x1> -> <1x14x8x1xf32, 112x8x1x1>
    %15 = migraphx.multibroadcast %14 {out_dyn_dims = [], out_lens = [1, 14, 8, 8]} : <1x14x8x1xf32, 112x8x1x1> -> <1x14x8x8xf32, 112x8x1x0>
    %16 = migraphx.sub %11, %15 : <1x14x8x8xf32, 896x64x8x1>, <1x14x8x8xf32, 112x8x1x0> -> <1x14x8x8xf32, 896x64x8x1>
    %17 = migraphx.exp %16 : <1x14x8x8xf32, 896x64x8x1> -> <1x14x8x8xf32, 896x64x8x1>
    %18 = migraphx.reshape %17 {dims = [1, 14, 8, 8]} : <1x14x8x8xf32, 896x64x8x1> -> <1x14x8x8xf32, 896x64x8x1>
    %19 = migraphx.reduce_sum %18 {axes = [3]} : <1x14x8x8xf32, 896x64x8x1> -> <1x14x8x1xf32, 112x8x1x1>
    %20 = migraphx.reshape %19 {dims = [1, 14, 8, 1]} : <1x14x8x1xf32, 112x8x1x1> -> <1x14x8x1xf32, 112x8x1x1>
    %21 = migraphx.multibroadcast %20 {out_dyn_dims = [], out_lens = [1, 14, 8, 8]} : <1x14x8x1xf32, 112x8x1x1> -> <1x14x8x8xf32, 112x8x1x0>
    %22 = migraphx.div %17, %21 : <1x14x8x8xf32, 896x64x8x1>, <1x14x8x8xf32, 112x8x1x0> -> <1x14x8x8xf32, 896x64x8x1>
    %23 = migraphx.convert %22 {target_type = 1 : i64} : <1x14x8x8xf32, 896x64x8x1> to <1x14x8x8xf16, 896x64x8x1>
    %24 = migraphx.dot %23, %arg2 : <1x14x8x8xf16, 896x64x8x1>, <1x14x8x64xf16, 7168x512x64x1> -> <1x14x8x64xf16, 7168x512x64x1>
    return %24 : !migraphx.shaped<1x14x8x64xf16, 7168x512x64x1>
  }
}
