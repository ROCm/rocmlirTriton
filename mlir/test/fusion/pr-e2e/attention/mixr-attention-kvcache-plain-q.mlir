// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand_min_int 0 -rand_max_int 7 -rand_type_int_for_inputs=2 -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand_min_int 8 -rand_max_int 8 -rand_type_int_for_inputs=2 -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

// Decode-shaped GQA kv-cache attention as MIGraphX emits it, with Q a plain
// kernel argument and one last-valid index per batch. Q's matmul operand is
// then an expand_shape rather than a collapse_shape, so the per-batch index
// must be broadcast across the batch * heads attention groups from the
// attention batch itself; before that fix the kernel read the index out of
// bounds for every group after the first and heads 1-3 came out wrong.

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<2x2x8x4xf16, 64x32x4x1>, %arg1: !migraphx.shaped<2x4x1x4xf16, 16x4x4x1>, %arg2: !migraphx.shaped<2x1xsi32, 1x1>, %arg3: !migraphx.shaped<2x2x8x4xf16, 64x32x4x1>) -> !migraphx.shaped<2x1x16xf16, 16x16x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xsi32>) : <8xsi32, 1>
    %1 = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %2 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 1>
    %3 = migraphx.reshape %arg0 {dims = [2, 2, 1, 8, 4]} : <2x2x8x4xf16, 64x32x4x1> -> <2x2x1x8x4xf16, 64x32x32x4x1>
    %4 = migraphx.transpose %3 {permutation = [0, 1, 2, 4, 3]} : <2x2x1x8x4xf16, 64x32x32x4x1> -> <2x2x1x4x8xf16, 64x32x32x1x4>
    %5 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [2, 2, 2, 4, 8]} : <2x2x1x4x8xf16, 64x32x32x1x4> -> <2x2x2x4x8xf16, 64x32x0x1x4>
    %6 = migraphx.reshape %5 {dims = [2, 4, 4, 8]} : <2x2x2x4x8xf16, 64x32x0x1x4> -> <2x4x4x8xf16, 128x32x8x1>
    %7 = migraphx.reshape %arg3 {dims = [2, 2, 1, 8, 4]} : <2x2x8x4xf16, 64x32x4x1> -> <2x2x1x8x4xf16, 64x32x32x4x1>
    %8 = migraphx.multibroadcast %7 {out_dyn_dims = [], out_lens = [2, 2, 2, 8, 4]} : <2x2x1x8x4xf16, 64x32x32x4x1> -> <2x2x2x8x4xf16, 64x32x0x4x1>
    %9 = migraphx.reshape %8 {dims = [2, 4, 8, 4]} : <2x2x2x8x4xf16, 64x32x0x4x1> -> <2x4x8x4xf16, 128x32x4x1>
    %10 = migraphx.dot %arg1, %6 : <2x4x1x4xf16, 16x4x4x1>, <2x4x4x8xf16, 128x32x8x1> -> <2x4x1x8xf16, 32x8x8x1>
    %11 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [2, 4, 1, 8]} : <1xf16, 1> -> <2x4x1x8xf16, 0x0x0x0>
    %12 = migraphx.mul %10, %11 : <2x4x1x8xf16, 32x8x8x1>, <2x4x1x8xf16, 0x0x0x0> -> <2x4x1x8xf16, 32x8x8x1>
    %13 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 4, 1, 8]} : <8xsi32, 1> -> <2x4x1x8xsi32, 0x0x0x1>
    %14 = migraphx.reshape %arg2 {dims = [2, 1]} : <2x1xsi32, 1x1> -> <2x1xsi32, 1x1>
    %15 = migraphx.multibroadcast %14 {out_dyn_dims = [], out_lens = [2, 4]} : <2x1xsi32, 1x1> -> <2x4xsi32, 1x0>
    %16 = migraphx.reshape %15 {dims = [2, 4, 1, 1]} : <2x4xsi32, 1x0> -> <2x4x1x1xsi32, 1x0x1x1>
    %17 = migraphx.multibroadcast %16 {out_dyn_dims = [], out_lens = [2, 4, 1, 8]} : <2x4x1x1xsi32, 1x0x1x1> -> <2x4x1x8xsi32, 1x0x1x0>
    %18 = migraphx.greater %13, %17 : <2x4x1x8xsi32, 0x0x0x1>, <2x4x1x8xsi32, 1x0x1x0> -> <2x4x1x8xsi32, 8x0x8x1>
    %19 = migraphx.convert %18 {target_type = 0 : i64} : <2x4x1x8xsi32, 8x0x8x1> to <2x4x1x8xsi8, 8x0x8x1>
    %20 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [2, 4, 1, 8]} : <1xf16, 1> -> <2x4x1x8xf16, 0x0x0x0>
    %21 = migraphx.where %19, %20, %12 : <2x4x1x8xsi8, 8x0x8x1>, <2x4x1x8xf16, 0x0x0x0>, <2x4x1x8xf16, 32x8x8x1> -> <2x4x1x8xf16, 32x8x8x1>
    %22 = migraphx.reshape %21 {dims = [2, 4, 1, 8]} : <2x4x1x8xf16, 32x8x8x1> -> <2x4x1x8xf16, 32x8x8x1>
    %23 = migraphx.reduce_max %22 {axes = [3]} : <2x4x1x8xf16, 32x8x8x1> -> <2x4x1x1xf16, 4x1x1x1>
    %24 = migraphx.reshape %23 {dims = [2, 4, 1, 1]} : <2x4x1x1xf16, 4x1x1x1> -> <2x4x1x1xf16, 4x1x1x1>
    %25 = migraphx.multibroadcast %24 {out_dyn_dims = [], out_lens = [2, 4, 1, 8]} : <2x4x1x1xf16, 4x1x1x1> -> <2x4x1x8xf16, 4x1x1x0>
    %26 = migraphx.sub %21, %25 : <2x4x1x8xf16, 32x8x8x1>, <2x4x1x8xf16, 4x1x1x0> -> <2x4x1x8xf16, 32x8x8x1>
    %27 = migraphx.exp %26 : <2x4x1x8xf16, 32x8x8x1> -> <2x4x1x8xf16, 32x8x8x1>
    %28 = migraphx.reshape %27 {dims = [2, 4, 1, 8]} : <2x4x1x8xf16, 32x8x8x1> -> <2x4x1x8xf16, 32x8x8x1>
    %29 = migraphx.reduce_sum %28 {axes = [3]} : <2x4x1x8xf16, 32x8x8x1> -> <2x4x1x1xf16, 4x1x1x1>
    %30 = migraphx.reshape %29 {dims = [2, 4, 1, 1]} : <2x4x1x1xf16, 4x1x1x1> -> <2x4x1x1xf16, 4x1x1x1>
    %31 = migraphx.multibroadcast %30 {out_dyn_dims = [], out_lens = [2, 4, 1, 8]} : <2x4x1x1xf16, 4x1x1x1> -> <2x4x1x8xf16, 4x1x1x0>
    %32 = migraphx.div %27, %31 : <2x4x1x8xf16, 32x8x8x1>, <2x4x1x8xf16, 4x1x1x0> -> <2x4x1x8xf16, 32x8x8x1>
    %33 = migraphx.dot %32, %9 : <2x4x1x8xf16, 32x8x8x1>, <2x4x8x4xf16, 128x32x4x1> -> <2x4x1x4xf16, 16x4x4x1>
    %34 = migraphx.transpose %33 {permutation = [0, 2, 1, 3]} : <2x4x1x4xf16, 16x4x4x1> -> <2x1x4x4xf16, 16x4x4x1>
    %35 = migraphx.reshape %34 {dims = [2, 1, 16]} : <2x1x4x4xf16, 16x4x4x1> -> <2x1x16xf16, 16x16x1>
    return %35 : !migraphx.shaped<2x1x16xf16, 16x16x1>
  }
}
