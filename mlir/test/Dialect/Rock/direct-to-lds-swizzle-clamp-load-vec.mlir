// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Regression test for triton-patches/patch-direct-to-lds-swizzle-check-vec.patch.
//
// The clamp from patch11295.patch measures the lane shuffle with the shared
// encoding's vec, but the lowering divides the offset delta by the load's
// vector, which the source contiguity and the target's LDS write widths cap.
// Both tiles below reach outside the warp with a load vector of one f32 and
// were left unclamped before that mismatch was fixed.

// RUN: rocmlir-opt --tritonamdgpu-pipeline='use_async_copy=1' --split-input-file %s | FileCheck %s

// Four warps over 64x128, one f32 per thread, so 64 lanes cover half the K
// columns and maxPhase = 16 has to halve once.

// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK-LABEL: @swizzle_clamped_one_elem_per_lane
// CHECK: ttg.async_copy_global_to_local {{.*}} -> <64x128xf32, #[[$SHARED]],

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @swizzle_clamped_one_elem_per_lane(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<64x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #dotB>
    %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked>
    %res = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x64xf32, #mma>) : i32 {
      %a = tt.load %ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f32>, #blocked>
      %aDot = ttg.convert_layout %a : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dotA>
      %aBf16 = arith.truncf %aDot : tensor<64x128xf32, #dotA> to tensor<64x128xbf16, #dotA>
      %dot = tt.dot %aBf16, %b, %acc {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xbf16, #dotA> * tensor<128x64xbf16, #dotB> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// Two f32 per thread, which CDNA4 cannot write to LDS in one go, so the load
// vector is still one element and maxPhase = 16 has to halve twice.

// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: @swizzle_clamped_two_elems_per_lane
// CHECK: ttg.async_copy_global_to_local {{.*}} -> <64x128xf32, #[[$SHARED]],

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @swizzle_clamped_two_elems_per_lane(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<64x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #dotB>
    %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked>
    %res = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x64xf32, #mma>) : i32 {
      %a = tt.load %ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f32>, #blocked>
      %aDot = ttg.convert_layout %a : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dotA>
      %aBf16 = arith.truncf %aDot : tensor<64x128xf32, #dotA> to tensor<64x128xbf16, #dotA>
      %dot = tt.dot %aBf16, %b, %acc {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xbf16, #dotA> * tensor<128x64xbf16, #dotB> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}
