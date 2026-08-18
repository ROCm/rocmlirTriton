// Regression test for triton-patches/patch11295.patch.
//
// Here 64 lanes cover only half of the 128 K columns, so the deduced
// maxPhase = 16 swizzle reaches outside the warp. Check that it is clamped to
// 8 and that the load keeps its direct-to-LDS fast path.

// RUN: rocmlir-opt --tritonamdgpu-pipeline='use_async_copy=1' %s | FileCheck %s

// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK: ttg.async_copy_global_to_local {{.*}} -> <128x128xf32, #[[$SHARED]],

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [32, 32, 16], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @direct_to_lds_swizzle_clamp(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128x32xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<128x32xf32, #mma>
    %b = arith.constant dense<0.000000e+00> : tensor<128x32xbf16, #dotB>
    %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    %res = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %acc0) -> (tensor<128x32xf32, #mma>) : i32 {
      %a = tt.load %ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
      %aDot = ttg.convert_layout %a : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #dotA>
      %aBf16 = arith.truncf %aDot : tensor<128x128xf32, #dotA> to tensor<128x128xbf16, #dotA>
      %dot = tt.dot %aBf16, %b, %acc {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<128x128xbf16, #dotA> * tensor<128x32xbf16, #dotB> -> tensor<128x32xf32, #mma>
      scf.yield %dot : tensor<128x32xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %res : tensor<128x32xf32, #mma>
  }
}
