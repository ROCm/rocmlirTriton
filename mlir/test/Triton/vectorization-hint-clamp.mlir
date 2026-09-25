// rock-transforms-to-pointer-arith stamps a `tt.contiguity`/`tt.divisibility`
// vectorization hint on the pointers it builds, derived from the global-tensor
// -> block-tile view only. The block-tile -> thread-tile mapping is Triton's
// choice and is invisible there, so the hint can claim a longer contiguous run
// than any single thread owns. It is safe anyway because
// ModuleAxisInfoAnalysis::getContiguity clamps it with getContigPerThread() of
// the final encoding. This test pins that clamp down.
//
// Both modules carry the same absurd hint (contiguity 1024) over divui/remui
// address math standing in for the flattened im2col coordinate chain. Only the
// encoding differs, and the load width follows getContigPerThread(), not the
// hint.

// RUN: rocmlir-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s

// Blocked encoding, 4 elements per thread on the fast axis: 1024 clamps to 4.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @hint_clamped_to_blocked_thread_tile
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<1> -> vector<4xf32>
  // CHECK-NOT: llvm.load
  tt.func public @hint_clamped_to_blocked_thread_tile(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<68> : tensor<8x32xi32, #blocked>
    %cst_0 = arith.constant dense<70> : tensor<8x32xi32, #blocked>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x32xi32, #blocked> -> tensor<8x32xi32, #blocked>
    %3 = arith.divui %2, %cst : tensor<8x32xi32, #blocked>
    %4 = arith.remui %2, %cst : tensor<8x32xi32, #blocked>
    %5 = arith.muli %3, %cst_0 : tensor<8x32xi32, #blocked>
    %6 = arith.addi %5, %4 : tensor<8x32xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %6 {tt.contiguity = dense<[1, 1024]> : tensor<2xi32>, tt.divisibility = dense<[1, 4096]> : tensor<2xi32>} : tensor<8x32x!tt.ptr<f32>, #blocked>, tensor<8x32xi32, #blocked>
    %9 = tt.load %8 : tensor<8x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// The case the hint cannot see coming: Triton bypasses LDS and loads straight
// into an MFMA dot-operand layout. f32 operands give kWidth 1, so a thread owns
// one element on the fast axis and 1024 clamps all the way to a scalar load.

#mfma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 2], isTransposed = false}>
#dotop = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: llvm.func @hint_clamped_to_mfma_thread_tile
  // CHECK: llvm.load %{{.*}} : !llvm.ptr<1> -> vector<1xf32>
  // CHECK-NOT: llvm.load
  tt.func public @hint_clamped_to_mfma_thread_tile(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<68> : tensor<32x2xi32, #dotop>
    %cst_0 = arith.constant dense<70> : tensor<32x2xi32, #dotop>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #dotop}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #dotop}>> -> tensor<1x2xi32, #dotop>
    %2 = tt.broadcast %1 : tensor<1x2xi32, #dotop> -> tensor<32x2xi32, #dotop>
    %3 = arith.divui %2, %cst : tensor<32x2xi32, #dotop>
    %4 = arith.remui %2, %cst : tensor<32x2xi32, #dotop>
    %5 = arith.muli %3, %cst_0 : tensor<32x2xi32, #dotop>
    %6 = arith.addi %5, %4 : tensor<32x2xi32, #dotop>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x2x!tt.ptr<f32>, #dotop>
    %8 = tt.addptr %7, %6 {tt.contiguity = dense<[1, 1024]> : tensor<2xi32>, tt.divisibility = dense<[1, 4096]> : tensor<2xi32>} : tensor<32x2x!tt.ptr<f32>, #dotop>, tensor<32x2xi32, #dotop>
    %9 = tt.load %8 : tensor<32x2x!tt.ptr<f32>, #dotop>
    tt.return
  }
}
