// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-epilogue | FileCheck %s

// CHECK-LABEL: one_op_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @one_op_in_chain(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = arith.truncf %1 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: store_dword_mfma32_small_n
// CHECK-NOT: tensor<32x8x!tt.ptr<f16>, #linear>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x8x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_mfma32_small_n(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x8xf32, #mma>
    %0 = ttg.convert_layout %cst : tensor<32x8xf32, #mma> -> tensor<32x8xf32, #blocked>
    %1 = arith.truncf %0 : tensor<32x8xf32, #blocked> to tensor<32x8xf16, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x8x!tt.ptr<f16>, #blocked>
    tt.store %2, %1 : tensor<32x8x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: two_ops_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @two_ops_in_chain(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = math.exp2 %1 : tensor<32x32xf32, #blocked>
    %3 = arith.truncf %2 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.store %4, %3 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [64, 0]], block = []}>
// CHECK-LABEL: store_dword_128x128
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_128x128(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 128], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [0, 64], [32, 0]], block = []}>
// CHECK-LABEL: store_dword_256x256
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<256x256x!tt.ptr<f16>, #mma> -> tensor<256x256x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<256x256x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_256x256(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<256x256xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<256x256xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<256x256xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<256x256xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
    %2 = arith.truncf %1 : tensor<256x256xf32, #blocked> to tensor<256x256xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x256x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<256x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// To validate if  warpsPerCTA is not expected, no linear layout will be created.
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: #linear
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// To validate if N of the input shape is not expected, larger or equal 16X2, no linear layout will be created.
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: #linear
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    %2 = arith.truncf %1 : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// FMA dot layout cases.

// Both contiguous sizePerThread are identical: bypass is profitable.
// CHECK-LABEL: fma_blocked_equal_contig
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128x!tt.ptr<f32>, #blocked1> -> tensor<64x128x!tt.ptr<f32>, #blocked>
// CHECK:       tt.store %[[PTR]], %{{.*}} : tensor<64x128x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_equal_contig(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<16x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x128xf32, #blocked>
    %1 = ttg.convert_layout %0 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %2, %1 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Source sizePerThread is bigger than the store layout's: bypass is profitable.
// CHECK-LABEL: fma_blocked_wider_contig
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %{{.*}} : tensor<64x128x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_wider_contig(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Source sizePerThread is smaller than the store layout's: bypass is not profitable.
// CHECK-LABEL: fma_blocked_narrower_contig
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<64x128x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_narrower_contig(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// The two layouts disagree on the order, so bypass is not profitable.
// CHECK-LABEL: fma_blocked_order_mismatch
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<64x128x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_order_mismatch(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// The source keeps a single lane on the contiguous dimension, so consecutive
// lanes do not continue each other's run and the store would not be coalesced,
// even though the contiguous sizePerThread match.
// CHECK-LABEL: fma_blocked_lanes_not_contig
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<64x128x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_lanes_not_contig(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// The source sizePerThread is smaller than the store layout's, but both already
// fill the 128 bit store width, so neither needs more stores than the other and
// the bypass is still profitable.
// CHECK-LABEL: fma_blocked_narrower_but_store_width_bound
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #blocked1>
// CHECK:       tt.store %{{.*}}, %{{.*}} : tensor<64x128x!tt.ptr<f16>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_narrower_but_store_width_bound(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// ttg.convert_layout only requires a ranked tensor, so its source may carry no
// encoding at all. Such a source has no layout to store in and must be left
// alone rather than inspected.
// CHECK-LABEL: unencoded_source
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128xf32> -> tensor<64x128xf32, #blocked>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<64x128x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unencoded_source(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32> -> tensor<64x128xf32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A tensor whose store dimension holds a single element cannot deliver the
// contiguous run its sizePerThread advertises, so the bypass must be rejected
// even though looking at sizePerThread alone would allow it.
// CHECK-LABEL: fma_blocked_tensor_narrower_than_tile
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<128x1x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_tensor_narrower_than_tile(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x1xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Elementwise DAG cases: the epilogue between the conversion and the store is
// not necessarily a chain.

// The accumulator is read twice, as it is in SiLU (`acc * sigmoid(acc)`) and in
// the GELU tanh approximation. The old linear walk could not express this and
// both hasOneUse checks rejected it, yet it costs nothing: the accumulator is
// one value in the accumulator layout, read more than once.
// CHECK-LABEL: reconvergent_accumulator
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK:       %[[EXP:.+]] = math.exp2 %[[ACC:.+]] : tensor<32x32xf32, #mma>
// CHECK:       %[[MUL:.+]] = arith.mulf %[[ACC]], %[[EXP]] : tensor<32x32xf32, #mma>
// CHECK:       tt.store %{{.*}}, %[[MUL]] : tensor<32x32x!tt.ptr<f32>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @reconvergent_accumulator(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = math.exp2 %1 : tensor<32x32xf32, #blocked>
    %3 = arith.mulf %1, %2 : tensor<32x32xf32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %4, %3 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A clamp against a splat constant, the shape of ReLU. Binary arith ops were
// absent from the old hand-maintained opcode list entirely, and the splat is
// rebuilt in the accumulator layout rather than converted: every thread holds
// the same scalar under any layout.
// CHECK-LABEL: clamp_against_splat_constant
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK-DAG:   %[[LIM:.+]] = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #mma>
// CHECK:       %[[MAX:.+]] = arith.maxnumf %{{.*}}, %[[LIM]] : tensor<32x32xf32, #mma>
// CHECK:       tt.store %{{.*}}, %[[MAX]] : tensor<32x32x!tt.ptr<f32>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @clamp_against_splat_constant(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lim = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #blocked>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = arith.maxnumf %1, %lim : tensor<32x32xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Alpha scaling by a runtime scalar. tt.splat is layout-agnostic for the same
// reason a splat constant is, so it is rebuilt rather than converted.
// CHECK-LABEL: scale_by_splatted_scalar
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK-DAG:   %[[ALPHA:.+]] = tt.splat %arg1 : f32 -> tensor<32x32xf32, #mma>
// CHECK:       %[[MUL:.+]] = arith.mulf %{{.*}}, %[[ALPHA]] : tensor<32x32xf32, #mma>
// CHECK:       tt.store %{{.*}}, %[[MUL]] : tensor<32x32x!tt.ptr<f32>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scale_by_splatted_scalar(%arg0: !tt.ptr<f32>, %arg1: f32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %alpha = tt.splat %arg1 : f32 -> tensor<32x32xf32, #blocked>
    %2 = arith.mulf %1, %alpha : tensor<32x32xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A bias add. The side operand comes from memory, so it is the load that is
// reissued in the accumulator layout: what was a conversion of the loaded data
// becomes a conversion of the pointer feeding it.
// CHECK-LABEL: side_load_bias
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK:       %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #mma>
// CHECK:       %[[BIAS:.+]] = tt.load %[[PTR]] : tensor<32x32x!tt.ptr<f32>, #mma>
// CHECK:       %[[SUM:.+]] = arith.addf %{{.*}}, %[[BIAS]] : tensor<32x32xf32, #mma>
// CHECK:       tt.store %{{.*}}, %[[SUM]] : tensor<32x32x!tt.ptr<f32>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @side_load_bias(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %biasptr = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %bias = tt.load %biasptr : tensor<32x32x!tt.ptr<f32>, #blocked>
    %2 = arith.addf %1, %bias : tensor<32x32xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A per-channel bias, which reaches the epilogue as a broadcast of an Mx1
// load rather than as a full-tile one. `tt.broadcast` is not elementwise, so
// halting the cone at it would put an unclassifiable value on the boundary and
// refuse the whole epilogue; it relayouts like an elementwise op instead, and
// the Mx1 load is the one reissued in the accumulator layout.
// CHECK-LABEL: side_load_broadcast_bias
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK:       %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x1x!tt.ptr<f32>, #mma>
// CHECK:       %[[BIAS:.+]] = tt.load %[[PTR]] : tensor<32x1x!tt.ptr<f32>, #mma>
// CHECK:       %[[BCAST:.+]] = tt.broadcast %[[BIAS]] : tensor<32x1xf32, #mma> -> tensor<32x32xf32, #mma>
// CHECK:       %[[SUM:.+]] = arith.addf %{{.*}}, %[[BCAST]] : tensor<32x32xf32, #mma>
// CHECK:       tt.store %{{.*}}, %[[SUM]] : tensor<32x32x!tt.ptr<f32>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @side_load_broadcast_bias(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %biasptr = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %bias = tt.load %biasptr : tensor<32x1x!tt.ptr<f32>, #blocked>
    %bcast = tt.broadcast %bias : tensor<32x1xf32, #blocked> -> tensor<32x32xf32, #blocked>
    %2 = arith.addf %1, %bcast : tensor<32x32xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// The epilogue value feeds two stores, which is how a kernel with several
// results reads one accumulator. Neither store can move alone, since the one
// left behind would be reading a value whose layout just changed, so both are
// rewritten together and the round trip goes away for both.
// CHECK-LABEL: two_stores_move_together
//   CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
//       CHECK:   ttg.convert_layout %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #mma>
//       CHECK:   tt.store %{{.*}} : tensor<32x32x!tt.ptr<f32>, #mma>
//       CHECK:   ttg.convert_layout %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #mma>
//       CHECK:   tt.store %{{.*}} : tensor<32x32x!tt.ptr<f32>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @two_stores_move_together(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = math.exp2 %1 : tensor<32x32xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %4, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// The second store consumes the epilogue as a *mask*, not as a value, so it is
// not a store this rewrite can carry along and the round trip has to stay.
// CHECK-LABEL: epilogue_used_as_mask
// CHECK:       ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK:       tt.store
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @epilogue_used_as_mask(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %zero = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = math.exp2 %1 : tensor<32x32xf32, #blocked>
    %m = arith.cmpf ogt, %2, %zero : tensor<32x32xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %4, %2, %m : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A non-splat constant would need its per-element values permuted into the
// accumulator layout, which this pattern does not build, so it bails.
// CHECK-LABEL: non_splat_constant
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<2x2xf32, #mma> -> tensor<2x2xf32, #blocked>
// CHECK:       arith.mulf %[[VAL]], %{{.*}} : tensor<2x2xf32, #blocked>
// CHECK:       tt.store %{{.*}} : tensor<2x2x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @non_splat_constant(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x2xf32, #mma>
    %tbl = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32, #blocked>
    %1 = ttg.convert_layout %cst : tensor<2x2xf32, #mma> -> tensor<2x2xf32, #blocked>
    %2 = arith.mulf %1, %tbl : tensor<2x2xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<2x2x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A guarded side load. The mask and the padding value are relayouted alongside
// the pointer, since a load only makes sense with all three in one layout.
// CHECK-LABEL: side_load_masked
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK:       %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #mma>
// CHECK:       %[[MASK:.+]] = ttg.convert_layout %{{.*}} : tensor<32x32xi1, #blocked> -> tensor<32x32xi1, #mma>
// CHECK:       %[[PAD:.+]] = ttg.convert_layout %{{.*}} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #mma>
// CHECK:       %[[BIAS:.+]] = tt.load %[[PTR]], %[[MASK]], %[[PAD]] : tensor<32x32x!tt.ptr<f32>, #mma>
// CHECK:       arith.addf %{{.*}}, %[[BIAS]] : tensor<32x32xf32, #mma>
// CHECK:       tt.store %{{.*}} : tensor<32x32x!tt.ptr<f32>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @side_load_masked(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %pad = arith.constant dense<7.000000e+00> : tensor<32x32xf32, #blocked>
    %inbounds = arith.constant dense<true> : tensor<32x32xi1, #blocked>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %biasptr = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %bias = tt.load %biasptr, %inbounds, %pad : tensor<32x32x!tt.ptr<f32>, #blocked>
    %2 = arith.addf %1, %bias : tensor<32x32xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// The side load is reissued in the accumulator layout rather than duplicated,
// so a second reader of the loaded data would have to read it from a second
// load of the same addresses. That costs more than the conversion being
// removed, so the round trip stays.
// CHECK-LABEL: side_load_read_twice
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK:       %[[BIAS:.+]] = tt.load %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked>
// CHECK:       arith.addf %[[VAL]], %[[BIAS]] : tensor<32x32xf32, #blocked>
// CHECK:       tt.store %{{.*}} : tensor<32x32x!tt.ptr<f32>, #blocked>
// CHECK:       tt.store %{{.*}}, %[[BIAS]] : tensor<32x32x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @side_load_read_twice(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %biasptr = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %bias = tt.load %biasptr : tensor<32x32x!tt.ptr<f32>, #blocked>
    %2 = arith.addf %1, %bias : tensor<32x32xf32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %4, %bias : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Budget override: a caller-supplied rock.max_lds outranks both of the
// performance vetoes, since keeping a round trip that does not fit only moves
// the failure to rock-resolve-kernel-launch-params.

// The rock.prefer_lds_epilogue veto, the one that is live on the MFMA path.
// This conversion needs 4096 bytes, more than the caller allowed the whole
// kernel, so the wider store it was asking for is not available.
// CHECK-LABEL: budget_overrides_prefer_lds_epilogue
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK:       tt.store %{{.*}} : tensor<32x32x!tt.ptr<f32>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @budget_overrides_prefer_lds_epilogue(%arg0: !tt.ptr<f32>) attributes {rock.max_lds = 2048 : i64, rock.prefer_lds_epilogue} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %0 = ttg.convert_layout %cst : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %1, %0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// The same kernel with room to spare: the veto decides, because there is a
// choice to make.
// CHECK-LABEL: prefer_lds_epilogue_honored_within_budget
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<32x32x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @prefer_lds_epilogue_honored_within_budget(%arg0: !tt.ptr<f32>) attributes {rock.max_lds = 65536 : i64, rock.prefer_lds_epilogue} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %0 = ttg.convert_layout %cst : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.store %1, %0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// The isProfitableBypassSource contiguity veto, the one that is live on the
// FMA/blocked path. Same layout pair as fma_blocked_narrower_contig, where the
// narrower source run keeps the round trip; here its 16384 bytes do not fit,
// so the narrower store is accepted as the cost of fitting.
// CHECK-LABEL: budget_overrides_contiguity_veto
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}} : tensor<64x128x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @budget_overrides_contiguity_veto(%arg0: !tt.ptr<f32>) attributes {rock.max_lds = 8192 : i64} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// The same layout pair with room to spare: the contiguity veto decides.
// CHECK-LABEL: contiguity_veto_honored_within_budget
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<64x128x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @contiguity_veto_honored_within_budget(%arg0: !tt.ptr<f32>) attributes {rock.max_lds = 65536 : i64} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
