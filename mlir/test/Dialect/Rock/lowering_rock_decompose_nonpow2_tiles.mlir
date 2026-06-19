// Tests for the rock-decompose-nonpow2-tiles pass.
//
// The IR in each case was captured from the real pipeline (rocmlir-driver)
// immediately before this pass runs, for a gemm with the tile sizes named in
// the test, then trimmed. Each blockwise_gemm sits in a K-loop scf.for with the
// accumulator carried as the single iter_arg.

// RUN: rocmlir-opt -rock-decompose-nonpow2-tiles -canonicalize -split-input-file -mlir-print-local-scope %s | FileCheck %s

// ============================================================
// Both M and N are non-power-of-two (80, 80): the GEMM splits
// into a 2x2 grid {64,16} x {64,16}.
// ============================================================

// CHECK-LABEL: func.func @test_m_and_n_nonpow2
// CHECK: scf.for
// CHECK-SAME: -> (tensor<64x64xf32>, tensor<64x16xf32>, tensor<16x64xf32>, tensor<16x16xf32>)
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<64x16xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x64xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x16xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x16xf16>
// CHECK-COUNT-4: rock.blockwise_gemm
// CHECK: scf.yield
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<64x64xf32>
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<64x16xf32>
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<16x64xf32>
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<16x16xf32>
func.func @test_m_and_n_nonpow2(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 91 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 2 : i64} {
  %c0_i32 = arith.constant 0 : i32
  %c49_i32 = arith.constant 49 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  %c91_i32 = arith.constant 91 : i32
  %c13_i32 = arith.constant 13 : i32
  %c7_i32 = arith.constant 7 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %2 = tt.get_program_id x : i32
  %3 = arith.divui %2, %c91_i32 : i32
  %4 = arith.remui %2, %c91_i32 : i32
  %5 = arith.divui %4, %c7_i32 : i32
  %6 = arith.subi %c13_i32, %5 : i32
  %7 = arith.minui %6, %c1_i32 : i32
  %8 = arith.remui %4, %7 : i32
  %9 = arith.addi %5, %8 : i32
  %10 = arith.remui %4, %c7_i32 : i32
  %11 = arith.divui %10, %7 : i32
  %12 = scf.for %arg3 = %c0_i32 to %c49_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<80x80xf32>)  : i32 {
    %17 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 15} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 784, 560] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x784x560xf16>
    %18 = rock.transform %17 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 16 + d4, d3 * 80 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{13} ["m_block"] at [2] -> [] at []>] bounds = [49, 1, 13, 7, 16, 80] -> [1, 784, 560]> : tensor<1x784x560xf16> to tensor<49x1x13x7x16x80xf16>
    %19 = rock.blockwise_load %18[%arg3, %3, %9, %11] : tensor<49x1x13x7x16x80xf16> -> tensor<16x80xf16>
    %20 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 15} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1040, 784] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1040x784xf16>
    %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 80 + d4, d0 * 16 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{7} ["n_block"] at [3] -> [] at []>] bounds = [49, 1, 13, 7, 80, 16] -> [1, 1040, 784]> : tensor<1x1040x784xf16> to tensor<49x1x13x7x80x16xf16>
    %22 = rock.blockwise_load %21[%arg3, %3, %9, %11] : tensor<49x1x13x7x80x16xf16> -> tensor<80x16xf16>
    %23 = rock.blockwise_gemm(%22, %19, %arg4) : tensor<80x16xf16>, tensor<16x80xf16>, tensor<80x80xf32> -> tensor<80x80xf32>
    scf.yield %23 : tensor<80x80xf32>
  }
  %13 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %14 = rock.transform %13 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1040, 560] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1040x560xf32>
  %15 = rock.transform %14 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d3, d2 * 80 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 13, 7, 80, 80] -> [1, 1040, 560]> : tensor<1x1040x560xf32> to tensor<1x13x7x80x80xf32>
  %16 = rock.blockwise_store %12 -> %15[%3, %9, %11] by set : tensor<80x80xf32> -> tensor<1x13x7x80x80xf32> -> tensor<524288xf32>
  return %16 : tensor<524288xf32>
}

// -----

// ============================================================
// Only M is non-power-of-two (80, 64): the GEMM splits along M
// into {64,16}, N stays whole.
// ============================================================

// CHECK-LABEL: func.func @test_m_nonpow2
// CHECK: scf.for
// CHECK-SAME: -> (tensor<64x64xf32>, tensor<16x64xf32>)
// A is sliced along M ({64,16}); B (16x64) is loaded whole.
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<64x16xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x16xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x64xf16>
// CHECK-COUNT-2: rock.blockwise_gemm
// CHECK-NOT: rock.blockwise_gemm
// CHECK: scf.yield
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<64x64xf32>
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<16x64xf32>
func.func @test_m_nonpow2(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 104 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 2 : i64} {
  %c0_i32 = arith.constant 0 : i32
  %c49_i32 = arith.constant 49 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<80x64xf32>
  %c104_i32 = arith.constant 104 : i32
  %c13_i32 = arith.constant 13 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %2 = tt.get_program_id x : i32
  %3 = arith.divui %2, %c104_i32 : i32
  %4 = arith.remui %2, %c104_i32 : i32
  %5 = arith.divui %4, %c8_i32 : i32
  %6 = arith.subi %c13_i32, %5 : i32
  %7 = arith.minui %6, %c1_i32 : i32
  %8 = arith.remui %4, %7 : i32
  %9 = arith.addi %5, %8 : i32
  %10 = arith.remui %4, %c8_i32 : i32
  %11 = arith.divui %10, %7 : i32
  %12 = scf.for %arg3 = %c0_i32 to %c49_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<80x64xf32>)  : i32 {
    %17 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 15} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 784, 512] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x784x512xf16>
    %18 = rock.transform %17 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 16 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{8, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{13} ["m_block"] at [2] -> [] at []>] bounds = [49, 1, 13, 8, 16, 64] -> [1, 784, 512]> : tensor<1x784x512xf16> to tensor<49x1x13x8x16x64xf16>
    %19 = rock.blockwise_load %18[%arg3, %3, %9, %11] : tensor<49x1x13x8x16x64xf16> -> tensor<16x64xf16>
    %20 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 15} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1040, 784] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1040x784xf16>
    %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 80 + d4, d0 * 16 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{8} ["n_block"] at [3] -> [] at []>] bounds = [49, 1, 13, 8, 80, 16] -> [1, 1040, 784]> : tensor<1x1040x784xf16> to tensor<49x1x13x8x80x16xf16>
    %22 = rock.blockwise_load %21[%arg3, %3, %9, %11] : tensor<49x1x13x8x80x16xf16> -> tensor<80x16xf16>
    %23 = rock.blockwise_gemm(%22, %19, %arg4) : tensor<80x16xf16>, tensor<16x64xf16>, tensor<80x64xf32> -> tensor<80x64xf32>
    scf.yield %23 : tensor<80x64xf32>
  }
  %13 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %14 = rock.transform %13 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1040, 512] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1040x512xf32>
  %15 = rock.transform %14 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{8, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 13, 8, 80, 64] -> [1, 1040, 512]> : tensor<1x1040x512xf32> to tensor<1x13x8x80x64xf32>
  %16 = rock.blockwise_store %12 -> %15[%3, %9, %11] by set : tensor<80x64xf32> -> tensor<1x13x8x80x64xf32> -> tensor<524288xf32>
  return %16 : tensor<524288xf32>
}

// -----

// ============================================================
// Only N is non-power-of-two (64, 80): the GEMM splits along N
// into {64,16}, M stays whole.
// ============================================================

// CHECK-LABEL: func.func @test_n_nonpow2
// CHECK: scf.for
// CHECK-SAME: -> (tensor<64x64xf32>, tensor<64x16xf32>)
// A (64x16) is loaded whole; B is sliced along N ({64,16}).
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<64x16xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x64xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x16xf16>
// CHECK-COUNT-2: rock.blockwise_gemm
// CHECK-NOT: rock.blockwise_gemm
// CHECK: scf.yield
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<64x64xf32>
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<64x16xf32>
func.func @test_n_nonpow2(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 112 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 2 : i64} {
  %c0_i32 = arith.constant 0 : i32
  %c49_i32 = arith.constant 49 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x80xf32>
  %c112_i32 = arith.constant 112 : i32
  %c16_i32 = arith.constant 16 : i32
  %c7_i32 = arith.constant 7 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %2 = tt.get_program_id x : i32
  %3 = arith.divui %2, %c112_i32 : i32
  %4 = arith.remui %2, %c112_i32 : i32
  %5 = arith.divui %4, %c7_i32 : i32
  %6 = arith.subi %c16_i32, %5 : i32
  %7 = arith.minui %6, %c1_i32 : i32
  %8 = arith.remui %4, %7 : i32
  %9 = arith.addi %5, %8 : i32
  %10 = arith.remui %4, %c7_i32 : i32
  %11 = arith.divui %10, %7 : i32
  %12 = scf.for %arg3 = %c0_i32 to %c49_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<64x80xf32>)  : i32 {
    %17 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 15} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 784, 560] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x784x560xf16>
    %18 = rock.transform %17 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 16 + d4, d3 * 80 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{16} ["m_block"] at [2] -> [] at []>] bounds = [49, 1, 16, 7, 16, 80] -> [1, 784, 560]> : tensor<1x784x560xf16> to tensor<49x1x16x7x16x80xf16>
    %19 = rock.blockwise_load %18[%arg3, %3, %9, %11] : tensor<49x1x16x7x16x80xf16> -> tensor<16x80xf16>
    %20 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 15} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1024, 784] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1024x784xf16>
    %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 16 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{16, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{7} ["n_block"] at [3] -> [] at []>] bounds = [49, 1, 16, 7, 64, 16] -> [1, 1024, 784]> : tensor<1x1024x784xf16> to tensor<49x1x16x7x64x16xf16>
    %22 = rock.blockwise_load %21[%arg3, %3, %9, %11] : tensor<49x1x16x7x64x16xf16> -> tensor<64x16xf16>
    %23 = rock.blockwise_gemm(%22, %19, %arg4) : tensor<64x16xf16>, tensor<16x80xf16>, tensor<64x80xf32> -> tensor<64x80xf32>
    scf.yield %23 : tensor<64x80xf32>
  }
  %13 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %14 = rock.transform %13 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1024, 560] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1024x560xf32>
  %15 = rock.transform %14 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 80 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{16, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 16, 7, 64, 80] -> [1, 1024, 560]> : tensor<1x1024x560xf32> to tensor<1x16x7x64x80xf32>
  %16 = rock.blockwise_store %12 -> %15[%3, %9, %11] by set : tensor<64x80xf32> -> tensor<1x16x7x64x80xf32> -> tensor<524288xf32>
  return %16 : tensor<524288xf32>
}

// -----

// ============================================================
// Nothing to do: both M and N are already powers of two (64, 64).
// The GEMM and store are left unchanged (one of each).
// ============================================================

// CHECK-LABEL: func.func @test_nothing_to_do
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x64xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<64x16xf16>
// CHECK: rock.blockwise_gemm{{.*}}tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK-NOT: rock.blockwise_gemm
// CHECK: rock.blockwise_store{{.*}} : tensor<64x64xf32>
// CHECK-NOT: rock.blockwise_store
func.func @test_nothing_to_do(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 128 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 2 : i64} {
  %c0_i32 = arith.constant 0 : i32
  %c49_i32 = arith.constant 49 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  %c128_i32 = arith.constant 128 : i32
  %c16_i32 = arith.constant 16 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %2 = tt.get_program_id x : i32
  %3 = arith.divui %2, %c128_i32 : i32
  %4 = arith.remui %2, %c128_i32 : i32
  %5 = arith.divui %4, %c8_i32 : i32
  %6 = arith.subi %c16_i32, %5 : i32
  %7 = arith.minui %6, %c1_i32 : i32
  %8 = arith.remui %4, %7 : i32
  %9 = arith.addi %5, %8 : i32
  %10 = arith.remui %4, %c8_i32 : i32
  %11 = arith.divui %10, %7 : i32
  %12 = scf.for %arg3 = %c0_i32 to %c49_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<64x64xf32>)  : i32 {
    %16 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 15} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 784, 512] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x784x512xf16>
    %17 = rock.transform %16 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 16 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{8, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{16} ["m_block"] at [2] -> [] at []>] bounds = [49, 1, 16, 8, 16, 64] -> [1, 784, 512]> : tensor<1x784x512xf16> to tensor<49x1x16x8x16x64xf16>
    %18 = rock.blockwise_load %17[%arg3, %3, %9, %11] : tensor<49x1x16x8x16x64xf16> -> tensor<16x64xf16>
    %19 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 15} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1024, 784] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1024x784xf16>
    %20 = rock.transform %19 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 16 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{16, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{8} ["n_block"] at [3] -> [] at []>] bounds = [49, 1, 16, 8, 64, 16] -> [1, 1024, 784]> : tensor<1x1024x784xf16> to tensor<49x1x16x8x64x16xf16>
    %21 = rock.blockwise_load %20[%arg3, %3, %9, %11] : tensor<49x1x16x8x64x16xf16> -> tensor<64x16xf16>
    %22 = rock.blockwise_gemm(%21, %18, %arg4) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
    scf.yield %22 : tensor<64x64xf32>
  }
  %13 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %14 = rock.transform %13 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{16, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{8, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 16, 8, 64, 64] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x16x8x64x64xf32>
  %15 = rock.blockwise_store %12 -> %14[%3, %9, %11] by set : tensor<64x64xf32> -> tensor<1x16x8x64x64xf32> -> tensor<524288xf32>
  return %15 : tensor<524288xf32>
}

// -----

// ============================================================
// Both M and N non-power-of-two (80, 80) with input AND output
// fusion. A and B are each the sum of two loads (input fusion);
// the GEMM result is combined with two extra loaded tensors via
// add then mul (output fusion). Splitting recurses through both
// fusion DAGs, producing a 2x2 grid of independent pipelines.
// ============================================================

// CHECK-LABEL: func.func @test_input_and_output_fusion
// CHECK: scf.for
// CHECK-SAME: -> (tensor<64x64xf32>, tensor<64x16xf32>, tensor<16x64xf32>, tensor<16x16xf32>)
// Input fusion: A and B are each addf of two sliced loads (8 loads, 4 addf),
// feeding a 2x2 grid of GEMMs.
// CHECK-COUNT-8: rock.blockwise_load {{.*}}xf16>
// CHECK: arith.addf {{.*}}xf16
// CHECK-COUNT-4: rock.blockwise_gemm
// CHECK: scf.yield
// Output fusion: each cell adds a sliced bias0 then multiplies a sliced bias1.
// CHECK-COUNT-4: rock.blockwise_load {{.*}}xf32>
// CHECK: arith.addf {{.*}}xf32
// CHECK-COUNT-4: rock.blockwise_load {{.*}}xf32>
// CHECK: arith.mulf {{.*}}xf32
// CHECK-COUNT-4: rock.blockwise_store
func.func @test_input_and_output_fusion(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf32>, %bias0: tensor<524288xf32>, %bias1: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 2 : i64} {
  %c0_i32 = arith.constant 0 : i32
  %c49_i32 = arith.constant 49 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  %c91_i32 = arith.constant 91 : i32
  %c13_i32 = arith.constant 13 : i32
  %c7_i32 = arith.constant 7 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %2 = tt.get_program_id x : i32
  %3 = arith.divui %2, %c91_i32 : i32
  %4 = arith.remui %2, %c91_i32 : i32
  %5 = arith.divui %4, %c7_i32 : i32
  %6 = arith.subi %c13_i32, %5 : i32
  %7 = arith.minui %6, %c1_i32 : i32
  %8 = arith.remui %4, %7 : i32
  %9 = arith.addi %5, %8 : i32
  %10 = arith.remui %4, %c7_i32 : i32
  %11 = arith.divui %10, %7 : i32
  %12 = scf.for %arg3 = %c0_i32 to %c49_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<80x80xf32>)  : i32 {
    %17 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 15} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 784, 560] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x784x560xf16>
    %18 = rock.transform %17 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 16 + d4, d3 * 80 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{13} ["m_block"] at [2] -> [] at []>] bounds = [49, 1, 13, 7, 16, 80] -> [1, 784, 560]> : tensor<1x784x560xf16> to tensor<49x1x13x7x16x80xf16>
    %19 = rock.blockwise_load %18[%arg3, %3, %9, %11] : tensor<49x1x13x7x16x80xf16> -> tensor<16x80xf16>
    %bload = rock.blockwise_load %18[%arg3, %3, %9, %11] : tensor<49x1x13x7x16x80xf16> -> tensor<16x80xf16>
    %fb = arith.addf %19, %bload : tensor<16x80xf16>
    %20 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 15} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1040, 784] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1040x784xf16>
    %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 80 + d4, d0 * 16 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{7} ["n_block"] at [3] -> [] at []>] bounds = [49, 1, 13, 7, 80, 16] -> [1, 1040, 784]> : tensor<1x1040x784xf16> to tensor<49x1x13x7x80x16xf16>
    %22 = rock.blockwise_load %21[%arg3, %3, %9, %11] : tensor<49x1x13x7x80x16xf16> -> tensor<80x16xf16>
    %aload = rock.blockwise_load %21[%arg3, %3, %9, %11] : tensor<49x1x13x7x80x16xf16> -> tensor<80x16xf16>
    %fa = arith.addf %22, %aload : tensor<80x16xf16>
    %23 = rock.blockwise_gemm(%fa, %fb, %arg4) : tensor<80x16xf16>, tensor<16x80xf16>, tensor<80x80xf32> -> tensor<80x80xf32>
    scf.yield %23 : tensor<80x80xf32>
  }
  %13 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %14 = rock.transform %13 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1040, 560] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1040x560xf32>
  %15 = rock.transform %14 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d3, d2 * 80 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 13, 7, 80, 80] -> [1, 1040, 560]> : tensor<1x1040x560xf32> to tensor<1x13x7x80x80xf32>
  %b0a = rock.transform %bias0 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %b0b = rock.transform %b0a by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1040, 560] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1040x560xf32>
  %b0c = rock.transform %b0b by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d3, d2 * 80 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 13, 7, 80, 80] -> [1, 1040, 560]> : tensor<1x1040x560xf32> to tensor<1x13x7x80x80xf32>
  %bl0 = rock.blockwise_load %b0c[%3, %9, %11] : tensor<1x13x7x80x80xf32> -> tensor<80x80xf32>
  %b1a = rock.transform %bias1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %b1b = rock.transform %b1a by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1040, 560] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1040x560xf32>
  %b1c = rock.transform %b1b by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d3, d2 * 80 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 13, 7, 80, 80] -> [1, 1040, 560]> : tensor<1x1040x560xf32> to tensor<1x13x7x80x80xf32>
  %bl1 = rock.blockwise_load %b1c[%3, %9, %11] : tensor<1x13x7x80x80xf32> -> tensor<80x80xf32>
  %add = arith.addf %12, %bl0 : tensor<80x80xf32>
  %mul = arith.mulf %add, %bl1 : tensor<80x80xf32>
  %16 = rock.blockwise_store %mul -> %15[%3, %9, %11] by set : tensor<80x80xf32> -> tensor<1x13x7x80x80xf32> -> tensor<524288xf32>
  return %16 : tensor<524288xf32>
}

// -----

// ============================================================
// Loopless variant of the input+output fusion case: the single
// K-step loop has been folded away, so the GEMM accumulates over
// a plain init value (no scf.for). The per-cell GEMMs are emitted
// directly and the same input/output fusion splitting applies.
// ============================================================

// CHECK-LABEL: func.func @test_input_and_output_fusion_loopless
// CHECK-NOT: scf.for
// Input fusion: A and B are each addf of two sliced loads (8 loads, 4 addf).
// CHECK-COUNT-8: rock.blockwise_load {{.*}}xf16>
// CHECK: arith.addf {{.*}}xf16
// CHECK-COUNT-4: rock.blockwise_gemm
// Output fusion: each cell adds a sliced bias0 then multiplies a sliced bias1.
// CHECK-COUNT-4: rock.blockwise_load {{.*}}xf32>
// CHECK: arith.addf {{.*}}xf32
// CHECK-COUNT-4: rock.blockwise_load {{.*}}xf32>
// CHECK: arith.mulf {{.*}}xf32
// CHECK-COUNT-4: rock.blockwise_store
func.func @test_input_and_output_fusion_loopless(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf32>, %bias0: tensor<524288xf32>, %bias1: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 2 : i64} {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<80x80xf32>
  %c91_i32 = arith.constant 91 : i32
  %c13_i32 = arith.constant 13 : i32
  %c7_i32 = arith.constant 7 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %2 = tt.get_program_id x : i32
  %3 = arith.divui %2, %c91_i32 : i32
  %4 = arith.remui %2, %c91_i32 : i32
  %5 = arith.divui %4, %c7_i32 : i32
  %6 = arith.subi %c13_i32, %5 : i32
  %7 = arith.minui %6, %c1_i32 : i32
  %8 = arith.remui %4, %7 : i32
  %9 = arith.addi %5, %8 : i32
  %10 = arith.remui %4, %c7_i32 : i32
  %11 = arith.divui %10, %7 : i32
  %17 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 15} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 784, 560] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x784x560xf16>
  %18 = rock.transform %17 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 16 + d4, d3 * 80 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{13} ["m_block"] at [2] -> [] at []>] bounds = [49, 1, 13, 7, 16, 80] -> [1, 784, 560]> : tensor<1x784x560xf16> to tensor<49x1x13x7x16x80xf16>
  %19 = rock.blockwise_load %18[%c0_i32, %3, %9, %11] : tensor<49x1x13x7x16x80xf16> -> tensor<16x80xf16>
  %bload = rock.blockwise_load %18[%c0_i32, %3, %9, %11] : tensor<49x1x13x7x16x80xf16> -> tensor<16x80xf16>
  %fb = arith.addf %19, %bload : tensor<16x80xf16>
  %20 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 15} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1040, 784] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1040x784xf16>
  %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 80 + d4, d0 * 16 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{7} ["n_block"] at [3] -> [] at []>] bounds = [49, 1, 13, 7, 80, 16] -> [1, 1040, 784]> : tensor<1x1040x784xf16> to tensor<49x1x13x7x80x16xf16>
  %22 = rock.blockwise_load %21[%c0_i32, %3, %9, %11] : tensor<49x1x13x7x80x16xf16> -> tensor<80x16xf16>
  %aload = rock.blockwise_load %21[%c0_i32, %3, %9, %11] : tensor<49x1x13x7x80x16xf16> -> tensor<80x16xf16>
  %fa = arith.addf %22, %aload : tensor<80x16xf16>
  %23 = rock.blockwise_gemm(%fa, %fb, %cst) : tensor<80x16xf16>, tensor<16x80xf16>, tensor<80x80xf32> -> tensor<80x80xf32>
  %13 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %14 = rock.transform %13 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1040, 560] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1040x560xf32>
  %15 = rock.transform %14 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d3, d2 * 80 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 13, 7, 80, 80] -> [1, 1040, 560]> : tensor<1x1040x560xf32> to tensor<1x13x7x80x80xf32>
  %b0a = rock.transform %bias0 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %b0b = rock.transform %b0a by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1040, 560] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1040x560xf32>
  %b0c = rock.transform %b0b by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d3, d2 * 80 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 13, 7, 80, 80] -> [1, 1040, 560]> : tensor<1x1040x560xf32> to tensor<1x13x7x80x80xf32>
  %bl0 = rock.blockwise_load %b0c[%3, %9, %11] : tensor<1x13x7x80x80xf32> -> tensor<80x80xf32>
  %b1a = rock.transform %bias1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %b1b = rock.transform %b1a by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 16} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 48} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1040, 560] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1040x560xf32>
  %b1c = rock.transform %b1b by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d3, d2 * 80 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{13, 80} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{7, 80} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 13, 7, 80, 80] -> [1, 1040, 560]> : tensor<1x1040x560xf32> to tensor<1x13x7x80x80xf32>
  %bl1 = rock.blockwise_load %b1c[%3, %9, %11] : tensor<1x13x7x80x80xf32> -> tensor<80x80xf32>
  %add = arith.addf %23, %bl0 : tensor<80x80xf32>
  %mul = arith.mulf %add, %bl1 : tensor<80x80xf32>
  %16 = rock.blockwise_store %mul -> %15[%3, %9, %11] by set : tensor<80x80xf32> -> tensor<1x13x7x80x80xf32> -> tensor<524288xf32>
  return %16 : tensor<524288xf32>
}

// -----

// ============================================================
// M is non-power-of-two with THREE set bits (112 = 64+32+16):
// the M dimension decomposes into three power-of-two segments,
// so the GEMM splits along M into {64,32,16}; N (64) stays whole.
// ============================================================

// CHECK-LABEL: func.func @test_m_three_segments
// CHECK: scf.for
// CHECK-SAME: -> (tensor<64x64xf32>, tensor<32x64xf32>, tensor<16x64xf32>)
// A is sliced along M into three segments ({64,32,16}); B (16x64) is whole.
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<64x16xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<32x16xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x16xf16>
// CHECK-DAG: rock.blockwise_load {{.*}} -> tensor<16x64xf16>
// CHECK-COUNT-3: rock.blockwise_gemm
// CHECK-NOT: rock.blockwise_gemm
// CHECK: scf.yield
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<64x64xf32>
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<32x64xf32>
// CHECK-DAG: rock.blockwise_store {{.*}} : tensor<16x64xf32>
func.func @test_m_three_segments(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 80 : i32, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 2 : i64} {
  %c0_i32 = arith.constant 0 : i32
  %c49_i32 = arith.constant 49 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<112x64xf32>
  %c80_i32 = arith.constant 80 : i32
  %c10_i32 = arith.constant 10 : i32
  %c8_i32 = arith.constant 8 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %2 = tt.get_program_id x : i32
  %3 = arith.divui %2, %c80_i32 : i32
  %4 = arith.remui %2, %c80_i32 : i32
  %5 = arith.divui %4, %c8_i32 : i32
  %6 = arith.subi %c10_i32, %5 : i32
  %7 = arith.minui %6, %c1_i32 : i32
  %8 = arith.remui %4, %7 : i32
  %9 = arith.addi %5, %8 : i32
  %10 = arith.remui %4, %c8_i32 : i32
  %11 = arith.divui %10, %7 : i32
  %12 = scf.for %arg3 = %c0_i32 to %c49_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<112x64xf32>)  : i32 {
    %17 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 15} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 784, 512] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x784x512xf16>
    %18 = rock.transform %17 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 16 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{8, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{10} ["m_block"] at [2] -> [] at []>] bounds = [49, 1, 10, 8, 16, 64] -> [1, 784, 512]> : tensor<1x784x512xf16> to tensor<49x1x10x8x16x64xf16>
    %19 = rock.blockwise_load %18[%arg3, %3, %9, %11] : tensor<49x1x10x8x16x64xf16> -> tensor<16x64xf16>
    %20 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 96} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 15} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1120, 784] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1120x784xf16>
    %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 112 + d4, d0 * 16 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{49, 16} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{10, 112} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{8} ["n_block"] at [3] -> [] at []>] bounds = [49, 1, 10, 8, 112, 16] -> [1, 1120, 784]> : tensor<1x1120x784xf16> to tensor<49x1x10x8x112x16xf16>
    %22 = rock.blockwise_load %21[%arg3, %3, %9, %11] : tensor<49x1x10x8x112x16xf16> -> tensor<112x16xf16>
    %23 = rock.blockwise_gemm(%22, %19, %arg4) : tensor<112x16xf16>, tensor<16x64xf16>, tensor<112x64xf32> -> tensor<112x64xf32>
    scf.yield %23 : tensor<112x64xf32>
  }
  %13 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %14 = rock.transform %13 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 96} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1120, 512] -> [1, 1024, 512]> : tensor<1x1024x512xf32> to tensor<1x1120x512xf32>
  %15 = rock.transform %14 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 112 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{10, 112} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{8, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 10, 8, 112, 64] -> [1, 1120, 512]> : tensor<1x1120x512xf32> to tensor<1x10x8x112x64xf32>
  %16 = rock.blockwise_store %12 -> %15[%3, %9, %11] by set : tensor<112x64xf32> -> tensor<1x10x8x112x64xf32> -> tensor<524288xf32>
  return %16 : tensor<524288xf32>
}

// -----

// ============================================================
// Backward-data-style store chaining plus non-power-of-two
// decomposition. Two independent GEMMs write disjoint slices of
// the same logical output. The second store's destination is a
// view of the first store's result, matching the multi-kernel
// bwd_data lowering shape.
// ============================================================

// CHECK-LABEL: func.func @test_chained_bwd_data_stores_nonpow2
// Each 80x64 GEMM splits along M into {64,16}; there are two original GEMMs.
// The first original store sits before the second GEMM, so the decomposed
// output is intentionally interleaved as:
//   first GEMM pair, first store pair, second GEMM pair, second store pair.
// CHECK: rock.blockwise_gemm{{.*}} : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: rock.blockwise_gemm{{.*}} : tensor<16x16xf16>, tensor<16x64xf16>, tensor<16x64xf32> -> tensor<16x64xf32>
// CHECK: rock.blockwise_store {{.*}} : tensor<64x64xf32>
// CHECK: rock.blockwise_store {{.*}} : tensor<16x64xf32>
// CHECK: rock.blockwise_gemm{{.*}} : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK: rock.blockwise_gemm{{.*}} : tensor<16x16xf16>, tensor<16x64xf16>, tensor<16x64xf32> -> tensor<16x64xf32>
// The second original store must still write through a view of the first
// decomposed store chain, not the original function argument.
// CHECK: rock.transform %{{.*}} by <affine_map<(d0, d1) -> (d0 + 80, d1)>
// CHECK: rock.blockwise_store {{.*}} : tensor<64x64xf32>
// CHECK: rock.blockwise_store {{.*}} : tensor<16x64xf32>
// CHECK: return %{{.*}} : tensor<160x64xf32>
func.func @test_chained_bwd_data_stores_nonpow2(
    %a0Src: tensor<80x16xf16>, %b0Src: tensor<16x64xf16>,
    %a1Src: tensor<80x16xf16>, %b1Src: tensor<16x64xf16>,
    %dest: tensor<160x64xf32>) -> tensor<160x64xf32>
    attributes {rock.kernel} {
  %cst = arith.constant dense<0.000000e+00> : tensor<80x64xf32>

  %a0 = rock.blockwise_load %a0Src : tensor<80x16xf16> -> tensor<80x16xf16>
  %b0 = rock.blockwise_load %b0Src : tensor<16x64xf16> -> tensor<16x64xf16>
  %g0 = rock.blockwise_gemm(%a0, %b0, %cst)
    : tensor<80x16xf16>, tensor<16x64xf16>, tensor<80x64xf32> -> tensor<80x64xf32>
  %dest0 = rock.transform %dest by
    <affine_map<(d0, d1) -> (d0, d1)> by [
      <Slice{0, 80, 0, 64} ["m0", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>]
    bounds = [80, 64] -> [160, 64]>
    : tensor<160x64xf32> to tensor<80x64xf32>
  %s0 = rock.blockwise_store %g0 -> %dest0 by set
    : tensor<80x64xf32> -> tensor<80x64xf32> -> tensor<160x64xf32>

  %a1 = rock.blockwise_load %a1Src : tensor<80x16xf16> -> tensor<80x16xf16>
  %b1 = rock.blockwise_load %b1Src : tensor<16x64xf16> -> tensor<16x64xf16>
  %g1 = rock.blockwise_gemm(%a1, %b1, %cst)
    : tensor<80x16xf16>, tensor<16x64xf16>, tensor<80x64xf32> -> tensor<80x64xf32>
  %dest1 = rock.transform %s0 by
    <affine_map<(d0, d1) -> (d0 + 80, d1)> by [
      <Slice{80, 160, 0, 64} ["m1", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>]
    bounds = [80, 64] -> [160, 64]>
    : tensor<160x64xf32> to tensor<80x64xf32>
  %s1 = rock.blockwise_store %g1 -> %dest1 by set
    : tensor<80x64xf32> -> tensor<80x64xf32> -> tensor<160x64xf32>
  return %s1 : tensor<160x64xf32>
}
