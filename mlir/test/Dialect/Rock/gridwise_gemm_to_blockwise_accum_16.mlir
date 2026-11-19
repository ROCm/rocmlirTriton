// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: rocmlir-opt -rock-gridwise-gemm-to-blockwise %s | FileCheck %s

func.func @gfx1250_fp8_bf8_f16(%arg0: tensor<787456xf8E4M3FN>, %arg1: tensor<393728xf8E5M2>, %arg2: tensor<524288xf16>) -> tensor<524288xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1250", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 32 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64, rock.use_bf16x3_for_f32 = -1 : i64, rock.use_optimize_epilogue = -1 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf8E4M3FN> to tensor<1x1024x769xf8E4M3FN>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 63} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1024, 832] -> [1, 1024, 769]> : tensor<1x1024x769xf8E4M3FN> to tensor<1x1024x832xf8E4M3FN>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf8E5M2> to tensor<1x769x512xf8E5M2>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 63} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 832, 512] -> [1, 769, 512]> : tensor<1x769x512xf8E5M2> to tensor<1x832x512xf8E5M2>
  %4 = rock.gridwise_gemm(%1, %3) {params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x1024x832xf8E4M3FN>, tensor<1x832x512xf8E5M2> -> tensor<1x1024x512xf16>
  %5 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf16> to tensor<1x1024x512xf16>
  %6 = rock.store %4 to %5 by set : tensor<1x1024x512xf16> -> tensor<524288xf16> to tensor<1x1024x512xf16>
  return %6 : tensor<524288xf16>
}

// CHECK-LABEL: @gfx1250_fp8_bf8_f16
// CHECK: rock.blockwise_gemm
// CHECK-SAME: tensor<128x64xf8E4M3FN>, tensor<64x128xf8E5M2>, tensor<128x128xf16> -> tensor<128x128xf16>


func.func @gfx1250_f16_f16_f16(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf16>) -> tensor<524288xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1250", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 32 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64, rock.use_bf16x3_for_f32 = -1 : i64, rock.use_optimize_epilogue = -1 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 63} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1024, 832] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1024x832xf16>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 63} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 832, 512] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x832x512xf16>
  %4 = rock.gridwise_gemm(%1, %3) {params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x1024x832xf16>, tensor<1x832x512xf16> -> tensor<1x1024x512xf16>
  %5 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf16> to tensor<1x1024x512xf16>
  %6 = rock.store %4 to %5 by set : tensor<1x1024x512xf16> -> tensor<524288xf16> to tensor<1x1024x512xf16>
  return %6 : tensor<524288xf16>
}

// CHECK-LABEL: @gfx1250_f16_f16_f16
// CHECK: rock.blockwise_gemm
// CHECK-SAME: tensor<128x64xf16>, tensor<64x128xf16>, tensor<128x128xf16> -> tensor<128x128xf16>

func.func @gfx1250_f16_f16_f32(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf16>, %arg2: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1250", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 32 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64, rock.use_bf16x3_for_f32 = -1 : i64, rock.use_optimize_epilogue = -1 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 769 + d2)> by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]> : tensor<787456xf16> to tensor<1x1024x769xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 63} ["gemmKPad"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 1024, 832] -> [1, 1024, 769]> : tensor<1x1024x769xf16> to tensor<1x1024x832xf16>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]> : tensor<393728xf16> to tensor<1x769x512xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 63} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 832, 512] -> [1, 769, 512]> : tensor<1x769x512xf16> to tensor<1x832x512xf16>
  %4 = rock.gridwise_gemm(%1, %3) {params = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<1x1024x832xf16>, tensor<1x832x512xf16> -> tensor<1x1024x512xf32>
  %5 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{1024, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 512] -> [524288]> : tensor<524288xf32> to tensor<1x1024x512xf32>
  %6 = rock.store %4 to %5 by set : tensor<1x1024x512xf32> -> tensor<524288xf32> to tensor<1x1024x512xf32>
  return %6 : tensor<524288xf32>
}

// CHECK-LABEL: @gfx1250_f16_f16_f32
// CHECK: rock.blockwise_gemm
// CHECK-SAME: tensor<128x64xf16>, tensor<64x128xf16>, tensor<128x128xf32> -> tensor<128x128xf32>
