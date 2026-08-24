// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics %s | FileCheck %s

// All tests share: G=2, M=64, K=400, N=400, mPerBlock=16, nPerBlock=16
// mBlocks = 64/16 = 4, nBlocks = 400/16 = 25, numCU = 256, numChiplets = 8
//
// makeGroupedGridLayout groupSize heuristic:
//   bitWidthOut = outputType.bitWidth
//   bitWidthIn  = min(inputType.bitWidth, outputType.bitWidth)
//   groupSize   = ceil(sqrt(numCU / numChiplets)) * (bitWidthOut / bitWidthIn)
//               = ceil(sqrt(256 / 8)) * ratio = 6 * ratio
//   blocksPerGroup = groupSize * nBlocks = groupSize * 25
//   mnBlocks       = mBlocks * nBlocks = 4 * 25 = 100
//
// The grid layout IR emitted after tt.get_program_id follows a fixed pattern:
//   1. XCC rearrangement (chiplet-aware reorder of bid)
//   2. Grid coordinate computation: g_block = bid / mnBlocks, then a grouped
//      m_block/n_block split using groupSize, blocksPerGroup and mBlocks.
// The CHECK lines anchor on tt.get_program_id -> arith.select (end of XCC
// rearrangement) -> groupSize/blocksPerGroup/mBlocks/mnBlocks constants ->
// their uses.

// -----

// Basic: i8 inputs, i32 output. Types resolve directly from block arguments.
// inputType = i8 (8-bit), outputType = i32 (32-bit)
// groupSize = 6 * (32 / min(8,32)) = 6 * 4 = 24
// blocksPerGroup = 24 * 25 = 600
// CHECK-LABEL: @gemm_basic_i8_i32
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 24 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 600 : i32
// CHECK:       %[[MBLK:.*]] = arith.constant 4 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[GROUPID:.*]] = arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       %[[FIRSTM:.*]] = arith.muli %[[GROUPID]], %[[GPSIZE]] : i32
// CHECK:       %[[MDIFF:.*]] = arith.subi %[[MBLK]], %[[FIRSTM]] : i32
// CHECK:       %[[THISMPG:.*]] = arith.minui %[[MDIFF]], %[[GPSIZE]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[THISMPG]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.divui %{{.*}}, %[[THISMPG]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_basic_i8_i32(%arg0: tensor<51200xi8>, %arg1: tensor<320000xi8>, %arg2: tensor<51200xi32>) -> tensor<51200xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xi8> to tensor<2x64x400xi8>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xi8> to tensor<2x400x400xi8>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xi32> to tensor<2x64x400xi32>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xi8> to tensor<2x400x400xi8>
  %4 = rock.gridwise_gemm(%0, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xi8>, tensor<2x400x400xi8> -> tensor<2x64x400xi32>
  %5 = rock.store %4 to %2 by set : tensor<2x64x400xi32> -> tensor<51200xi32> to tensor<2x64x400xi32>
  return %5 : tensor<51200xi32>
}

// -----

// f32 inputs, f32 output. All 32-bit.
// inputType = f32 (32-bit), outputType = f32 (32-bit)
// groupSize = 6 * (32 / min(32,32)) = 6
// blocksPerGroup = 6 * 25 = 150
// CHECK-LABEL: @gemm_basic_f32_f32
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 6 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 150 : i32
// CHECK:       %[[MBLK:.*]] = arith.constant 4 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[GROUPID:.*]] = arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       %[[FIRSTM:.*]] = arith.muli %[[GROUPID]], %[[GPSIZE]] : i32
// CHECK:       %[[MDIFF:.*]] = arith.subi %[[MBLK]], %[[FIRSTM]] : i32
// CHECK:       %[[THISMPG:.*]] = arith.minui %[[MDIFF]], %[[GPSIZE]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[THISMPG]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.divui %{{.*}}, %[[THISMPG]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_basic_f32_f32(%arg0: tensor<51200xf32>, %arg1: tensor<320000xf32>, %arg2: tensor<51200xf32>) -> tensor<51200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf32> to tensor<2x64x400xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xf32> to tensor<2x400x400xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf32> to tensor<2x64x400xf32>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xf32> to tensor<2x400x400xf32>
  %4 = rock.gridwise_gemm(%0, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xf32>, tensor<2x400x400xf32> -> tensor<2x64x400xf32>
  %5 = rock.store %4 to %2 by set : tensor<2x64x400xf32> -> tensor<51200xf32> to tensor<2x64x400xf32>
  return %5 : tensor<51200xf32>
}

// -----

// f16 inputs, f16 output. All 16-bit.
// inputType = f16 (16-bit), outputType = f16 (16-bit)
// groupSize = 6 * (16 / min(16,16)) = 6
// blocksPerGroup = 6 * 25 = 150
// CHECK-LABEL: @gemm_basic_f16_f16
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 6 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 150 : i32
// CHECK:       %[[MBLK:.*]] = arith.constant 4 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[GROUPID:.*]] = arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       %[[FIRSTM:.*]] = arith.muli %[[GROUPID]], %[[GPSIZE]] : i32
// CHECK:       %[[MDIFF:.*]] = arith.subi %[[MBLK]], %[[FIRSTM]] : i32
// CHECK:       %[[THISMPG:.*]] = arith.minui %[[MDIFF]], %[[GPSIZE]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[THISMPG]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.divui %{{.*}}, %[[THISMPG]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_basic_f16_f16(%arg0: tensor<51200xf16>, %arg1: tensor<320000xf16>, %arg2: tensor<51200xf16>) -> tensor<51200xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf16> to tensor<2x64x400xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xf16> to tensor<2x400x400xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf16> to tensor<2x64x400xf16>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xf16> to tensor<2x400x400xf16>
  %4 = rock.gridwise_gemm(%0, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xf16>, tensor<2x400x400xf16> -> tensor<2x64x400xf16>
  %5 = rock.store %4 to %2 by set : tensor<2x64x400xf16> -> tensor<51200xf16> to tensor<2x64x400xf16>
  return %5 : tensor<51200xf16>
}

// -----

// f16 inputs, f32 output. Mixed precision widens the groupSize ratio.
// inputType = f16 (16-bit), outputType = f32 (32-bit)
// groupSize = 6 * (32 / min(16,32)) = 6 * 2 = 12
// blocksPerGroup = 12 * 25 = 300
// CHECK-LABEL: @gemm_basic_f16_f32
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 12 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 300 : i32
// CHECK:       %[[MBLK:.*]] = arith.constant 4 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[GROUPID:.*]] = arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       %[[FIRSTM:.*]] = arith.muli %[[GROUPID]], %[[GPSIZE]] : i32
// CHECK:       %[[MDIFF:.*]] = arith.subi %[[MBLK]], %[[FIRSTM]] : i32
// CHECK:       %[[THISMPG:.*]] = arith.minui %[[MDIFF]], %[[GPSIZE]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[THISMPG]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.divui %{{.*}}, %[[THISMPG]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_basic_f16_f32(%arg0: tensor<51200xf16>, %arg1: tensor<320000xf16>, %arg2: tensor<51200xf32>) -> tensor<51200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf16> to tensor<2x64x400xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xf16> to tensor<2x400x400xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf32> to tensor<2x64x400xf32>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xf16> to tensor<2x400x400xf16>
  %4 = rock.gridwise_gemm(%0, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xf16>, tensor<2x400x400xf16> -> tensor<2x64x400xf32>
  %5 = rock.store %4 to %2 by set : tensor<2x64x400xf32> -> tensor<51200xf32> to tensor<2x64x400xf32>
  return %5 : tensor<51200xf32>
}

// -----

// bf16 inputs, bf16 output. All 16-bit.
// inputType = bf16 (16-bit), outputType = bf16 (16-bit)
// groupSize = 6 * (16 / min(16,16)) = 6
// blocksPerGroup = 6 * 25 = 150
// CHECK-LABEL: @gemm_basic_bf16_bf16
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 6 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 150 : i32
// CHECK:       %[[MBLK:.*]] = arith.constant 4 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[GROUPID:.*]] = arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       %[[FIRSTM:.*]] = arith.muli %[[GROUPID]], %[[GPSIZE]] : i32
// CHECK:       %[[MDIFF:.*]] = arith.subi %[[MBLK]], %[[FIRSTM]] : i32
// CHECK:       %[[THISMPG:.*]] = arith.minui %[[MDIFF]], %[[GPSIZE]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[THISMPG]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.divui %{{.*}}, %[[THISMPG]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_basic_bf16_bf16(%arg0: tensor<51200xbf16>, %arg1: tensor<320000xbf16>, %arg2: tensor<51200xbf16>) -> tensor<51200xbf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xbf16> to tensor<2x64x400xbf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xbf16> to tensor<2x400x400xbf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xbf16> to tensor<2x64x400xbf16>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xbf16> to tensor<2x400x400xbf16>
  %4 = rock.gridwise_gemm(%0, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xbf16>, tensor<2x400x400xbf16> -> tensor<2x64x400xbf16>
  %5 = rock.store %4 to %2 by set : tensor<2x64x400xbf16> -> tensor<51200xbf16> to tensor<2x64x400xbf16>
  return %5 : tensor<51200xbf16>
}

// -----

// bf16 inputs, f32 output. Mixed precision widens the groupSize ratio.
// inputType = bf16 (16-bit), outputType = f32 (32-bit)
// groupSize = 6 * (32 / min(16,32)) = 6 * 2 = 12
// blocksPerGroup = 12 * 25 = 300
// CHECK-LABEL: @gemm_basic_bf16_f32
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 12 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 300 : i32
// CHECK:       %[[MBLK:.*]] = arith.constant 4 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[GROUPID:.*]] = arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       %[[FIRSTM:.*]] = arith.muli %[[GROUPID]], %[[GPSIZE]] : i32
// CHECK:       %[[MDIFF:.*]] = arith.subi %[[MBLK]], %[[FIRSTM]] : i32
// CHECK:       %[[THISMPG:.*]] = arith.minui %[[MDIFF]], %[[GPSIZE]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[THISMPG]] : i32
// CHECK:       arith.remui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.divui %{{.*}}, %[[THISMPG]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_basic_bf16_f32(%arg0: tensor<51200xbf16>, %arg1: tensor<320000xbf16>, %arg2: tensor<51200xf32>) -> tensor<51200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xbf16> to tensor<2x64x400xbf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xbf16> to tensor<2x400x400xbf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf32> to tensor<2x64x400xf32>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xbf16> to tensor<2x400x400xbf16>
  %4 = rock.gridwise_gemm(%0, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xbf16>, tensor<2x400x400xbf16> -> tensor<2x64x400xf32>
  %5 = rock.store %4 to %2 by set : tensor<2x64x400xf32> -> tensor<51200xf32> to tensor<2x64x400xf32>
  return %5 : tensor<51200xf32>
}
