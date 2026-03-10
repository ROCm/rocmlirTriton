// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics %s | FileCheck %s

// All tests share: G=2, M=64, K=400, N=400, mPerBlock=16, nPerBlock=16
// mBlocks = 64/16 = 4, nBlocks = 400/16 = 25, numCU = 256, numChiplets = 8
//
// makeGroupedGridLayout groupSize heuristic:
//   bitWidthIn  = min(inputType.bitWidth, outputType.bitWidth)
//   bitWidthOut = outputType.bitWidth
//   groupSize   = ceil(sqrt(numCU)) * (bitWidthOut / bitWidthIn) = 16 * ratio
//   blocksPerGroup = groupSize * nBlocks = groupSize * 25
//
// The grid layout IR emitted after tt.get_program_id follows a fixed pattern:
//   1. XCC rearrangement (chiplet-aware reorder of bid)
//   2. Grid coordinate computation using groupSize and blocksPerGroup
// The CHECK lines anchor on tt.get_program_id -> arith.select (end of XCC
// rearrangement) -> groupSize/blocksPerGroup constants -> their uses.

// -----

// Input fusion: arith.extf widens f16 block arg to f32 for the gemm.
// getInputFusionElementType traces through extf to find f16.
// inputType = f16 (16-bit), outputType = f32 (32-bit)
// groupSize = 16 * (32 / min(16,32)) = 16 * 2 = 32
// blocksPerGroup = 32 * 25 = 800
// CHECK-LABEL: @gemm_input_fusion_extf
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 32 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 800 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.muli %{{.*}}, %[[GPSIZE]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_input_fusion_extf(%arg0: tensor<51200xf16>, %arg1: tensor<320000xf32>, %arg2: tensor<51200xf32>) -> tensor<51200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf16> to tensor<2x64x400xf16>
  %ext = arith.extf %0 : tensor<2x64x400xf16> to tensor<2x64x400xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xf32> to tensor<2x400x400xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf32> to tensor<2x64x400xf32>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xf32> to tensor<2x400x400xf32>
  %4 = rock.gridwise_gemm(%ext, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xf32>, tensor<2x400x400xf32> -> tensor<2x64x400xf32>
  %5 = rock.store %4 to %2 by set : tensor<2x64x400xf32> -> tensor<51200xf32> to tensor<2x64x400xf32>
  return %5 : tensor<51200xf32>
}

// -----

// Output fusion: arith.truncf narrows f32 gemm result to f16 before store.
// getOutputFusionElementType traces through truncf + store to find f16 dest arg.
// inputType = f32 (32-bit), outputType = f16 (16-bit)
// bitWidthIn = min(32,16) = 16, bitWidthOut = 16
// groupSize = 16 * (16 / 16) = 16
// blocksPerGroup = 16 * 25 = 400
// CHECK-LABEL: @gemm_output_fusion_truncf
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 16 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 400 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.muli %{{.*}}, %[[GPSIZE]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_output_fusion_truncf(%arg0: tensor<51200xf32>, %arg1: tensor<320000xf32>, %arg2: tensor<51200xf16>) -> tensor<51200xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf32> to tensor<2x64x400xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xf32> to tensor<2x400x400xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf16> to tensor<2x64x400xf16>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xf32> to tensor<2x400x400xf32>
  %4 = rock.gridwise_gemm(%0, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xf32>, tensor<2x400x400xf32> -> tensor<2x64x400xf32>
  %trunc = arith.truncf %4 : tensor<2x64x400xf32> to tensor<2x64x400xf16>
  %5 = rock.store %trunc to %2 by set : tensor<2x64x400xf16> -> tensor<51200xf16> to tensor<2x64x400xf16>
  return %5 : tensor<51200xf16>
}

// -----

// Combined input + output fusion: f16 input extended to f32 for gemm,
// f32 result truncated to f16 for storage.
// inputType = f16 (16-bit), outputType = f16 (16-bit)
// bitWidthIn = min(16,16) = 16, bitWidthOut = 16
// groupSize = 16 * (16 / 16) = 16
// blocksPerGroup = 16 * 25 = 400
// CHECK-LABEL: @gemm_input_output_fusion
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 16 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 400 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.muli %{{.*}}, %[[GPSIZE]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_input_output_fusion(%arg0: tensor<51200xf16>, %arg1: tensor<320000xf32>, %arg2: tensor<51200xf16>) -> tensor<51200xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf16> to tensor<2x64x400xf16>
  %ext = arith.extf %0 : tensor<2x64x400xf16> to tensor<2x64x400xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xf32> to tensor<2x400x400xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf16> to tensor<2x64x400xf16>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xf32> to tensor<2x400x400xf32>
  %4 = rock.gridwise_gemm(%ext, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xf32>, tensor<2x400x400xf32> -> tensor<2x64x400xf32>
  %trunc = arith.truncf %4 : tensor<2x64x400xf32> to tensor<2x64x400xf16>
  %5 = rock.store %trunc to %2 by set : tensor<2x64x400xf16> -> tensor<51200xf16> to tensor<2x64x400xf16>
  return %5 : tensor<51200xf16>
}

// -----

// Biggest tensor: input fusion traces arith.addf to two block args of different
// sizes and types. arg0 (51200 x f16) is larger than arg_bias (400 x f32), so
// getInputFusionElementType picks arg0's element type (f16).
// inputType = f16 (16-bit), outputType = f32 (32-bit)
// groupSize = 16 * (32 / min(16,32)) = 16 * 2 = 32
// blocksPerGroup = 32 * 25 = 800
// If the logic incorrectly chose the smaller tensor (f32), groupSize would be
// 16 instead of 32, causing the CHECKs below to fail.
// CHECK-LABEL: @gemm_input_fusion_biggest_tensor
// CHECK:       %[[BID:.*]] = tt.get_program_id x : i32
// CHECK:       %[[ADJUSTED:.*]] = arith.select %{{.*}}, %[[BID]], %{{.*}} : i32
// CHECK:       %[[GPSIZE:.*]] = arith.constant 32 : i32
// CHECK:       %[[BPG:.*]] = arith.constant 800 : i32
// CHECK:       %[[MNBLK:.*]] = arith.constant 100 : i32
// CHECK:       arith.divui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       %[[BID_IN_G:.*]] = arith.remui %[[ADJUSTED]], %[[MNBLK]] : i32
// CHECK:       arith.divui %[[BID_IN_G]], %[[BPG]] : i32
// CHECK:       arith.muli %{{.*}}, %[[GPSIZE]] : i32
// CHECK:       rock.blockwise_gemm
// CHECK:       rock.store_marker
func.func @gemm_input_fusion_biggest_tensor(%arg0: tensor<51200xf16>, %arg1: tensor<320000xf32>, %arg2: tensor<51200xf32>, %arg_bias: tensor<400xf32>) -> tensor<51200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc-:xnack+", rock.block_size = 128 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 200 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf16> to tensor<2x64x400xf16>
  %ext2 = arith.extf %0 : tensor<2x64x400xf16> to tensor<2x64x400xf32>
  %bias_3d = rock.transform %arg_bias by <affine_map<(d0, d1, d2) -> (d2)> by [<AddDim{2} ["g"] at [0] -> [] at []>, <AddDim{64} ["m"] at [1] -> [] at []>, <PassThrough ["k"] at [2] -> ["k"] at [0]>] bounds = [2, 64, 400] -> [400]> : tensor<400xf32> to tensor<2x64x400xf32>
  %fused = arith.addf %ext2, %bias_3d : tensor<2x64x400xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 400 + d1) * 400 + d2)> by [<Unmerge{2, 400, 400} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 400, 400] -> [320000]> : tensor<320000xf32> to tensor<2x400x400xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 400 + d2)> by [<Unmerge{2, 64, 400} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [2, 64, 400] -> [51200]> : tensor<51200xf32> to tensor<2x64x400xf32>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmN"] at [1, 2] -> ["gemmK", "gemmN"] at [2, 1]>] bounds = [2, 400, 400] -> [2, 400, 400]> : tensor<2x400x400xf32> to tensor<2x400x400xf32>
  %4 = rock.gridwise_gemm(%fused, %3) {params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>} : tensor<2x64x400xf32>, tensor<2x400x400xf32> -> tensor<2x64x400xf32>
  %5 = rock.store %4 to %2 by set : tensor<2x64x400xf32> -> tensor<51200xf32> to tensor<2x64x400xf32>
  return %5 : tensor<51200xf32>
}
