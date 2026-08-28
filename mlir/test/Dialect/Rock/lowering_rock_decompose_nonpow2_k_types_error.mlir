// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -rock-decompose-nonpow2-k -canonicalize -verify-diagnostics %s

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 18, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
func.func @gemm_nonpow2_kperblock_i8_narrow_segment_unsupported(%arg0: tensor<1x64x72xi8>, %arg1: tensor<1x72x64xi8>, %arg2: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.num_cu = 228 : i32} {
  // expected-error @+1 {{non-power-of-two K tile is not supported for integer operands when it decomposes into K segments narrower than 4}}
  %0 = rock.gridwise_gemm(%arg0, %arg1) {params = #gemm_params} : tensor<1x64x72xi8>, tensor<1x72x64xi8> -> tensor<1x64x64xi32>
  %1 = rock.store %0 to %arg2 by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  return %1 : tensor<1x64x64xi32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 48, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
func.func @gemm_nonpow2_kperblock_scaled_unsupported(%arg0: tensor<1x64x96xf4E2M1FN>, %arg1: tensor<1x96x64xf4E2M1FN>, %arg2: tensor<1x64x6xf8E8M0FNU>, %arg3: tensor<1x64x6xf8E8M0FNU>, %arg4: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.num_cu = 256 : i32} {
  // expected-error @+1 {{non-power-of-two K tile is not supported for scaled gemm}}
  %0 = rock.gridwise_gemm(%arg0, %arg1, %arg2, %arg3) {params = #gemm_params, quantBlockSize = 16 : i64} : tensor<1x64x96xf4E2M1FN>, tensor<1x96x64xf4E2M1FN>, tensor<1x64x6xf8E8M0FNU>, tensor<1x64x6xf8E8M0FNU> -> tensor<1x64x64xf32>
  %1 = rock.store %0 to %arg4 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %1 : tensor<1x64x64xf32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 48, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
func.func @gemm_nonpow2_kperblock_f4_unsupported(%arg0: tensor<1x64x96xf4E2M1FN>, %arg1: tensor<1x96x64xf4E2M1FN>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.num_cu = 256 : i32} {
  // expected-error @+1 {{non-power-of-two K tile is not supported for sub-byte operands}}
  %0 = rock.gridwise_gemm(%arg0, %arg1) {params = #gemm_params} : tensor<1x64x96xf4E2M1FN>, tensor<1x96x64xf4E2M1FN> -> tensor<1x64x64xf32>
  %1 = rock.store %0 to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %1 : tensor<1x64x64xf32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 48, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
func.func @gemm_nonpow2_kperblock_i4_unsupported(%arg0: tensor<1x64x96xi4>, %arg1: tensor<1x96x64xi4>, %arg2: tensor<1x64x64xi32>) -> tensor<1x64x64xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.num_cu = 256 : i32} {
  // expected-error @+1 {{non-power-of-two K tile is not supported for sub-byte operands}}
  %0 = rock.gridwise_gemm(%arg0, %arg1) {params = #gemm_params} : tensor<1x64x96xi4>, tensor<1x96x64xi4> -> tensor<1x64x64xi32>
  %1 = rock.store %0 to %arg2 by set : tensor<1x64x64xi32> -> tensor<1x64x64xi32> to tensor<1x64x64xi32>
  return %1 : tensor<1x64x64xi32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 48, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#map = affine_map<(d0, d1, d2) -> (d1 * 96 + d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{64, 96} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 96] -> [6144]>
func.func @gemm_nonpow2_kperblock_dequant_i4_unsupported(%arg0: tensor<6144xi4>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.num_cu = 256 : i32} {
  %cst = arith.constant dense<2.500000e-01> : tensor<1x64x96xf16>
  %0 = rock.transform %arg0 by #transform_map : tensor<6144xi4> to tensor<1x64x96xi4>
  %1 = arith.sitofp %0 : tensor<1x64x96xi4> to tensor<1x64x96xf16>
  %2 = arith.mulf %1, %cst : tensor<1x64x96xf16>
  // expected-error @+1 {{non-power-of-two K tile is not supported for sub-byte operands}}
  %3 = rock.gridwise_gemm(%2, %arg1) {params = #gemm_params} : tensor<1x64x96xf16>, tensor<1x96x64xf16> -> tensor<1x64x64xf32>
  %4 = rock.store %3 to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %4 : tensor<1x64x64xf32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 48, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
func.func @gemm_nonpow2_kperblock_quant_f4_unsupported(%arg0: tensor<1x64x96xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.num_cu = 256 : i32} {
  %0 = arith.truncf %arg0 : tensor<1x64x96xf16> to tensor<1x64x96xf4E2M1FN>
  %1 = arith.truncf %arg1 : tensor<1x96x64xf16> to tensor<1x96x64xf4E2M1FN>
  // expected-error @+1 {{non-power-of-two K tile is not supported for sub-byte operands}}
  %2 = rock.gridwise_gemm(%0, %1) {params = #gemm_params} : tensor<1x64x96xf4E2M1FN>, tensor<1x96x64xf4E2M1FN> -> tensor<1x64x64xf32>
  %3 = rock.store %2 to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %3 : tensor<1x64x64xf32>
}
