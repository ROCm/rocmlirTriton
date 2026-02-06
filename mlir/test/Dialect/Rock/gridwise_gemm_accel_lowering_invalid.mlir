// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics %s

// -----

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// Test case: LDS size exceeds architecture limit
// For gfx942, maxSharedMemPerWG = 65536 bytes
// This test allocates LDS larger than 65536 bytes
// kPerBlock * mPerBlock * kpack * sizeof(f32) + kPerBlock * nPerBlock * kpack * sizeof(f32)
// = 32 * 256 * 8 * 4 + 32 * 256 * 8 * 4 = 262144 + 262144 = 524288 bytes > 65536
// Format: A (G x K x M), B (G x K x N), C (G x M x N)
// #xdlops_gemm_params_too_much_lds = #rock.gemm_params<kPerBlock = 256, mPerBlock = 256, nPerBlock = 256, kpack = 8, numWaves = 16, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
// func.func @excessive_lds_usage(%arg0: tensor<1x256x256xf32>, %arg1: tensor<1x256x256xf32>, %arg2: tensor<1x256x256xf32>) -> tensor<1x256x256xf32> attributes {rock.block_size = 256 : i32, grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.numCU = 304 : i32} {
//   // expected-disabled-error @+2 {{requires too much LDS}}
//   // expected-disabled-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
//   %result = rock.gridwise_gemm(%arg0, %arg1) {blockSize = 256 : i32, gridSize = 1 : i32, params = #xdlops_gemm_params_too_much_lds} : tensor<1x256x256xf32>, tensor<1x256x256xf32> -> tensor<1x256x256xf32>
//   %out = rock.store %result to %arg2 by set : tensor<1x256x256xf32> -> tensor<1x256x256xf32> to tensor<1x256x256xf32>
//   return %out : tensor<1x256x256xf32>
// }

// -----

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// Test case: LDS size exceeds limit for gfx950 (160KB limit)
// kPerBlock * mPerBlock * kpack * sizeof(f32) + kPerBlock * nPerBlock * kpack * sizeof(f32) > 163840
// 16 * 512 * 8 * 4 + 16 * 512 * 8 * 4 = 262144 + 262144 = 524288 bytes > 163840
// Format: A (G x K x M), B (G x K x N), C (G x M x N)
// #xdlops_gemm_params_gfx950_lds = #rock.gemm_params<kPerBlock = 128, mPerBlock = 512, nPerBlock = 512, kpack = 8, numWaves = 64, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
// func.func @gfx950_lds_exceeded(%arg0: tensor<1x128x512xf32>, %arg1: tensor<1x128x512xf32>, %arg2: tensor<1x512x512xf32>) -> tensor<1x512x512xf32> attributes {rock.block_size = 256 : i32, grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.numCU = 256 : i32} {
//   // expected-disabled-error @+2 {{requires too much LDS}}
//   // expected-disabled-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
//   %result = rock.gridwise_gemm(%arg0, %arg1) {blockSize = 256 : i32, gridSize = 1 : i32, params = #xdlops_gemm_params_gfx950_lds} : tensor<1x128x512xf32>, tensor<1x128x512xf32> -> tensor<1x512x512xf32>
//   %out = rock.store %result to %arg2 by set : tensor<1x512x512xf32> -> tensor<1x512x512xf32> to tensor<1x512x512xf32>
//   return %out : tensor<1x512x512xf32>
// }

// -----

// TODO(roctriton): Scaled gemm tests need rework
// Test case: Scaled GEMM with f4E2M1FN exceeds LDS limit for gfx950 (160KB limit)
// For f4E2M1FN (4-bit float), sizeof(f4E2M1FN) = 0.5 bytes
// kPerBlock * mPerBlock * kpack * sizeof(f4E2M1FN) + kPerBlock * nPerBlock * kpack * sizeof(f4E2M1FN)
// = 32 * 256 * 32 * 0.5 + 32 * 256 * 32 * 0.5 = 131072 + 131072 = 262144 bytes > 163840
// Format: A (G x K x M), B (G x K x N), C (G x M x N), scaleA (G x K x M), scaleB (G x K x N)
// #xdlops_gemm_params_scaled_lds_exceeded = #rock.gemm_params<kPerBlock = 1024, mPerBlock = 256, nPerBlock = 256, kpack = 32, numWaves = 64, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
// func.func @scaled_gemm_lds_exceeded(%arg0: tensor<1x1024x256xf4E2M1FN>, %arg1: tensor<1x1024x256xf4E2M1FN>, %arg2: tensor<1x256x256xf32>, %scaleA: tensor<1x1024x256xf8E8M0FNU>, %scaleB: tensor<1x1024x256xf8E8M0FNU>) -> tensor<1x256x256xf32> attributes {rock.block_size = 256 : i32, grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.numCU = 256 : i32} {
//   // expected-disabled-error @+2 {{requires too much LDS}}
//   // expected-disabled-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
//   %result = rock.gridwise_gemm(%arg0, %arg1, %scaleA, %scaleB) features = mfma {blockSize = 256 : i32, gridSize = 1 : i32, params = #xdlops_gemm_params_scaled_lds_exceeded} : tensor<1x1024x256xf4E2M1FN>, tensor<1x1024x256xf4E2M1FN>, tensor<1x1024x256xf8E8M0FNU>, tensor<1x1024x256xf8E8M0FNU> -> tensor<1x256x256xf32>
//   %out = rock.store %result to %arg2 by set : tensor<1x256x256xf32> -> tensor<1x256x256xf32> to tensor<1x256x256xf32>
//   return %out : tensor<1x256x256xf32>
// }

// -----

// TODO(roctriton): Scaled gemm tests need rework
// Test case: Another scaled GEMM configuration exceeding LDS for gfx950
// kPerBlock * mPerBlock * kpack * sizeof(f4E2M1FN) + kPerBlock * nPerBlock * kpack * sizeof(f4E2M1FN)
// = 32 * 512 * 32 * 0.5 + 32 * 512 * 32 * 0.5 = 262144 + 262144 = 524288 bytes > 163840
// Format: A (G x K x M), B (G x K x N), C (G x M x N), scaleA (G x K x M), scaleB (G x K x N)
// #xdlops_gemm_params_scaled_lds_exceeded2 = #rock.gemm_params<kPerBlock = 1024, mPerBlock = 512, nPerBlock = 512, kpack = 32, numWaves = 256, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
// func.func @scaled_gemm_lds_exceeded_alt(%arg0: tensor<1x1024x512xf4E2M1FN>, %arg1: tensor<1x1024x512xf4E2M1FN>, %arg2: tensor<1x512x512xf32>, %scaleA: tensor<1x1024x512xf8E8M0FNU>, %scaleB: tensor<1x1024x512xf8E8M0FNU>) -> tensor<1x512x512xf32> attributes {rock.block_size = 256 : i32, grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.numCU = 256 : i32} {
//   // expected-disabled-error @+2 {{requires too much LDS}}
//   // expected-disabled-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
//   %result = rock.gridwise_gemm(%arg0, %arg1, %scaleA, %scaleB) features = mfma {blockSize = 256 : i32, gridSize = 1 : i32, params = #xdlops_gemm_params_scaled_lds_exceeded2} : tensor<1x1024x512xf4E2M1FN>, tensor<1x1024x512xf4E2M1FN>, tensor<1x1024x512xf8E8M0FNU>, tensor<1x1024x512xf8E8M0FNU> -> tensor<1x512x512xf32>
//   %out = rock.store %result to %arg2 by set : tensor<1x512x512xf32> -> tensor<1x512x512xf32> to tensor<1x512x512xf32>
//   return %out : tensor<1x512x512xf32>
// }
