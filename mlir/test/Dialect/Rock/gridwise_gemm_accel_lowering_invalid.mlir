// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics %s
// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics %s \
// RUN:   --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NA --implicit-check-not=rock.not_applicable

// -----

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// Test case: LDS size exceeds architecture limit
// For gfx942, maxSharedMemPerWG = 65536 bytes
// This test allocates LDS larger than 65536 bytes
// kPerBlock * mPerBlock * kpack * sizeof(f32) + kPerBlock * nPerBlock * kpack * sizeof(f32)
// = 32 * 256 * 8 * 4 + 32 * 256 * 8 * 4 = 262144 + 262144 = 524288 bytes > 65536
// Format: A (G x K x M), B (G x K x N), C (G x M x N)
// #xdlops_gemm_params_too_much_lds = #rock.gemm_params<kPerBlock = 256, mPerBlock = 256, nPerBlock = 256, kpack = 8, numWaves = 16, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
// func.func @excessive_lds_usage(%arg0: tensor<1x256x256xf32>, %arg1: tensor<1x256x256xf32>, %arg2: tensor<1x256x256xf32>) -> tensor<1x256x256xf32> attributes {rock.block_size = 256 : i32, grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 304 : i32} {
//   // expected-disabled-error @+2 {{requires too much LDS}}
//   // expected-disabled-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
//   %result = rock.gridwise_gemm(%arg0, %arg1) {params = #xdlops_gemm_params_too_much_lds} : tensor<1x256x256xf32>, tensor<1x256x256xf32> -> tensor<1x256x256xf32>
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
// func.func @gfx950_lds_exceeded(%arg0: tensor<1x128x512xf32>, %arg1: tensor<1x128x512xf32>, %arg2: tensor<1x512x512xf32>) -> tensor<1x512x512xf32> attributes {rock.block_size = 256 : i32, grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i32} {
//   // expected-disabled-error @+2 {{requires too much LDS}}
//   // expected-disabled-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
//   %result = rock.gridwise_gemm(%arg0, %arg1) {params = #xdlops_gemm_params_gfx950_lds} : tensor<1x128x512xf32>, tensor<1x128x512xf32> -> tensor<1x512x512xf32>
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
// func.func @scaled_gemm_lds_exceeded(%arg0: tensor<1x1024x256xf4E2M1FN>, %arg1: tensor<1x1024x256xf4E2M1FN>, %arg2: tensor<1x256x256xf32>, %scaleA: tensor<1x1024x256xf8E8M0FNU>, %scaleB: tensor<1x1024x256xf8E8M0FNU>) -> tensor<1x256x256xf32> attributes {rock.block_size = 256 : i32, grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i32} {
//   // expected-disabled-error @+2 {{requires too much LDS}}
//   // expected-disabled-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
//   %result = rock.gridwise_gemm(%arg0, %arg1, %scaleA, %scaleB) features = mfma {params = #xdlops_gemm_params_scaled_lds_exceeded} : tensor<1x1024x256xf4E2M1FN>, tensor<1x1024x256xf4E2M1FN>, tensor<1x1024x256xf8E8M0FNU>, tensor<1x1024x256xf8E8M0FNU> -> tensor<1x256x256xf32>
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
// func.func @scaled_gemm_lds_exceeded_alt(%arg0: tensor<1x1024x512xf4E2M1FN>, %arg1: tensor<1x1024x512xf4E2M1FN>, %arg2: tensor<1x512x512xf32>, %scaleA: tensor<1x1024x512xf8E8M0FNU>, %scaleB: tensor<1x1024x512xf8E8M0FNU>) -> tensor<1x512x512xf32> attributes {rock.block_size = 256 : i32, grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 256 : i32} {
//   // expected-disabled-error @+2 {{requires too much LDS}}
//   // expected-disabled-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
//   %result = rock.gridwise_gemm(%arg0, %arg1, %scaleA, %scaleB) features = mfma {params = #xdlops_gemm_params_scaled_lds_exceeded2} : tensor<1x1024x512xf4E2M1FN>, tensor<1x1024x512xf4E2M1FN>, tensor<1x1024x512xf8E8M0FNU>, tensor<1x1024x512xf8E8M0FNU> -> tensor<1x512x512xf32>
//   %out = rock.store %result to %arg2 by set : tensor<1x512x512xf32> -> tensor<1x512x512xf32> to tensor<1x512x512xf32>
//   return %out : tensor<1x512x512xf32>
// }

// -----

// Verifies that a scaled gemm whose `kPerBlock` is not a multiple of
// `quantBlockSize` is classified as not-applicable (rather than a real
// compilation bug): the pass marks the enclosing module with
// `rock.not_applicable` before signalling failure so the tuning driver can
// distinguish "config doesn't fit" from a real compiler error.
//
// kPerBlock=4, quantBlockSize=8, so 4 % 8 != 0 triggers the divisibility check.
#bad_qb_params = #rock.gemm_params<
  kPerBlock = 4, kpack = 1, mPerBlock = 8, nPerBlock = 16,
  numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2,
  wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// NA-LABEL: 'func.func' operation: @gridwise_gemm_kperblock_not_multiple_of_quantblock
// NA: module attributes {rock.not_applicable
func.func @gridwise_gemm_kperblock_not_multiple_of_quantblock(
    %A: tensor<1x8x64xf4E2M1FN>, %B: tensor<1x64x16xf4E2M1FN>,
    %scaleA: tensor<1x8x8xf8E8M0FNU>, %scaleB: tensor<1x16x8xf8E8M0FNU>,
    %C: tensor<1x8x16xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950",
                                        rock.block_size = 64 : i32,
                                        rock.grid_size = 1 : i32,
                                        rock.kernel} {
  // expected-error @+2 {{kPerBlock is not a multiple of quantBlockSize}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) {
    quantBlockSize = 8 : i64,
    params = #bad_qb_params
  } : tensor<1x8x64xf4E2M1FN>, tensor<1x64x16xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>, tensor<1x16x8xf8E8M0FNU> -> tensor<1x8x16xf32>
  %stored = rock.store %result to %C by set : tensor<1x8x16xf32> -> tensor<1x8x16xf32> to tensor<1x8x16xf32>
  func.return
}
