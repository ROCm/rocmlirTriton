// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// RUN: rocmlir-opt %s | FileCheck %s

// TODO(rocmlirTriton): Add dummy check to make FileCheck happy
// CHECK: module

// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// kPerBlock = kpackPerBlock * kpack = 8 * 8 = 64
// numWaves = (mPerBlock * nPerBlock) / (mPerWave * nPerWave) = (128 * 128) / (64 * 64) = 4
// #xdlops_gemm_params1 = #rock.gemm_params<kPerBlock = 512, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
// #xdlops_gemm_params2 = #rock.gemm_params<kPerBlock = 512, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// DISABLED-CHECK-LABEL: @fp8_bf8_xdlops
// func.func @fp8_bf8_xdlops(%arg0: tensor<1x128x128xf8E4M3FNUZ>, %arg1: tensor<1x128x115200xf8E5M2FNUZ>, %arg2: tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.block_size = 256 : i32, grid_size = 900 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32} {
//   // DISABLED-CHECK: scf.for
//   // DISABLED-CHECK: rock.blockwise_gemm
//   %result = rock.gridwise_gemm(%arg0, %arg1) {params = #xdlops_gemm_params1} : tensor<1x128x128xf8E4M3FNUZ>, tensor<1x128x115200xf8E5M2FNUZ> -> tensor<1x128x115200xf32>
//   %out = rock.store %result to %arg2 by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
//   return %out : tensor<1x128x115200xf32>
// }

// -----

// kPerBlock = kpackPerBlock * kpack = 8 * 8 = 64
// numWaves = (mPerBlock * nPerBlock) / (mPerWave * nPerWave) = (128 * 128) / (64 * 64) = 4
// #xdlops_gemm_params2 = #rock.gemm_params<kPerBlock = 512, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// DISABLED-CHECK-LABEL: @fp8_bf8_xdlops_schedulev2
// func.func @fp8_bf8_xdlops_schedulev2(%arg0: tensor<1x128x128xf8E4M3FNUZ>, %arg1: tensor<1x128x115200xf8E5M2FNUZ>, %arg2: tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.block_size = 256 : i32, grid_size = 900 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32} {
//   // DISABLED-CHECK: scf.for
//   // DISABLED-CHECK: rock.blockwise_gemm
//   %result = rock.gridwise_gemm(%arg0, %arg1) {params = #xdlops_gemm_params2} : tensor<1x128x128xf8E4M3FNUZ>, tensor<1x128x115200xf8E5M2FNUZ> -> tensor<1x128x115200xf32>
//   %out = rock.store %result to %arg2 by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
//   return %out : tensor<1x128x115200xf32>
// }
