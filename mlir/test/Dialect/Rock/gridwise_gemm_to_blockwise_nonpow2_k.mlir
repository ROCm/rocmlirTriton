// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// kPerBlock is a power of two, but K need not be a multiple of it. The
// [0, kMain) region (kMain = floor(K/kPerBlock)*kPerBlock) is reduced by the
// main loop, one kPerBlock-wide blockwise_gemm per iteration, and the leftover
// K tail is peeled into power-of-two segments contracted *after* the loop.

// K = 176, kPerBlock = 64 -> main loop over [0, 128), tail 48 -> {32, 16}.
// CHECK-NOT: rock.gridwise_gemm
// CHECK-LABEL: @gemm_k_tail_two_segments
// CHECK-SAME: -> tensor<1x64x64xf32>
func.func @gemm_k_tail_two_segments(%arg0: tensor<1x64x176xf16>, %arg1: tensor<1x176x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // CHECK-DAG: %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  // CHECK:     %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<64x64xf32>)
  // CHECK:     %[[ACCL:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     scf.yield %[[ACCL]] : tensor<64x64xf32>

  // Peeled tail, contracted after the loop into the loop result.
  // CHECK:     %[[T0:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[OUT]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     %[[T1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[T0]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     rock.store_marker %[[T1]] {{.*}} : tensor<64x64xf32> -> tensor<1x64x64xf32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x176xf16>, tensor<1x176x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}

// -----

// K = 368, kPerBlock = 128 -> main loop over [0, 256), tail 112 -> {64, 32, 16}.
// CHECK-NOT: rock.gridwise_gemm
// CHECK-LABEL: @gemm_k_tail_three_segments
// CHECK-SAME: -> tensor<1x64x64xf32>
func.func @gemm_k_tail_three_segments(%arg0: tensor<1x64x368xf16>, %arg1: tensor<1x368x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // CHECK-DAG: %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  // CHECK:     %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<64x64xf32>)
  // CHECK:     %[[ACCL:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<64x128xf16>, tensor<128x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     scf.yield %[[ACCL]] : tensor<64x64xf32>

  // Peeled tail {64, 32, 16}, contracted after the loop.
  // CHECK:     %[[T0:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[OUT]]) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     %[[T1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[T0]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     %[[T2:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[T1]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     rock.store_marker %[[T2]] {{.*}} : tensor<64x64xf32> -> tensor<1x64x64xf32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 128, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x368xf16>, tensor<1x368x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}

// -----

// K = 128, kPerBlock = 64 -> K is a multiple of kPerBlock, no tail to peel.
// The operands are used untouched (no slice transform) and a single
// blockwise_gemm runs in the loop, identical to the pre-peeling fast path.
// CHECK-LABEL: @gemm_pow2_k_no_peel
// CHECK:     %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
// CHECK-NOT: rock.transform
// CHECK:     %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<64x64xf32>)
// CHECK:     %[[ACC1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
// CHECK:     scf.yield %[[ACC1]] : tensor<64x64xf32>
// CHECK-NOT: rock.blockwise_gemm
func.func @gemm_pow2_k_no_peel(%arg0: tensor<1x64x128xf16>, %arg1: tensor<1x128x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x128xf16>, tensor<1x128x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}

// -----

// K = 48 < kPerBlock = 64 -> no main loop at all (kMain = 0); the whole K is
// peeled into {32, 16} and reduced into the zero accumulator directly.
// CHECK-NOT: rock.gridwise_gemm
// CHECK-LABEL: @gemm_k_below_kperblock
// CHECK-SAME: -> tensor<1x64x64xf32>
func.func @gemm_k_below_kperblock(%arg0: tensor<1x64x48xf16>, %arg1: tensor<1x48x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // CHECK:     %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  // CHECK-NOT: scf.for
  // CHECK:     %[[T0:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC0]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     %[[T1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[T0]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK:     rock.store_marker %[[T1]] {{.*}} : tensor<64x64xf32> -> tensor<1x64x64xf32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 64, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x48xf16>, tensor<1x48x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}
