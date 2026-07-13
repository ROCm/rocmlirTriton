// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// Non-power-of-two per-block K tile (kPerBlock = 48). The Triton layouts
// produced downstream require power-of-two tensor shapes, so the pass peels the
// K tile into the minimal largest-first power-of-two cover 48 -> {32, 16}. Each
// segment contributes one blockwise_gemm that accumulates into the *same*
// per-workgroup accumulator tile (an in-register reduction along K), so the
// outer K-loop trip count, grid layout, and the entire output/store side stay
// identical to the pow2 case and no atomics are needed.

// Config: G=1, M=64, N=64, K=96, mPerBlock=64, nPerBlock=64, kPerBlock=48,
// kpack=1. mBlocks = nBlocks = 1, so the grid is a single tile. Outer K-loop
// iterations = K / kPerBlock = 96 / 48 = 2.

// rock.gridwise_gemm must be fully consumed by this pass.
// CHECK-NOT: rock.gridwise_gemm

// CHECK-LABEL: @gemm_nonpow2_kperblock_48
// CHECK-SAME: -> tensor<1x64x64xf32>
func.func @gemm_nonpow2_kperblock_48(%arg0: tensor<1x64x96xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // Zero-initialized accumulator (mPerBlock x nPerBlock = 64 x 64), threaded as
  // the K-loop iter_arg.
  // CHECK-DAG: %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  // CHECK: %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<64x64xf32>)

  // Segment 0: 32-wide (largest power-of-two <= 48), offset 0. Loads read a
  // pow2 tile (mPerBlock x 32 for A, 32 x nPerBlock for B) from the K-sliced
  // operand views.
  // CHECK-DAG: rock.load_marker %{{.*}} -> tensor<32x64xf16>
  // CHECK-DAG: rock.load_marker %{{.*}} -> tensor<64x32xf16>
  // CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>

  // Segment 1: 16-wide remainder, offset 32, accumulating on top of segment 0.
  // CHECK-DAG: rock.load_marker %{{.*}} -> tensor<16x64xf16>
  // CHECK-DAG: rock.load_marker %{{.*}} -> tensor<64x16xf16>
  // CHECK: %[[ACC2:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC1]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: scf.yield %[[ACC2]] : tensor<64x64xf32>

  // No non-power-of-two (48-wide) contraction tiles are handed to blockwise_gemm.
  // CHECK-NOT: tensor<64x48xf16>
  // CHECK-NOT: tensor<48x64xf16>

  // Result written back through the output view, unchanged from the pow2 case.
  // CHECK: rock.store_marker %[[OUT]] {{.*}} : tensor<64x64xf32> -> tensor<1x64x64xf32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 48, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x96xf16>, tensor<1x96x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}
