// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// CHECK-NOT: rock.gridwise_gemm
// CHECK-LABEL: @gemm_nonpow2_kperblock_48
// CHECK-SAME: -> tensor<1x64x64xf32>
func.func @gemm_nonpow2_kperblock_48(%arg0: tensor<1x64x96xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32, rock.kernel} {
  // CHECK-DAG: %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  // CHECK:     %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<64x64xf32>)

  // CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: %[[ACC2:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC1]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: scf.yield %[[ACC2]] : tensor<64x64xf32>

  // CHECK: rock.store_marker %[[OUT]] {{.*}} : tensor<64x64xf32> -> tensor<1x64x64xf32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #rock.gemm_params<kPerBlock = 48, mPerBlock = 64, nPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>} : tensor<1x64x96xf16>, tensor<1x96x64xf16> -> tensor<1x64x64xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}
