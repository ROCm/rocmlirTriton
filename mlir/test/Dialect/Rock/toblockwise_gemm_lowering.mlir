// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// kPerBlock = kpackPerBlock * kpack = 8 * 8 = 64
// numWaves = (mPerBlock * nPerBlock) / (mPerWave * nPerWave) = (128 * 128) / (64 * 64) = 4
#xdlops_gemm_params1 = #rock.gemm_params<kPerBlock = 64, mPerBlock = 128, nPerBlock = 128, kpack = 8, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// rock.gridwise_gemm must be fully consumed by this pass.
// CHECK-NOT: rock.gridwise_gemm

// CHECK-LABEL: @fp8_bf8_xdlops
// CHECK-SAME: -> tensor<1x128x115200xf32>
func.func @fp8_bf8_xdlops(%arg0: tensor<1x128x128xf8E4M3FNUZ>, %arg1: tensor<1x128x115200xf8E5M2FNUZ>, %arg2: tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.block_size = 256 : i32, rock.grid_size = 900 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 228 : i32} {
  // Zero-initialized accumulator (mPerBlock x nPerBlock = 128 x 128) and
  // Triton-style program id replaces the old workgroup/workitem id chain.
  // CHECK-DAG: %[[ACC0:.+]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
  // CHECK-DAG: tt.get_program_id x : i32

  // Outer K-iteration loop (kIterations = K/kPerBlock = 128/64 = 2) with the
  // accumulator threaded as an iter_arg.
  // CHECK: %[[OUT:.+]] = scf.for {{.*}} iter_args(%[[ACC:.+]] = %[[ACC0]]) -> (tensor<128x128xf32>)

  // Both operands are loaded through view transforms; tile shapes pin down
  // kPerBlock x nPerBlock for B and mPerBlock x kPerBlock for A.
  // CHECK-DAG: rock.load_marker %arg1{{.*}} : tensor<1x128x115200xf8E5M2FNUZ> -> tensor<64x128xf8E5M2FNUZ>
  // CHECK-DAG: rock.load_marker %arg0{{.*}} : tensor<1x128x128xf8E4M3FNUZ> -> tensor<128x64xf8E4M3FNUZ>

  // Blockwise GEMM consumes both loaded tiles + the accumulator iter_arg and
  // yields the next accumulator value. Shapes pin down the full tiling:
  // mPerBlock=128, nPerBlock=128, kPerBlock=64.
  // CHECK: %[[NEWACC:.+]] = rock.blockwise_gemm(%{{.*}}, %{{.*}}, %[[ACC]]) : tensor<128x64xf8E4M3FNUZ>, tensor<64x128xf8E5M2FNUZ>, tensor<128x128xf32> -> tensor<128x128xf32>
  // CHECK: scf.yield %[[NEWACC]] : tensor<128x128xf32>

  // Result is written back through the output view into the global output.
  // CHECK: rock.store_marker %[[OUT]] {{.*}} : tensor<128x128xf32> -> tensor<1x128x115200xf32>
  %result = rock.gridwise_gemm(%arg0, %arg1) {params = #xdlops_gemm_params1} : tensor<1x128x128xf8E4M3FNUZ>, tensor<1x128x115200xf8E5M2FNUZ> -> tensor<1x128x115200xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out : tensor<1x128x115200xf32>
}
