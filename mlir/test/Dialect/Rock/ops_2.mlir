// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// Run: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

func.func @rock_blockwise_gemm(%A : tensor<8x128x1xf32>, %B : tensor<8x128x1xf32>, %C : tensor<8x8xf32>) -> tensor<8x8xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %result = rock.blockwise_gemm(%A, %B, %C) {
    rock.arch = "amdgcn-amd-amdhsa:gfx90a",
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.gemm_params<
      mPerBlock = 128,
      nPerBlock = 128,
      kPerBlock = 8,
      kpack = 1,
      numWaves = 1,
      matrixInstrNonkdim = 0,
      splitKFactor = 1,
      numStages = 2,
      wavesPerEU = 0,
      gridGroupSize = 0,
      numCTAs = 1>
  } : tensor<8x128x1xf32>, tensor<8x128x1xf32>, tensor<8x8xf32> -> tensor<8x8xf32>
  %stored = rock.store %result to %C by set : tensor<8x8xf32> -> tensor<8x8xf32> to tensor<8x8xf32>
  return %stored : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @rock_blockwise_gemm
// CHECK: rock.blockwise_gemm
// CHECK: rock.store
