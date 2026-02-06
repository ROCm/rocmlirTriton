// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// Run: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

func.func @rock_blockwise_gemm_f16(%A : tensor<8x128x1xf16>, %B : tensor<8x128x1xf16>, %C : tensor<8x8xf16>) -> tensor<8x8xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  %result = rock.blockwise_gemm(%A, %B, %C) {
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.gemm_params<
      kPerBlock = 8,
      mPerBlock = 256,
      nPerBlock = 256,
      kpack = 1,
      numWaves = 1,
      matrixInstrNonkdim = 0,
      splitKFactor = 1,
      numStages = 2,
      wavesPerEU = 0,
      gridGroupSize = 0,
      numCTAs = 1>
  } : tensor<8x128x1xf16>, tensor<8x128x1xf16>, tensor<8x8xf16> -> tensor<8x8xf16>
  %out = rock.store %result to %C by set : tensor<8x8xf16> -> tensor<8x8xf16> to tensor<8x8xf16>
  return %out : tensor<8x8xf16>
}

// CHECK-LABEL: func.func @rock_blockwise_gemm_f16
//  CHECK: rock.blockwise_gemm
