// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

#gemm_params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_i32_wants_i8(%a: tensor<1x16x16xf32>,
                        %b: tensor<1x16x16xf32>,
                        %c: tensor<1x16x16xi32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op floating-point input type 'f32' requires a floating-point output type, but the output type is 'i32'}}
  %result = rock.gridwise_gemm(%a, %b) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #gemm_params0}
  : tensor<1x16x16xf32>, tensor<1x16x16xf32> -> tensor<1x16x16xi32>
  func.return
}

// -----

#gemm_params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_i8_wants_i32(%a: tensor<1x16x16xi8>,
                        %b: tensor<1x16x16xi8>,
                        %c: tensor<1x16x16xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op integer input type 'i8' requires an integer output type, but the output type is 'f32'}}
  %result = rock.gridwise_gemm(%a, %b) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #gemm_params1}
  : tensor<1x16x16xi8>, tensor<1x16x16xi8> -> tensor<1x16x16xf32>
  func.return
}

// -----

#gemm_params2 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_m_too_big(%a: tensor<1x2147483648x1xf32>,
                        %b: tensor<1x1x1xf32>,
                        %c: tensor<1x2147483648x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op M dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #gemm_params2}
  : tensor<1x2147483648x1xf32>, tensor<1x1x1xf32> -> tensor<1x2147483648x1xf32>
  func.return
}

// -----

#gemm_params3 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_k_too_big(%a: tensor<1x1x2147483648xf32>,
                        %b: tensor<1x2147483648x1xf32>,
                        %c: tensor<1x1x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op K dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #gemm_params3}
  : tensor<1x1x2147483648xf32>, tensor<1x2147483648x1xf32> -> tensor<1x1x1xf32>
  func.return
}
// -----

#gemm_params4 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_n_too_big(%a: tensor<1x1x1xf32>,
                        %b: tensor<1x1x2147483648xf32>,
                        %c: tensor<1x1x2147483648xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op N dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #gemm_params4}
  : tensor<1x1x1xf32>, tensor<1x1x2147483648xf32> -> tensor<1x1x2147483648xf32>
  func.return
}
