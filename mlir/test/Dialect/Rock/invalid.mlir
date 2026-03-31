// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

#gemm_params2 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_m_too_big(%a: tensor<1x2147483648x1xf32>,
                        %b: tensor<1x1x1xf32>,
                        %c: tensor<1x2147483648x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op M dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    gridSize = 1 : i32,
    rock.numCU = 64 : i32,
    params = #gemm_params2}
  : tensor<1x2147483648x1xf32>, tensor<1x1x1xf32> -> tensor<1x2147483648x1xf32>
  func.return
}

// -----

#gemm_params3 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_k_too_big(%a: tensor<1x1x2147483648xf32>,
                        %b: tensor<1x2147483648x1xf32>,
                        %c: tensor<1x1x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op K dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    gridSize = 1 : i32,
    rock.numCU = 64 : i32,
    params = #gemm_params3}
  : tensor<1x1x2147483648xf32>, tensor<1x2147483648x1xf32> -> tensor<1x1x1xf32>
  func.return
}
// -----

#gemm_params4 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_n_too_big(%a: tensor<1x1x1xf32>,
                        %b: tensor<1x1x2147483648xf32>,
                        %c: tensor<1x1x2147483648xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op N dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
    gridSize = 1 : i32,
    rock.numCU = 64 : i32,
    params = #gemm_params4}
  : tensor<1x1x1xf32>, tensor<1x1x2147483648xf32> -> tensor<1x1x2147483648xf32>
  func.return
}

// -----

func.func @expand_strides_rank_mismatch(%input: tensor<4x24xf16>) -> tensor<4x48x24xf16> {
  // expected-error@+1 {{'rock.expand_strides' op input and result must have the same rank}}
  %result = rock.expand_strides %input : tensor<4x24xf16> -> tensor<4x48x24xf16>
  return %result : tensor<4x48x24xf16>
}

// -----

func.func @expand_strides_output_too_small(%input: tensor<4x24x24xf16>) -> tensor<4x20x24xf16> {
  // expected-error@+1 {{'rock.expand_strides' op result dimension 20 is smaller than input dimension 24}}
  %result = rock.expand_strides %input : tensor<4x24x24xf16> -> tensor<4x20x24xf16>
  return %result : tensor<4x20x24xf16>
}

// -----

func.func @expand_strides_element_type_mismatch(%input: tensor<4x24x24xf16>) -> tensor<4x48x24xf32> {
  // expected-error@+1 {{'rock.expand_strides' op input and result must have the same element type}}
  %result = rock.expand_strides %input : tensor<4x24x24xf16> -> tensor<4x48x24xf32>
  return %result : tensor<4x48x24xf32>
}

