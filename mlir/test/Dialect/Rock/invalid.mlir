// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

#gemm_params2 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
func.func @gridwise_gemm_m_too_big(%a: tensor<1x2147483648x1xf32>,
                        %b: tensor<1x1x1xf32>,
                        %c: tensor<1x2147483648x1xf32>) attributes {rock.arch = "gfx906"} {
  // expected-error@+1 {{'rock.gridwise_gemm' op M dimmension 2147483648 cannot be greater than int32_max 2147483647}}
  %result = rock.gridwise_gemm(%a, %b) {
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
    params = #gemm_params4}
  : tensor<1x1x1xf32>, tensor<1x1x2147483648xf32> -> tensor<1x1x2147483648xf32>
  func.return
}

// -----

// Only Pad transforms may be marked as tile-alignment padding.
// expected-error @+1 {{Only a Pad can align a gemm dimension to the tile size}}
#invalid_tile_alignment = #rock.transform<PassThrough tileAlignment ["m"] at [0] -> ["m"] at [0]>
