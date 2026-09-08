// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -verify-diagnostics

// Of the gemm dimensions only M may be dynamic here. `rock-gemm-to-gridwise`
// already rejects the others, so these cases guard the pass boundary itself.

#params = #rock.gemm_params<
  mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numCTAs = 1,
  numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2,
  wavesPerEU = 0, gridGroupSize = 0>

// N fixes how many blocks the grid is grouped into.
func.func @gridwise_gemm_dynamic_n(%A: tensor<1x64x64xf32>, %B: tensor<1x64x?xf32>,
                                   %C: tensor<1x64x?xf32>)
    attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 256 : i32,
                rock.kernel} {
  // expected-error @+2 {{lowering a gemm with a dynamic G, N or K dimension is not implemented}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %0 = rock.gridwise_gemm(%A, %B) {params = #params}
    : tensor<1x64x64xf32>, tensor<1x64x?xf32> -> tensor<1x64x?xf32>
  %1 = rock.store %0 to %C by set
    : tensor<1x64x?xf32> -> tensor<1x64x?xf32> to tensor<1x64x?xf32>
  func.return
}

// -----

#params = #rock.gemm_params<
  mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numCTAs = 1,
  numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2,
  wavesPerEU = 0, gridGroupSize = 0>

// K is the trip count of the emitted loop.
func.func @gridwise_gemm_dynamic_k(%A: tensor<1x64x?xf32>, %B: tensor<1x?x64xf32>,
                                   %C: tensor<1x64x64xf32>)
    attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 256 : i32,
                rock.kernel} {
  // expected-error @+2 {{lowering a gemm with a dynamic G, N or K dimension is not implemented}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %0 = rock.gridwise_gemm(%A, %B) {params = #params}
    : tensor<1x64x?xf32>, tensor<1x?x64xf32> -> tensor<1x64x64xf32>
  %1 = rock.store %0 to %C by set
    : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  func.return
}

// -----

#params = #rock.gemm_params<
  mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numCTAs = 1,
  numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2,
  wavesPerEU = 0, gridGroupSize = 0>

// G is a grid coordinate of its own.
func.func @gridwise_gemm_dynamic_g(%A: tensor<?x64x64xf32>, %B: tensor<?x64x64xf32>,
                                   %C: tensor<?x64x64xf32>)
    attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 256 : i32,
                rock.kernel} {
  // expected-error @+2 {{lowering a gemm with a dynamic G, N or K dimension is not implemented}}
  // expected-error @+1 {{failed to legalize operation 'rock.gridwise_gemm'}}
  %0 = rock.gridwise_gemm(%A, %B) {params = #params}
    : tensor<?x64x64xf32>, tensor<?x64x64xf32> -> tensor<?x64x64xf32>
  %1 = rock.store %0 to %C by set
    : tensor<?x64x64xf32> -> tensor<?x64x64xf32> to tensor<?x64x64xf32>
  func.return
}
