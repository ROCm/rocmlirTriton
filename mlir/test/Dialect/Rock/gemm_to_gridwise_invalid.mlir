// RUN: rocmlir-opt -rock-gemm-to-gridwise -split-input-file -verify-diagnostics %s

// The number of blocks the grid is grouped into has to be a compile-time
// constant, so only the M extent of a gemm can be left until run time. A
// dynamic G or N has to be rejected with a diagnostic instead of computing a
// nonsensical grid.

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

func.func @gemm_dynamic_g(%a: tensor<?x64x72xf32>, %b: tensor<?x72x64xf32>, %c: tensor<?x64x64xf32>) -> tensor<?x64x64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  // expected-error @+2 {{cannot compute a grid size for a gemm with a dynamic G or N dimension}}
  // expected-error @+1 {{failed to legalize operation 'rock.gemm'}}
  %result = rock.gemm %a * %b {
    params = #gemm_params
  } : tensor<?x64x72xf32> * tensor<?x72x64xf32> -> tensor<?x64x64xf32>
  %out = rock.store %result to %c by set : tensor<?x64x64xf32> -> tensor<?x64x64xf32> to tensor<?x64x64xf32>
  func.return %out : tensor<?x64x64xf32>
}

// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

func.func @gemm_dynamic_n(%a: tensor<1x64x72xf32>, %b: tensor<1x72x?xf32>, %c: tensor<1x64x?xf32>) -> tensor<1x64x?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  // expected-error @+2 {{cannot compute a grid size for a gemm with a dynamic G or N dimension}}
  // expected-error @+1 {{failed to legalize operation 'rock.gemm'}}
  %result = rock.gemm %a * %b {
    params = #gemm_params
  } : tensor<1x64x72xf32> * tensor<1x72x?xf32> -> tensor<1x64x?xf32>
  %out = rock.store %result to %c by set : tensor<1x64x?xf32> -> tensor<1x64x?xf32> to tensor<1x64x?xf32>
  func.return %out : tensor<1x64x?xf32>
}
