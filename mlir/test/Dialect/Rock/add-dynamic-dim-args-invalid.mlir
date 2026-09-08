// RUN: rocmlir-opt -rock-add-dynamic-dim-args -split-input-file -verify-diagnostics %s

// The appended arguments carry G, M, N and K positionally, so a kernel holding
// more than one gemm has to be rejected rather than numbered ambiguously.

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// expected-error @+1 {{dynamic shapes are only supported for kernels with exactly one rock.gemm, found 2}}
func.func @two_gemms(%a: tensor<1x?x72xf32>, %b: tensor<1x72x64xf32>) -> tensor<1x?x64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel = "mixr"} {
  %0 = rock.gemm %a * %b {params = #gemm_params} : tensor<1x?x72xf32> * tensor<1x72x64xf32> -> tensor<1x?x64xf32>
  %1 = rock.gemm %a * %b {params = #gemm_params} : tensor<1x?x72xf32> * tensor<1x72x64xf32> -> tensor<1x?x64xf32>
  return %1 : tensor<1x?x64xf32>
}

// -----

// A kernel with no gemm at all, such as a pure elementwise fusion, is equally
// outside what the positional convention can describe.

// expected-error @+1 {{dynamic shapes are only supported for kernels with exactly one rock.gemm, found 0}}
func.func @no_gemm(%a: tensor<?xf32>) -> tensor<?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel = "mixr"} {
  return %a : tensor<?xf32>
}
