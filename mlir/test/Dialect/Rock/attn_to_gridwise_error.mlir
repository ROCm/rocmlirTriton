// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Errors reported by attn-to-gridwise when setting up split-k for
// gemm-gemm-like ops.

// RUN: rocmlir-opt -rock-attn-to-gridwise -split-input-file -verify-diagnostics %s

#params_g0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 4, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#params_g1_splitk = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// Attention can never use split-k: softmax reduces over gemmN, the very
// dimension split-k partitions, and the padding introduced by the split would
// need -inf rather than zero in the masked lanes. Keep this rejection pinned,
// since the inter-gemm split-k path relies on it to stay out of the softmax
// case entirely.
func.func @attention_rejects_splitk(%q: tensor<1x64x1024xf32>, %k: tensor<1x64x1024xf32>, %v: tensor<1x1024x64xf32>, %out: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // expected-error @+2 {{split-k is not supported for attention}}
  // expected-error @+1 {{failed to legalize operation 'rock.attention'}}
  %result = rock.attention{
    qk = tr %q * %k : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %v : tensor<1x1024x64xf32>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32,
     params0 = #params_g0, params1 = #params_g1_splitk} -> tensor<1x1024x64xf32>
  %stored = rock.store %result to %out by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
  return %stored : tensor<1x1024x64xf32>
}

// -----

#params_g0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 4, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#params_g1_splitk = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// Split-k reshapes the inter-gemm body into the split space, which is only
// possible for a constant whose value does not depend on its position. A
// non-splat constant in the body therefore cannot be carried across the split.
func.func @intergemm_body_non_splat_constant(%a: tensor<1x4x4xf32>, %b: tensor<1x4x4xf32>, %c: tensor<1x4x4xf32>, %out: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // expected-error @+2 {{non-splat constant in elementwise body cannot be reshaped}}
  // expected-error @+1 {{failed to legalize operation 'rock.gemm_elementwise_gemm'}}
  %result = rock.gemm_elementwise_gemm{
    ab = %a * %b : tensor<1x4x4xf32>, tensor<1x4x4xf32>
    ab = elementwise {
    ^bb0(%ab_in: tensor<1x4x4xf32>):
      %cst = arith.constant dense<[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]> : tensor<1x4x4xf32>
      %scaled = arith.mulf %ab_in, %cst : tensor<1x4x4xf32>
      rock.yield %scaled : tensor<1x4x4xf32>
    }
    out = ab * %c : tensor<1x4x4xf32>
  } {params0 = #params_g0, params1 = #params_g1_splitk} -> tensor<1x4x4xf32>
  %stored = rock.store %result to %out by set : tensor<1x4x4xf32> -> tensor<1x4x4xf32> to tensor<1x4x4xf32>
  return %stored : tensor<1x4x4xf32>
}
