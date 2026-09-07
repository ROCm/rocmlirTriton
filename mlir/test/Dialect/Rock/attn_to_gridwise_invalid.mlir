// RUN: rocmlir-opt -rock-attn-to-gridwise -split-input-file -verify-diagnostics %s

// The grid size is a compile-time constant that gets baked into both the kernel
// binary and the launch, so attention operands with a dynamic dimension have to
// be rejected with a diagnostic instead of computing a nonsensical grid.

#attn_params_g0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 4, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#attn_params_g1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

func.func @attention_dynamic_seq_len_q(%q: tensor<1x64x?xf32>, %k: tensor<1x64x1024xf32>, %v: tensor<1x1024x64xf32>, %o: tensor<1x?x64xf32>) -> tensor<1x?x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  // expected-error @+2 {{cannot compute a static grid size for an attention op with dynamically shaped operands}}
  // expected-error @+1 {{failed to legalize operation 'rock.attention'}}
  %result = rock.attention{
    qk = tr %q * %k : tensor<1x64x?xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %v : tensor<1x1024x64xf32>
  } {
    params0 = #attn_params_g0,
    params1 = #attn_params_g1,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<1x?x64xf32>
  %out = rock.store %result to %o by set : tensor<1x?x64xf32> -> tensor<1x?x64xf32> to tensor<1x?x64xf32>
  return %out : tensor<1x?x64xf32>
}

// -----

#attn_params_g0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 4, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#attn_params_g1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

func.func @attention_dynamic_seq_len_kv(%q: tensor<1x64x1024xf32>, %k: tensor<1x64x?xf32>, %v: tensor<1x?x64xf32>, %o: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  // expected-error @+2 {{cannot compute a static grid size for an attention op with dynamically shaped operands}}
  // expected-error @+1 {{failed to legalize operation 'rock.attention'}}
  %result = rock.attention{
    qk = tr %q * %k : tensor<1x64x1024xf32>, tensor<1x64x?xf32>
    softmax(qk) * %v : tensor<1x?x64xf32>
  } {
    params0 = #attn_params_g0,
    params1 = #attn_params_g1,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<1x1024x64xf32>
  %out = rock.store %result to %o by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
  return %out : tensor<1x1024x64xf32>
}
