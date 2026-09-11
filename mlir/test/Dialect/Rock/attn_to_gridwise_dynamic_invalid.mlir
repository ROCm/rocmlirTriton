// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-attn-to-gridwise -split-input-file -verify-diagnostics

// Only the query sequence length may be dynamic, and only when it is the
// slowest-moving axis of the operands it appears on. Anything else either sizes
// a loop, a tile view or a pad amount at compile time, or leaves the strides
// outside it dynamic, which a transform map cannot express.

#attn_params_g0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 4, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#attn_params_g1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// A transposed Q puts the query sequence length on the fastest-moving axis.
func.func @attention_dynamic_seq_len_q_transposed(%q: tensor<1x64x?xf32>, %k: tensor<1x64x1024xf32>, %v: tensor<1x1024x64xf32>, %o: tensor<1x?x64xf32>) -> tensor<1x?x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+2 {{only the M dimension of the first gemm may be dynamic}}
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

// The key sequence length is the first gemm's N and the second gemm's K, so it
// bounds the flash-attention loop.
func.func @attention_dynamic_seq_len_kv(%q: tensor<1x64x1024xf32>, %k: tensor<1x64x?xf32>, %v: tensor<1x?x64xf32>, %o: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "##TOKEN_ARCH##"} {
  // expected-error @+2 {{only the M dimension of the first gemm may be dynamic}}
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
