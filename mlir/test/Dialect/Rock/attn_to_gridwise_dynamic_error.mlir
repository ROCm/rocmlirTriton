// Attention variants that rock-attn-to-gridwise does not support with
// dynamic shapes.

// RUN: rocmlir-opt -rock-attn-to-gridwise -split-input-file -verify-diagnostics %s

#params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 32, kPerBlock = 32, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 32, kPerBlock = 32, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

func.func @dynamic_sliding_window(%q: tensor<?x?x32xf16>, %k: tensor<?x32x?xf16>, %v: tensor<?x?x32xf16>, %kv: tensor<?xi32>, %out: tensor<?x?x32xf16>) -> tensor<?x?x32xf16> attributes {rock.kernel, rock.block_size = 128 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+2 {{sliding-window attention is not supported with dynamic shapes}}
  // expected-error @+1 {{failed to legalize operation 'rock.attention'}}
  %result = rock.attention{
    qk = %q * %k : tensor<?x?x32xf16>, tensor<?x32x?xf16>
    lastValidKVIndex = (%kv : tensor<?xi32>)
    softmax(qk) * %v : tensor<?x?x32xf16>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32,
     slidingWindowLookBack = 16 : i32, params0 = #params0, params1 = #params1} -> tensor<?x?x32xf16>
  %stored = rock.store %result to %out by set : tensor<?x?x32xf16> -> tensor<?x?x32xf16> to tensor<?x?x32xf16>
  return %stored : tensor<?x?x32xf16>
}

// -----

#params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 32, kPerBlock = 32, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 32, kPerBlock = 32, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// The elementwise inputs of a split-KV attention fused by MIGraphX already
// carry the splits in their batch.
func.func @dynamic_split_kv_elementwise_inputs(%q: tensor<?x?x32xf16>, %k: tensor<?x32x?xf16>, %v: tensor<?x?x32xf16>, %bias: tensor<?x?x?xf16>, %out: tensor<?x?x32xf16>, %lseOut: tensor<?x?xf16>) -> (tensor<?x?x32xf16>, tensor<?x?xf16>) attributes {rock.kernel, rock.block_size = 128 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+2 {{split-KV elementwise inputs are not supported with dynamic shapes}}
  // expected-error @+1 {{failed to legalize operation 'rock.attention'}}
  %result, %lse = rock.attention{
    qk = %q * %k : tensor<?x?x32xf16>, tensor<?x32x?xf16>
    qk = elementwise otherIns(%bias : tensor<?x?x?xf16>) {
    ^bb0(%qk: tensor<?x?x?xf16>, %b: tensor<?x?x?xf16>):
      %sum = arith.addf %qk, %b : tensor<?x?x?xf16>
      rock.yield %sum : tensor<?x?x?xf16>
    }
    softmax(qk) * %v : tensor<?x?x32xf16>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 2 : i32,
     preSoftmaxHasSplitKVTransforms = true, params0 = #params0, params1 = #params1} -> tensor<?x?x32xf16>, tensor<?x?xf16>
  %stored = rock.store %result to %out by set : tensor<?x?x32xf16> -> tensor<?x?x32xf16> to tensor<?x?x32xf16>
  %storedLse = rock.store %lse to %lseOut by set : tensor<?x?xf16> -> tensor<?x?xf16> to tensor<?x?xf16>
  return %stored, %storedLse : tensor<?x?x32xf16>, tensor<?x?xf16>
}
