// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-attn-to-gridwise -mlir-print-local-scope | FileCheck %s

// Lowering a cross-attention kernel whose query sequence length is only known
// at run time. The grid depends on it and on nothing else, so instead of a size
// the kernel carries the two factors the launch multiplies: the tile height and
// the number of group-by-splitKV blocks.

#attn_params_g0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 4, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#attn_params_g1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// CHECK-LABEL: func.func @attention_dynamic_seq_len_q
// CHECK-SAME: rock.block_size = 64 : i32
// CHECK-NOT: rock.grid_size
// CHECK-SAME: rock.dyn_grid_size = {gnBlocks = 1 : i64, mPerBlock = 32 : i64}
func.func @attention_dynamic_seq_len_q(%q: tensor<1x?x64xf32>, %k: tensor<1x64x1024xf32>, %v: tensor<1x1024x64xf32>, %o: tensor<1x?x64xf32>) -> tensor<1x?x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "##TOKEN_ARCH##"} {
  // CHECK: rock.gridwise_attention(%{{.*}}, %{{.*}}, %{{.*}})
  %result = rock.attention{
    qk = %q * %k : tensor<1x?x64xf32>, tensor<1x64x1024xf32>
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
