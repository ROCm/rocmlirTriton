// Ensures that the padding application, group application, etc. in
// attn-to-gridwise function as expected for rock.attention and
// rock.gemm_elementwise_gemm.

// RUN: rocmlir-opt -rock-lower-reduce -rock-regularize-output -rock-regularize-inter-gemm-fusion -rock-fusion-splitk-regularization -rock-attn-to-gridwise -mlir-print-local-scope %s | FileCheck %s

#xldops_attn_params_g0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 4, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xldops_attn_params_g1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xldops_attn_params_g1_splitk = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
// nPerBlock here plays the role of the tiled second-gemm N tile (nPerBlockG1).
#xldops_attn_params_g1_npb16 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 16, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// CHECK-LABEL: func.func @rock_attention_simple
// CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[o:.*]]: tensor<1x1024x64xf32>)
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 32 : i32
func.func @rock_attention_simple(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK: rock.gridwise_attention(%[[trQ]], %[[k]], %[[v]])
  %result = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %arg2 : tensor<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<1x1024x64xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
  return %out : tensor<1x1024x64xf32>
}

// CHECK-LABEL: func.func @rock_attention_tr_padded
// CHECK-SAME: (%[[q:.*]]: tensor<1x7x49xf32>, %[[k:.*]]: tensor<1x7x49xf32>, %[[v:.*]]: tensor<1x49x7xf32>, %[[o:.*]]: tensor<1x49x7xf32>)
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 2 : i32
func.func @rock_attention_tr_padded(%arg0: tensor<1x7x49xf32>, %arg1: tensor<1x7x49xf32>, %arg2: tensor<1x49x7xf32>, %arg3: tensor<1x49x7xf32>) -> tensor<1x49x7xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x7x49xf32> to tensor<1x49x7xf32>
  // CHECK-DAG: %[[paddedTrQ:.*]] = rock.transform %[[trQ]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x64x8xf32>
  // CHECK-DAG: %[[paddedK:.*]] = rock.transform %[[k]] by {{.*}} : tensor<1x7x49xf32> to tensor<1x8x64xf32>
  // CHECK-DAG: %[[paddedV:.*]] = rock.transform %[[v]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x64x32xf32>
  // CHECK: rock.gridwise_attention(%[[paddedTrQ]], %[[paddedK]], %[[paddedV]])
  // CHECK: prePadG0M = 49 : index, prePadG0N = 49 : index
  %result = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x7x49xf32>, tensor<1x7x49xf32>
    softmax(qk) * %arg2 : tensor<1x49x7xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<1x49x7xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x49x7xf32> -> tensor<1x49x7xf32> to tensor<1x49x7xf32>
  return %out : tensor<1x49x7xf32>
}

// CHECK-LABEL: func.func @rock_attention_kvcache
// CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[o:.*]]: tensor<1x1024x64xf32>, %[[lastValidKVIndex:.*]]: tensor<1xi32>)
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 32 : i32
func.func @rock_attention_kvcache(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>, %arg4: tensor<1xi32>) -> tensor<1x1024x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK: rock.gridwise_attention(%[[trQ]], %[[k]], %[[v]], %[[lastValidKVIndex]])
  %result = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    lastValidKVIndex = (%arg4 : tensor<1xi32>)
    softmax(qk) * %arg2 : tensor<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<1x1024x64xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
  return %out : tensor<1x1024x64xf32>
}

// CHECK-LABEL: func.func @rock_attention_causal
// CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[o:.*]]: tensor<1x1024x64xf32>)
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 32 : i32
func.func @rock_attention_causal(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK: rock.gridwise_attention(%[[trQ]], %[[k]], %[[v]])
  // CHECK: causal
  %result = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    causal
    softmax(qk) * %arg2 : tensor<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<1x1024x64xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
  return %out : tensor<1x1024x64xf32>
}

// CHECK-LABEL: func.func @rock_attention_lse
// CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[lse:.*]]: tensor<1x1024xf32>, %[[o:.*]]: tensor<1x1024x64xf32>)
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 32 : i32
func.func @rock_attention_lse(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024xf32>, %arg4: tensor<1x1024x64xf32>) -> (tensor<1x1024x64xf32>, tensor<1x1024xf32>) attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK: %[[result:.*]], %[[lseOut:.*]] = rock.gridwise_attention(%[[trQ]], %[[k]], %[[v]])
  %result, %lseOut = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %arg2 : tensor<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<1x1024x64xf32>, tensor<1x1024xf32>
  // CHECK: rock.store %[[result]] to %[[o]]
  %out = rock.store %result to %arg4 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
  // CHECK: rock.store %[[lseOut]] to %[[lse]]
  %lseStore = rock.store %lseOut to %arg3 by set : tensor<1x1024xf32> -> tensor<1x1024xf32> to tensor<1x1024xf32>
  return %out, %lseStore : tensor<1x1024x64xf32>, tensor<1x1024xf32>
}

// CHECK-LABEL: func.func @rock_attention_splitkv
// CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[lse:.*]]: tensor<4x1024xf32>, %[[o:.*]]: tensor<4x1024x64xf32>)
// CHECK-SAME: rock.grid_size = 128
func.func @rock_attention_splitkv(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<4x1024xf32>, %arg4: tensor<4x1024x64xf32>) -> (tensor<4x1024x64xf32>, tensor<4x1024xf32>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.grid_size = 1024 : i32} {
  // CHECK: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK: %[[result:.*]], %[[lseOut:.*]] = rock.gridwise_attention(%[[trQ]], %[[k]], %[[v]])
  // CHECK: splitKV = 4
  %result, %lseOut = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %arg2 : tensor<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    splitKV = 4 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<4x1024x64xf32>, tensor<4x1024xf32>
  %out = rock.store %result to %arg4 by set : tensor<4x1024x64xf32> -> tensor<4x1024x64xf32> to tensor<4x1024x64xf32>
  %lseStore = rock.store %lseOut to %arg3 by set : tensor<4x1024xf32> -> tensor<4x1024xf32> to tensor<4x1024xf32>
  return %out, %lseStore : tensor<4x1024x64xf32>, tensor<4x1024xf32>
}

// CHECK-LABEL: func.func @rock_attention_splitkv_padding
// CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x384xf32>, %[[v:.*]]: tensor<1x384x64xf32>, %[[lse:.*]]: tensor<8x1024xf32>, %[[o:.*]]: tensor<8x1024x64xf32>)
// CHECK-SAME: rock.grid_size = 256
func.func @rock_attention_splitkv_padding(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<8x1024xf32>, %arg4: tensor<8x1024x64xf32>) -> (tensor<8x1024x64xf32>, tensor<8x1024xf32>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.grid_size = 1024 : i32} {
  // CHECK-DAG: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK-DAG: %[[kPadding:.*]] = rock.transform %[[k]] by {{.*}} : tensor<1x64x384xf32> to tensor<1x64x512xf32>
  // CHECK-DAG: %[[vPadding:.*]] = rock.transform %[[v]] by {{.*}} : tensor<1x384x64xf32> to tensor<1x512x64xf32>
  // CHECK: rock.gridwise_attention(%[[trQ]], %[[kPadding]], %[[vPadding]])
  // CHECK: splitKV = 8
  %result, %lseOut = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x384xf32>
    softmax(qk) * %arg2 : tensor<1x384x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    splitKV = 8 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<8x1024x64xf32>, tensor<8x1024xf32>
  %out = rock.store %result to %arg4 by set : tensor<8x1024x64xf32> -> tensor<8x1024x64xf32> to tensor<8x1024x64xf32>
  %lseStore = rock.store %lseOut to %arg3 by set : tensor<8x1024xf32> -> tensor<8x1024xf32> to tensor<8x1024xf32>
  return %out, %lseStore : tensor<8x1024x64xf32>, tensor<8x1024xf32>
}

// CHECK-LABEL: func.func @rock_attention_softmaxtype
// CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf16>, %[[k:.*]]: tensor<1x64x1024xf16>, %[[v:.*]]: tensor<1x1024x64xf16>, %[[lse:.*]]: tensor<1x1024xf16>, %[[o:.*]]: tensor<1x1024x64xf16>)
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 32 : i32
func.func @rock_attention_softmaxtype(%arg0: tensor<1x64x1024xf16>, %arg1: tensor<1x64x1024xf16>, %arg2: tensor<1x1024x64xf16>, %arg3: tensor<1x1024xf16>, %arg4: tensor<1x1024x64xf16>) -> (tensor<1x1024x64xf16>, tensor<1x1024xf16>) attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x64x1024xf16> to tensor<1x1024x64xf16>
  // CHECK: rock.gridwise_attention(%[[trQ]], %[[k]], %[[v]])
  // CHECK: softmaxType = f32
  %result, %lseOut = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x64x1024xf16>, tensor<1x64x1024xf16>
    softmax(qk) * %arg2 : tensor<1x1024x64xf16>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32,
    softmaxType = f32
  } -> tensor<1x1024x64xf16>, tensor<1x1024xf16>
  %out = rock.store %result to %arg4 by set : tensor<1x1024x64xf16> -> tensor<1x1024x64xf16> to tensor<1x1024x64xf16>
  %lseStore = rock.store %lseOut to %arg3 by set : tensor<1x1024xf16> -> tensor<1x1024xf16> to tensor<1x1024xf16>
  return %out, %lseStore : tensor<1x1024x64xf16>, tensor<1x1024xf16>
}

// When the second gemm's N dim (head_dim_v) is tiled (nPerBlockG1 < gemm1N),
// AttnToGridwise rounds gemm1N up to a power of two so the per-chunk outputs
// fold back together with pairwise tt.join. Here head_dim_v = 48 is already a
// multiple of nPerBlock (16) but not a power of two, so it is padded to 64.
// CHECK-LABEL: func.func @rock_attention_nperblockg1_pow2_pad
// CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x48xf32>, %[[o:.*]]: tensor<1x1024x48xf32>)
func.func @rock_attention_nperblockg1_pow2_pad(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x48xf32>, %arg3: tensor<1x1024x48xf32>) -> tensor<1x1024x48xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK-DAG: %[[vPad:.*]] = rock.transform %[[v]] by {{.*}} : tensor<1x1024x48xf32> to tensor<1x1024x64xf32>
  // CHECK-DAG: %[[oPad:.*]] = rock.transform %[[o]] by {{.*}} : tensor<1x1024x48xf32> to tensor<1x1024x64xf32>
  // CHECK: %[[result:.*]] = rock.gridwise_attention(%[[trQ]], %[[k]], %[[vPad]])
  // CHECK: rock.store %[[result]] to %[[oPad]] by set
  %result = rock.attention{
    qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    softmax(qk) * %arg2 : tensor<1x1024x48xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1_npb16,
    splitKV = 1 : i32,
    numHeadsKV = 1 : i32,
    numHeadsQ = 1 : i32
  } -> tensor<1x1024x48xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x1024x48xf32> -> tensor<1x1024x48xf32> to tensor<1x1024x48xf32>
  return %out : tensor<1x1024x48xf32>
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_simple
// CHECK-SAME: (%[[a:.*]]: tensor<1x64x1024xf32>, %[[b:.*]]: tensor<1x64x1024xf32>, %[[c:.*]]: tensor<1x1024x64xf32>, %[[o:.*]]: tensor<1x1024x64xf32>)
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 32 : i32
func.func @rock_gemmelementwisegemm_simple(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: %[[trA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK: rock.gridwise_attention(%[[trA]], %[[b]], %[[c]])
  // CHECK: enableSoftmax = false
  %result = rock.gemm_elementwise_gemm{
    ab = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    out = ab * %arg2 : tensor<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1
  } -> tensor<1x1024x64xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
  return %out : tensor<1x1024x64xf32>
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_tr_padded
// CHECK-SAME: (%[[a:.*]]: tensor<1x7x49xf32>, %[[b:.*]]: tensor<1x7x49xf32>, %[[c:.*]]: tensor<1x49x7xf32>, %[[o:.*]]: tensor<1x49x7xf32>)
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 2 : i32
func.func @rock_gemmelementwisegemm_tr_padded(%arg0: tensor<1x7x49xf32>, %arg1: tensor<1x7x49xf32>, %arg2: tensor<1x49x7xf32>, %arg3: tensor<1x49x7xf32>) -> tensor<1x49x7xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[trA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x7x49xf32> to tensor<1x49x7xf32>
  // CHECK-DAG: %[[paddedTrA:.*]] = rock.transform %[[trA]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x64x8xf32>
  // CHECK-DAG: %[[paddedB:.*]] = rock.transform %[[b]] by {{.*}} : tensor<1x7x49xf32> to tensor<1x8x64xf32>
  // CHECK-DAG: %[[paddedC:.*]] = rock.transform %[[c]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x64x32xf32>
  // CHECK: rock.gridwise_attention(%[[paddedTrA]], %[[paddedB]], %[[paddedC]])
  // CHECK: enableSoftmax = false
  // CHECK-SAME: prePadG0M = 49 : index, prePadG0N = 49 : index
  %result = rock.gemm_elementwise_gemm{
    ab = tr %arg0 * %arg1 : tensor<1x7x49xf32>, tensor<1x7x49xf32>
    out = ab * %arg2 : tensor<1x49x7xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1
  } -> tensor<1x49x7xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x49x7xf32> -> tensor<1x49x7xf32> to tensor<1x49x7xf32>
  return %out : tensor<1x49x7xf32>
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_splitk
// CHECK-SAME: (%[[aRaw:.*]]: tensor<1x64x1024xf32>, %[[bRaw:.*]]: tensor<1x64x1024xf32>, %[[cRaw:.*]]: tensor<1x1024x64xf32>, %[[oRaw:.*]]: tensor<1x1024x64xf32>
// CHECK-SAME: {rock.prefill = 0.000000e+00 : f32})
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 128 : i32
func.func @rock_gemmelementwisegemm_splitk(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32> {rock.prefill = 0.000000e+00 : f32}) -> tensor<1x1024x64xf32> attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[trA:.*]] = rock.transform %[[aRaw]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x1024x64xf32>
  // CHECK-DAG: %[[bSplit:.*]] = rock.transform %[[bRaw]] by {{.*}} : tensor<1x64x1024xf32> to tensor<1x4x256x64xf32>
  // CHECK-DAG: %[[b:.*]] = rock.transform %[[bSplit]] by {{.*}} : tensor<1x4x256x64xf32> to tensor<4x64x256xf32>
  // CHECK-DAG: %[[cSplit:.*]] = rock.transform %[[cRaw]] by {{.*}} : tensor<1x1024x64xf32> to tensor<1x4x256x64xf32>
  // CHECK-DAG: %[[c:.*]] = rock.transform %[[cSplit]] by {{.*}} : tensor<1x4x256x64xf32> to tensor<4x256x64xf32>
  // CHECK-DAG: %[[aSplit:.*]] = rock.transform %[[trA]] by {{.*}} : tensor<1x1024x64xf32> to tensor<1x1024x64x4xf32>
  // CHECK-DAG: %[[a:.*]] = rock.transform %[[aSplit]] by {{.*}} : tensor<1x1024x64x4xf32> to tensor<4x1024x64xf32>
  // CHECK-DAG: %[[oSplit:.*]] = rock.transform %[[oRaw]] by {{.*}} : tensor<1x1024x64xf32> to tensor<1x4x1024x64xf32>
  // CHECK-DAG: %[[o:.*]] = rock.transform %[[oSplit]] by {{.*}} : tensor<1x4x1024x64xf32> to tensor<4x1024x64xf32>
  // CHECK: %[[result:.*]] = rock.gridwise_attention(%[[a]], %[[b]], %[[c]])
  // CHECK: enableSoftmax = false
  // CHECK: rock.store %[[result]] to %[[o]] by atomic_add
  %result = rock.gemm_elementwise_gemm{
    ab = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
    out = ab * %arg2 : tensor<1x1024x64xf32>
  } {
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1_splitk
  } -> tensor<1x1024x64xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
  return %out : tensor<1x1024x64xf32>
}

// CHECK-LABEL: func.func @rock_gemmelementwisegemm_splitk_two_outputs
// CHECK-SAME: (%[[a:.*]]: tensor<1x64x64xf32>, %[[b:.*]]: tensor<1x64x64xf32>, %[[c:.*]]: tensor<1x64x64xf32>, %[[oRaw:.*]]: tensor<1x64x64xf32> {rock.prefill = 0.000000e+00 : f32},
// CHECK-SAME: %[[reduceOut:.*]]: tensor<1x64x1xf32> {rock.prefill = 0.000000e+00 : f32})
// CHECK-SAME: rock.block_size = 64 : i32, rock.grid_size = 8 : i32
func.func @rock_gemmelementwisegemm_splitk_two_outputs(%arg0: tensor<1x64x64xf32>, %arg1: tensor<1x64x64xf32>, %arg2: tensor<1x64x64xf32>, %arg3: tensor<1x64x64xf32> {rock.prefill = 0.000000e+00 : f32}, %arg4: tensor<1x64x1xf32> {rock.prefill = 0.000000e+00 : f32}) -> (tensor<1x64x64xf32>, tensor<1x64x1xf32>) attributes {rock.kernel, rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK-DAG: %[[bSplit:.*]] = rock.transform %[[b]] by {{.*}} : tensor<1x64x64xf32> to tensor<1x4x16x64xf32>
  // CHECK-DAG: %[[bMerged:.*]] = rock.transform %[[bSplit]] by {{.*}} : tensor<1x4x16x64xf32> to tensor<4x64x16xf32>
  // CHECK-DAG: %[[bPad:.*]] = rock.transform %[[bMerged]] by {{.*}} : tensor<4x64x16xf32> to tensor<4x64x64xf32>
  // CHECK-DAG: %[[cSplit:.*]] = rock.transform %[[c]] by {{.*}} : tensor<1x64x64xf32> to tensor<1x4x16x64xf32>
  // CHECK-DAG: %[[cMerged:.*]] = rock.transform %[[cSplit]] by {{.*}} : tensor<1x4x16x64xf32> to tensor<4x16x64xf32>
  // CHECK-DAG: %[[cPad:.*]] = rock.transform %[[cMerged]] by {{.*}} : tensor<4x16x64xf32> to tensor<4x64x64xf32>
  // CHECK-DAG: %[[aAddDim:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x64x64xf32> to tensor<1x64x64x4xf32>
  // CHECK-DAG: %[[aPad:.*]] = rock.transform %[[aAddDim]] by {{.*}} : tensor<1x64x64x4xf32> to tensor<4x64x64xf32>
  // CHECK-DAG: %[[oSplit:.*]] = rock.transform %[[oRaw]] by {{.*}} : tensor<1x64x64xf32> to tensor<1x4x64x64xf32>
  // CHECK-DAG: %[[o:.*]] = rock.transform %[[oSplit]] by {{.*}} : tensor<1x4x64x64xf32> to tensor<4x64x64xf32>
  // CHECK: %[[result:.*]] = rock.gridwise_attention(%[[aPad]], %[[bPad]], %[[cPad]])
  // CHECK: enableSoftmax = false
  // CHECK: rock.store %[[result]] to %[[o]] by atomic_add
  %result = rock.gemm_elementwise_gemm{
    ab = %arg0 * %arg1 : tensor<1x64x64xf32>, tensor<1x64x64xf32>
    out = ab * %arg2 : tensor<1x64x64xf32>
  } {
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 64, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 1, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    perf_config = "attn:v1:32,64,64,1,1,4,0,4,1,0,0"
  } -> tensor<1x64x64xf32>
  // CHECK-DAG: %[[reduceBcast:.*]] = rock.transform %[[reduceOut]] by {{.*}} : tensor<1x64x1xf32> to tensor<1x64x64xf32>
  // CHECK-DAG: %[[reduceSplit:.*]] = rock.transform %[[reduceBcast]] by {{.*}} : tensor<1x64x64xf32> to tensor<1x4x64x64xf32>
  // CHECK-DAG: %[[reduceMerged:.*]] = rock.transform %[[reduceSplit]] by {{.*}} : tensor<1x4x64x64xf32> to tensor<4x64x64xf32>
  // CHECK: rock.store %[[result]] to %[[reduceMerged]] by atomic_add
  %reduce_result = rock.reduce sum %result {axis = 2 : index} : tensor<1x64x64xf32> -> tensor<1x64x1xf32>
  %out1 = rock.store %result to %arg3 by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  %out2 = rock.store %reduce_result to %arg4 by set : tensor<1x64x1xf32> -> tensor<1x64x1xf32> to tensor<1x64x1xf32>
  return %out1, %out2 : tensor<1x64x64xf32>, tensor<1x64x1xf32>
}

// CHECK-LABEL: func.func @rock_attention_gqa
// CHECK-SAME: (%[[q:.*]]: tensor<64x1x128xf16>, %[[k:.*]]: tensor<8x128x8192xf16>, %[[v:.*]]: tensor<8x8192x128xf16>, %[[lse:.*]]: tensor<256x1xf16>, %[[o:.*]]: tensor<256x1x128xf16>)
// CHECK-SAME: rock.grid_size = 32
func.func @rock_attention_gqa(%arg0: tensor<64x1x128xf16>, %arg1: tensor<8x128x8192xf16>, %arg2: tensor<8x8192x128xf16>, %arg3: tensor<256x1xf16>, %arg4: tensor<256x1x128xf16>) -> (tensor<256x1x128xf16>, tensor<256x1xf16>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 64 : i32, rock.grid_size = 1024 : i32} {
  // CHECK-DAG: %[[qExtractNumRepeats:.+]] = rock.transform %[[q]] by {{.*}} : tensor<64x1x128xf16> to tensor<8x1x8x128xf16>
  // CHECK-DAG: %[[qMoveToSeqLen:.*]] = rock.transform %[[qExtractNumRepeats]] by {{.*}} : tensor<8x1x8x128xf16> to tensor<8x8x128xf16>
  // CHECK-DAG: %[[qPad:.+]] = rock.transform %[[qMoveToSeqLen]] by {{.*}} : tensor<8x8x128xf16> to tensor<8x32x128xf16>
  // CHECK-DAG: %[[outUnmerge:.*]] = rock.transform %[[o]] by {{.*}} : tensor<256x1x128xf16> to tensor<8x4x1x8x128xf16>
  // CHECK-DAG: %[[outMerge:.*]] = rock.transform %[[outUnmerge]] by {{.*}} : tensor<8x4x1x8x128xf16> to tensor<32x8x128xf16>
  // CHECK-DAG: %[[outPad:.*]] = rock.transform %[[outMerge]] by {{.*}} : tensor<32x8x128xf16> to tensor<32x32x128xf16>
  // CHECK-DAG: %[[lseUnmerge:.*]] = rock.transform %[[lse]] by {{.*}} : tensor<256x1xf16> to tensor<8x4x1x8xf16>
  // CHECK-DAG: %[[lseMerge:.*]] = rock.transform %[[lseUnmerge]] by {{.*}} : tensor<8x4x1x8xf16> to tensor<32x8xf16>
  // CHECK-DAG: %[[lsePad:.*]] = rock.transform %[[lseMerge]] by {{.*}} : tensor<32x8xf16> to tensor<32x32xf16>
  // CHECK: %[[result:.*]], %[[lseOut:.*]] = rock.gridwise_attention(%[[qPad]], %[[k]], %[[v]])
  // CHECK: splitKV = 4
  // CHECK: rock.store %[[result]] to %[[outPad]] by set
  // CHECK: rock.store %[[lseOut]] to %[[lsePad]] by set
  %result, %lseOut = rock.attention{
    qk = %arg0 * %arg1 : tensor<64x1x128xf16>, tensor<8x128x8192xf16>
    softmax(qk) * %arg2 : tensor<8x8192x128xf16>
  } {numHeadsKV = 8 : i32, numHeadsQ = 64 : i32, params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, softmaxType = f32, splitKV = 4 : i32} -> tensor<256x1x128xf16>, tensor<256x1xf16>
  %out = rock.store %result to %arg4 by set : tensor<256x1x128xf16> -> tensor<256x1x128xf16> to tensor<256x1x128xf16>
  %lseStore = rock.store %lseOut to %arg3 by set : tensor<256x1xf16> -> tensor<256x1xf16> to tensor<256x1xf16>
  return %out, %lseStore : tensor<256x1x128xf16>, tensor<256x1xf16>
}

