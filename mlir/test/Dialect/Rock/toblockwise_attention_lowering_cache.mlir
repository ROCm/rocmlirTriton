// Cache-modifier heuristics for rock-gridwise-attn-to-blockwise.
//
// Q is reused across the whole nLoop and always stays cached. K and V are only
// reused across the seqQ tiles (reuse factor = gemm0 mBlocks), so when seqQ is
// skinny (decode: a single gemm0 M block) they are read once and are streamed
// (CacheModifier cs) -- but only under cache pressure (Q + K + V do not fit in
// the last-level cache). K and V are decided independently: an operand whose
// load reloads data (a non-injective view such as a broadcast) relies on
// caching and is never streamed. All tests use gfx90a (8 MiB LLC).

// RUN: rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -canonicalize -verify-diagnostics %s | FileCheck %s

// seqQ = 16 = gemm0 mPerBlock -> a single seqQ tile (skinny / decode). K (8 MiB)
// + V (8 MiB) + Q exceed the 8 MiB LLC -> cache pressure. K and V are read once,
// so both are streamed; Q stays cached.
// CHECK-LABEL: @attn_skinny_pressure_streams_kv
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x16x64xf32> -> tensor<16x16xf32>
// CHECK-DAG: rock.load_marker %arg1 {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x64x32768xf32> -> tensor<16x32xf32>
// CHECK-DAG: rock.load_marker %arg2 {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x32768x64xf32> -> tensor<32x64xf32>
func.func @attn_skinny_pressure_streams_kv(
    %q: tensor<1x16x64xf32>, %k: tensor<1x64x32768xf32>, %v: tensor<1x32768x64xf32>) -> tensor<1x16x64xf32>
    attributes {rock.block_size = 64 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x16x64xf32>, tensor<1x64x32768xf32>, tensor<1x32768x64xf32> -> tensor<1x16x64xf32>
  return %result : tensor<1x16x64xf32>
}

// -----

// seqQ is skinny, but Q + K + V fit easily in the 8 MiB LLC, so there is no
// cache pressure and nothing is streamed: Q, K and V all stay cached.
// CHECK-LABEL: @attn_skinny_no_pressure
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x16x64xf32> -> tensor<16x16xf32>
// CHECK-DAG: rock.load_marker %arg1 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x64x128xf32> -> tensor<16x32xf32>
// CHECK-DAG: rock.load_marker %arg2 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x128x64xf32> -> tensor<32x64xf32>
func.func @attn_skinny_no_pressure(
    %q: tensor<1x16x64xf32>, %k: tensor<1x64x128xf32>, %v: tensor<1x128x64xf32>) -> tensor<1x16x64xf32>
    attributes {rock.block_size = 64 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x16x64xf32>, tensor<1x64x128xf32>, tensor<1x128x64xf32> -> tensor<1x16x64xf32>
  return %result : tensor<1x16x64xf32>
}

// -----

// seqQ = 64 = 4 * gemm0 mPerBlock -> gemm0 mBlocks = 4 (not skinny): K and V are
// reused across the 4 seqQ tiles, so even under cache pressure they stay cached.
// CHECK-LABEL: @attn_seqq_not_skinny
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x64x64xf32> -> tensor<16x16xf32>
// CHECK-DAG: rock.load_marker %arg1 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x64x32768xf32> -> tensor<16x32xf32>
// CHECK-DAG: rock.load_marker %arg2 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x32768x64xf32> -> tensor<32x64xf32>
func.func @attn_seqq_not_skinny(
    %q: tensor<1x64x64xf32>, %k: tensor<1x64x32768xf32>, %v: tensor<1x32768x64xf32>) -> tensor<1x64x64xf32>
    attributes {rock.block_size = 64 : i32, rock.grid_size = 4 : i32, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x64xf32>, tensor<1x64x32768xf32>, tensor<1x32768x64xf32> -> tensor<1x64x64xf32>
  return %result : tensor<1x64x64xf32>
}

// -----

// K is fed through a Broadcast view (its single head-dim row is replicated to
// headDim = 64): iterating the gemm0 contraction re-reads the same element, a
// non-injective ("reload") view. Even though seqQ is skinny and under cache
// pressure, K relies on caching and is NOT streamed, while V (injective, read
// once) is still streamed. Q stays cached.
// CHECK-LABEL: @attn_skinny_pressure_noninjective_k
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x16x64xf32> -> tensor<16x16xf32>
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x64x32768xf32> -> tensor<16x32xf32>
// CHECK-DAG: rock.load_marker %arg2 {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x32768x64xf32> -> tensor<32x64xf32>
#bcastH = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Broadcast{1} ["h"] at [1] -> ["h"] at [1]>, <PassThrough ["s"] at [2] -> ["s"] at [2]>] bounds = [1, 64, 32768] -> [1, 1, 32768]>
func.func @attn_skinny_pressure_noninjective_k(
    %q: tensor<1x16x64xf32>, %k: tensor<1x1x32768xf32>, %v: tensor<1x32768x64xf32>) -> tensor<1x16x64xf32>
    attributes {rock.block_size = 64 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %kb = rock.transform %k by #bcastH : tensor<1x1x32768xf32> to tensor<1x64x32768xf32>
  %result = rock.gridwise_attention(%q, %kb, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x16x64xf32>, tensor<1x64x32768xf32>, tensor<1x32768x64xf32> -> tensor<1x16x64xf32>
  return %result : tensor<1x16x64xf32>
}

// -----

// K is built by an AddDim of size 64 along the head dim: the added gemm0
// contraction steps re-read the same element (a reload). Even with seqQ skinny
// and under pressure, K is NOT streamed, while V (injective, read once) is. Q
// stays cached.
// CHECK-LABEL: @attn_skinny_pressure_adddim_k
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x16x64xf32> -> tensor<16x16xf32>
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x64x32768xf32> -> tensor<16x32xf32>
// CHECK-DAG: rock.load_marker %arg2 {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x32768x64xf32> -> tensor<32x64xf32>
#adddimH = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <AddDim{64} ["h"] at [1] -> [] at []>, <PassThrough ["s"] at [2] -> ["s"] at [1]>] bounds = [1, 64, 32768] -> [1, 32768]>
func.func @attn_skinny_pressure_adddim_k(
    %q: tensor<1x16x64xf32>, %k: tensor<1x32768xf32>, %v: tensor<1x32768x64xf32>) -> tensor<1x16x64xf32>
    attributes {rock.block_size = 64 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %kb = rock.transform %k by #adddimH : tensor<1x32768xf32> to tensor<1x64x32768xf32>
  %result = rock.gridwise_attention(%q, %kb, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x16x64xf32>, tensor<1x64x32768xf32>, tensor<1x32768x64xf32> -> tensor<1x16x64xf32>
  return %result : tensor<1x16x64xf32>
}

// -----

// Input-fusion chain rock.transform -> fusion -> rock.transform -> fusion ->
// rock.transform on K, all views injective. The walk traverses the fusion ops
// and finds no reload, so with seqQ skinny and under pressure K is still
// streamed (as is V).
// CHECK-LABEL: @attn_skinny_pressure_fusion_chain_injective
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x16x64xf32> -> tensor<16x16xf32>
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x64x32768xf32> -> tensor<16x32xf32>
// CHECK-DAG: rock.load_marker %arg2 {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x32768x64xf32> -> tensor<32x64xf32>
#idK = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g", "h", "s"] at [0, 1, 2] -> ["g", "h", "s"] at [0, 1, 2]>] bounds = [1, 64, 32768] -> [1, 64, 32768]>
func.func @attn_skinny_pressure_fusion_chain_injective(
    %q: tensor<1x16x64xf32>, %k: tensor<1x64x32768xf32>, %v: tensor<1x32768x64xf32>) -> tensor<1x16x64xf32>
    attributes {rock.block_size = 64 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %t1 = rock.transform %k by #idK : tensor<1x64x32768xf32> to tensor<1x64x32768xf32>
  %f1 = math.absf %t1 : tensor<1x64x32768xf32>
  %t2 = rock.transform %f1 by #idK : tensor<1x64x32768xf32> to tensor<1x64x32768xf32>
  %f2 = arith.addf %t2, %t2 : tensor<1x64x32768xf32>
  %t3 = rock.transform %f2 by #idK : tensor<1x64x32768xf32> to tensor<1x64x32768xf32>
  %result = rock.gridwise_attention(%q, %t3, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x16x64xf32>, tensor<1x64x32768xf32>, tensor<1x32768x64xf32> -> tensor<1x16x64xf32>
  return %result : tensor<1x16x64xf32>
}

// -----

// Same shape of chain, but a Broadcast (reload) sits behind the fusion ops. The
// walk must see through the fusions to the broadcast: K is NOT streamed, while V
// still is.
// CHECK-LABEL: @attn_skinny_pressure_fusion_chain_reload
// CHECK-DAG: rock.load_marker %arg0 {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x16x64xf32> -> tensor<16x16xf32>
// CHECK-DAG: rock.load_marker %{{.*}} {{.*}}{cacheModifier = #rock<CacheModifier none>{{.*}}} : tensor<1x64x32768xf32> -> tensor<16x32xf32>
// CHECK-DAG: rock.load_marker %arg2 {{.*}}{cacheModifier = #rock<CacheModifier cs>{{.*}}} : tensor<1x32768x64xf32> -> tensor<32x64xf32>
#idK2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g", "h", "s"] at [0, 1, 2] -> ["g", "h", "s"] at [0, 1, 2]>] bounds = [1, 64, 32768] -> [1, 64, 32768]>
#bcastH2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Broadcast{1} ["h"] at [1] -> ["h"] at [1]>, <PassThrough ["s"] at [2] -> ["s"] at [2]>] bounds = [1, 64, 32768] -> [1, 1, 32768]>
func.func @attn_skinny_pressure_fusion_chain_reload(
    %q: tensor<1x16x64xf32>, %k: tensor<1x1x32768xf32>, %v: tensor<1x32768x64xf32>) -> tensor<1x16x64xf32>
    attributes {rock.block_size = 64 : i32, rock.grid_size = 1 : i32, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  %b0 = rock.transform %k by #bcastH2 : tensor<1x1x32768xf32> to tensor<1x64x32768xf32>
  %f1 = math.absf %b0 : tensor<1x64x32768xf32>
  %t2 = rock.transform %f1 by #idK2 : tensor<1x64x32768xf32> to tensor<1x64x32768xf32>
  %f2 = arith.addf %t2, %t2 : tensor<1x64x32768xf32>
  %t3 = rock.transform %f2 by #idK2 : tensor<1x64x32768xf32> to tensor<1x64x32768xf32>
  %result = rock.gridwise_attention(%q, %t3, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x16x64xf32>, tensor<1x64x32768xf32>, tensor<1x32768x64xf32> -> tensor<1x16x64xf32>
  return %result : tensor<1x16x64xf32>
}
