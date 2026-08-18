// Tests for the rock-decompose-nonpow2-tiles pass.
//
// The pass runs at the gridwise layer (after gemm/attn-to-gridwise, before
// gridwise-to-blockwise), so each case is a rock.gridwise_gemm whose tuning
// params carry a non-power-of-two mPerBlock/nPerBlock, feeding a rock.store
// (optionally through output fusion). The pass splits it into a grid of
// power-of-two-tile sub-gridwise_gemms over sliced A/B/output views.
//
// -canonicalize is used to drop the now-dead original gridwise_gemm/store
// (the real pipeline DCEs them via addWithDCE).

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-decompose-nonpow2-tiles -canonicalize -split-input-file -mlir-print-local-scope | FileCheck %s

#p80x80 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Both M and N are non-power-of-two (80, 80): the gemm splits
// into a 2x2 grid {64,16} x {64,16}.
// ============================================================

// CHECK-LABEL: func.func @test_m_and_n_nonpow2
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 64{{.*}} -> tensor<1x128x128xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 16{{.*}} -> tensor<1x128x32xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 16, nPerBlock = 64{{.*}} -> tensor<1x32x128xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 16, nPerBlock = 16{{.*}} -> tensor<1x32x32xf32>
// CHECK-NOT: mPerBlock = 80
// CHECK-COUNT-4: rock.store
func.func @test_m_and_n_nonpow2(%a: tensor<1x160x64xf16>, %b: tensor<1x64x160xf16>, %c: tensor<1x160x160xf32>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #p80x80} : tensor<1x160x64xf16>, tensor<1x64x160xf16> -> tensor<1x160x160xf32>
  %out = rock.store %r to %c by set : tensor<1x160x160xf32> -> tensor<1x160x160xf32> to tensor<1x160x160xf32>
  return %out : tensor<1x160x160xf32>
}

// -----

#pk48 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 48, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Both M and N (80, 80) and the contraction tile kPerBlock (48)
// are non-power-of-two. This pass only peels M/N into a 2x2 grid
// {64,16} x {64,16}; the non-pow2 kPerBlock rides along unchanged
// on every sub-gemm (K peeling happens later in
// gridwise-gemm-to-blockwise), so K stays 96 on each sub-view.
// ============================================================

// CHECK-LABEL: func.func @test_m_and_n_nonpow2_with_nonpow2_k
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 64, kPerBlock = 48{{.*}} : tensor<1x128x96xf16>, tensor<1x96x128xf16> -> tensor<1x128x128xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 16, kPerBlock = 48{{.*}} -> tensor<1x128x32xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 16, nPerBlock = 64, kPerBlock = 48{{.*}} -> tensor<1x32x128xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 16, nPerBlock = 16, kPerBlock = 48{{.*}} -> tensor<1x32x32xf32>
// CHECK-NOT: mPerBlock = 80
// CHECK-COUNT-4: rock.store
func.func @test_m_and_n_nonpow2_with_nonpow2_k(%a: tensor<1x160x96xf16>, %b: tensor<1x96x160xf16>, %c: tensor<1x160x160xf32>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #pk48} : tensor<1x160x96xf16>, tensor<1x96x160xf16> -> tensor<1x160x160xf32>
  %out = rock.store %r to %c by set : tensor<1x160x160xf32> -> tensor<1x160x160xf32> to tensor<1x160x160xf32>
  return %out : tensor<1x160x160xf32>
}

// -----

#p80x64 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 64, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Only M is non-power-of-two (80, 64): the gemm splits along M
// into {64,16}; N (64) stays whole, so B is not sliced.
// ============================================================

// CHECK-LABEL: func.func @test_m_nonpow2
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 64{{.*}} -> tensor<1x128x128xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 16, nPerBlock = 64{{.*}} -> tensor<1x32x128xf32>
// CHECK-NOT: mPerBlock = 80
// CHECK-COUNT-2: rock.store
func.func @test_m_nonpow2(%a: tensor<1x160x64xf16>, %b: tensor<1x64x128xf16>, %c: tensor<1x160x128xf32>) -> tensor<1x160x128xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #p80x64} : tensor<1x160x64xf16>, tensor<1x64x128xf16> -> tensor<1x160x128xf32>
  %out = rock.store %r to %c by set : tensor<1x160x128xf32> -> tensor<1x160x128xf32> to tensor<1x160x128xf32>
  return %out : tensor<1x160x128xf32>
}

// -----

#p64x80 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 80, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Only N is non-power-of-two (64, 80): the gemm splits along N
// into {64,16}; M (64) stays whole, so A is not sliced.
// ============================================================

// CHECK-LABEL: func.func @test_n_nonpow2
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 64{{.*}} -> tensor<1x128x128xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 16{{.*}} -> tensor<1x128x32xf32>
// CHECK-NOT: nPerBlock = 80
// CHECK-COUNT-2: rock.store
func.func @test_n_nonpow2(%a: tensor<1x128x64xf16>, %b: tensor<1x64x160xf16>, %c: tensor<1x128x160xf32>) -> tensor<1x128x160xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #p64x80} : tensor<1x128x64xf16>, tensor<1x64x160xf16> -> tensor<1x128x160xf32>
  %out = rock.store %r to %c by set : tensor<1x128x160xf32> -> tensor<1x128x160xf32> to tensor<1x128x160xf32>
  return %out : tensor<1x128x160xf32>
}

// -----

#p64x64 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Nothing to do: both M and N are already powers of two (64, 64).
// The gemm and store are left unchanged (one of each).
// ============================================================

// CHECK-LABEL: func.func @test_nothing_to_do
// CHECK: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 64
// CHECK-NOT: rock.gridwise_gemm
// CHECK: rock.store
// CHECK-NOT: rock.store
func.func @test_nothing_to_do(%a: tensor<1x128x64xf16>, %b: tensor<1x64x128xf16>, %c: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #p64x64} : tensor<1x128x64xf16>, tensor<1x64x128xf16> -> tensor<1x128x128xf32>
  %out = rock.store %r to %c by set : tensor<1x128x128xf32> -> tensor<1x128x128xf32> to tensor<1x128x128xf32>
  return %out : tensor<1x128x128xf32>
}

// -----

#p32x32 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Non-power-of-two GEMM dimensions (M = N = 160) but power-of-two
// tiles (mPerBlock = nPerBlock = 32): the pass keys off the TILE,
// not the padded dimension, so decomposePow2(32) yields a single
// segment and this is a no-op (one gemm, one store, unchanged).
// ============================================================

// CHECK-LABEL: func.func @test_nonpow2_dim_pow2_tile
// CHECK: rock.gridwise_gemm{{.*}}mPerBlock = 32, nPerBlock = 32
// CHECK-NOT: rock.gridwise_gemm
// CHECK: rock.store
// CHECK-NOT: rock.store
func.func @test_nonpow2_dim_pow2_tile(%a: tensor<1x160x64xf16>, %b: tensor<1x64x160xf16>, %c: tensor<1x160x160xf32>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #p32x32} : tensor<1x160x64xf16>, tensor<1x64x160xf16> -> tensor<1x160x160xf32>
  %out = rock.store %r to %c by set : tensor<1x160x160xf32> -> tensor<1x160x160xf32> to tensor<1x160x160xf32>
  return %out : tensor<1x160x160xf32>
}

// -----

#pf80x80 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Output fusion: the gemm result is combined with two extra
// inputs via add then mul. The fusion chain is replicated per
// cell, with each extra input block-sliced to match (4 adds,
// 4 muls, 4 stores).
// ============================================================

// CHECK-LABEL: func.func @test_output_fusion
// CHECK-COUNT-4: rock.gridwise_gemm
// CHECK-NOT: rock.gridwise_gemm
// CHECK-COUNT-4: arith.addf
// CHECK-COUNT-4: arith.mulf
// CHECK-COUNT-4: rock.store
func.func @test_output_fusion(%a: tensor<1x160x64xf16>, %b: tensor<1x64x160xf16>, %c: tensor<1x160x160xf32>, %bias0: tensor<1x160x160xf32>, %bias1: tensor<1x160x160xf32>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #pf80x80} : tensor<1x160x64xf16>, tensor<1x64x160xf16> -> tensor<1x160x160xf32>
  %add = arith.addf %r, %bias0 : tensor<1x160x160xf32>
  %mul = arith.mulf %add, %bias1 : tensor<1x160x160xf32>
  %out = rock.store %mul to %c by set : tensor<1x160x160xf32> -> tensor<1x160x160xf32> to tensor<1x160x160xf32>
  return %out : tensor<1x160x160xf32>
}

// -----

#pcst80x80 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Output fusion against a splat arith.constant: the gemm result
// is multiplied by a splat constant before the store. The splat
// is re-materialized per cell at the matching sub-tile shape, so
// the chain replicates to 4 gemms, 4 constants, 4 muls, 4 stores.
// ============================================================

// canonicalize hoists the per-cell splat constants to the top of the function,
// ahead of the sub-gemms.
// CHECK-LABEL: func.func @test_output_fusion_splat_constant
// CHECK-COUNT-4: arith.constant dense<2.000000e+00>
// CHECK-COUNT-4: rock.gridwise_gemm
// CHECK-NOT: rock.gridwise_gemm
// CHECK-COUNT-4: arith.mulf
// CHECK-COUNT-4: rock.store
func.func @test_output_fusion_splat_constant(%a: tensor<1x160x64xf16>, %b: tensor<1x64x160xf16>, %c: tensor<1x160x160xf32>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #pcst80x80} : tensor<1x160x64xf16>, tensor<1x64x160xf16> -> tensor<1x160x160xf32>
  %cst = arith.constant dense<2.000000e+00> : tensor<1x160x160xf32>
  %mul = arith.mulf %r, %cst : tensor<1x160x160xf32>
  %out = rock.store %mul to %c by set : tensor<1x160x160xf32> -> tensor<1x160x160xf32> to tensor<1x160x160xf32>
  return %out : tensor<1x160x160xf32>
}

// -----

#p112x64 = #rock.gemm_params<mPerBlock = 112, nPerBlock = 64, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// M is non-power-of-two with three set bits (112 = 64+32+16):
// the M dimension decomposes into three segments {64,32,16};
// N (64) stays whole. M = 2*112 = 224.
// ============================================================

// CHECK-LABEL: func.func @test_m_three_segments
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 64, nPerBlock = 64{{.*}} -> tensor<1x128x128xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 32, nPerBlock = 64{{.*}} -> tensor<1x64x128xf32>
// CHECK-DAG: rock.gridwise_gemm{{.*}}mPerBlock = 16, nPerBlock = 64{{.*}} -> tensor<1x32x128xf32>
// CHECK-NOT: mPerBlock = 112
// CHECK-COUNT-3: rock.store
func.func @test_m_three_segments(%a: tensor<1x224x64xf16>, %b: tensor<1x64x128xf16>, %c: tensor<1x224x128xf32>) -> tensor<1x224x128xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #p112x64} : tensor<1x224x64xf16>, tensor<1x64x128xf16> -> tensor<1x224x128xf32>
  %out = rock.store %r to %c by set : tensor<1x224x128xf32> -> tensor<1x224x128xf32> to tensor<1x224x128xf32>
  return %out : tensor<1x224x128xf32>
}

// -----

#pknob80x80 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 80, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1, useReductionLayout = 1>

// ============================================================
// Knob propagation: the input params opt into the reduction-layout gate
// (useReductionLayout = 1, introduced in perfConfig v4). Every power-of-two sub-tile
// produced by the 2x2 split must carry the same knob so a tuned config is not
// silently dropped during decomposition.
// ============================================================

// CHECK-LABEL: func.func @test_reduction_layout_knob_propagates
// CHECK-COUNT-4: rock.gridwise_gemm{{.*}}useReductionLayout = 1
// CHECK-NOT: rock.gridwise_gemm
// CHECK-NOT: mPerBlock = 80
func.func @test_reduction_layout_knob_propagates(%a: tensor<1x160x64xf16>, %b: tensor<1x64x160xf16>, %c: tensor<1x160x160xf32>) -> tensor<1x160x160xf32> attributes {rock.kernel} {
  %r = rock.gridwise_gemm(%a, %b) {params = #pknob80x80} : tensor<1x160x64xf16>, tensor<1x64x160xf16> -> tensor<1x160x160xf32>
  %out = rock.store %r to %c by set : tensor<1x160x160xf32> -> tensor<1x160x160xf32> to tensor<1x160x160xf32>
  return %out : tensor<1x160x160xf32>
}

// -----

#pc80x64 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 64, kPerBlock = 64, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// ============================================================
// Chained stores (bwd-data shape): two gemms write disjoint M
// halves of the same output; the second store's destination is
// another view of the original output. Each gemm splits along M
// into {64,16}, and the resultAlias chain must be preserved so the
// final store result represents both writes.
// ============================================================

// The first original store sits before the second gemm, so the decomposed
// output is interleaved: gemm0 cells, store0 cells, gemm1 cells, store1 cells.
// CHECK-LABEL: func.func @test_chained_stores_nonpow2
// CHECK: rock.gridwise_gemm
// CHECK: rock.gridwise_gemm
// CHECK: rock.store
// CHECK: rock.store
// CHECK: rock.gridwise_gemm
// CHECK: rock.gridwise_gemm
// The second store writes through the original argument with offset +160 while
// aliasing the first store result.
// CHECK: rock.transform %{{.*}} -> (d0, d1 + 160, d2)
// CHECK: rock.store
// CHECK: rock.store
// CHECK: return
func.func @test_chained_stores_nonpow2(%a0: tensor<1x160x64xf16>, %b0: tensor<1x64x128xf16>, %a1: tensor<1x160x64xf16>, %b1: tensor<1x64x128xf16>, %dest: tensor<1x320x128xf32>) -> tensor<1x320x128xf32> attributes {rock.kernel} {
  %r0 = rock.gridwise_gemm(%a0, %b0) {params = #pc80x64} : tensor<1x160x64xf16>, tensor<1x64x128xf16> -> tensor<1x160x128xf32>
  %dest0 = rock.transform %dest by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Slice{0, 160} ["m0"] at [1] -> ["m"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>] bounds = [1, 160, 128] -> [1, 320, 128]> : tensor<1x320x128xf32> to tensor<1x160x128xf32>
  %s0 = rock.store %r0 to %dest0 alias %dest by set : tensor<1x160x128xf32> -> tensor<1x320x128xf32> to tensor<1x160x128xf32> alias tensor<1x320x128xf32>
  %r1 = rock.gridwise_gemm(%a1, %b1) {params = #pc80x64} : tensor<1x160x64xf16>, tensor<1x64x128xf16> -> tensor<1x160x128xf32>
  %dest1 = rock.transform %dest by <affine_map<(d0, d1, d2) -> (d0, d1 + 160, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Slice{160, 320} ["m1"] at [1] -> ["m"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>] bounds = [1, 160, 128] -> [1, 320, 128]> : tensor<1x320x128xf32> to tensor<1x160x128xf32>
  %s1 = rock.store %r1 to %dest1 alias %s0 by set : tensor<1x160x128xf32> -> tensor<1x320x128xf32> to tensor<1x160x128xf32> alias tensor<1x320x128xf32>
  return %s1 : tensor<1x320x128xf32>
}

// -----

// The attention path (processGridwiseAttention) splits a rock.gridwise_attention
// along its two output axes: seqLenQ (params0.mPerBlock, the grid M tile) and
// headDimV (result N = gemm1N). seqLenK (params0.nPerBlock, the softmax
// reduction) stays whole. Each (mSeg, nSeg) cell becomes a sub-attention over
// sliced queries (M) / values (head dim) / output (M and N).

#a0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#a1n16 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 16, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// Only the head dim (result N = 80) is non-power-of-two: the
// attention splits into head-dim segments {64,16}. seqLenQ (M)
// stays whole, so queries are not sliced; only V and the output
// are sliced along the head dim.
// ============================================================

// CHECK-LABEL: func.func @test_attn_headdim_nonpow2
// CHECK-COUNT-2: rock.gridwise_attention
// CHECK-NOT: rock.gridwise_attention
// CHECK-DAG: rock.store {{.*}} : tensor<1x64x64xf32> -> tensor<1x64x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x64x16xf32> -> tensor<1x64x80xf32>
func.func @test_attn_headdim_nonpow2(%q: tensor<1x64x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x80xf32>, %o: tensor<1x64x80xf32>) -> tensor<1x64x80xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #a0, params1 = #a1n16, splitKV = 1 : i32} : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x80xf32> -> tensor<1x64x80xf32>
  %out = rock.store %result to %o by set : tensor<1x64x80xf32> -> tensor<1x64x80xf32> to tensor<1x64x80xf32>
  return %out : tensor<1x64x80xf32>
}

// -----

#ap0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#ap1n16 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 16, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// Head dim (result N = 80) non-power-of-two, seqLenQ (M) stays
// whole, but seqLenQ = 128 is padded (prePadG0M = 100) across
// mBlocks = 128/32 = 4 blocks. Since M is not split, both head-dim
// sub-attentions must carry the original prePadG0M unchanged (it is
// a full-M quantity that exceeds a single mPerBlock tile).
// ============================================================

// CHECK-LABEL: func.func @test_attn_headdim_nonpow2_mpad_multiblock
// CHECK: rock.gridwise_attention
// CHECK: prePadG0M = 100 : index
// CHECK: rock.gridwise_attention
// CHECK: prePadG0M = 100 : index
// CHECK-NOT: rock.gridwise_attention
func.func @test_attn_headdim_nonpow2_mpad_multiblock(%q: tensor<1x128x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x80xf32>, %o: tensor<1x128x80xf32>) -> tensor<1x128x80xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, prePadG0M = 100 : index, params0 = #ap0, params1 = #ap1n16, splitKV = 1 : i32} : tensor<1x128x32xf32>, tensor<1x32x64xf32>, tensor<1x64x80xf32> -> tensor<1x128x80xf32>
  %out = rock.store %result to %o by set : tensor<1x128x80xf32> -> tensor<1x128x80xf32> to tensor<1x128x80xf32>
  return %out : tensor<1x128x80xf32>
}

// -----

#am0 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#am1 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// Only seqLenQ is non-power-of-two (mPerBlock = 80, seqLenQ =
// 160 -> mBlocks = 2), with an LSE output. M splits into {64,16}
// (per-block sizes 2*64 = 128 and 2*16 = 32). LSE is head-dim
// independent, so each M-segment gets its own LSE store.
// ============================================================

// CHECK-LABEL: func.func @test_attn_seqq_nonpow2_lse
// CHECK-COUNT-2: rock.gridwise_attention
// CHECK-NOT: rock.gridwise_attention
// CHECK-DAG: rock.store {{.*}} : tensor<1x128x64xf32> -> tensor<1x160x64xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x32x64xf32> -> tensor<1x160x64xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x128xf32> -> tensor<1x160xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x32xf32> -> tensor<1x160xf32>
func.func @test_attn_seqq_nonpow2_lse(%q: tensor<1x160x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x64xf32>, %lse: tensor<1x160xf32>, %o: tensor<1x160x64xf32>) -> (tensor<1x160x64xf32>, tensor<1x160xf32>) attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result, %lseOut = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #am0, params1 = #am1, splitKV = 1 : i32} : tensor<1x160x32xf32>, tensor<1x32x64xf32>, tensor<1x64x64xf32> -> tensor<1x160x64xf32>, tensor<1x160xf32>
  %out = rock.store %result to %o by set : tensor<1x160x64xf32> -> tensor<1x160x64xf32> to tensor<1x160x64xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<1x160xf32> -> tensor<1x160xf32> to tensor<1x160xf32>
  return %out, %lseStore : tensor<1x160x64xf32>, tensor<1x160xf32>
}

// -----

#am0 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#am1 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// Same M-split as above but the LSE output carries an elementwise
// fusion (add then mul against rank-2 [G, seqLenQ] biases). The
// LSE fusion DAG is replicated per M segment, so each segment gets
// its own add/mul before the sliced store.
// ============================================================

// CHECK-LABEL: func.func @test_attn_lse_output_fusion
// CHECK-COUNT-2: rock.gridwise_attention
// CHECK-NOT: rock.gridwise_attention
// CHECK-COUNT-2: arith.addf
// CHECK-COUNT-2: arith.mulf
// CHECK-DAG: rock.store {{.*}} : tensor<1x128xf32> -> tensor<1x160xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x32xf32> -> tensor<1x160xf32>
func.func @test_attn_lse_output_fusion(%q: tensor<1x160x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x64xf32>, %lseBias0: tensor<1x160xf32>, %lseBias1: tensor<1x160xf32>, %lse: tensor<1x160xf32>, %o: tensor<1x160x64xf32>) -> (tensor<1x160x64xf32>, tensor<1x160xf32>) attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result, %lseOut = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #am0, params1 = #am1, splitKV = 1 : i32} : tensor<1x160x32xf32>, tensor<1x32x64xf32>, tensor<1x64x64xf32> -> tensor<1x160x64xf32>, tensor<1x160xf32>
  %out = rock.store %result to %o by set : tensor<1x160x64xf32> -> tensor<1x160x64xf32> to tensor<1x160x64xf32>
  %lseAdd = arith.addf %lseOut, %lseBias0 : tensor<1x160xf32>
  %lseMul = arith.mulf %lseAdd, %lseBias1 : tensor<1x160xf32>
  %lseStore = rock.store %lseMul to %lse by set : tensor<1x160xf32> -> tensor<1x160xf32> to tensor<1x160xf32>
  return %out, %lseStore : tensor<1x160x64xf32>, tensor<1x160xf32>
}

// -----

#amp0 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#amp1 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// seqLenQ non-power-of-two (mPerBlock = 80, seqLenQ = 160 ->
// mBlocks = 2) AND padded (prePadG0M = 150). M splits into
// {64,16}. Each segment forwards the original prePadG0M unchanged
// and records the pre-split tile (gemm0MOrigPerBlock = 80) and its
// slice offset so lowering can replay the (mBlocks, tile) + slice +
// pad on the first-gemm mask (the padded rows are interleaved per
// block, which no single scalar per segment can express).
// ============================================================

// CHECK-LABEL: func.func @test_attn_seqq_nonpow2_mpad
// CHECK: rock.gridwise_attention
// CHECK: gemm0MOrigPerBlock = 80 : index
// CHECK-SAME: gemm0MSliceOffset = 0 : index
// CHECK-SAME: prePadG0M = 150 : index
// CHECK: rock.gridwise_attention
// CHECK: gemm0MOrigPerBlock = 80 : index
// CHECK-SAME: gemm0MSliceOffset = 64 : index
// CHECK-SAME: prePadG0M = 150 : index
// CHECK-NOT: rock.gridwise_attention
func.func @test_attn_seqq_nonpow2_mpad(%q: tensor<1x160x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x64xf32>, %o: tensor<1x160x64xf32>) -> tensor<1x160x64xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, prePadG0M = 150 : index, params0 = #amp0, params1 = #amp1, splitKV = 1 : i32} : tensor<1x160x32xf32>, tensor<1x32x64xf32>, tensor<1x64x64xf32> -> tensor<1x160x64xf32>
  %out = rock.store %result to %o by set : tensor<1x160x64xf32> -> tensor<1x160x64xf32> to tensor<1x160x64xf32>
  return %out : tensor<1x160x64xf32>
}

// -----

#ab0 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#ab1 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 16, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// Both axes non-power-of-two (seqLenQ = 160/mPerBlock = 80, head
// dim = 80): a 2x2 grid of sub-attentions over M x head-dim =
// {64,16} x {64,16}.
// ============================================================

// CHECK-LABEL: func.func @test_attn_both_nonpow2
// CHECK-COUNT-4: rock.gridwise_attention
// CHECK-NOT: rock.gridwise_attention
// CHECK-DAG: rock.store {{.*}} : tensor<1x128x64xf32> -> tensor<1x160x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x128x16xf32> -> tensor<1x160x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x32x64xf32> -> tensor<1x160x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x32x16xf32> -> tensor<1x160x80xf32>
func.func @test_attn_both_nonpow2(%q: tensor<1x160x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x80xf32>, %o: tensor<1x160x80xf32>) -> tensor<1x160x80xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #ab0, params1 = #ab1, splitKV = 1 : i32} : tensor<1x160x32xf32>, tensor<1x32x64xf32>, tensor<1x64x80xf32> -> tensor<1x160x80xf32>
  %out = rock.store %result to %o by set : tensor<1x160x80xf32> -> tensor<1x160x80xf32> to tensor<1x160x80xf32>
  return %out : tensor<1x160x80xf32>
}

// -----

#abl0 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#abl1 = #rock.gemm_params<mPerBlock = 80, nPerBlock = 16, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// Both axes non-power-of-two (seqLenQ = 160/mPerBlock = 80, head
// dim = 80) AND an LSE output: the main result splits into a 2x2
// grid over M x head-dim = {64,16} x {64,16}, but the LSE is
// head-dim independent so it only splits along M into {64,16}
// (per-block 128 and 32). Both head-dim sub-tiles that share an
// M-segment therefore store into the same LSE slice.
// ============================================================

// CHECK-LABEL: func.func @test_attn_both_nonpow2_lse
// CHECK-COUNT-4: rock.gridwise_attention
// CHECK-NOT: rock.gridwise_attention
// CHECK-DAG: rock.store {{.*}} : tensor<1x128x64xf32> -> tensor<1x160x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x128x16xf32> -> tensor<1x160x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x32x64xf32> -> tensor<1x160x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x32x16xf32> -> tensor<1x160x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x128xf32> -> tensor<1x160xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x32xf32> -> tensor<1x160xf32>
func.func @test_attn_both_nonpow2_lse(%q: tensor<1x160x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x80xf32>, %lse: tensor<1x160xf32>, %o: tensor<1x160x80xf32>) -> (tensor<1x160x80xf32>, tensor<1x160xf32>) attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result, %lseOut = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #abl0, params1 = #abl1, splitKV = 1 : i32} : tensor<1x160x32xf32>, tensor<1x32x64xf32>, tensor<1x64x80xf32> -> tensor<1x160x80xf32>, tensor<1x160xf32>
  %out = rock.store %result to %o by set : tensor<1x160x80xf32> -> tensor<1x160x80xf32> to tensor<1x160x80xf32>
  %lseStore = rock.store %lseOut to %lse by set : tensor<1x160xf32> -> tensor<1x160xf32> to tensor<1x160xf32>
  return %out, %lseStore : tensor<1x160x80xf32>, tensor<1x160xf32>
}

// -----

#ap0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#ap1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 16, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// Pre-softmax fusion (a scale mul) with a non-power-of-two head
// dim (80). The preSoftmax region is cloned into each head-dim
// sub-attention; the scale input is gemm0-shaped (seqLenQ x
// seqLenK), so it is NOT sliced along the head dim (both cells
// share the same scale operand).
// ============================================================

// CHECK-LABEL: func.func @test_attn_presoftmax_headdim_nonpow2
// CHECK-DAG: rock.gridwise_attention
// CHECK-DAG: rock.gridwise_attention
// CHECK-DAG: arith.mulf
// CHECK-DAG: arith.mulf
// CHECK-DAG: rock.store {{.*}} : tensor<1x64x64xf32> -> tensor<1x64x80xf32>
// CHECK-DAG: rock.store {{.*}} : tensor<1x64x16xf32> -> tensor<1x64x80xf32>
func.func @test_attn_presoftmax_headdim_nonpow2(%q: tensor<1x64x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x80xf32>, %scale: tensor<1xf32>, %o: tensor<1x64x80xf32>) -> tensor<1x64x80xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %s0 = rock.transform %scale by <affine_map<(d0, d1, d2) -> (d0 + d1 + d2)> by [<Unmerge{1, 1, 1} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 1, 1] -> [1]> : tensor<1xf32> to tensor<1x1x1xf32>
  %s1 = rock.transform %s0 by <affine_map<(d0, d1, d2) -> (d0, 0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [1, 64, 64] -> [1, 1, 1]> : tensor<1x1x1xf32> to tensor<1x64x64xf32>
  %result = rock.gridwise_attention(%q, %k, %v, %s1) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>, %arg_scale: tensor<1x64x64xf32>):
    %0 = arith.mulf %arg_qk, %arg_scale : tensor<1x64x64xf32>
    rock.yield %0 : tensor<1x64x64xf32>
  } {operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>, params0 = #ap0, params1 = #ap1, splitKV = 1 : i32} : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x80xf32>, tensor<1x64x64xf32> -> tensor<1x64x80xf32>
  %out = rock.store %result to %o by set : tensor<1x64x80xf32> -> tensor<1x64x80xf32> to tensor<1x64x80xf32>
  return %out : tensor<1x64x80xf32>
}

// -----

#anop0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#anop1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>

// ============================================================
// Nothing to do: seqLenQ tile (32) and head dim (64) are both
// powers of two. The attention and store are left unchanged.
// ============================================================

// CHECK-LABEL: func.func @test_attn_nothing_to_do
// CHECK: rock.gridwise_attention
// CHECK-NOT: rock.gridwise_attention
// CHECK: rock.store
// CHECK-NOT: rock.store
func.func @test_attn_nothing_to_do(%q: tensor<1x64x32xf32>, %k: tensor<1x32x64xf32>, %v: tensor<1x64x64xf32>, %o: tensor<1x64x64xf32>) -> tensor<1x64x64xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  } {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #anop0, params1 = #anop1, splitKV = 1 : i32} : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x64xf32> -> tensor<1x64x64xf32>
  %out = rock.store %result to %o by set : tensor<1x64x64xf32> -> tensor<1x64x64xf32> to tensor<1x64x64xf32>
  return %out : tensor<1x64x64xf32>
}
