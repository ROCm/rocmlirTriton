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

// RUN: rocmlir-opt -rock-decompose-nonpow2-tiles -canonicalize -split-input-file -mlir-print-local-scope %s | FileCheck %s

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
