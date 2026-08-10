// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -canonicalize -verify-diagnostics | FileCheck %s

// Same shape as gridwise_attn_simple in toblockwise_attention_lowering.mlir, but
// params1 tiles the second gemm's N dim (head_dim_v = 64) into nPerBlockG1 = 32
// wide chunks => gemm1NChunks = 2. Each chunk keeps its own accumulator and runs
// its own V load + second GEMM; the per-chunk [16x32] output tiles are folded
// back into a single [16x64] tile with pairwise tt.join + tt.trans + tt.reshape.

// CHECK-LABEL: func @gridwise_attn_nperblockg1
// CHECK-SAME: (%[[Q:.+]]: tensor<1x384x64xf32>, %[[K:.+]]: tensor<1x64x384xf32>, %[[V:.+]]: tensor<1x384x64xf32>)

// The flash accumulators are now nPerBlockG1-wide ([16x32]), not the full
// head-dim width ([16x64]).
// CHECK-DAG: %[[ln2:.+]] = arith.constant dense<1.44269502> : tensor<16x32xf32>
// CHECK-DAG: %[[zero32:.+]] = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[c12:.+]] = arith.constant 12 : i32
// CHECK-DAG: %[[zeroRow:.+]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG: %[[negInf:.+]] = arith.constant dense<0xFF800000> : tensor<16xf32>
// CHECK-DAG: %[[c24:.+]] = arith.constant 24 : i32
// CHECK-DAG: %[[c4:.+]] = arith.constant 4 : i32
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : i32

// CHECK: %[[PID:.+]] = tt.get_program_id x : i32

// Outer N-tile loop now carries one accumulator per head-dim chunk (acc0, acc1)
// plus running max/sum, so it yields 4 results instead of 3.
// CHECK: %[[OUT:.+]]:4 = scf.for %{{.*}} = %[[c0]] to %[[c12]] step %[[c1]] iter_args(%[[acc0:.+]] = %[[zero32]], %[[acc1:.+]] = %[[zero32]], %[[rmax:.+]] = %[[negInf]], %[[rsum:.+]] = %[[zeroRow]])

// gemm0 = Q @ K^T.
// CHECK: %[[G0:.+]] = scf.for %{{.*}} = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[g0acc:.+]] = %[[zero32]])
// CHECK: %[[g0gemm:.+]] = rock.blockwise_gemm
// CHECK: scf.yield %[[g0gemm]]

// Softmax (shared across chunks): scale, online max, exp2, row-sum.
// CHECK: %[[scaled:.+]] = arith.mulf %[[G0]], %[[ln2]]
// CHECK: %[[tileMax:.+]] = rock.blockwise_reduce max %[[scaled]] {axis = 1 : index}
// CHECK: %[[P:.+]] = math.exp2
// CHECK: rock.blockwise_reduce sum %[[P]] {axis = 1 : index}

// gemm1 chunk 0: load V chunk n_block = 0, second GEMM into a fresh [16x32]
// accumulator, then flash-correct against acc0.
// CHECK: %[[vTile0:.+]] = rock.load_marker %[[V]] views [#{{.*}}][%{{.*}}, %{{.*}}, %{{.*}}, %[[c0]]]
// CHECK: %[[g1gemm0:.+]] = rock.blockwise_gemm(%[[P]], %[[vTile0]], %[[zero32]])
// CHECK: %[[accScaled0:.+]] = arith.mulf %[[acc0]], %{{.*}}
// CHECK: %[[newAcc0:.+]] = arith.addf %[[accScaled0]], %[[g1gemm0]]

// gemm1 chunk 1: load V chunk n_block = 1, independent second GEMM, correct
// against acc1.
// CHECK: %[[vTile1:.+]] = rock.load_marker %[[V]] views [#{{.*}}][%{{.*}}, %{{.*}}, %{{.*}}, %[[c1]]]
// CHECK: %[[g1gemm1:.+]] = rock.blockwise_gemm(%[[P]], %[[vTile1]], %[[zero32]])
// CHECK: %[[accScaled1:.+]] = arith.mulf %[[acc1]], %{{.*}}
// CHECK: %[[newAcc1:.+]] = arith.addf %[[accScaled1]], %[[g1gemm1]]

// CHECK: scf.yield %[[newAcc0]], %[[newAcc1]], %{{.*}}, %{{.*}}

// Final normalize each chunk by the row-sum, then fold the two [16x32] tiles
// into one [16x64] tile via tt.join -> tt.trans -> tt.reshape, and store.
// CHECK: %[[norm0:.+]] = arith.divf %[[OUT]]#0, %{{.*}}
// CHECK: %[[norm1:.+]] = arith.divf %[[OUT]]#1, %{{.*}}
// CHECK: %[[joined:.+]] = tt.join %[[norm0]], %[[norm1]] : tensor<16x32xf32> -> tensor<16x32x2xf32>
// CHECK: %[[transed:.+]] = tt.trans %[[joined]] {order = array<i32: 0, 2, 1>} : tensor<16x32x2xf32> -> tensor<16x2x32xf32>
// CHECK: %[[reshaped:.+]] = tt.reshape %[[transed]] : tensor<16x2x32xf32> -> tensor<16x64xf32>
// CHECK: rock.store_marker %[[reshaped]] views [#{{.*}}]
func.func @gridwise_attn_nperblockg1(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x384xf32>,
    %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 24 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}
