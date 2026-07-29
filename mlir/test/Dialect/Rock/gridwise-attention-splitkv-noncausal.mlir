// Non-causal / non-KV-cache split-KV attention lowering.
//
// This locks in the per-split N-loop iteration math for the non-causal
// split-KV branch. (Split-KV partitions the first GEMM along the key-sequence
// dimension, which is GEMM0's N axis here; rocMLIR computes the transposed
// product and calls the same axis M.) The bounds must use ceil-division
// together with a minui clamp to the block count so that trailing splits never
// iterate past the available blocks, and so configurations with fewer blocks
// than splits still get at least one iteration per split -- truncating
// division would give zero, skipping the softmax for every split and leaving
// 0/0 in the final rescale.
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -canonicalize -verify-diagnostics | FileCheck %s

// 12 blocks (seqK 384 / nPerBlock 32) over 5 splits, so itersPerSplit = 3 and
// the last split would reach 15 without the clamp.

// CHECK-LABEL: func @gridwise_attn_splitkv_indivisible
// CHECK-DAG: %[[cIters:.+]] = arith.constant 3 : i32
// CHECK-DAG: %[[cBlocks:.+]] = arith.constant 12 : i32
// CHECK-DAG: %[[cSplit:.+]] = arith.constant 5 : i32
// CHECK-DAG: %[[cOne:.+]] = arith.constant 1 : i32
// splitBlock = workgroup_id % splitKV
// CHECK: %[[block:.+]] = arith.remui %{{.+}}, %[[cSplit]] : i32
// start = splitBlock * itersPerSplit
// CHECK: %[[start:.+]] = arith.muli %[[block]], %[[cIters]] : i32
// end = min((splitBlock + 1) * itersPerSplit, gemm0NBlocks)
// CHECK: %[[blockP1:.+]] = arith.addi %[[block]], %[[cOne]] : i32
// CHECK: %[[endRaw:.+]] = arith.muli %[[blockP1]], %[[cIters]] : i32
// CHECK: %[[end:.+]] = arith.minui %[[endRaw]], %[[cBlocks]] : i32
// CHECK: scf.for %{{.+}} = %[[start]] to %[[end]] step %[[cOne]]
func.func @gridwise_attn_splitkv_indivisible(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x384xf32>,
    %v: tensor<1x384x64xf32>) -> tensor<5x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 120 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 5 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<5x384x64xf32>
  return %result : tensor<5x384x64xf32>
}

// -----

// A single block (seqK 32 / nPerBlock 32) over 2 splits. Ceil-division keeps
// itersPerSplit at 1 rather than 0, so split 0 still runs the softmax; the
// clamp then leaves split 1 empty. Since itersPerSplit is 1 here, canonicalize
// folds the start multiply away and both constants collapse to %c1_i32.

// CHECK-LABEL: func @gridwise_attn_splitkv_fewer_blocks_than_splits
// CHECK-DAG: %[[cOne2:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[cSplit2:.+]] = arith.constant 2 : i32
// CHECK: %[[block2:.+]] = arith.remui %{{.+}}, %[[cSplit2]] : i32
// CHECK: %[[endRaw2:.+]] = arith.addi %[[block2]], %[[cOne2]] : i32
// CHECK: %[[end2:.+]] = arith.minui %[[endRaw2]], %[[cOne2]] : i32
// CHECK: scf.for %{{.+}} = %[[block2]] to %[[end2]] step %[[cOne2]]
func.func @gridwise_attn_splitkv_fewer_blocks_than_splits(
    %q: tensor<1x384x64xf32>,
    %k: tensor<1x64x32xf32>,
    %v: tensor<1x32x64xf32>) -> tensor<2x384x64xf32>
    attributes {
      rock.block_size = 64 : i32,
      rock.grid_size = 48 : i32,
      rock.kernel,
      rock.arch = "##TOKEN_ARCH##"
    } {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x16x32xf32>):
    rock.yield %arg_qk : tensor<1x16x32xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 32, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 2 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x32xf32>, tensor<1x32x64xf32> -> tensor<2x384x64xf32>
  return %result : tensor<2x384x64xf32>
}
