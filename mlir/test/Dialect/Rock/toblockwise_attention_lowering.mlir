// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -canonicalize -verify-diagnostics | FileCheck %s
// The causal-masking bound math in getNLoopInfo() is folded by -canonicalize
// (segLast + nPerBlock gets combined), so re-run without it to check the raw
// stride/offset arithmetic for the M-decomposed causal kernel below.
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file -rock-gridwise-attn-to-blockwise -verify-diagnostics | FileCheck %s --check-prefix=CAUSAL

// CHECK-LABEL: func @gridwise_attn_simple
// CHECK-SAME: (%[[Q:.+]]: tensor<1x384x64xf32>, %[[K:.+]]: tensor<1x64x384xf32>, %[[V:.+]]: tensor<1x384x64xf32>)

// Constants: ln2 = log2(e) softmax scale; zero32/zero64/zeroRow/negInf are
// the flash-attention accumulator inits; c12 = gemmN/nPerBlock outer count,
// c4 = gemmK/kPerBlock inner count, c24 = grid_size (block-coord divisor).
// CHECK-DAG: %[[ln2:.+]] = arith.constant dense<1.44269502> : tensor<16x32xf32>
// CHECK-DAG: %[[zero32:.+]] = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[zero64:.+]] = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
// CHECK-DAG: %[[c12:.+]] = arith.constant 12 : i32
// CHECK-DAG: %[[zeroRow:.+]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK-DAG: %[[negInf:.+]] = arith.constant dense<0xFF800000> : tensor<16xf32>
// CHECK-DAG: %[[c24:.+]] = arith.constant 24 : i32
// CHECK-DAG: %[[c4:.+]] = arith.constant 4 : i32
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : i32

// Triton-style block id.
// CHECK: %[[PID:.+]] = tt.get_program_id x : i32

// Outer N-tile loop, iter_args = (attn_output, running_max, running_sum).
// CHECK: %[[OUT:.+]]:3 = scf.for %{{.*}} = %[[c0]] to %[[c12]] step %[[c1]] iter_args(%[[acc:.+]] = %[[zero64]], %[[rmax:.+]] = %[[negInf]], %[[rsum:.+]] = %[[zeroRow]])

// Per-iteration block-M / block-N from PID.
// CHECK: arith.divui %{{.*}}, %[[c24]] : i32
// CHECK: arith.remui %{{.*}}, %[[c24]] : i32

// gemm0 = Q @ K^T : inner KpacksPerBlock loop accumulating into zero32.
// CHECK: %[[G0:.+]] = scf.for %{{.*}} = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[g0acc:.+]] = %[[zero32]])
// CHECK: %[[qTile:.+]] = rock.load_marker %[[Q]] views [#{{.*}}]
// CHECK: %[[kTile:.+]] = rock.load_marker %[[K]] views [#{{.*}}]
// CHECK: %[[g0gemm:.+]] = rock.blockwise_gemm(%[[qTile]], %[[kTile]], %[[g0acc]])
// CHECK: scf.yield %[[g0gemm]]

// Softmax: scale by ln2, online row-max, exp2(scores - newMax), row-sum.
// CHECK: %[[scaled:.+]] = arith.mulf %[[G0]], %[[ln2]]
// CHECK: %[[tileMax:.+]] = rock.blockwise_reduce max %[[scaled]] {axis = 1 : index}
// CHECK: %[[newMax:.+]] = arith.maximumf %[[rmax]], %[[tileMax]]
// CHECK: %[[maxExp:.+]] = tt.expand_dims %[[newMax]]
// CHECK: %[[maxBcast:.+]] = tt.broadcast %[[maxExp]]
// CHECK: %[[sub:.+]] = arith.subf %[[scaled]], %[[maxBcast]]
// CHECK: %[[P:.+]] = math.exp2 %[[sub]]
// CHECK: %[[tileSum:.+]] = rock.blockwise_reduce sum %[[P]] {axis = 1 : index}

// Row-sum rescale: l = exp2(m_{j-1} - m_j) * l_{j-1} + rowsum(P).
// CHECK: %[[newMax2:.+]] = arith.maximumf %[[rmax]], %[[tileMax]]
// CHECK: %[[maxDiff:.+]] = arith.subf %[[rmax]], %[[newMax2]]
// CHECK: %[[corr:.+]] = math.exp2 %[[maxDiff]]
// CHECK: %[[sumScaled:.+]] = arith.mulf %[[corr]], %[[rsum]]
// CHECK: %[[newSum:.+]] = arith.addf %[[sumScaled]], %[[tileSum]]

// gemm1 = P @ V.
// CHECK: arith.divui %{{.*}}, %[[c24]] : i32
// CHECK: arith.remui %{{.*}}, %[[c24]] : i32
// CHECK: %[[vTile:.+]] = rock.load_marker %[[V]] views [#{{.*}}]
// CHECK: %[[g1gemm:.+]] = rock.blockwise_gemm(%[[P]], %[[vTile]], %[[zero64]])

// Output rescale: acc = acc * corr + gemm1.
// CHECK: %[[corrExp:.+]] = tt.expand_dims %[[corr]]
// CHECK: %[[corrBcast:.+]] = tt.broadcast %[[corrExp]]
// CHECK: %[[accScaled:.+]] = arith.mulf %[[acc]], %[[corrBcast]]
// CHECK: %[[newAcc:.+]] = arith.addf %[[accScaled]], %[[g1gemm]]
// CHECK: scf.yield %[[newAcc]], %[[newMax2]], %[[newSum]]

// Final normalize by row-sum and store.
// CHECK: %[[sumExp:.+]] = tt.expand_dims %[[OUT]]#2
// CHECK: %[[sumBcast:.+]] = tt.broadcast %[[sumExp]]
// CHECK: %[[norm:.+]] = arith.divf %[[OUT]]#0, %[[sumBcast]]
// CHECK: arith.divui %{{.*}}, %[[c24]] : i32
// CHECK: arith.remui %{{.*}}, %[[c24]] : i32
// CHECK: rock.store_marker %[[norm]] views [#{{.*}}]
func.func @gridwise_attn_simple(
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
    params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}

// -----

// Real decomposition grid captured from the full pipeline
//   rocmlir-gen --operation attention -seq_len_q 150 -seq_len_k 64
//     -head_dim_qk 32 -head_dim_v 48 -perf_config attn:v6:80,64,16,32,...
//   | rocmlir-driver -c   (IR after rock-decompose-nonpow2-tiles +
//                          remove-dead-values, i.e. the input to this pass)
//
// seqLenQ = 150 is padded to 160 (prePadG0M = 150) and the non-pow2 tile
// gemm0MOrigPerBlock = 80 splits M into {64, 16} (mBlocks = 2); the non-pow2
// head_dim_v = 48 splits gemm1 N into {32, 16}. That yields a 2x2 grid of
// sub-attentions: M-offset {0, 64} x head-dim-offset {0, 32}. Each first-gemm
// mask replays the (mBlocks = 2, tile = 80) restructure + per-M-segment slice +
// pad to the valid seqLenQ; seqLenK is never split. The mask views are emitted
// as module-level aliases (canonicalize dedups shared ones) before the
// function, so match them first.
// CHECK-DAG: Unmerge{2, 80} ["m_b", "m_i"]
// CHECK-DAG: Pad{0, 10, 0, 0} ["paddedDim1", "paddedDim2"]
// CHECK-DAG: Slice{0, 64} ["m_i"]
// CHECK-DAG: Merge{2, 64} ["m"]
// CHECK-DAG: Slice{64, 80} ["m_i"]
// CHECK-DAG: Merge{2, 16} ["m"]
// CHECK: func{{.*}}@gridwise_attn_decomposed_grid
// The 2x2 grid lowers to four flash-attention bodies, each with its own store.
// CHECK-DAG: rock.store_marker %{{.*}} : tensor<64x32xf16> -> tensor<1x128x32xf16>
// CHECK-DAG: rock.store_marker %{{.*}} : tensor<64x16xf16> -> tensor<1x128x16xf16>
// CHECK-DAG: rock.store_marker %{{.*}} : tensor<16x32xf16> -> tensor<1x32x32xf16>
// CHECK-DAG: rock.store_marker %{{.*}} : tensor<16x16xf16> -> tensor<1x32x16xf16>
#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 16, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params2 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params3 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#map = affine_map<(d0, d1, d2) -> (d1 * 32 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1 * 48 + d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 80 + d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d1 mod 64, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 48 + d3)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 + 32)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + 64, d3)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 16, d1 mod 16, d2)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d2, d3 * 48 + d4)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map14 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d1 mod 64, 0, d2)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4 + 32)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 + 64, d3, d4)>
#map17 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 16, d1 mod 16, 0, d2)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 + 64, d3, d4 + 32)>
#transform_map = #rock.transform_map<#map by [<Unmerge{150, 32} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 32] -> [4800]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm0MPad"] at [1] -> ["gemm0M"] at [1]>, <PassThrough ["gemm0K"] at [2] -> ["gemm0K"] at [2]>] bounds = [1, 160, 32] -> [1, 150, 32]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{32, 64} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]>
#transform_map3 = #rock.transform_map<#map3 by [<Unmerge{64, 48} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 48] -> [3072]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <PassThrough ["d2"] at [3] -> ["d2"] at [2]>] bounds = [1, 2, 80, 32] -> [1, 160, 32]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{0, 64} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 2, 64, 32] -> [1, 2, 80, 32]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 128, 32] -> [1, 2, 64, 32]>
#transform_map7 = #rock.transform_map<#map7 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [2, 3] -> ["d2"] at [2]>] bounds = [1, 64, 1, 48] -> [1, 64, 48]>
#transform_map8 = #rock.transform_map<#map5 by [<PassThrough ["d0", "d1", "d2b"] at [0, 1, 2] -> ["d0", "d1", "d2b"] at [0, 1, 2]>, <Slice{0, 32} ["d2i"] at [3] -> ["d2i"] at [3]>] bounds = [1, 64, 1, 32] -> [1, 64, 1, 48]>
#transform_map9 = #rock.transform_map<#map8 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [2, 3]>] bounds = [1, 64, 32] -> [1, 64, 1, 32]>
#transform_map10 = #rock.transform_map<#map9 by [<PassThrough ["d0", "d1", "d2b"] at [0, 1, 2] -> ["d0", "d1", "d2b"] at [0, 1, 2]>, <Slice{32, 48} ["d2i"] at [3] -> ["d2i"] at [3]>] bounds = [1, 64, 1, 16] -> [1, 64, 1, 48]>
#transform_map11 = #rock.transform_map<#map8 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [2, 3]>] bounds = [1, 64, 16] -> [1, 64, 1, 16]>
#transform_map12 = #rock.transform_map<#map10 by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{64, 80} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 2, 16, 32] -> [1, 2, 80, 32]>
#transform_map13 = #rock.transform_map<#map11 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 32, 32] -> [1, 2, 16, 32]>
#transform_map14 = #rock.transform_map<#map3 by [<Unmerge{150, 48} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 48] -> [7200]>
#transform_map15 = #rock.transform_map<#map1 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm1MPad"] at [1] -> ["gemm1M"] at [1]>, <PassThrough ["gemm1N"] at [2] -> ["gemm1N"] at [2]>] bounds = [1, 160, 48] -> [1, 150, 48]>
#transform_map16 = #rock.transform_map<#map12 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [3, 4] -> ["d2"] at [2]>] bounds = [1, 2, 80, 1, 48] -> [1, 160, 48]>
#transform_map17 = #rock.transform_map<#map13 by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{0, 64, 0, 32} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 64, 1, 32] -> [1, 2, 80, 1, 48]>
#transform_map18 = #rock.transform_map<#map14 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 128, 32] -> [1, 2, 64, 1, 32]>
#transform_map19 = #rock.transform_map<#map15 by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{0, 64, 32, 48} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 64, 1, 16] -> [1, 2, 80, 1, 48]>
#transform_map20 = #rock.transform_map<#map14 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 128, 16] -> [1, 2, 64, 1, 16]>
#transform_map21 = #rock.transform_map<#map16 by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{64, 80, 0, 32} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 16, 1, 32] -> [1, 2, 80, 1, 48]>
#transform_map22 = #rock.transform_map<#map17 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 32, 32] -> [1, 2, 16, 1, 32]>
#transform_map23 = #rock.transform_map<#map18 by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{64, 80, 32, 48} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 16, 1, 16] -> [1, 2, 80, 1, 48]>
#transform_map24 = #rock.transform_map<#map17 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 32, 16] -> [1, 2, 16, 1, 16]>
func.func @gridwise_attn_decomposed_grid(%arg0: tensor<4800xf16>, %arg1: tensor<2048xf16>, %arg2: tensor<3072xf16>, %arg3: tensor<7200xf16>) -> tensor<7200xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 64 : i32, rock.grid_size = 2 : i32, rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<4800xf16> to tensor<1x150x32xf16>
    %1 = rock.transform %0 by #transform_map1 : tensor<1x150x32xf16> to tensor<1x160x32xf16>
    %2 = rock.transform %arg1 by #transform_map2 : tensor<2048xf16> to tensor<1x32x64xf16>
    %3 = rock.transform %arg2 by #transform_map3 : tensor<3072xf16> to tensor<1x64x48xf16>
    %4 = rock.transform %1 by #transform_map4 : tensor<1x160x32xf16> to tensor<1x2x80x32xf16>
    %5 = rock.transform %4 by #transform_map5 : tensor<1x2x80x32xf16> to tensor<1x2x64x32xf16>
    %6 = rock.transform %5 by #transform_map6 : tensor<1x2x64x32xf16> to tensor<1x128x32xf16>
    %7 = rock.transform %3 by #transform_map7 : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
    %8 = rock.transform %7 by #transform_map8 : tensor<1x64x1x48xf16> to tensor<1x64x1x32xf16>
    %9 = rock.transform %8 by #transform_map9 : tensor<1x64x1x32xf16> to tensor<1x64x32xf16>
    %result = rock.gridwise_attention(%6, %2, %9) preSoftmaxOps = {
    } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 0 : index, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #gemm_params, params1 = #gemm_params1, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x128x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16> -> tensor<1x128x32xf16>
    %10 = rock.transform %3 by #transform_map7 : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
    %11 = rock.transform %10 by #transform_map10 : tensor<1x64x1x48xf16> to tensor<1x64x1x16xf16>
    %12 = rock.transform %11 by #transform_map11 : tensor<1x64x1x16xf16> to tensor<1x64x16xf16>
    %result_0 = rock.gridwise_attention(%6, %2, %12) preSoftmaxOps = {
    } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 0 : index, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #gemm_params, params1 = #gemm_params1, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x128x32xf16>, tensor<1x32x64xf16>, tensor<1x64x16xf16> -> tensor<1x128x16xf16>
    %13 = rock.transform %1 by #transform_map4 : tensor<1x160x32xf16> to tensor<1x2x80x32xf16>
    %14 = rock.transform %13 by #transform_map12 : tensor<1x2x80x32xf16> to tensor<1x2x16x32xf16>
    %15 = rock.transform %14 by #transform_map13 : tensor<1x2x16x32xf16> to tensor<1x32x32xf16>
    %16 = rock.transform %3 by #transform_map7 : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
    %17 = rock.transform %16 by #transform_map8 : tensor<1x64x1x48xf16> to tensor<1x64x1x32xf16>
    %18 = rock.transform %17 by #transform_map9 : tensor<1x64x1x32xf16> to tensor<1x64x32xf16>
    %result_1 = rock.gridwise_attention(%15, %2, %18) preSoftmaxOps = {
    } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 64 : index, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #gemm_params2, params1 = #gemm_params3, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x32x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16> -> tensor<1x32x32xf16>
    %19 = rock.transform %3 by #transform_map7 : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
    %20 = rock.transform %19 by #transform_map10 : tensor<1x64x1x48xf16> to tensor<1x64x1x16xf16>
    %21 = rock.transform %20 by #transform_map11 : tensor<1x64x1x16xf16> to tensor<1x64x16xf16>
    %result_2 = rock.gridwise_attention(%15, %2, %21) preSoftmaxOps = {
    } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 64 : index, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #gemm_params2, params1 = #gemm_params3, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x32x32xf16>, tensor<1x32x64xf16>, tensor<1x64x16xf16> -> tensor<1x32x16xf16>
    %22 = rock.transform %arg3 by #transform_map14 : tensor<7200xf16> to tensor<1x150x48xf16>
    %23 = rock.transform %22 by #transform_map15 : tensor<1x150x48xf16> to tensor<1x160x48xf16>
    %24 = rock.transform %23 by #transform_map16 : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
    %25 = rock.transform %24 by #transform_map17 : tensor<1x2x80x1x48xf16> to tensor<1x2x64x1x32xf16>
    %26 = rock.transform %25 by #transform_map18 : tensor<1x2x64x1x32xf16> to tensor<1x128x32xf16>
    %27 = rock.store %result to %26 alias %arg3 by set : tensor<1x128x32xf16> -> tensor<7200xf16> to tensor<1x128x32xf16> alias tensor<7200xf16>
    %28 = rock.transform %arg3 by #transform_map14 : tensor<7200xf16> to tensor<1x150x48xf16>
    %29 = rock.transform %28 by #transform_map15 : tensor<1x150x48xf16> to tensor<1x160x48xf16>
    %30 = rock.transform %29 by #transform_map16 : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
    %31 = rock.transform %30 by #transform_map19 : tensor<1x2x80x1x48xf16> to tensor<1x2x64x1x16xf16>
    %32 = rock.transform %31 by #transform_map20 : tensor<1x2x64x1x16xf16> to tensor<1x128x16xf16>
    %33 = rock.store %result_0 to %32 alias %27 by set : tensor<1x128x16xf16> -> tensor<7200xf16> to tensor<1x128x16xf16> alias tensor<7200xf16>
    %34 = rock.transform %arg3 by #transform_map14 : tensor<7200xf16> to tensor<1x150x48xf16>
    %35 = rock.transform %34 by #transform_map15 : tensor<1x150x48xf16> to tensor<1x160x48xf16>
    %36 = rock.transform %35 by #transform_map16 : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
    %37 = rock.transform %36 by #transform_map21 : tensor<1x2x80x1x48xf16> to tensor<1x2x16x1x32xf16>
    %38 = rock.transform %37 by #transform_map22 : tensor<1x2x16x1x32xf16> to tensor<1x32x32xf16>
    %39 = rock.store %result_1 to %38 alias %33 by set : tensor<1x32x32xf16> -> tensor<7200xf16> to tensor<1x32x32xf16> alias tensor<7200xf16>
    %40 = rock.transform %arg3 by #transform_map14 : tensor<7200xf16> to tensor<1x150x48xf16>
    %41 = rock.transform %40 by #transform_map15 : tensor<1x150x48xf16> to tensor<1x160x48xf16>
    %42 = rock.transform %41 by #transform_map16 : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
    %43 = rock.transform %42 by #transform_map23 : tensor<1x2x80x1x48xf16> to tensor<1x2x16x1x16xf16>
    %44 = rock.transform %43 by #transform_map24 : tensor<1x2x16x1x16xf16> to tensor<1x32x16xf16>
    %45 = rock.store %result_2 to %44 alias %39 by set : tensor<1x32x16xf16> -> tensor<7200xf16> to tensor<1x32x16xf16> alias tensor<7200xf16>
    return %45 : tensor<7200xf16>
  }

// -----

// Same non-power-of-two decomposition as above, but with a NON-EMPTY preSoftmax
// region: an attention bias add (`arith.addf %qk, %bias`) whose bias input is
// the full, unpadded, pre-GQA [1, 150, 64] score-space tensor (%arg3). The
// decompose pass clones the region into each of the four (M-seg x head-dim-seg)
// sub-attentions and forwards the whole bias unsliced; the lowering must realign
// each sub-op's first-gemm tile back to that full space
// (decomposeTileView + unpadTileView) before loading + applying the bias. So the
// two M-offset-0 sub-ops load a 64x64 bias tile and the two M-offset-64 sub-ops
// a 16x64 tile (the differing view chains encode the per-segment slice offset),
// and each cloned add consumes exactly that realigned bias tile. Captured (like
// the test above) from the full pipeline with --with-attn-bias, as the input to
// rock-gridwise-attn-to-blockwise.
// CHECK-LABEL: func{{.*}}@gridwise_attn_decomposed_bias
// CHECK-DAG: %[[BIAS64A:.+]] = rock.load_marker %arg3 {{.*}} : tensor<9600xf16> -> tensor<64x64xf16>
// CHECK-DAG: %[[BIAS64B:.+]] = rock.load_marker %arg3 {{.*}} : tensor<9600xf16> -> tensor<64x64xf16>
// CHECK-DAG: %[[BIAS16A:.+]] = rock.load_marker %arg3 {{.*}} : tensor<9600xf16> -> tensor<16x64xf16>
// CHECK-DAG: %[[BIAS16B:.+]] = rock.load_marker %arg3 {{.*}} : tensor<9600xf16> -> tensor<16x64xf16>
// CHECK-DAG: arith.addf %{{.+}}, %[[BIAS64A]] : tensor<64x64xf16>
// CHECK-DAG: arith.addf %{{.+}}, %[[BIAS64B]] : tensor<64x64xf16>
// CHECK-DAG: arith.addf %{{.+}}, %[[BIAS16A]] : tensor<16x64xf16>
// CHECK-DAG: arith.addf %{{.+}}, %[[BIAS16B]] : tensor<16x64xf16>
// The 2x2 grid still lowers to four flash-attention bodies + stores.
// CHECK-DAG: rock.store_marker %{{.*}} : tensor<64x32xf16> -> tensor<1x128x32xf16>
// CHECK-DAG: rock.store_marker %{{.*}} : tensor<64x16xf16> -> tensor<1x128x16xf16>
// CHECK-DAG: rock.store_marker %{{.*}} : tensor<16x32xf16> -> tensor<1x32x32xf16>
// CHECK-DAG: rock.store_marker %{{.*}} : tensor<16x16xf16> -> tensor<1x32x16xf16>
func.func @gridwise_attn_decomposed_bias(%arg0: tensor<4800xf16>, %arg1: tensor<2048xf16>, %arg2: tensor<3072xf16>, %arg3: tensor<9600xf16>, %arg4: tensor<7200xf16>) -> tensor<7200xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 128 : i32, rock.grid_size = 2 : i32, rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 32 + d2)> by [<Unmerge{150, 32} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 32] -> [4800]> : tensor<4800xf16> to tensor<1x150x32xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm0MPad"] at [1] -> ["gemm0M"] at [1]>, <PassThrough ["gemm0K"] at [2] -> ["gemm0K"] at [2]>] bounds = [1, 160, 32] -> [1, 150, 32]> : tensor<1x150x32xf16> to tensor<1x160x32xf16>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{32, 64} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]> : tensor<2048xf16> to tensor<1x32x64xf16>
  %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 48 + d2)> by [<Unmerge{64, 48} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 48] -> [3072]> : tensor<3072xf16> to tensor<1x64x48xf16>
  %4 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{150, 64} ["seq_q", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 64] -> [9600]> : tensor<9600xf16> to tensor<1x150x64xf16>
  %5 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 80 + d2, d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <PassThrough ["d2"] at [3] -> ["d2"] at [2]>] bounds = [1, 2, 80, 32] -> [1, 160, 32]> : tensor<1x160x32xf16> to tensor<1x2x80x32xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{0, 64} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 2, 64, 32] -> [1, 2, 80, 32]> : tensor<1x2x80x32xf16> to tensor<1x2x64x32xf16>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d1 mod 64, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 128, 32] -> [1, 2, 64, 32]> : tensor<1x2x64x32xf16> to tensor<1x128x32xf16>
  %8 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 48 + d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [2, 3] -> ["d2"] at [2]>] bounds = [1, 64, 1, 48] -> [1, 64, 48]> : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
  %9 = rock.transform %8 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> by [<PassThrough ["d0", "d1", "d2b"] at [0, 1, 2] -> ["d0", "d1", "d2b"] at [0, 1, 2]>, <Slice{0, 32} ["d2i"] at [3] -> ["d2i"] at [3]>] bounds = [1, 64, 1, 32] -> [1, 64, 1, 48]> : tensor<1x64x1x48xf16> to tensor<1x64x1x32xf16>
  %10 = rock.transform %9 by <affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [2, 3]>] bounds = [1, 64, 32] -> [1, 64, 1, 32]> : tensor<1x64x1x32xf16> to tensor<1x64x32xf16>
  %result = rock.gridwise_attention(%7, %2, %10, %4) preSoftmaxOps = {
  ^bb0(%arg5: tensor<1x150x64xf16>, %arg6: tensor<1x150x64xf16>):
    %47 = arith.addf %arg5, %arg6 : tensor<1x150x64xf16>
    rock.yield %47 : tensor<1x150x64xf16>
  } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 0 : index, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>, params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>, params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 16, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x128x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16>, tensor<1x150x64xf16> -> tensor<1x128x32xf16>
  %11 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 48 + d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [2, 3] -> ["d2"] at [2]>] bounds = [1, 64, 1, 48] -> [1, 64, 48]> : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
  %12 = rock.transform %11 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 + 32)> by [<PassThrough ["d0", "d1", "d2b"] at [0, 1, 2] -> ["d0", "d1", "d2b"] at [0, 1, 2]>, <Slice{32, 48} ["d2i"] at [3] -> ["d2i"] at [3]>] bounds = [1, 64, 1, 16] -> [1, 64, 1, 48]> : tensor<1x64x1x48xf16> to tensor<1x64x1x16xf16>
  %13 = rock.transform %12 by <affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [2, 3]>] bounds = [1, 64, 16] -> [1, 64, 1, 16]> : tensor<1x64x1x16xf16> to tensor<1x64x16xf16>
  %result_0 = rock.gridwise_attention(%7, %2, %13, %4) preSoftmaxOps = {
  ^bb0(%arg5: tensor<1x150x64xf16>, %arg6: tensor<1x150x64xf16>):
    %47 = arith.addf %arg5, %arg6 : tensor<1x150x64xf16>
    rock.yield %47 : tensor<1x150x64xf16>
  } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 0 : index, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>, params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>, params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 16, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x128x32xf16>, tensor<1x32x64xf16>, tensor<1x64x16xf16>, tensor<1x150x64xf16> -> tensor<1x128x16xf16>
  %14 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 80 + d2, d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <PassThrough ["d2"] at [3] -> ["d2"] at [2]>] bounds = [1, 2, 80, 32] -> [1, 160, 32]> : tensor<1x160x32xf16> to tensor<1x2x80x32xf16>
  %15 = rock.transform %14 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + 64, d3)> by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{64, 80} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 2, 16, 32] -> [1, 2, 80, 32]> : tensor<1x2x80x32xf16> to tensor<1x2x16x32xf16>
  %16 = rock.transform %15 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 16, d1 mod 16, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 32, 32] -> [1, 2, 16, 32]> : tensor<1x2x16x32xf16> to tensor<1x32x32xf16>
  %17 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 48 + d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [2, 3] -> ["d2"] at [2]>] bounds = [1, 64, 1, 48] -> [1, 64, 48]> : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
  %18 = rock.transform %17 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> by [<PassThrough ["d0", "d1", "d2b"] at [0, 1, 2] -> ["d0", "d1", "d2b"] at [0, 1, 2]>, <Slice{0, 32} ["d2i"] at [3] -> ["d2i"] at [3]>] bounds = [1, 64, 1, 32] -> [1, 64, 1, 48]> : tensor<1x64x1x48xf16> to tensor<1x64x1x32xf16>
  %19 = rock.transform %18 by <affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [2, 3]>] bounds = [1, 64, 32] -> [1, 64, 1, 32]> : tensor<1x64x1x32xf16> to tensor<1x64x32xf16>
  %result_1 = rock.gridwise_attention(%16, %2, %19, %4) preSoftmaxOps = {
  ^bb0(%arg5: tensor<1x150x64xf16>, %arg6: tensor<1x150x64xf16>):
    %47 = arith.addf %arg5, %arg6 : tensor<1x150x64xf16>
    rock.yield %47 : tensor<1x150x64xf16>
  } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 64 : index, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>, params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>, params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x32x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16>, tensor<1x150x64xf16> -> tensor<1x32x32xf16>
  %20 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 48 + d3)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [2, 3] -> ["d2"] at [2]>] bounds = [1, 64, 1, 48] -> [1, 64, 48]> : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
  %21 = rock.transform %20 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 + 32)> by [<PassThrough ["d0", "d1", "d2b"] at [0, 1, 2] -> ["d0", "d1", "d2b"] at [0, 1, 2]>, <Slice{32, 48} ["d2i"] at [3] -> ["d2i"] at [3]>] bounds = [1, 64, 1, 16] -> [1, 64, 1, 48]> : tensor<1x64x1x48xf16> to tensor<1x64x1x16xf16>
  %22 = rock.transform %21 by <affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [2, 3]>] bounds = [1, 64, 16] -> [1, 64, 1, 16]> : tensor<1x64x1x16xf16> to tensor<1x64x16xf16>
  %result_2 = rock.gridwise_attention(%16, %2, %22, %4) preSoftmaxOps = {
  ^bb0(%arg5: tensor<1x150x64xf16>, %arg6: tensor<1x150x64xf16>):
    %47 = arith.addf %arg5, %arg6 : tensor<1x150x64xf16>
    rock.yield %47 : tensor<1x150x64xf16>
  } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 64 : index, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0>, params0 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>, params1 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x32x32xf16>, tensor<1x32x64xf16>, tensor<1x64x16xf16>, tensor<1x150x64xf16> -> tensor<1x32x16xf16>
  %23 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> (d1 * 48 + d2)> by [<Unmerge{150, 48} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 48] -> [7200]> : tensor<7200xf16> to tensor<1x150x48xf16>
  %24 = rock.transform %23 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm1MPad"] at [1] -> ["gemm1M"] at [1]>, <PassThrough ["gemm1N"] at [2] -> ["gemm1N"] at [2]>] bounds = [1, 160, 48] -> [1, 150, 48]> : tensor<1x150x48xf16> to tensor<1x160x48xf16>
  %25 = rock.transform %24 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d2, d3 * 48 + d4)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [3, 4] -> ["d2"] at [2]>] bounds = [1, 2, 80, 1, 48] -> [1, 160, 48]> : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
  %26 = rock.transform %25 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{0, 64, 0, 32} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 64, 1, 32] -> [1, 2, 80, 1, 48]> : tensor<1x2x80x1x48xf16> to tensor<1x2x64x1x32xf16>
  %27 = rock.transform %26 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d1 mod 64, 0, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 128, 32] -> [1, 2, 64, 1, 32]> : tensor<1x2x64x1x32xf16> to tensor<1x128x32xf16>
  %28 = rock.store %result to %27 alias %arg4 by set : tensor<1x128x32xf16> -> tensor<7200xf16> to tensor<1x128x32xf16> alias tensor<7200xf16>
  %29 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> (d1 * 48 + d2)> by [<Unmerge{150, 48} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 48] -> [7200]> : tensor<7200xf16> to tensor<1x150x48xf16>
  %30 = rock.transform %29 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm1MPad"] at [1] -> ["gemm1M"] at [1]>, <PassThrough ["gemm1N"] at [2] -> ["gemm1N"] at [2]>] bounds = [1, 160, 48] -> [1, 150, 48]> : tensor<1x150x48xf16> to tensor<1x160x48xf16>
  %31 = rock.transform %30 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d2, d3 * 48 + d4)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [3, 4] -> ["d2"] at [2]>] bounds = [1, 2, 80, 1, 48] -> [1, 160, 48]> : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
  %32 = rock.transform %31 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4 + 32)> by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{0, 64, 32, 48} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 64, 1, 16] -> [1, 2, 80, 1, 48]> : tensor<1x2x80x1x48xf16> to tensor<1x2x64x1x16xf16>
  %33 = rock.transform %32 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d1 mod 64, 0, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 128, 16] -> [1, 2, 64, 1, 16]> : tensor<1x2x64x1x16xf16> to tensor<1x128x16xf16>
  %34 = rock.store %result_0 to %33 alias %28 by set : tensor<1x128x16xf16> -> tensor<7200xf16> to tensor<1x128x16xf16> alias tensor<7200xf16>
  %35 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> (d1 * 48 + d2)> by [<Unmerge{150, 48} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 48] -> [7200]> : tensor<7200xf16> to tensor<1x150x48xf16>
  %36 = rock.transform %35 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm1MPad"] at [1] -> ["gemm1M"] at [1]>, <PassThrough ["gemm1N"] at [2] -> ["gemm1N"] at [2]>] bounds = [1, 160, 48] -> [1, 150, 48]> : tensor<1x150x48xf16> to tensor<1x160x48xf16>
  %37 = rock.transform %36 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d2, d3 * 48 + d4)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [3, 4] -> ["d2"] at [2]>] bounds = [1, 2, 80, 1, 48] -> [1, 160, 48]> : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
  %38 = rock.transform %37 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 + 64, d3, d4)> by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{64, 80, 0, 32} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 16, 1, 32] -> [1, 2, 80, 1, 48]> : tensor<1x2x80x1x48xf16> to tensor<1x2x16x1x32xf16>
  %39 = rock.transform %38 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 16, d1 mod 16, 0, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 32, 32] -> [1, 2, 16, 1, 32]> : tensor<1x2x16x1x32xf16> to tensor<1x32x32xf16>
  %40 = rock.store %result_1 to %39 alias %34 by set : tensor<1x32x32xf16> -> tensor<7200xf16> to tensor<1x32x32xf16> alias tensor<7200xf16>
  %41 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> (d1 * 48 + d2)> by [<Unmerge{150, 48} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 48] -> [7200]> : tensor<7200xf16> to tensor<1x150x48xf16>
  %42 = rock.transform %41 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm1MPad"] at [1] -> ["gemm1M"] at [1]>, <PassThrough ["gemm1N"] at [2] -> ["gemm1N"] at [2]>] bounds = [1, 160, 48] -> [1, 150, 48]> : tensor<1x150x48xf16> to tensor<1x160x48xf16>
  %43 = rock.transform %42 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d2, d3 * 48 + d4)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [3, 4] -> ["d2"] at [2]>] bounds = [1, 2, 80, 1, 48] -> [1, 160, 48]> : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
  %44 = rock.transform %43 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 + 64, d3, d4 + 32)> by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{64, 80, 32, 48} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 16, 1, 16] -> [1, 2, 80, 1, 48]> : tensor<1x2x80x1x48xf16> to tensor<1x2x16x1x16xf16>
  %45 = rock.transform %44 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 16, d1 mod 16, 0, d2)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 32, 16] -> [1, 2, 16, 1, 16]> : tensor<1x2x16x1x16xf16> to tensor<1x32x16xf16>
  %46 = rock.store %result_2 to %45 alias %40 by set : tensor<1x32x16xf16> -> tensor<7200xf16> to tensor<1x32x16xf16> alias tensor<7200xf16>
  return %46 : tensor<7200xf16>
}


// -----

#gemm_params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 16, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params2 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#gemm_params3 = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
#map = affine_map<(d0, d1, d2) -> (d1 * 32 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1 * 48 + d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 80 + d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d1 mod 64, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 48 + d3)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 + 32)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + 64, d3)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 16, d1 mod 16, d2)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 80 + d2, d3 * 48 + d4)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map14 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 64, d1 mod 64, 0, d2)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4 + 32)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 + 64, d3, d4)>
#map17 = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 16, d1 mod 16, 0, d2)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 + 64, d3, d4 + 32)>
#transform_map = #rock.transform_map<#map by [<Unmerge{150, 32} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 32] -> [4800]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm0MPad"] at [1] -> ["gemm0M"] at [1]>, <PassThrough ["gemm0K"] at [2] -> ["gemm0K"] at [2]>] bounds = [1, 160, 32] -> [1, 150, 32]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{32, 64} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]>
#transform_map3 = #rock.transform_map<#map3 by [<Unmerge{64, 48} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 48] -> [3072]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <PassThrough ["d2"] at [3] -> ["d2"] at [2]>] bounds = [1, 2, 80, 32] -> [1, 160, 32]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{0, 64} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 2, 64, 32] -> [1, 2, 80, 32]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 128, 32] -> [1, 2, 64, 32]>
#transform_map7 = #rock.transform_map<#map7 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [2, 3] -> ["d2"] at [2]>] bounds = [1, 64, 1, 48] -> [1, 64, 48]>
#transform_map8 = #rock.transform_map<#map5 by [<PassThrough ["d0", "d1", "d2b"] at [0, 1, 2] -> ["d0", "d1", "d2b"] at [0, 1, 2]>, <Slice{0, 32} ["d2i"] at [3] -> ["d2i"] at [3]>] bounds = [1, 64, 1, 32] -> [1, 64, 1, 48]>
#transform_map9 = #rock.transform_map<#map8 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [2, 3]>] bounds = [1, 64, 32] -> [1, 64, 1, 32]>
#transform_map10 = #rock.transform_map<#map9 by [<PassThrough ["d0", "d1", "d2b"] at [0, 1, 2] -> ["d0", "d1", "d2b"] at [0, 1, 2]>, <Slice{32, 48} ["d2i"] at [3] -> ["d2i"] at [3]>] bounds = [1, 64, 1, 16] -> [1, 64, 1, 48]>
#transform_map11 = #rock.transform_map<#map8 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d1"] at [1]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [2, 3]>] bounds = [1, 64, 16] -> [1, 64, 1, 16]>
#transform_map12 = #rock.transform_map<#map10 by [<PassThrough ["d0", "d1b", "d2"] at [0, 1, 3] -> ["d0", "d1b", "d2"] at [0, 1, 3]>, <Slice{64, 80} ["d1i"] at [2] -> ["d1i"] at [2]>] bounds = [1, 2, 16, 32] -> [1, 2, 80, 32]>
#transform_map13 = #rock.transform_map<#map11 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <PassThrough ["d2"] at [2] -> ["d2"] at [3]>] bounds = [1, 32, 32] -> [1, 2, 16, 32]>
#transform_map14 = #rock.transform_map<#map3 by [<Unmerge{150, 48} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 150, 48] -> [7200]>
#transform_map15 = #rock.transform_map<#map1 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 10} ["gemm1MPad"] at [1] -> ["gemm1M"] at [1]>, <PassThrough ["gemm1N"] at [2] -> ["gemm1N"] at [2]>] bounds = [1, 160, 48] -> [1, 150, 48]>
#transform_map16 = #rock.transform_map<#map12 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Unmerge{2, 80} ["d1b", "d1i"] at [1, 2] -> ["d1"] at [1]>, <Unmerge{1, 48} ["d2b", "d2i"] at [3, 4] -> ["d2"] at [2]>] bounds = [1, 2, 80, 1, 48] -> [1, 160, 48]>
#transform_map17 = #rock.transform_map<#map13 by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{0, 64, 0, 32} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 64, 1, 32] -> [1, 2, 80, 1, 48]>
#transform_map18 = #rock.transform_map<#map14 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 128, 32] -> [1, 2, 64, 1, 32]>
#transform_map19 = #rock.transform_map<#map15 by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{0, 64, 32, 48} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 64, 1, 16] -> [1, 2, 80, 1, 48]>
#transform_map20 = #rock.transform_map<#map14 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 64} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 128, 16] -> [1, 2, 64, 1, 16]>
#transform_map21 = #rock.transform_map<#map16 by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{64, 80, 0, 32} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 16, 1, 32] -> [1, 2, 80, 1, 48]>
#transform_map22 = #rock.transform_map<#map17 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 32} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 32, 32] -> [1, 2, 16, 1, 32]>
#transform_map23 = #rock.transform_map<#map18 by [<PassThrough ["d0", "d1b", "d2b"] at [0, 1, 3] -> ["d0", "d1b", "d2b"] at [0, 1, 3]>, <Slice{64, 80, 32, 48} ["d1i", "d2i"] at [2, 4] -> ["d1i", "d2i"] at [2, 4]>] bounds = [1, 2, 16, 1, 16] -> [1, 2, 80, 1, 48]>
#transform_map24 = #rock.transform_map<#map17 by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <Merge{2, 16} ["d1"] at [1] -> ["d1b", "d1i"] at [1, 2]>, <Merge{1, 16} ["d2"] at [2] -> ["d2b", "d2i"] at [3, 4]>] bounds = [1, 32, 16] -> [1, 2, 16, 1, 16]>
// Causal masking on an M-decomposed attention produced by
// rock-decompose-nonpow2-tiles: seqLenQ = 150 (padded to 160) with
// gemm0MOrigPerBlock = 80 splits M into {64, 16} sub-tiles at slice offsets
// {0, 64}; head_dim_v = 48 splits gemm1 N into {32, 16}, giving four sub-ops.
// getNLoopInfo() derives the causal K-loop bound from the ORIGINAL query rows:
// the last row of block m_block is
//   m_block*gemm0MOrigPerBlock + gemm0MSliceOffset + (mPerBlock - 1).
// So every sub-op strides m_block by 80 (the pre-split tile, NOT the sliced
// mPerBlock), and the per-segment last-row constant is
// gemm0MSliceOffset + mPerBlock - 1: 63 (= 0 + 64 - 1) for the offset-0
// M-segment and 79 (= 64 + 16 - 1) for the offset-64 M-segment. Using the
// sliced stride/offset here would under-count live seqLenK blocks and break
// causal masking. Matched pre-canonicalize (see the second RUN line) so the
// stride/offset arithmetic isn't folded together with nPerBlock.
//
// The last query row is then bounded by gemm0N - 1 = 63, because the original
// query sequence (2 * 80 = 160 padded rows) is longer than seqLenK = 64. That
// test must also be made against the ORIGINAL M: the offset-64 sub-ops carry a
// compacted M of only 2 * 16 = 32 rows, which is shorter than seqLenK, so
// comparing the sliced M would wrongly skip the bound.
// CAUSAL-LABEL: func{{.*}}@gridwise_attn_decomposed_grid_causal
// The two offset-0 sub-ops (M-tile 64): stride 80, last row 0 + 64 - 1 = 63.
// CAUSAL: arith.muli %{{.+}}, %c80_i32
// CAUSAL: arith.addi %{{.+}}, %c63_i32
// CAUSAL: arith.minui %{{.+}}, %c63_i32
// CAUSAL: arith.muli %{{.+}}, %c80_i32
// CAUSAL: arith.addi %{{.+}}, %c63_i32
// CAUSAL: arith.minui %{{.+}}, %c63_i32
// The two offset-64 sub-ops (M-tile 16): stride 80, last row 64 + 16 - 1 = 79.
// CAUSAL: arith.muli %{{.+}}, %c80_i32
// CAUSAL: arith.addi %{{.+}}, %c79_i32
// CAUSAL: arith.minui %{{.+}}, %c63_i32
// CAUSAL: arith.muli %{{.+}}, %c80_i32
// CAUSAL: arith.addi %{{.+}}, %c79_i32
// CAUSAL: arith.minui %{{.+}}, %c63_i32
func.func @gridwise_attn_decomposed_grid_causal(%arg0: tensor<4800xf16>, %arg1: tensor<2048xf16>, %arg2: tensor<3072xf16>, %arg3: tensor<7200xf16>) -> tensor<7200xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.block_size = 64 : i32, rock.grid_size = 2 : i32, rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<4800xf16> to tensor<1x150x32xf16>
    %1 = rock.transform %0 by #transform_map1 : tensor<1x150x32xf16> to tensor<1x160x32xf16>
    %2 = rock.transform %arg1 by #transform_map2 : tensor<2048xf16> to tensor<1x32x64xf16>
    %3 = rock.transform %arg2 by #transform_map3 : tensor<3072xf16> to tensor<1x64x48xf16>
    %4 = rock.transform %1 by #transform_map4 : tensor<1x160x32xf16> to tensor<1x2x80x32xf16>
    %5 = rock.transform %4 by #transform_map5 : tensor<1x2x80x32xf16> to tensor<1x2x64x32xf16>
    %6 = rock.transform %5 by #transform_map6 : tensor<1x2x64x32xf16> to tensor<1x128x32xf16>
    %7 = rock.transform %3 by #transform_map7 : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
    %8 = rock.transform %7 by #transform_map8 : tensor<1x64x1x48xf16> to tensor<1x64x1x32xf16>
    %9 = rock.transform %8 by #transform_map9 : tensor<1x64x1x32xf16> to tensor<1x64x32xf16>
    %result = rock.gridwise_attention(%6, %2, %9) preSoftmaxOps = {
    } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 0 : index, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #gemm_params, params1 = #gemm_params1, causal, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x128x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16> -> tensor<1x128x32xf16>
    %10 = rock.transform %3 by #transform_map7 : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
    %11 = rock.transform %10 by #transform_map10 : tensor<1x64x1x48xf16> to tensor<1x64x1x16xf16>
    %12 = rock.transform %11 by #transform_map11 : tensor<1x64x1x16xf16> to tensor<1x64x16xf16>
    %result_0 = rock.gridwise_attention(%6, %2, %12) preSoftmaxOps = {
    } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 0 : index, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #gemm_params, params1 = #gemm_params1, causal, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x128x32xf16>, tensor<1x32x64xf16>, tensor<1x64x16xf16> -> tensor<1x128x16xf16>
    %13 = rock.transform %1 by #transform_map4 : tensor<1x160x32xf16> to tensor<1x2x80x32xf16>
    %14 = rock.transform %13 by #transform_map12 : tensor<1x2x80x32xf16> to tensor<1x2x16x32xf16>
    %15 = rock.transform %14 by #transform_map13 : tensor<1x2x16x32xf16> to tensor<1x32x32xf16>
    %16 = rock.transform %3 by #transform_map7 : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
    %17 = rock.transform %16 by #transform_map8 : tensor<1x64x1x48xf16> to tensor<1x64x1x32xf16>
    %18 = rock.transform %17 by #transform_map9 : tensor<1x64x1x32xf16> to tensor<1x64x32xf16>
    %result_1 = rock.gridwise_attention(%15, %2, %18) preSoftmaxOps = {
    } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 64 : index, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #gemm_params2, params1 = #gemm_params3, causal, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x32x32xf16>, tensor<1x32x64xf16>, tensor<1x64x32xf16> -> tensor<1x32x32xf16>
    %19 = rock.transform %3 by #transform_map7 : tensor<1x64x48xf16> to tensor<1x64x1x48xf16>
    %20 = rock.transform %19 by #transform_map10 : tensor<1x64x1x48xf16> to tensor<1x64x1x16xf16>
    %21 = rock.transform %20 by #transform_map11 : tensor<1x64x1x16xf16> to tensor<1x64x16xf16>
    %result_2 = rock.gridwise_attention(%15, %2, %21) preSoftmaxOps = {
    } {gemm0MOrigPerBlock = 80 : index, gemm0MSliceOffset = 64 : index, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>, params0 = #gemm_params2, params1 = #gemm_params3, causal, prePadG0M = 150 : index, softmaxType = f32, splitKV = 1 : i32} : tensor<1x32x32xf16>, tensor<1x32x64xf16>, tensor<1x64x16xf16> -> tensor<1x32x16xf16>
    %22 = rock.transform %arg3 by #transform_map14 : tensor<7200xf16> to tensor<1x150x48xf16>
    %23 = rock.transform %22 by #transform_map15 : tensor<1x150x48xf16> to tensor<1x160x48xf16>
    %24 = rock.transform %23 by #transform_map16 : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
    %25 = rock.transform %24 by #transform_map17 : tensor<1x2x80x1x48xf16> to tensor<1x2x64x1x32xf16>
    %26 = rock.transform %25 by #transform_map18 : tensor<1x2x64x1x32xf16> to tensor<1x128x32xf16>
    %27 = rock.store %result to %26 alias %arg3 by set : tensor<1x128x32xf16> -> tensor<7200xf16> to tensor<1x128x32xf16> alias tensor<7200xf16>
    %28 = rock.transform %arg3 by #transform_map14 : tensor<7200xf16> to tensor<1x150x48xf16>
    %29 = rock.transform %28 by #transform_map15 : tensor<1x150x48xf16> to tensor<1x160x48xf16>
    %30 = rock.transform %29 by #transform_map16 : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
    %31 = rock.transform %30 by #transform_map19 : tensor<1x2x80x1x48xf16> to tensor<1x2x64x1x16xf16>
    %32 = rock.transform %31 by #transform_map20 : tensor<1x2x64x1x16xf16> to tensor<1x128x16xf16>
    %33 = rock.store %result_0 to %32 alias %27 by set : tensor<1x128x16xf16> -> tensor<7200xf16> to tensor<1x128x16xf16> alias tensor<7200xf16>
    %34 = rock.transform %arg3 by #transform_map14 : tensor<7200xf16> to tensor<1x150x48xf16>
    %35 = rock.transform %34 by #transform_map15 : tensor<1x150x48xf16> to tensor<1x160x48xf16>
    %36 = rock.transform %35 by #transform_map16 : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
    %37 = rock.transform %36 by #transform_map21 : tensor<1x2x80x1x48xf16> to tensor<1x2x16x1x32xf16>
    %38 = rock.transform %37 by #transform_map22 : tensor<1x2x16x1x32xf16> to tensor<1x32x32xf16>
    %39 = rock.store %result_1 to %38 alias %33 by set : tensor<1x32x32xf16> -> tensor<7200xf16> to tensor<1x32x32xf16> alias tensor<7200xf16>
    %40 = rock.transform %arg3 by #transform_map14 : tensor<7200xf16> to tensor<1x150x48xf16>
    %41 = rock.transform %40 by #transform_map15 : tensor<1x150x48xf16> to tensor<1x160x48xf16>
    %42 = rock.transform %41 by #transform_map16 : tensor<1x160x48xf16> to tensor<1x2x80x1x48xf16>
    %43 = rock.transform %42 by #transform_map23 : tensor<1x2x80x1x48xf16> to tensor<1x2x16x1x16xf16>
    %44 = rock.transform %43 by #transform_map24 : tensor<1x2x16x1x16xf16> to tensor<1x32x16xf16>
    %45 = rock.store %result_2 to %44 alias %39 by set : tensor<1x32x16xf16> -> tensor<7200xf16> to tensor<1x32x16xf16> alias tensor<7200xf16>
    return %45 : tensor<7200xf16>
  }
