// Error tests for the rock-decompose-nonpow2-k pass.
//
// The pass runs at the blockwise layer, so each case is a rock.blockwise_gemm
// with a non-power-of-two K tile whose operands hit one of the pass's
// diagnostics. These operand shapes are not reachable from the gridwise layer,
// which is why they are checked here rather than in the pipeline tests.

// RUN: rocmlir-opt -split-input-file -rock-decompose-nonpow2-k -canonicalize -verify-diagnostics %s

// Only an operand loaded from global memory can be re-sliced per segment.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 48 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 48, 64] -> [1, 96, 64]>

func.func @operand_not_loaded(%arg0: tensor<64x48xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x96x64xf16> -> tensor<48x64xf16>
  // expected-error @+1 {{non-power-of-two K tile requires both operands to be loaded by rock.load_marker}}
  %1 = rock.blockwise_gemm(%arg0, %0, %arg2) : tensor<64x48xf16>, tensor<48x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// -----

// A marker that verifies but was not built by rock::loadTile: its source is 2D,
// so there is no [G, D, K] operand to slice a K segment out of.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 48 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2 * 64 + d4, d0 * 48 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 48, 64] -> [1, 96, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [1]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [0]>, <AddDim{1} ["g_block"] at [1] -> [] at []>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 48] -> [64, 96]>

func.func @operand_tiling_unrecoverable(%arg0: tensor<64x96xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x96x64xf16> -> tensor<48x64xf16>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<64x96xf16> -> tensor<64x48xf16>
  // expected-error @+1 {{could not recover the operand tiling of a non-power-of-two K tile}}
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x48xf16>, tensor<48x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 32 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 48 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 32} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 32, 64] -> [1, 64, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 48] -> [1, 64, 96]>

func.func @operands_disagree_on_k_tile(%arg0: tensor<1x64x96xf16>, %arg1: tensor<1x64x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x64xf16> -> tensor<32x64xf16>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x96xf16> -> tensor<64x48xf16>
  // expected-error @+1 {{operands of a non-power-of-two K tile must share their K tile, but got 48 and 32}}
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x48xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}

// -----

// A marker whose view has the shape rock::loadTile produces, but which tiles a
// dimension the blockwise_gemm does not contract over: its loop runs over "z",
// so slicing its tiles would not slice K.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 48 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 48 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 48, 64] -> [1, 96, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["z_loop", "z_iter"] at [0, 5] -> ["z"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 48] -> [1, 64, 96]>

func.func @operand_tiling_not_over_k(%arg0: tensor<1x64x96xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x96x64xf16> -> tensor<48x64xf16>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x96xf16> -> tensor<64x48xf16>
  // expected-error @+1 {{could not recover the operand tiling of a non-power-of-two K tile}}
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x48xf16>, tensor<48x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}
