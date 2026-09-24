// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// rock-decompose-nonpow2-k splits a non-power-of-two K tile into a chain of
// power-of-two segment gemms. When rock-add-triton-metadata runs, every segment
// must receive the same rock.o_transposed as the final store requires. 
// Otherwise the segments' dots end up in different accelerator layouts and the 
// accumulator is converted between them on every K iteration, making performance
// much worse.
//
// RUN: rocmlir-opt -split-input-file -rock-decompose-nonpow2-k -rock-add-triton-metadata %s | FileCheck %s

// A K tile of 48 decomposes into {32, 16}; the output store is column-major.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 48 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 48 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 48, 64] -> [1, 96, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 48] -> [1, 64, 96]>
#tmapT = #rock.transform_map<affine_map<(m, n) -> (n * 64 + m)> by [<Unmerge{64, 64} ["n", "m"] at [1, 0] -> ["raw"] at [0]>] bounds = [64, 64] -> [4096]>

// CHECK-LABEL: @two_segments_col_major
// CHECK-COUNT-2: rock.blockwise_gemm{{.*}}rock.o_transposed = #rock.o_transposed<true>
// CHECK-NOT: rock.blockwise_gemm
func.func @two_segments_col_major(%arg0: tensor<1x64x96xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<64x64xf32>, %dest_raw: tensor<4096xf32>) attributes {rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x96x64xf16> -> tensor<48x64xf16>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x96xf16> -> tensor<64x48xf16>
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x48xf16>, tensor<48x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %dest = rock.transform %dest_raw by #tmapT : tensor<4096xf32> to tensor<64x64xf32>
  %r = rock.blockwise_store %2 -> %dest by set : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<4096xf32>
  return
}

// -----

// A K tile of 112 decomposes into {64, 32, 16}; the output store is row-major.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 112 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 112 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 112} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 112, 64] -> [1, 224, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 112} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 112] -> [1, 64, 224]>

// CHECK-LABEL: @three_segments_row_major
// CHECK-COUNT-3: rock.blockwise_gemm{{.*}}rock.o_transposed = #rock.o_transposed<false>
// CHECK-NOT: rock.blockwise_gemm
func.func @three_segments_row_major(%arg0: tensor<1x64x224xf16>, %arg1: tensor<1x224x64xf16>, %arg2: tensor<64x64xf32>, %dest: tensor<64x64xf32>) attributes {rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x224x64xf16> -> tensor<112x64xf16>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x224xf16> -> tensor<64x112xf16>
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x112xf16>, tensor<112x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  %r = rock.blockwise_store %2 -> %dest by set : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<64x64xf32>
  return
}
