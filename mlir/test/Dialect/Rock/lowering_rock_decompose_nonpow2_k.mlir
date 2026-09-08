// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: rocmlir-opt -split-input-file -rock-decompose-nonpow2-k -canonicalize %s | FileCheck %s

// A K tile of 48 decomposes into {32, 16}.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 48 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 48 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 48, 64] -> [1, 96, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 48] -> [1, 64, 96]>

// CHECK-LABEL: @decompose_two_segments
// CHECK-SAME:  %[[ACC:[^:]+]]: tensor<64x64xf32>
func.func @decompose_two_segments(%arg0: tensor<1x64x96xf16>, %arg1: tensor<1x96x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
  // CHECK: %[[BT32:.+]] = rock.load_marker {{.*}} : tensor<1x64x64xf16> -> tensor<32x64xf16>
  // CHECK: %[[AT32:.+]] = rock.load_marker {{.*}} : tensor<1x64x64xf16> -> tensor<64x32xf16>
  // CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%[[AT32]], %[[BT32]], %[[ACC]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: %[[BT16:.+]] = rock.load_marker {{.*}} : tensor<1x32x64xf16> -> tensor<16x64xf16>
  // CHECK: %[[AT16:.+]] = rock.load_marker {{.*}} : tensor<1x64x32xf16> -> tensor<64x16xf16>
  // CHECK: %[[ACC2:.+]] = rock.blockwise_gemm(%[[AT16]], %[[BT16]], %[[ACC1]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK-NOT: rock.blockwise_gemm
  // CHECK: return %[[ACC2]]
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x96x64xf16> -> tensor<48x64xf16>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x96xf16> -> tensor<64x48xf16>
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x48xf16>, tensor<48x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}

// -----

// A K tile of 112 has three set bits and decomposes into {64, 32, 16}.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 112 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 112 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 112} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 112, 64] -> [1, 224, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 112} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 112] -> [1, 64, 224]>

// CHECK-LABEL: @decompose_three_segments
// CHECK-SAME:  %[[ACC:[^:]+]]: tensor<64x64xf32>
func.func @decompose_three_segments(%arg0: tensor<1x64x224xf16>, %arg1: tensor<1x224x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
  // CHECK: %[[BT64:.+]] = rock.load_marker {{.*}} : tensor<1x128x64xf16> -> tensor<64x64xf16>
  // CHECK: %[[AT64:.+]] = rock.load_marker {{.*}} : tensor<1x64x128xf16> -> tensor<64x64xf16>
  // CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%[[AT64]], %[[BT64]], %[[ACC]]) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: %[[BT32:.+]] = rock.load_marker {{.*}} : tensor<1x64x64xf16> -> tensor<32x64xf16>
  // CHECK: %[[AT32:.+]] = rock.load_marker {{.*}} : tensor<1x64x64xf16> -> tensor<64x32xf16>
  // CHECK: %[[ACC2:.+]] = rock.blockwise_gemm(%[[AT32]], %[[BT32]], %[[ACC1]]) : tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: %[[BT16:.+]] = rock.load_marker {{.*}} : tensor<1x32x64xf16> -> tensor<16x64xf16>
  // CHECK: %[[AT16:.+]] = rock.load_marker {{.*}} : tensor<1x64x32xf16> -> tensor<64x16xf16>
  // CHECK: %[[ACC3:.+]] = rock.blockwise_gemm(%[[AT16]], %[[BT16]], %[[ACC2]]) : tensor<64x16xf16>, tensor<16x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK-NOT: rock.blockwise_gemm
  // CHECK: return %[[ACC3]]
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x224x64xf16> -> tensor<112x64xf16>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x224xf16> -> tensor<64x112xf16>
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x112xf16>, tensor<112x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}

// -----

// The attributes describing how the operands were packed are not derivable from
// the segment shapes, so every segment gemm must inherit them from the original.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 48 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 48 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 48, 64] -> [1, 96, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 48} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 48] -> [1, 64, 96]>

// CHECK-LABEL: @packing_attributes_propagate
// CHECK-COUNT-2: rock.blockwise_gemm{{.*}}matrixAKPack = true, matrixAOrigElemType = f4E2M1FN, matrixBKPack = false, matrixBOrigElemType = f4E2M1FN
// CHECK-NOT: rock.blockwise_gemm
func.func @packing_attributes_propagate(%arg0: tensor<1x64x96xi8>, %arg1: tensor<1x96x64xi8>, %arg2: tensor<64x64xi32>) -> tensor<64x64xi32> attributes {rock.kernel} {
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x96x64xi8> -> tensor<48x64xi8>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x96xi8> -> tensor<64x48xi8>
  %2 = rock.blockwise_gemm(%1, %0, %arg2) {matrixAKPack = true, matrixAOrigElemType = f4E2M1FN, matrixBKPack = false, matrixBOrigElemType = f4E2M1FN} : tensor<64x48xi8>, tensor<48x64xi8>, tensor<64x64xi32> -> tensor<64x64xi32>
  return %2 : tensor<64x64xi32>
}

// -----

// Integer operands are accepted as long as no segment is narrower than 4, since
// a narrower one has no integer dot instruction. A K tile of 20 decomposes into
// {16, 4}, which sits exactly on that boundary.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 20 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 20 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 20} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 20, 64] -> [1, 40, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 20} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 20] -> [1, 64, 40]>

// CHECK-LABEL: @decompose_i8_narrowest_segment
// CHECK-SAME:  %[[ACC:[^:]+]]: tensor<64x64xi32>
func.func @decompose_i8_narrowest_segment(%arg0: tensor<1x64x40xi8>, %arg1: tensor<1x40x64xi8>, %arg2: tensor<64x64xi32>) -> tensor<64x64xi32> attributes {rock.kernel} {
  // CHECK: %[[BT16:.+]] = rock.load_marker {{.*}} : tensor<1x32x64xi8> -> tensor<16x64xi8>
  // CHECK: %[[AT16:.+]] = rock.load_marker {{.*}} : tensor<1x64x32xi8> -> tensor<64x16xi8>
  // CHECK: %[[ACC1:.+]] = rock.blockwise_gemm(%[[AT16]], %[[BT16]], %[[ACC]]) : tensor<64x16xi8>, tensor<16x64xi8>, tensor<64x64xi32> -> tensor<64x64xi32>
  // CHECK: %[[BT4:.+]] = rock.load_marker {{.*}} : tensor<1x8x64xi8> -> tensor<4x64xi8>
  // CHECK: %[[AT4:.+]] = rock.load_marker {{.*}} : tensor<1x64x8xi8> -> tensor<64x4xi8>
  // CHECK: %[[ACC2:.+]] = rock.blockwise_gemm(%[[AT4]], %[[BT4]], %[[ACC1]]) : tensor<64x4xi8>, tensor<4x64xi8>, tensor<64x64xi32> -> tensor<64x64xi32>
  // CHECK-NOT: rock.blockwise_gemm
  // CHECK: return %[[ACC2]]
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x40x64xi8> -> tensor<20x64xi8>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x40xi8> -> tensor<64x20xi8>
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x20xi8>, tensor<20x64xi8>, tensor<64x64xi32> -> tensor<64x64xi32>
  return %2 : tensor<64x64xi32>
}

// -----

// We bail on a power-of-two K tile.

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 64, 64] -> [1, 128, 64]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 64] -> [1, 64, 128]>

// CHECK-LABEL: @pow2_tile_untouched
// CHECK-NOT: rock.transform
func.func @pow2_tile_untouched(%arg0: tensor<1x64x128xf16>, %arg1: tensor<1x128x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
  // CHECK: rock.blockwise_gemm(%{{.*}}, %{{.*}}, %{{.*}}) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK-NOT: rock.blockwise_gemm
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.load_marker %arg1 views [#transform_map][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x128x64xf16> -> tensor<64x64xf16>
  %1 = rock.load_marker %arg0 views [#transform_map1][%c0_i32, %c0_i32, %c0_i32, %c0_i32] {cacheModifier = #rock<CacheModifier none>} : tensor<1x64x128xf16> -> tensor<64x64xf16>
  %2 = rock.blockwise_gemm(%1, %0, %arg2) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}
