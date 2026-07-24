// Unit tests for rock-regularize-input pass.
// Tests that load_markers whose source contains fusion ops are distributed
// so each leaf (block arg or constant) gets its own load_marker, and fusion
// ops are cloned to operate on tile types.

// RUN: rocmlir-opt -rock-regularize-input -canonicalize -mlir-print-local-scope %s | FileCheck %s
// RUN: rocmlir-opt -rock-regularize-input -rock-lower-loads -mlir-print-local-scope %s | FileCheck %s --check-prefix=LOWER

#tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d3, d2 * 16 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 16} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{1, 16} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 1, 16, 16] -> [1, 16, 16]>

#tmap_two_n_blocks = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d3, d2 * 16 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 16} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{2, 16} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 2, 16, 16] -> [1, 16, 32]>

#tmap_broadcast_n = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <PassThrough ["m"] at [1] -> ["m"] at [0]>, <AddDim{32} ["n"] at [2] -> [] at []>] bounds = [1, 16, 32] -> [16]>

#tmap_broadcast_m = #rock.transform_map<affine_map<(d0, d1, d2) -> (d2)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{16} ["m"] at [1] -> [] at []>, <PassThrough ["n"] at [2] -> ["n"] at [0]>] bounds = [1, 16, 32] -> [32]>

#tmap_broadcast_n_14 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <PassThrough ["m"] at [1] -> ["m"] at [0]>, <AddDim{14} ["n"] at [2] -> [] at []>] bounds = [1, 16, 14] -> [16]>

#tmap_pad_n_14_to_16 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2 - 1)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Pad{1, 1} ["n_pad"] at [2] -> ["n"] at [2]>] bounds = [1, 16, 16] -> [1, 16, 14]>

#tmap_broadcast_scalar = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0)> by [<PassThrough ["scalar"] at [0] -> ["scalar"] at [0]>, <AddDim{16} ["m"] at [1] -> [] at []>, <AddDim{16} ["n"] at [2] -> [] at []>] bounds = [1, 16, 16] -> [1]>

#tmap_6d_to_4d = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 floordiv 4, d2 mod 4, d3 floordiv 4, d3 mod 4)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>, <PassThrough ["b"] at [1] -> ["b"] at [1]>, <Merge{4, 4} ["m"] at [2] -> ["m0", "m1"] at [2, 3]>, <Merge{4, 4} ["n"] at [3] -> ["n0", "n1"] at [4, 5]>] bounds = [1, 1, 16, 16] -> [1, 1, 4, 4, 4, 4]>

#tmap_5d_to_4d = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d2 floordiv 4, d2 mod 4, d3 floordiv 4, d3 mod 4)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>, <AddDim{1} ["b"] at [1] -> [] at []>, <Merge{4, 4} ["m"] at [2] -> ["m0", "m1"] at [1, 2]>, <Merge{4, 4} ["n"] at [3] -> ["n0", "n1"] at [3, 4]>] bounds = [1, 1, 16, 16] -> [1, 4, 4, 4, 4]>

#tmap_2d_to_4d = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d2, d3)> by [<AddDim{1} ["a"] at [0] -> [] at []>, <AddDim{1} ["b"] at [1] -> [] at []>, <PassThrough ["m"] at [2] -> ["m"] at [0]>, <PassThrough ["n"] at [3] -> ["n"] at [1]>] bounds = [1, 1, 16, 16] -> [16, 16]>

#tmap_broadcast_1d = #rock.transform_map<affine_map<(d0, d1, d2) -> (d2)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{16} ["m"] at [1] -> [] at []>, <PassThrough ["n"] at [2] -> ["n"] at [0]>] bounds = [1, 16, 16] -> [16]>

#tmap_transpose_3d = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["n"] at [2]>, <PassThrough ["n"] at [2] -> ["m"] at [1]>] bounds = [1, 16, 16] -> [1, 16, 16]>

#tmap_4d_to_3d = #rock.transform_map<affine_map<(d0, d1, d2) -> (0, 0, d1, d2)> by [<Merge{1, 1} ["g"] at [0] -> ["g0", "g1"] at [0, 1]>, <PassThrough ["m"] at [1] -> ["m"] at [2]>, <PassThrough ["n"] at [2] -> ["n"] at [3]>] bounds = [1, 16, 16] -> [1, 1, 16, 16]>

module {

  // ============================================================
  // Direct block arg source: load_marker already has a pure source.
  // Pass recreates but the structure is the same.
  // ============================================================

  // CHECK-LABEL: func.func @test_direct_blockarg
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_direct_blockarg(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %lm = rock.load_marker %bias views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Fusion in load_marker source: addf(arg1, arg2).
  // Distributed into two load_markers + tile-level addf.
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_addf
  // CHECK: %[[SM:.*]] = rock.store_marker
  // Two load_markers on individual block args
  // CHECK: %[[LM1:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[LM2:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // Tile-level addf replaces the original load_marker
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM1]], %[[LM2]] : tensor<16x16xf16>
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_ADD]] : tensor<16x16xf16> -> tensor<1x16x16xf16>
  // Output fusion uses the untile result
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]] : tensor<1x16x16xf16>
  // CHECK: rock.store %[[FUSED]]
  func.func @test_fusion_addf(%tile: tensor<16x16xf16>, %t1: tensor<1x16x16xf16>, %t2: tensor<1x16x16xf16>, %dest: tensor<1x16x16xf16>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf16> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf16> -> tensor<1x16x16xf16>
    %sum = arith.addf %t1, %t2 : tensor<1x16x16xf16>
    %lm = rock.load_marker %sum views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %ut = rock.untile %lm : tensor<16x16xf16> -> tensor<1x16x16xf16>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf16>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf16> -> tensor<1x16x16xf16> to tensor<1x16x16xf16>
    return %r : tensor<1x16x16xf16>
  }

  // ============================================================
  // Splat constant source: load_marker replaced by tile-shaped
  // constant (no load_marker needed for constants).
  // ============================================================

  // CHECK-LABEL: func.func @test_splat_constant
  // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[UT:.*]] = rock.untile %[[CST]] : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK: %[[FUSED:.*]] = arith.subf %[[SM]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: rock.store %[[FUSED]]
  func.func @test_splat_constant(%tile: tensor<16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<1x16x16xf32>
    %lm = rock.load_marker %cst views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %sub = arith.subf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %sub to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Transform chain source: load_marker(transform(blockarg)).
  // Transform is accumulated and applied at the block arg leaf.
  // ============================================================

  // CHECK-LABEL: func.func @test_transform_chain
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[T:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[LM:.*]] = rock.load_marker %[[T]] views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_transform_chain(%tile: tensor<16x16xf32>, %bias_raw: tensor<16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %bias_3d = rock.transform %bias_raw by <affine_map<(d0, d1, d2) -> (d2)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{16} ["m"] at [1] -> [] at []>, <PassThrough ["n"] at [2] -> ["n"] at [0]>] bounds = [1, 16, 16] -> [16]> : tensor<16xf32> to tensor<1x16x16xf32>
    %lm = rock.load_marker %bias_3d views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Mixed fusion: addf(blockarg, constant) in load_marker source.
  // Block arg gets load_marker, constant becomes tile-shaped,
  // addf is cloned at tile level.
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_with_constant
  // CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<16x16xf32>
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM]], %[[CST]] : tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_ADD]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_fusion_with_constant(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %cst = arith.constant dense<2.000000e+00> : tensor<1x16x16xf32>
    %sum = arith.addf %bias, %cst : tensor<1x16x16xf32>
    %lm = rock.load_marker %sum views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Fusion chain: mulf(addf(a, b), c) in load_marker source.
  // Three load_markers + tile-level addf and mulf.
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_chain
  // CHECK: %[[SM:.*]] = rock.store_marker
  // Three load_markers on individual block args
  // CHECK: %[[LM_A:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[LM_B:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM_A]], %[[LM_B]] : tensor<16x16xf32>
  // CHECK: %[[LM_C:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[TILE_MUL:.*]] = arith.mulf %[[TILE_ADD]], %[[LM_C]] : tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_MUL]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_fusion_chain(%tile: tensor<16x16xf32>, %a: tensor<1x16x16xf32>, %b: tensor<1x16x16xf32>, %c: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %a, %b : tensor<1x16x16xf32>
    %mul = arith.mulf %add, %c : tensor<1x16x16xf32>
    %lm = rock.load_marker %mul views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %fused = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %fused to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Complex: three inputs with different transforms
  // (6D→4D, 5D→4D, 2D→4D), fused at 4D (addf then mulf),
  // then transformed 4D→3D before load_marker.
  // Each leaf gets its own transform chain + load_marker.
  // ============================================================

  // CHECK-LABEL: func.func @test_complex_transform_fusion
  // CHECK: %[[SM:.*]] = rock.store_marker
  // arg1: 6D → 4D → 3D → load_marker
  // CHECK: %[[T1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x1x4x4x4x4xf32> to tensor<1x1x16x16xf32>
  // CHECK: %[[T1_3D:.*]] = rock.transform %[[T1]] by
  // CHECK-SAME: tensor<1x1x16x16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[LM1:.*]] = rock.load_marker %[[T1_3D]] views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // arg2: 5D → 4D → 3D → load_marker
  // CHECK: %[[T2:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x4x4x4x4xf32> to tensor<1x1x16x16xf32>
  // CHECK: %[[T2_3D:.*]] = rock.transform %[[T2]] by
  // CHECK-SAME: tensor<1x1x16x16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[LM2:.*]] = rock.load_marker %[[T2_3D]] views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // Tile-level addf
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM1]], %[[LM2]] : tensor<16x16xf32>
  // arg3: 2D → 4D → 3D → load_marker
  // CHECK: %[[T3:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16x16xf32> to tensor<1x1x16x16xf32>
  // CHECK: %[[T3_3D:.*]] = rock.transform %[[T3]] by
  // CHECK-SAME: tensor<1x1x16x16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[LM3:.*]] = rock.load_marker %[[T3_3D]] views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // Tile-level mulf
  // CHECK: %[[TILE_MUL:.*]] = arith.mulf %[[TILE_ADD]], %[[LM3]] : tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_MUL]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_complex_transform_fusion(
      %tile: tensor<16x16xf32>,
      %arg1: tensor<1x1x4x4x4x4xf32>,
      %arg2: tensor<1x4x4x4x4xf32>,
      %arg3: tensor<16x16xf32>,
      %dest: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %t1 = rock.transform %arg1 by #tmap_6d_to_4d : tensor<1x1x4x4x4x4xf32> to tensor<1x1x16x16xf32>
    %t2 = rock.transform %arg2 by #tmap_5d_to_4d : tensor<1x4x4x4x4xf32> to tensor<1x1x16x16xf32>
    %sum_4d = arith.addf %t1, %t2 : tensor<1x1x16x16xf32>
    %t3 = rock.transform %arg3 by #tmap_2d_to_4d : tensor<16x16xf32> to tensor<1x1x16x16xf32>
    %prod_4d = arith.mulf %sum_4d, %t3 : tensor<1x1x16x16xf32>
    %result_3d = rock.transform %prod_4d by #tmap_4d_to_3d : tensor<1x1x16x16xf32> to tensor<1x16x16xf32>
    %lm = rock.load_marker %result_3d views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Splat constant (1D) broadcast to 3D then transposed, fused
  // with a block arg. The pass drops all transforms on the
  // constant (splat values are invariant to broadcast/transpose)
  // and creates a tile-shaped constant directly.
  // ============================================================

  // CHECK-LABEL: func.func @test_constant_broadcast_transpose
  // CHECK: %[[CST:.*]] = arith.constant dense<3.000000e+00> : tensor<16x16xf32>
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM]], %[[CST]] : tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_ADD]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_constant_broadcast_transpose(
      %tile: tensor<16x16xf32>,
      %bias: tensor<1x16x16xf32>,
      %dest: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %cst = arith.constant dense<3.000000e+00> : tensor<16xf32>
    %broadcast = rock.transform %cst by #tmap_broadcast_1d : tensor<16xf32> to tensor<1x16x16xf32>
    %transposed = rock.transform %broadcast by #tmap_transpose_3d : tensor<1x16x16xf32> to tensor<1x16x16xf32>
    %sum = arith.addf %transposed, %bias : tensor<1x16x16xf32>
    %lm = rock.load_marker %sum views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Input fusion (no store_marker): load_marker(addf(A, B)).
  // Distributed into two load_markers + tile-level addf,
  // result returned directly (e.g. feeds blockwise_gemm).
  // ============================================================

  // CHECK-LABEL: func.func @test_input_fusion_simple
  // CHECK: %[[LM1:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[LM2:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM1]], %[[LM2]] : tensor<16x16xf16>
  // CHECK: return %[[TILE_ADD]]
  func.func @test_input_fusion_simple(
      %A: tensor<1x16x16xf16>, %B: tensor<1x16x16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16> attributes {rock.kernel} {
    %fused = arith.addf %A, %B : tensor<1x16x16xf16>
    %lm = rock.load_marker %fused views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    return %lm : tensor<16x16xf16>
  }

  // ============================================================
  // Input fusion with full activation and broadcast scale/bias.
  // N has two blocks, so narrowing must remove both n_block and
  // n_iter before expanding back to the 2-D tile.
  // ============================================================

  // CHECK-LABEL: func.func @test_input_fusion_narrow_broadcast_multiblock
  // CHECK: %[[ACTIVATION:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x32xf16> -> tensor<16x16xf16>
  // CHECK: %[[SCALE:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<16xf16> -> tensor<16xf16>
  // CHECK: %[[SCALE_EXPANDED:.*]] = tt.expand_dims %[[SCALE]] {axis = 1 : i32}
  // CHECK-SAME: tensor<16xf16> -> tensor<16x1xf16>
  // CHECK: %[[SCALE_BROADCAST:.*]] = tt.broadcast %[[SCALE_EXPANDED]]
  // CHECK-SAME: tensor<16x1xf16> -> tensor<16x16xf16>
  // CHECK: %[[SCALED:.*]] = arith.mulf %[[ACTIVATION]], %[[SCALE_BROADCAST]]
  // CHECK: %[[BIAS:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<16xf16> -> tensor<16xf16>
  // CHECK: %[[BIAS_EXPANDED:.*]] = tt.expand_dims %[[BIAS]] {axis = 1 : i32}
  // CHECK-SAME: tensor<16xf16> -> tensor<16x1xf16>
  // CHECK: %[[BIAS_BROADCAST:.*]] = tt.broadcast %[[BIAS_EXPANDED]]
  // CHECK-SAME: tensor<16x1xf16> -> tensor<16x16xf16>
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SCALED]], %[[BIAS_BROADCAST]]
  // CHECK: return %[[FUSED]]
  //
  // LOWER-LABEL: func.func @test_input_fusion_narrow_broadcast_multiblock
  // LOWER: %[[ACTIVATION:.*]] = rock.blockwise_load
  // LOWER-SAME: tensor<1x1x2x16x16xf16> -> tensor<16x16xf16>
  // LOWER: %[[SCALE:.*]] = rock.blockwise_load
  // LOWER-SAME: -> tensor<16xf16>
  // LOWER: %[[SCALE_EXPANDED:.*]] = tt.expand_dims %[[SCALE]]
  // LOWER: %[[SCALE_BROADCAST:.*]] = tt.broadcast %[[SCALE_EXPANDED]]
  // LOWER: %[[SCALED:.*]] = arith.mulf %[[ACTIVATION]], %[[SCALE_BROADCAST]]
  // LOWER: %[[BIAS:.*]] = rock.blockwise_load
  // LOWER-SAME: -> tensor<16xf16>
  // LOWER: %[[BIAS_EXPANDED:.*]] = tt.expand_dims %[[BIAS]]
  // LOWER: %[[BIAS_BROADCAST:.*]] = tt.broadcast %[[BIAS_EXPANDED]]
  // LOWER: %[[FUSED:.*]] = arith.addf %[[SCALED]], %[[BIAS_BROADCAST]]
  // LOWER: return %[[FUSED]]
  func.func @test_input_fusion_narrow_broadcast_multiblock(
      %activation: tensor<1x16x32xf16>,
      %scaleRaw: tensor<16xf16>,
      %biasRaw: tensor<16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16>
      attributes {rock.kernel} {
    %scale = rock.transform %scaleRaw by #tmap_broadcast_n : tensor<16xf16> to tensor<1x16x32xf16>
    %bias = rock.transform %biasRaw by #tmap_broadcast_n : tensor<16xf16> to tensor<1x16x32xf16>
    %scaled = arith.mulf %activation, %scale : tensor<1x16x32xf16>
    %fused = arith.addf %scaled, %bias : tensor<1x16x32xf16>
    %lm = rock.load_marker %fused views [#tmap_two_n_blocks] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x32xf16> -> tensor<16x16xf16>
    return %lm : tensor<16x16xf16>
  }

  // ============================================================
  // Input fusion with a broadcast along M instead of N. This narrows the
  // load to the N tile and broadcasts it back along tile axis 0.
  // ============================================================

  // CHECK-LABEL: func.func @test_input_fusion_narrow_broadcast_m_axis
  // CHECK: %[[ACTIVATION:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x32xf16> -> tensor<16x16xf16>
  // CHECK: %[[BIAS:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<32xf16> -> tensor<16xf16>
  // CHECK: %[[BIAS_EXPANDED:.*]] = tt.expand_dims %[[BIAS]] {axis = 0 : i32}
  // CHECK-SAME: tensor<16xf16> -> tensor<1x16xf16>
  // CHECK: %[[BIAS_BROADCAST:.*]] = tt.broadcast %[[BIAS_EXPANDED]]
  // CHECK-SAME: tensor<1x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[FUSED:.*]] = arith.addf %[[ACTIVATION]], %[[BIAS_BROADCAST]]
  // CHECK: return %[[FUSED]]
  //
  // LOWER-LABEL: func.func @test_input_fusion_narrow_broadcast_m_axis
  // LOWER: %[[ACTIVATION:.*]] = rock.blockwise_load
  // LOWER-SAME: tensor<1x1x2x16x16xf16> -> tensor<16x16xf16>
  // LOWER: %[[BIAS:.*]] = rock.blockwise_load
  // LOWER-SAME: -> tensor<16xf16>
  // LOWER: %[[BIAS_EXPANDED:.*]] = tt.expand_dims %[[BIAS]] {axis = 0 : i32}
  // LOWER: %[[BIAS_BROADCAST:.*]] = tt.broadcast %[[BIAS_EXPANDED]]
  // LOWER: %[[FUSED:.*]] = arith.addf %[[ACTIVATION]], %[[BIAS_BROADCAST]]
  // LOWER: return %[[FUSED]]
  func.func @test_input_fusion_narrow_broadcast_m_axis(
      %activation: tensor<1x16x32xf16>,
      %biasRaw: tensor<32xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16>
      attributes {rock.kernel} {
    %bias = rock.transform %biasRaw by #tmap_broadcast_m : tensor<32xf16> to tensor<1x16x32xf16>
    %fused = arith.addf %activation, %bias : tensor<1x16x32xf16>
    %lm = rock.load_marker %fused views [#tmap_two_n_blocks] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x32xf16> -> tensor<16x16xf16>
    return %lm : tensor<16x16xf16>
  }

  // ============================================================
  // Broadcast path with a path-specific pad. Even though the address is
  // independent of N, the pad validity mask depends on N, so the load must
  // remain full-rank instead of being narrowed and broadcast.
  // ============================================================

  // CHECK-LABEL: func.func @test_input_fusion_does_not_narrow_path_specific_pad
  // CHECK: %[[ACTIVATION:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[BROADCAST:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16xf16> to tensor<1x16x14xf16>
  // CHECK: %[[PAD:.*]] = rock.transform %[[BROADCAST]] by
  // CHECK-SAME: tensor<1x16x14xf16> to tensor<1x16x16xf16>
  // CHECK: %[[SCALE:.*]] = rock.load_marker %[[PAD]] views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK-NOT: tt.expand_dims
  // CHECK: %[[FUSED:.*]] = arith.addf %[[ACTIVATION]], %[[SCALE]]
  // CHECK: return %[[FUSED]]
  //
  // LOWER-LABEL: func.func @test_input_fusion_does_not_narrow_path_specific_pad
  // LOWER: %[[ACTIVATION:.*]] = rock.blockwise_load
  // LOWER-SAME: tensor<1x1x1x16x16xf16> -> tensor<16x16xf16>
  // LOWER: %[[SCALE:.*]] = rock.blockwise_load
  // LOWER-SAME: -> tensor<16x16xf16>
  // LOWER-NOT: tt.expand_dims
  // LOWER: %[[FUSED:.*]] = arith.addf %[[ACTIVATION]], %[[SCALE]]
  // LOWER: return %[[FUSED]]
  func.func @test_input_fusion_does_not_narrow_path_specific_pad(
      %activation: tensor<1x16x16xf16>,
      %scaleRaw: tensor<16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16>
      attributes {rock.kernel} {
    %scaleBase = rock.transform %scaleRaw by #tmap_broadcast_n_14 : tensor<16xf16> to tensor<1x16x14xf16>
    %scale = rock.transform %scaleBase by #tmap_pad_n_14_to_16 : tensor<1x16x14xf16> to tensor<1x16x16xf16>
    %fused = arith.addf %activation, %scale : tensor<1x16x16xf16>
    %lm = rock.load_marker %fused views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    return %lm : tensor<16x16xf16>
  }

  // ============================================================
  // Scalar broadcast has two removable tile axes. Since the pass only narrows
  // a single tile axis today, this leaf remains a full-rank load.
  // ============================================================

  // CHECK-LABEL: func.func @test_input_fusion_does_not_narrow_scalar_broadcast
  // CHECK: %[[ACTIVATION:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[SCALAR_VIEW:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1xf16> to tensor<1x16x16xf16>
  // CHECK: %[[SCALAR:.*]] = rock.load_marker %[[SCALAR_VIEW]] views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK-NOT: tt.expand_dims
  // CHECK: %[[FUSED:.*]] = arith.addf %[[ACTIVATION]], %[[SCALAR]]
  // CHECK: return %[[FUSED]]
  func.func @test_input_fusion_does_not_narrow_scalar_broadcast(
      %activation: tensor<1x16x16xf16>,
      %scalarRaw: tensor<1xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16>
      attributes {rock.kernel} {
    %scalar = rock.transform %scalarRaw by #tmap_broadcast_scalar : tensor<1xf16> to tensor<1x16x16xf16>
    %fused = arith.addf %activation, %scalar : tensor<1x16x16xf16>
    %lm = rock.load_marker %fused views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    return %lm : tensor<16x16xf16>
  }

  // ============================================================
  // Input fusion with full-rank transforms, nested fusions, and
  // interleaved transforms:
  //   transpose(A) \
  //                 addf → transpose → \
  //   transpose(B) /                    mulf → transpose → load_marker
  //                                    /
  //                  transpose(C) ----
  // Each leaf accumulates the transforms between it and the
  // load_marker: A,B get 3 transforms, C gets 2. Full-rank inputs keep
  // this test focused on transform accumulation rather than load narrowing.
  // ============================================================

  // CHECK-LABEL: func.func @test_input_fusion_with_transforms
  // A: transpose → transpose → transpose → load_marker
  // CHECK: %[[TA1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[TA2:.*]] = rock.transform %[[TA1]] by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[TA3:.*]] = rock.transform %[[TA2]] by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[LM_A:.*]] = rock.load_marker %[[TA3]] views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // B: transpose → transpose → transpose → load_marker
  // CHECK: %[[TB1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[TB2:.*]] = rock.transform %[[TB1]] by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[TB3:.*]] = rock.transform %[[TB2]] by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[LM_B:.*]] = rock.load_marker %[[TB3]] views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM_A]], %[[LM_B]] : tensor<16x16xf16>
  // C: transpose → transpose → load_marker
  // CHECK: %[[TC1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[TC2:.*]] = rock.transform %[[TC1]] by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[LM_C:.*]] = rock.load_marker %[[TC2]] views
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[TILE_MUL:.*]] = arith.mulf %[[TILE_ADD]], %[[LM_C]] : tensor<16x16xf16>
  // CHECK: return %[[TILE_MUL]]
  func.func @test_input_fusion_with_transforms(
      %A_raw: tensor<1x16x16xf16>, %B_raw: tensor<1x16x16xf16>,
      %C_raw: tensor<1x16x16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16> attributes {rock.kernel} {
    %A = rock.transform %A_raw by #tmap_transpose_3d : tensor<1x16x16xf16> to tensor<1x16x16xf16>
    %B = rock.transform %B_raw by #tmap_transpose_3d : tensor<1x16x16xf16> to tensor<1x16x16xf16>
    %sum = arith.addf %A, %B : tensor<1x16x16xf16>
    %transposed1 = rock.transform %sum by #tmap_transpose_3d : tensor<1x16x16xf16> to tensor<1x16x16xf16>
    %C = rock.transform %C_raw by #tmap_transpose_3d : tensor<1x16x16xf16> to tensor<1x16x16xf16>
    %prod = arith.mulf %transposed1, %C : tensor<1x16x16xf16>
    %transposed2 = rock.transform %prod by #tmap_transpose_3d : tensor<1x16x16xf16> to tensor<1x16x16xf16>
    %lm = rock.load_marker %transposed2 views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    return %lm : tensor<16x16xf16>
  }

  // ============================================================
  // No load_marker at all: only store_marker → store.
  // Pass has nothing to process.
  // ============================================================

  // CHECK-LABEL: func.func @test_no_load_marker
  // CHECK: rock.store_marker
  // CHECK-NOT: rock.load_marker
  // CHECK: rock.store
  func.func @test_no_load_marker(%tile: tensor<16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %r = rock.store %sm to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Non-kernel function: pass skips entirely.
  // ============================================================

  // CHECK-LABEL: func.func @test_non_kernel
  // CHECK: rock.load_marker %{{.*}} views
  // CHECK-NOT: rock.untile
  func.func @test_non_kernel(%bias: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<16x16xf32> {
    %sum = arith.addf %bias, %bias : tensor<1x16x16xf32>
    %lm = rock.load_marker %sum views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    return %lm : tensor<16x16xf32>
  }

  // CHECK-LABEL: func.func @test_shared_arg_different_transforms
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[LM_DIRECT:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[T:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[LM_TRANSP:.*]] = rock.load_marker %[[T]] views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM_DIRECT]], %[[LM_TRANSP]] : tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_ADD]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_shared_arg_different_transforms(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %bias_t = rock.transform %bias by #tmap_transpose_3d : tensor<1x16x16xf32> to tensor<1x16x16xf32>
    %sum = arith.addf %bias, %bias_t : tensor<1x16x16xf32>
    %lm = rock.load_marker %sum views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Cache modifier propagation: a load_marker with a non-default cache
  // modifier (cs) whose source is a fusion (addf) must distribute the cache
  // modifier to every leaf load_marker created for the fusion operands.
  // ============================================================

  // CHECK-LABEL: func.func @test_cache_modifier_propagation
  // CHECK: %[[LM1:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: {cacheModifier = #rock<CacheModifier cs>}
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[LM2:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: {cacheModifier = #rock<CacheModifier cs>}
  // CHECK-SAME: tensor<1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: arith.addf %[[LM1]], %[[LM2]] : tensor<16x16xf16>
  func.func @test_cache_modifier_propagation(
      %A: tensor<1x16x16xf16>, %B: tensor<1x16x16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16> attributes {rock.kernel} {
    %fused = arith.addf %A, %B : tensor<1x16x16xf16>
    %lm = rock.load_marker %fused views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier cs>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    return %lm : tensor<16x16xf16>
  }
}
