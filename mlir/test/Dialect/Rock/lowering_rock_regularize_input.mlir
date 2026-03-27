// Unit tests for rock-regularize-input pass.
// Tests that load_markers whose source contains fusion ops are distributed
// so each leaf (block arg or constant) gets its own load_marker, and fusion
// ops are cloned to operate on tile types.

// RUN: rocmlir-opt -rock-regularize-input -mlir-print-local-scope %s | FileCheck %s

#tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d3, d2 * 16 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 16} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{1, 16} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 1, 16, 16] -> [1, 16, 16]>

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
    %lm = rock.load_marker %bias views [#tmap] [%g, %m, %n] : tensor<1x16x16xf32> -> tensor<16x16xf32>
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
  // Original full-type addf left as dead code
  // CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<1x16x16xf16>
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
    %lm = rock.load_marker %sum views [#tmap] [%g, %m, %n] : tensor<1x16x16xf16> -> tensor<16x16xf16>
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
  // CHECK: %[[SM:.*]] = rock.store_marker
  // Tile-shaped constant replaces the load_marker
  // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[CST]] : tensor<16x16xf32> -> tensor<1x16x16xf32>
  // CHECK: %[[FUSED:.*]] = arith.subf %[[SM]], %[[UT]] : tensor<1x16x16xf32>
  // CHECK: rock.store %[[FUSED]]
  func.func @test_splat_constant(%tile: tensor<16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<1x16x16xf32>
    %lm = rock.load_marker %cst views [#tmap] [%g, %m, %n] : tensor<1x16x16xf32> -> tensor<16x16xf32>
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
  // Pass re-applies the transform to blockarg and creates new load_marker
  // CHECK: rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[LM]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_transform_chain(%tile: tensor<16x16xf32>, %bias_raw: tensor<16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %bias_3d = rock.transform %bias_raw by <affine_map<(d0, d1, d2) -> (d2)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{16} ["m"] at [1] -> [] at []>, <PassThrough ["n"] at [2] -> ["n"] at [0]>] bounds = [1, 16, 16] -> [16]> : tensor<16xf32> to tensor<1x16x16xf32>
    %lm = rock.load_marker %bias_3d views [#tmap] [%g, %m, %n] : tensor<1x16x16xf32> -> tensor<16x16xf32>
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
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[LM:.*]] = rock.load_marker %{{.*}} views
  // CHECK-SAME: tensor<1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<16x16xf32>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[LM]], %[[CST]] : tensor<16x16xf32>
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_ADD]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_fusion_with_constant(%tile: tensor<16x16xf32>, %bias: tensor<1x16x16xf32>, %dest: tensor<1x16x16xf32>, %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %cst = arith.constant dense<2.000000e+00> : tensor<1x16x16xf32>
    %sum = arith.addf %bias, %cst : tensor<1x16x16xf32>
    %lm = rock.load_marker %sum views [#tmap] [%g, %m, %n] : tensor<1x16x16xf32> -> tensor<16x16xf32>
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
    %lm = rock.load_marker %mul views [#tmap] [%g, %m, %n] : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %fused = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %fused to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
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
    %lm = rock.load_marker %sum views [#tmap] [%g, %m, %n] : tensor<1x16x16xf32> -> tensor<16x16xf32>
    return %lm : tensor<16x16xf32>
  }
}
