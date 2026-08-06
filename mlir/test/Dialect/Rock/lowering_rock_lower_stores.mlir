// Unit tests for rock-lower-stores pass.
// The input to this pass is the output of rock-lower-loads:
// - StoreMarkerOp maps tiles to full tensors for fusion ops
// - Fusion ops (arith.addf, etc.) operate on full tensor types
// - rock.store stores the fused result
//
// The pass traces from rock.store back through fusions to StoreMarkerOp,
// clones fusion ops at tile level, combines StoreMarkerOp transforms with
// destination transforms, and creates rock.blockwise_store.

// RUN: rocmlir-opt -rock-lower-stores -canonicalize -mlir-print-local-scope %s | FileCheck %s

// --- Tiling transform: 3D [1,256,128] -> 5D [1,4,2,64,64] block/iter split ---
#tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{4, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 4, 2, 64, 64] -> [1, 256, 128]>

// --- Destination flat-to-3D transform: 32768 -> [1,256,128] ---
#tmap_dest = #rock.transform_map<affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 128 + d2)> by [<Unmerge{1, 256, 128} ["d0", "d1", "d2"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [1, 256, 128] -> [32768]>
#reduce_tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d3, d2 * 4 + d4)> by [<PassThrough ["g_block"] at [0] -> ["g"] at [0]>, <Unmerge{2, 4} ["m_block", "m_iter"] at [1, 3] -> ["m"] at [1]>, <Unmerge{2, 4} ["n_block", "n_iter"] at [2, 4] -> ["n"] at [2]>] bounds = [1, 2, 2, 4, 4] -> [1, 8, 8]>
#reduce_flat = #rock.transform_map<affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 8, 1} ["flat"] at [0] -> ["g", "m", "n"] at [0, 1, 2]>] bounds = [8] -> [1, 8, 1]>
#spanning_tmap = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 16 + d1)> by [<Unmerge{4, 16} ["block", "iter"] at [0, 1] -> ["flat"] at [0]>] bounds = [4, 16] -> [64]>
#unflatten_8x8 = #rock.transform_map<affine_map<(d0, d1, d2) -> ((d0 * 8 + d1) * 8 + d2)> by [<Unmerge{1, 8, 8} ["g", "m", "n"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [1, 8, 8] -> [64]>

module {

  // ============================================================
  // No fusion: store_marker feeds directly into rock.store.
  // The tile is stored via blockwise_store with extraViews
  // transforms applied to the destination.
  // ============================================================

  // CHECK-LABEL: func.func @test_no_fusion
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: %[[DEST:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: to tensor<1x4x2x64x64xf32>
  // CHECK: rock.blockwise_store %{{.*}} -> %[[DEST]][%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}set
  // CHECK-SAME: tensor<64x64xf32> -> tensor<1x4x2x64x64xf32>
  func.func @test_no_fusion(
      %tile: tensor<64x64xf32>,
      %dest: tensor<1x256x128xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %r = rock.store %sm to %dest by set : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
    return %r : tensor<1x256x128xf32>
  }

  // ============================================================
  // Single addf fusion: store_marker → addf(marker, untile(load))
  // → store.  The addf is cloned at tile level; the untile/load
  // input feeds the tile-level addf directly.
  // ============================================================

  // CHECK-LABEL: func.func @test_single_addf
  // CHECK: %[[BL:.*]] = rock.blockwise_load
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[BL]] : tensor<64x64xf32>
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: %[[DEST:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: to tensor<1x4x2x64x64xf32>
  // CHECK: rock.blockwise_store %[[ADD]] -> %[[DEST]][%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}set
  // CHECK-SAME: tensor<64x64xf32> -> tensor<1x4x2x64x64xf32>
  func.func @test_single_addf(
      %tile: tensor<64x64xf32>,
      %bias_tile: tensor<64x64xf32>,
      %dest: tensor<1x256x128xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %bl = rock.blockwise_load %bias_tile {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf32> -> tensor<64x64xf32>
    %ut = rock.untile %bl : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %add = arith.addf %sm, %ut : tensor<1x256x128xf32>
    %r = rock.store %add to %dest by set : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
    return %r : tensor<1x256x128xf32>
  }

  // ============================================================
  // Fusion chain: addf then mulf, each with an extra loaded
  // input via untile(blockwise_load).
  // Both fusions are cloned at tile level.
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_chain
  // CHECK: %[[BL1:.*]] = rock.blockwise_load %{{.*}} : tensor<64x64xf16> -> tensor<64x64xf16>
  // CHECK: %[[BL2:.*]] = rock.blockwise_load %{{.*}} : tensor<64x64xf16> -> tensor<64x64xf16>
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[BL1]] : tensor<64x64xf16>
  // CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[BL2]] : tensor<64x64xf16>
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: rock.blockwise_store %[[MUL]] -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}set
  // CHECK-SAME: tensor<64x64xf16>
  func.func @test_fusion_chain(
      %tile: tensor<64x64xf16>,
      %bias_tile: tensor<64x64xf16>,
      %scale_tile: tensor<64x64xf16>,
      %dest: tensor<1x256x128xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf16> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf16> -> tensor<1x256x128xf16>
    %bl1 = rock.blockwise_load %bias_tile {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf16> -> tensor<64x64xf16>
    %ut1 = rock.untile %bl1 : tensor<64x64xf16> -> tensor<1x256x128xf16>
    %add = arith.addf %sm, %ut1 : tensor<1x256x128xf16>
    %bl2 = rock.blockwise_load %scale_tile {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf16> -> tensor<64x64xf16>
    %ut2 = rock.untile %bl2 : tensor<64x64xf16> -> tensor<1x256x128xf16>
    %mul = arith.mulf %add, %ut2 : tensor<1x256x128xf16>
    %r = rock.store %mul to %dest by set : tensor<1x256x128xf16> -> tensor<1x256x128xf16> to tensor<1x256x128xf16>
    return %r : tensor<1x256x128xf16>
  }

  // ============================================================
  // Type-changing fusion: arith.extf (f16 → f32).
  // The tile type for the blockwise_store uses f32 element type
  // while the StoreMarkerOp tile shape is preserved.
  // ============================================================

  // CHECK-LABEL: func.func @test_extf
  // CHECK: %[[EXT:.*]] = arith.extf %{{.*}} : tensor<64x64xf16> to tensor<64x64xf32>
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: rock.blockwise_store %[[EXT]] -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}set
  // CHECK-SAME: tensor<64x64xf32> -> tensor<1x4x2x64x64xf32>
  func.func @test_extf(
      %tile: tensor<64x64xf16>,
      %dest: tensor<1x256x128xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf16> -> tensor<1x256x128xf16>
    %ext = arith.extf %sm : tensor<1x256x128xf16> to tensor<1x256x128xf32>
    %r = rock.store %ext to %dest by set : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
    return %r : tensor<1x256x128xf32>
  }

  // ============================================================
  // Cascaded type-changing fusion: addf (f16) → extf (f16→f32).
  // The addf operands use f16 tile type, extf result uses f32.
  // ============================================================

  // CHECK-LABEL: func.func @test_cascaded_addf_extf
  // CHECK: %[[BL:.*]] = rock.blockwise_load %{{.*}} : tensor<64x64xf16> -> tensor<64x64xf16>
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[BL]] : tensor<64x64xf16>
  // CHECK: %[[EXT:.*]] = arith.extf %[[ADD]] : tensor<64x64xf16> to tensor<64x64xf32>
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: rock.blockwise_store %[[EXT]] -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}set
  // CHECK-SAME: tensor<64x64xf32>
  func.func @test_cascaded_addf_extf(
      %tile: tensor<64x64xf16>,
      %bias_tile: tensor<64x64xf16>,
      %dest: tensor<1x256x128xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf16> -> tensor<1x256x128xf16>
    %bl = rock.blockwise_load %bias_tile {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf16> -> tensor<64x64xf16>
    %ut = rock.untile %bl : tensor<64x64xf16> -> tensor<1x256x128xf16>
    %add = arith.addf %sm, %ut : tensor<1x256x128xf16>
    %ext = arith.extf %add : tensor<1x256x128xf16> to tensor<1x256x128xf32>
    %r = rock.store %ext to %dest by set : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
    return %r : tensor<1x256x128xf32>
  }

  // ============================================================
  // Destination with existing transforms: store dest already has
  // a rock.transform chain (flat 32768 → 3D [1,256,128]).
  // The pass combines StoreMarkerOp's extraViews with the
  // existing destination transforms, stacking the tiling
  // transform on top.
  // ============================================================

  // CHECK-LABEL: func.func @test_dest_transforms
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // dest_transform (flat→3D) then tiling (3D→5D) stacked
  // CHECK: %[[D1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<32768xf32> to tensor<1x256x128xf32>
  // CHECK: %[[D2:.*]] = rock.transform %[[D1]] by
  // CHECK-SAME: tensor<1x256x128xf32> to tensor<1x4x2x64x64xf32>
  // CHECK: rock.blockwise_store %{{.*}} -> %[[D2]][%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}set
  // CHECK-SAME: tensor<64x64xf32> -> tensor<1x4x2x64x64xf32> -> tensor<32768xf32>
  func.func @test_dest_transforms(
      %tile: tensor<64x64xf32>,
      %dest_raw: tensor<32768xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<32768xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %dest = rock.transform %dest_raw by #tmap_dest : tensor<32768xf32> to tensor<1x256x128xf32>
    %r = rock.store %sm to %dest by set : tensor<1x256x128xf32> -> tensor<32768xf32> to tensor<1x256x128xf32>
    return %r : tensor<32768xf32>
  }

  // ============================================================
  // Fusion with destination transforms: addf fusion + dest with
  // existing transform chain. Both fusion tiling and transform
  // combining happen.
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_with_dest_transforms
  // CHECK: %[[BL:.*]] = rock.blockwise_load
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[BL]] : tensor<64x64xf32>
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: %[[D1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<32768xf32> to tensor<1x256x128xf32>
  // CHECK: %[[D2:.*]] = rock.transform %[[D1]] by
  // CHECK-SAME: tensor<1x256x128xf32> to tensor<1x4x2x64x64xf32>
  // CHECK: rock.blockwise_store %[[ADD]] -> %[[D2]][%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}set
  // CHECK-SAME: tensor<64x64xf32> -> tensor<1x4x2x64x64xf32> -> tensor<32768xf32>
  func.func @test_fusion_with_dest_transforms(
      %tile: tensor<64x64xf32>,
      %bias_tile: tensor<64x64xf32>,
      %dest_raw: tensor<32768xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<32768xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %bl = rock.blockwise_load %bias_tile {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf32> -> tensor<64x64xf32>
    %ut = rock.untile %bl : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %add = arith.addf %sm, %ut : tensor<1x256x128xf32>
    %dest = rock.transform %dest_raw by #tmap_dest : tensor<32768xf32> to tensor<1x256x128xf32>
    %r = rock.store %add to %dest by set : tensor<1x256x128xf32> -> tensor<32768xf32> to tensor<1x256x128xf32>
    return %r : tensor<32768xf32>
  }

  // ============================================================
  // atomic_add store method: the store method is preserved on
  // the blockwise_store.
  // ============================================================

  // CHECK-LABEL: func.func @test_atomic_add
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: rock.blockwise_store %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}atomic_add
  // CHECK-SAME: tensor<64x64xf32> -> tensor<1x4x2x64x64xf32>
  func.func @test_atomic_add(
      %tile: tensor<64x64xf32>,
      %dest: tensor<1x256x128xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %r = rock.store %sm to %dest by atomic_add : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
    return %r : tensor<1x256x128xf32>
  }

  // ============================================================
  // atomic_add with fusion: store method preserved even when
  // fusion ops are cloned to tile level.
  // ============================================================

  // CHECK-LABEL: func.func @test_atomic_add_with_fusion
  // CHECK: %[[BL:.*]] = rock.blockwise_load
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[BL]] : tensor<64x64xf32>
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: rock.blockwise_store %[[ADD]] -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}atomic_add
  // CHECK-SAME: tensor<64x64xf32>
  func.func @test_atomic_add_with_fusion(
      %tile: tensor<64x64xf32>,
      %bias_tile: tensor<64x64xf32>,
      %dest: tensor<1x256x128xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %bl = rock.blockwise_load %bias_tile {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf32> -> tensor<64x64xf32>
    %ut = rock.untile %bl : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %add = arith.addf %sm, %ut : tensor<1x256x128xf32>
    %r = rock.store %add to %dest by atomic_add : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
    return %r : tensor<1x256x128xf32>
  }

  // ============================================================
  // Fusion with splat constant (already tile-shaped from
  // regularize-input): constant → blockwise_load → untile.
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_with_constant
  // CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<64x64xf32>
  // CHECK: %[[BL:.*]] = rock.blockwise_load %[[CST]] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf32> -> tensor<64x64xf32>
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[BL]] : tensor<64x64xf32>
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: rock.blockwise_store %[[ADD]] -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] by {{.*}}set
  func.func @test_fusion_with_constant(
      %tile: tensor<64x64xf32>,
      %dest: tensor<1x256x128xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf32> attributes {rock.kernel} {
    %cst = arith.constant dense<2.0> : tensor<64x64xf32>
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %bl = rock.blockwise_load %cst {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf32> -> tensor<64x64xf32>
    %ut = rock.untile %bl : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %add = arith.addf %sm, %ut : tensor<1x256x128xf32>
    %r = rock.store %add to %dest by set : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
    return %r : tensor<1x256x128xf32>
  }

  // ============================================================
  // Empty combined transforms: store_marker has no extraViews
  // and store dest is a plain block arg (no rock.transform).
  // The blockwise_store destination is the block arg directly,
  // with no transform inserted.
  // ============================================================

  // CHECK-LABEL: func.func @test_no_combined_transforms
  // CHECK-NOT: = rock.transform
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: rock.blockwise_store %{{.*}} -> %{{.*}} by {{.*}}set
  // CHECK-SAME: tensor<64x64xf32> -> tensor<64x64xf32>
  func.func @test_no_combined_transforms(
      %tile: tensor<64x64xf32>,
      %dest: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [] : tensor<64x64xf32> -> tensor<64x64xf32>
    %r = rock.store %sm to %dest by set : tensor<64x64xf32> -> tensor<64x64xf32> to tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }

  // ============================================================
  // Empty combined transforms with fusion: same as above but
  // with an addf fusion. Fusion is cloned at tile level, but
  // the blockwise_store destination has no transforms.
  // ============================================================

  // CHECK-LABEL: func.func @test_no_combined_transforms_with_fusion
  // CHECK: %[[BL:.*]] = rock.blockwise_load
  // CHECK: %[[ADD:.*]] = arith.addf %{{.*}}, %[[BL]] : tensor<64x64xf32>
  // CHECK-NOT: = rock.transform
  // CHECK-NOT: rock.store {{.*}} to {{.*}} by
  // CHECK: rock.blockwise_store %[[ADD]] -> %{{.*}} by {{.*}}set
  // CHECK-SAME: tensor<64x64xf32> -> tensor<64x64xf32>
  func.func @test_no_combined_transforms_with_fusion(
      %tile: tensor<64x64xf32>,
      %bias_tile: tensor<64x64xf32>,
      %dest: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [] : tensor<64x64xf32> -> tensor<64x64xf32>
    %bl = rock.blockwise_load %bias_tile {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xf32> -> tensor<64x64xf32>
    %ut = rock.untile %bl : tensor<64x64xf32> -> tensor<64x64xf32>
    %add = arith.addf %sm, %ut : tensor<64x64xf32>
    %r = rock.store %add to %dest by set : tensor<64x64xf32> -> tensor<64x64xf32> to tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }

  // CHECK-LABEL: func.func @test_blockwise_reduction
  // CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce sum {{.*}} {axis = 1 : index} : tensor<4x4xf32> -> tensor<4xf32>
  // CHECK: rock.blockwise_store %[[REDUCED]] -> {{.*}} by atomic_add
  // CHECK-SAME: tensor<4xf32>
  func.func @test_blockwise_reduction(
      %tile: tensor<4x4xf32>, %dest: tensor<8xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<8xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#reduce_tmap] [%g, %m, %n] : tensor<4x4xf32> -> tensor<1x8x8xf32>
    %reduced = rock.reduce sum %sm {axis = 2 : index, blockwise} : tensor<1x8x8xf32> -> tensor<1x8x1xf32>
    %flat = rock.transform %reduced by #reduce_flat : tensor<1x8x1xf32> to tensor<8xf32>
    %result = rock.store %flat to %dest by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
    return %result : tensor<8xf32>
  }

  // A tile spanning both retained and reduced coordinates has no complete
  // invariant local axis. Keep the prepared per-element atomic store.
  // CHECK-LABEL: func.func @test_reduction_no_local_axis
  // CHECK-NOT: rock.blockwise_reduce
  // CHECK: rock.blockwise_store {{.*}} by atomic_add
  // CHECK-SAME: tensor<16xf32>
  func.func @test_reduction_no_local_axis(
      %tile: tensor<16xf32>, %dest: tensor<8xf32>,
      %block: i32) -> tensor<8xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#spanning_tmap] [%block] : tensor<16xf32> -> tensor<64xf32>
    %logical = rock.transform %sm by #unflatten_8x8 : tensor<64xf32> to tensor<1x8x8xf32>
    %reduced = rock.reduce sum %logical {axis = 2 : index, blockwise} : tensor<1x8x8xf32> -> tensor<1x8x1xf32>
    %flat = rock.transform %reduced by #reduce_flat : tensor<1x8x1xf32> to tensor<8xf32>
    %result = rock.store %flat to %dest by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
    return %result : tensor<8xf32>
  }

  // ============================================================
  // Non-kernel function: pass is skipped entirely.
  // rock.store remains unchanged.
  // ============================================================

  // CHECK-LABEL: func.func @test_non_kernel
  // CHECK: rock.store_marker
  // CHECK: rock.store
  // CHECK-NOT: rock.blockwise_store
  func.func @test_non_kernel(
      %tile: tensor<64x64xf32>,
      %dest: tensor<1x256x128xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x256x128xf32> {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<64x64xf32> -> tensor<1x256x128xf32>
    %r = rock.store %sm to %dest by set : tensor<1x256x128xf32> -> tensor<1x256x128xf32> to tensor<1x256x128xf32>
    return %r : tensor<1x256x128xf32>
  }
}
