// Unit tests for rock-lower-loads pass.
// The input to this pass is the output of rock-regularize-input:
// every rock.load_marker's source is a pure transform chain (no fusions).
// The pass replaces each rock.load_marker with rock.blockwise_load,
// applying the marker's extraViews on top of the source first.

// RUN: rocmlir-opt -rock-lower-loads -mlir-print-local-scope %s | FileCheck %s

// --- Tiling transform: 3D -> 5D block/iter split ---
#tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d3, d2 * 16 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 16} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{1, 16} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 1, 1, 16, 16] -> [1, 16, 16]>

// --- Broadcast: 1D -> 3D ---
#tmap_broadcast = #rock.transform_map<affine_map<(d0, d1, d2) -> (d2)> by [<AddDim{1} ["d0"] at [0] -> [] at []>, <AddDim{16} ["d1"] at [1] -> [] at []>, <PassThrough ["d2"] at [2] -> ["dim0"] at [0]>] bounds = [1, 16, 16] -> [16]>

// --- Transpose: 3D -> 3D (swap last two dims) ---
#tmap_transpose = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["d0"] at [0] -> ["d0"] at [0]>, <PassThrough ["d1"] at [1] -> ["d2"] at [2]>, <PassThrough ["d2"] at [2] -> ["d1"] at [1]>] bounds = [1, 16, 16] -> [1, 16, 16]>

// --- AddDim: 1D -> 2D (scalar-like, as in GridwiseAttnToBlockwise) ---
#tmap_adddim = #rock.transform_map<affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["dummy"] at [0] -> [] at []>, <PassThrough ["gemmG"] at [1] -> ["gemmG"] at [0]>] bounds = [1, 1] -> [1]>

module {

  // ============================================================
  // Direct block arg: load_marker(block_arg) → blockwise_load.
  // ExtraViews are applied as transforms before the load.
  // ============================================================

  // CHECK-LABEL: func.func @test_direct_blockarg
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[T:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf32> to tensor<1x1x1x16x16xf32>
  // CHECK: %[[BL:.*]] = rock.blockwise_load %[[T]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: tensor<1x1x1x16x16xf32> -> tensor<16x16xf32>
  // CHECK-NOT: rock.load_marker
  // CHECK: %[[UT:.*]] = rock.untile %[[BL]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_direct_blockarg(
      %tile: tensor<16x16xf32>,
      %bias: tensor<1x16x16xf32>,
      %dest: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %lm = rock.load_marker %bias views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // No extraViews: scalar-like load as in GridwiseAttnToBlockwise.
  // A 1D tensor gets an addDim transform, then load_marker with
  // empty views but with an index. No extra transform is inserted
  // by the pass; blockwise_load uses the pre-existing transform
  // chain directly.
  // ============================================================

  // CHECK-LABEL: func.func @test_no_extraviews
  // CHECK: %[[T:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1xi32> to tensor<1x1xi32>
  // CHECK: %[[BL:.*]] = rock.blockwise_load %[[T]][%{{.*}}]
  // CHECK-SAME: tensor<1x1xi32> -> tensor<1xi32>
  // CHECK-NOT: rock.load_marker
  // CHECK: return %[[BL]]
  func.func @test_no_extraviews(
      %seqlen: tensor<1xi32>, %g: i32) -> tensor<1xi32> attributes {rock.kernel} {
    %t = rock.transform %seqlen by #tmap_adddim : tensor<1xi32> to tensor<1x1xi32>
    %lm = rock.load_marker %t views [] [%g] {cacheModifier = #rock<CacheModifier none>} : tensor<1x1xi32> -> tensor<1xi32>
    return %lm : tensor<1xi32>
  }

  // ============================================================
  // Splat constant: load_marker(constant) with no extraViews.
  // No transform is inserted; blockwise_load takes the constant
  // directly (no indices).
  // ============================================================

  // CHECK-LABEL: func.func @test_splat_constant
  // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<16x16xf32>
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[BL:.*]] = rock.blockwise_load %[[CST]] {cacheModifier = #rock<CacheModifier none>} : tensor<16x16xf32> -> tensor<16x16xf32>
  // CHECK-NOT: rock.load_marker
  // CHECK: %[[UT:.*]] = rock.untile %[[BL]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_splat_constant(
      %tile: tensor<16x16xf32>,
      %dest: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %cst = arith.constant dense<1.0> : tensor<16x16xf32>
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %lm = rock.load_marker %cst views [] {cacheModifier = #rock<CacheModifier none>} : tensor<16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Transform chain before load_marker: broadcast(1D→3D) then
  // tiling. The existing transform is preserved; extraViews
  // transform is stacked on top of it.
  // ============================================================

  // CHECK-LABEL: func.func @test_transform_chain
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[BC:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[TILE:.*]] = rock.transform %[[BC]] by
  // CHECK-SAME: tensor<1x16x16xf32> to tensor<1x1x1x16x16xf32>
  // CHECK: %[[BL:.*]] = rock.blockwise_load %[[TILE]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: tensor<1x1x1x16x16xf32> -> tensor<16x16xf32>
  // CHECK-NOT: rock.load_marker
  // CHECK: %[[UT:.*]] = rock.untile %[[BL]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_transform_chain(
      %tile: tensor<16x16xf32>,
      %bias_raw: tensor<16xf32>,
      %dest: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %bias = rock.transform %bias_raw by #tmap_broadcast : tensor<16xf32> to tensor<1x16x16xf32>
    %lm = rock.load_marker %bias views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %ut = rock.untile %lm : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Tile-level fusion (output of regularize-input): two
  // load_markers feeding arith.addf at tile level.
  // Both markers are independently replaced by blockwise_load.
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_addf
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[T1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL1:.*]] = rock.blockwise_load %[[T1]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: tensor<1x1x1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[T2:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL2:.*]] = rock.blockwise_load %[[T2]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: tensor<1x1x1x16x16xf16> -> tensor<16x16xf16>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[BL1]], %[[BL2]] : tensor<16x16xf16>
  // CHECK-NOT: rock.load_marker
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_ADD]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_fusion_addf(
      %tile: tensor<16x16xf16>,
      %bias1: tensor<1x16x16xf16>,
      %bias2: tensor<1x16x16xf16>,
      %dest: tensor<1x16x16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf16> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf16> -> tensor<1x16x16xf16>
    %lm1 = rock.load_marker %bias1 views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %lm2 = rock.load_marker %bias2 views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %tile_add = arith.addf %lm1, %lm2 : tensor<16x16xf16>
    %ut = rock.untile %tile_add : tensor<16x16xf16> -> tensor<1x16x16xf16>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf16>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf16> -> tensor<1x16x16xf16> to tensor<1x16x16xf16>
    return %r : tensor<1x16x16xf16>
  }

  // ============================================================
  // Fusion with a constant: one marker has extraViews (block arg)
  // and the other has no views (splat constant).
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_with_constant
  // CHECK: %[[CST:.*]] = arith.constant dense<2.000000e+00> : tensor<16x16xf32>
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[T:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf32> to tensor<1x1x1x16x16xf32>
  // CHECK: %[[BL1:.*]] = rock.blockwise_load %[[T]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: tensor<1x1x1x16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[BL2:.*]] = rock.blockwise_load %[[CST]] {cacheModifier = #rock<CacheModifier none>} : tensor<16x16xf32> -> tensor<16x16xf32>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[BL1]], %[[BL2]] : tensor<16x16xf32>
  // CHECK-NOT: rock.load_marker
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_ADD]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_fusion_with_constant(
      %tile: tensor<16x16xf32>,
      %bias: tensor<1x16x16xf32>,
      %dest: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %cst = arith.constant dense<2.0> : tensor<16x16xf32>
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %lm1 = rock.load_marker %bias views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %lm2 = rock.load_marker %cst views [] {cacheModifier = #rock<CacheModifier none>} : tensor<16x16xf32> -> tensor<16x16xf32>
    %tile_add = arith.addf %lm1, %lm2 : tensor<16x16xf32>
    %ut = rock.untile %tile_add : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Fusion chain: addf then mulf at tile level, three markers.
  // ============================================================

  // CHECK-LABEL: func.func @test_fusion_chain
  // CHECK: %[[SM:.*]] = rock.store_marker
  // CHECK: %[[T1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL1:.*]] = rock.blockwise_load %[[T1]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf16>
  // CHECK: %[[T2:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL2:.*]] = rock.blockwise_load %[[T2]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf16>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[BL1]], %[[BL2]] : tensor<16x16xf16>
  // CHECK: %[[T3:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL3:.*]] = rock.blockwise_load %[[T3]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf16>
  // CHECK: %[[TILE_MUL:.*]] = arith.mulf %[[TILE_ADD]], %[[BL3]] : tensor<16x16xf16>
  // CHECK-NOT: rock.load_marker
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_MUL]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_fusion_chain(
      %tile: tensor<16x16xf16>,
      %a: tensor<1x16x16xf16>,
      %b: tensor<1x16x16xf16>,
      %c: tensor<1x16x16xf16>,
      %dest: tensor<1x16x16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf16> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf16> -> tensor<1x16x16xf16>
    %lm1 = rock.load_marker %a views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %lm2 = rock.load_marker %b views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %tile_add = arith.addf %lm1, %lm2 : tensor<16x16xf16>
    %lm3 = rock.load_marker %c views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %tile_mul = arith.mulf %tile_add, %lm3 : tensor<16x16xf16>
    %ut = rock.untile %tile_mul : tensor<16x16xf16> -> tensor<1x16x16xf16>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf16>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf16> -> tensor<1x16x16xf16> to tensor<1x16x16xf16>
    return %r : tensor<1x16x16xf16>
  }

  // ============================================================
  // Complex transform chains: each leaf block arg has a
  // different pre-existing transform before the load_marker.
  // broadcast(1D→3D) and transpose(3D→3D) chains stacked with
  // the tiling extraViews.
  // ============================================================

  // CHECK-LABEL: func.func @test_complex_transforms
  // CHECK: %[[SM:.*]] = rock.store_marker
  // arg1: broadcast → tiling → blockwise_load
  // CHECK: %[[BC1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[TILE1:.*]] = rock.transform %[[BC1]] by
  // CHECK-SAME: tensor<1x16x16xf32> to tensor<1x1x1x16x16xf32>
  // CHECK: %[[BL1:.*]] = rock.blockwise_load %[[TILE1]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf32>
  // arg2: transpose → tiling → blockwise_load
  // CHECK: %[[TR:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf32> to tensor<1x16x16xf32>
  // CHECK: %[[TILE2:.*]] = rock.transform %[[TR]] by
  // CHECK-SAME: tensor<1x16x16xf32> to tensor<1x1x1x16x16xf32>
  // CHECK: %[[BL2:.*]] = rock.blockwise_load %[[TILE2]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf32>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[BL1]], %[[BL2]] : tensor<16x16xf32>
  // CHECK-NOT: rock.load_marker
  // CHECK: %[[UT:.*]] = rock.untile %[[TILE_ADD]]
  // CHECK: %[[FUSED:.*]] = arith.addf %[[SM]], %[[UT]]
  // CHECK: rock.store %[[FUSED]]
  func.func @test_complex_transforms(
      %tile: tensor<16x16xf32>,
      %raw1: tensor<16xf32>,
      %raw2: tensor<1x16x16xf32>,
      %dest: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<1x16x16xf32> attributes {rock.kernel} {
    %sm = rock.store_marker %tile views [#tmap] [%g, %m, %n] : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %bc = rock.transform %raw1 by #tmap_broadcast : tensor<16xf32> to tensor<1x16x16xf32>
    %lm1 = rock.load_marker %bc views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %tr = rock.transform %raw2 by #tmap_transpose : tensor<1x16x16xf32> to tensor<1x16x16xf32>
    %lm2 = rock.load_marker %tr views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    %tile_add = arith.addf %lm1, %lm2 : tensor<16x16xf32>
    %ut = rock.untile %tile_add : tensor<16x16xf32> -> tensor<1x16x16xf32>
    %add = arith.addf %sm, %ut : tensor<1x16x16xf32>
    %r = rock.store %add to %dest by set : tensor<1x16x16xf32> -> tensor<1x16x16xf32> to tensor<1x16x16xf32>
    return %r : tensor<1x16x16xf32>
  }

  // ============================================================
  // Input fusion (no store_marker): two load_markers + tile addf.
  // Result returned directly (feeds e.g. blockwise_gemm).
  // ============================================================

  // CHECK-LABEL: func.func @test_input_fusion_simple
  // CHECK: %[[T1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL1:.*]] = rock.blockwise_load %[[T1]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf16>
  // CHECK: %[[T2:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL2:.*]] = rock.blockwise_load %[[T2]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf16>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[BL1]], %[[BL2]] : tensor<16x16xf16>
  // CHECK-NOT: rock.load_marker
  // CHECK: return %[[TILE_ADD]]
  func.func @test_input_fusion_simple(
      %A: tensor<1x16x16xf16>, %B: tensor<1x16x16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16> attributes {rock.kernel} {
    %lm1 = rock.load_marker %A views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %lm2 = rock.load_marker %B views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %add = arith.addf %lm1, %lm2 : tensor<16x16xf16>
    return %add : tensor<16x16xf16>
  }

  // ============================================================
  // Input fusion with pre-existing transform chains:
  // broadcast(A) and broadcast(B) feed load_markers.
  // ============================================================

  // CHECK-LABEL: func.func @test_input_fusion_with_transforms
  // CHECK: %[[BC1:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[TILE1:.*]] = rock.transform %[[BC1]] by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL1:.*]] = rock.blockwise_load %[[TILE1]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf16>
  // CHECK: %[[BC2:.*]] = rock.transform %{{.*}} by
  // CHECK-SAME: tensor<16xf16> to tensor<1x16x16xf16>
  // CHECK: %[[TILE2:.*]] = rock.transform %[[BC2]] by
  // CHECK-SAME: tensor<1x16x16xf16> to tensor<1x1x1x16x16xf16>
  // CHECK: %[[BL2:.*]] = rock.blockwise_load %[[TILE2]][%{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK-SAME: -> tensor<16x16xf16>
  // CHECK: %[[TILE_ADD:.*]] = arith.addf %[[BL1]], %[[BL2]] : tensor<16x16xf16>
  // CHECK-NOT: rock.load_marker
  // CHECK: return %[[TILE_ADD]]
  func.func @test_input_fusion_with_transforms(
      %A_raw: tensor<16xf16>, %B_raw: tensor<16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16> attributes {rock.kernel} {
    %A = rock.transform %A_raw by #tmap_broadcast : tensor<16xf16> to tensor<1x16x16xf16>
    %lm1 = rock.load_marker %A views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %B = rock.transform %B_raw by #tmap_broadcast : tensor<16xf16> to tensor<1x16x16xf16>
    %lm2 = rock.load_marker %B views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    %add = arith.addf %lm1, %lm2 : tensor<16x16xf16>
    return %add : tensor<16x16xf16>
  }

  // ============================================================
  // Non-kernel function: load_marker is NOT replaced because
  // the pass only operates on rock.kernel functions.
  // ============================================================

  // CHECK-LABEL: func.func @test_non_kernel
  // CHECK: rock.load_marker
  // CHECK-NOT: rock.blockwise_load
  func.func @test_non_kernel(
      %bias: tensor<1x16x16xf32>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf32> {
    %lm = rock.load_marker %bias views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier none>} : tensor<1x16x16xf32> -> tensor<16x16xf32>
    return %lm : tensor<16x16xf32>
  }

  // ============================================================
  // Cache modifier propagation: a load_marker carrying a non-default
  // cache modifier (cs) must produce a blockwise_load with the same
  // cache modifier.
  // ============================================================

  // CHECK-LABEL: func.func @test_cache_modifier_propagation
  // CHECK: %[[BL:.*]] = rock.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {cacheModifier = #rock<CacheModifier cs>}
  // CHECK-SAME: tensor<1x1x1x16x16xf16> -> tensor<16x16xf16>
  // CHECK-NOT: rock.load_marker
  func.func @test_cache_modifier_propagation(
      %A: tensor<1x16x16xf16>,
      %g: i32, %m: i32, %n: i32) -> tensor<16x16xf16> attributes {rock.kernel} {
    %lm = rock.load_marker %A views [#tmap] [%g, %m, %n] {cacheModifier = #rock<CacheModifier cs>} : tensor<1x16x16xf16> -> tensor<16x16xf16>
    return %lm : tensor<16x16xf16>
  }
}
