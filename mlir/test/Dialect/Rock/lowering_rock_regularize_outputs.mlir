// Unit tests for rock-regularize-output pass.
// Tests output fusions through various invertible transforms for gemm, conv,
// and attention FusionRoot ops.

// RUN: rocmlir-opt -rock-lower-reduce -rock-regularize-output -mlir-print-local-scope %s | FileCheck %s

// --- Common gemm flat-to-3D transforms ---
#map_flat = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#tf_a = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#tf_b = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#tf_c = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>

// --- Unmerge n: [1,100,100] -> [1,100,10,10] ---
#map_unmerge_n = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 10 + d3)>
#tf_unmerge_n = #rock.transform_map<#map_unmerge_n by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Unmerge{10, 10} ["n0", "n1"] at [2, 3] -> ["n"] at [2]>] bounds = [1, 100, 10, 10] -> [1, 100, 100]>

// --- Merge n0,n1: [1,100,10,10] -> [1,100,100] ---
#map_merge_n = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 10, d2 mod 10)>
#tf_merge_n = #rock.transform_map<#map_merge_n by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Merge{10, 10} ["n"] at [2] -> ["n0", "n1"] at [2, 3]>] bounds = [1, 100, 100] -> [1, 100, 10, 10]>

// --- Flat -> 4D [1,100,10,10] ---
#map_flat_4d = affine_map<(d0, d1, d2, d3) -> ((d1 * 10 + d2) * 10 + d3)>
#tf_flat_4d = #rock.transform_map<#map_flat_4d by [<Unmerge{100, 10, 10} ["m", "n0", "n1"] at [1, 2, 3] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 10, 10] -> [10000]>

// --- Merge m,n: [1,100,100] -> [1,10000] ---
#map_merge_mn = affine_map<(d0, d1) -> (d0, d1 floordiv 100, d1 mod 100)>
#tf_merge_mn = #rock.transform_map<#map_merge_mn by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Merge{100, 100} ["mn"] at [1] -> ["m", "n"] at [1, 2]>] bounds = [1, 10000] -> [1, 100, 100]>

// --- Flat -> 2D [1,10000] ---
#map_flat_2d = affine_map<(d0, d1) -> (d1)>
#tf_flat_2d = #rock.transform_map<#map_flat_2d by [<Unmerge{10000} ["mn"] at [1] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 10000] -> [10000]>

// --- AddDim{1}: [1,100,100] -> [1,100,100,1] ---
#map_adddim = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#tf_adddim = #rock.transform_map<#map_adddim by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>, <AddDim{1} ["extra"] at [3] -> [] at []>] bounds = [1, 100, 100, 1] -> [1, 100, 100]>

// --- Conv transforms ---
#map_conv_fil = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 3 + d2) * 3 + d3) * 8 + d4)>
#map_conv_in = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 8 + d2) * 8 + d3) * 8 + d4)>
#map_conv_out = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)>
#tf_fil = #rock.transform_map<#map_conv_fil by [<Unmerge{4, 3, 3, 8} ["k", "0", "1", "c"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4, 3, 3, 8] -> [288]>
#tf_in = #rock.transform_map<#map_conv_in by [<Unmerge{2, 8, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [2, 1, 8, 8, 8] -> [1024]>
#tf_out = #rock.transform_map<#map_conv_out by [<Unmerge{2, 4, 8, 8} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]>

// --- Conv Unmerge 1o: [2,1,4,8,8] -> [2,1,4,8,4,2] ---
#map_conv_unmerge = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 * 2 + d5)>
#tf_conv_unmerge = #rock.transform_map<#map_conv_unmerge by [<PassThrough ["no"] at [0] -> ["no"] at [0]>, <PassThrough ["go"] at [1] -> ["go"] at [1]>, <PassThrough ["ko"] at [2] -> ["ko"] at [2]>, <PassThrough ["0o"] at [3] -> ["0o"] at [3]>, <Unmerge{4, 2} ["1oa", "1ob"] at [4, 5] -> ["1o"] at [4]>] bounds = [2, 1, 4, 8, 4, 2] -> [2, 1, 4, 8, 8]>

// --- Attention transforms ---
#map_attn_q = affine_map<(d0, d1, d2) -> (d1 * 32 + d2)>
#map_attn_k = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#tf_attn_q = #rock.transform_map<#map_attn_q by [<Unmerge{64, 32} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 32] -> [2048]>
#tf_attn_k = #rock.transform_map<#map_attn_k by [<Unmerge{32, 64} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 64] -> [2048]>
#tf_attn_v = #rock.transform_map<#map_attn_q by [<Unmerge{64, 32} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 32] -> [2048]>
#tf_attn_o = #rock.transform_map<#map_attn_q by [<Unmerge{64, 32} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 32] -> [2048]>

// --- Attention Unmerge seq_q: [1,64,32] -> [1,8,8,32] ---
#map_attn_unmerge = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 8 + d2, d3)>
#tf_attn_unmerge = #rock.transform_map<#map_attn_unmerge by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Unmerge{8, 8} ["sq0", "sq1"] at [1, 2] -> ["seq_q"] at [1]>, <PassThrough ["head_v"] at [3] -> ["head_v"] at [2]>] bounds = [1, 8, 8, 32] -> [1, 64, 32]>

module {

  // ============================================================
  // NO-OP: Output fusion without transforms (pass does nothing)
  // ============================================================

  // CHECK-LABEL: func.func @test_noop
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x100x100xf16>
  // CHECK: rock.store %[[F]] to %{{.*}} by {{.*}}set : tensor<1x100x100xf16>
  func.func @test_noop(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %arg2: tensor<10000xf16>, %arg3: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.kernel} {
    %0 = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %1 = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %2 = rock.transform %arg2 by #tf_c : tensor<10000xf16> to tensor<1x100x100xf16>
    %3 = rock.transform %arg3 by #tf_c : tensor<10000xf16> to tensor<1x100x100xf16>
    %4 = rock.gemm %0 * %1 : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %5 = arith.addf %4, %2 : tensor<1x100x100xf16>
    %6 = rock.store %5 to %3 by set : tensor<1x100x100xf16> -> tensor<10000xf16> to tensor<1x100x100xf16>
    return %6 : tensor<10000xf16>
  }

  // ============================================================
  // GEMM + Unmerge: addf with external tensor through Unmerge
  // Input:  arith.addf in tensor<1x100x10x10xf16>
  // Expect: arith.addf in tensor<1x100x100xf16> using gemm directly
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_unmerge
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x100x100xf16>
  // CHECK: rock.store %[[F]] to %{{.*}} {{.*}}: tensor<1x100x100xf16>
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x10x10xf16>
  func.func @test_gemm_unmerge(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %ext: tensor<10000xf16>, %dest: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.kernel} {
    %a = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_4d = rock.transform %gemm by #tf_unmerge_n : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    %ext_4d = rock.transform %ext by #tf_flat_4d : tensor<10000xf16> to tensor<1x100x10x10xf16>
    %fused = arith.addf %gemm_4d, %ext_4d : tensor<1x100x10x10xf16>
    %d = rock.transform %dest by #tf_flat_4d : tensor<10000xf16> to tensor<1x100x10x10xf16>
    %r = rock.store %fused to %d by set : tensor<1x100x10x10xf16> -> tensor<10000xf16> to tensor<1x100x10x10xf16>
    return %r : tensor<10000xf16>
  }

  // ============================================================
  // GEMM + Merge: addf with external tensor through Merge(m,n)
  // Input:  arith.addf in tensor<1x10000xf16>
  // Expect: arith.addf in tensor<1x100x100xf16> using gemm directly
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_merge
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x10000xf16>
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x100x100xf16>
  // CHECK: rock.store %[[F]] to %{{.*}} {{.*}}: tensor<1x100x100xf16>
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x10000xf16>
  func.func @test_gemm_merge(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %ext: tensor<10000xf16>, %dest: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.kernel} {
    %a = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_2d = rock.transform %gemm by #tf_merge_mn : tensor<1x100x100xf16> to tensor<1x10000xf16>
    %ext_2d = rock.transform %ext by #tf_flat_2d : tensor<10000xf16> to tensor<1x10000xf16>
    %fused = arith.addf %gemm_2d, %ext_2d : tensor<1x10000xf16>
    %d = rock.transform %dest by #tf_flat_2d : tensor<10000xf16> to tensor<1x10000xf16>
    %r = rock.store %fused to %d by set : tensor<1x10000xf16> -> tensor<10000xf16> to tensor<1x10000xf16>
    return %r : tensor<10000xf16>
  }

  // ============================================================
  // GEMM + AddDim{1}: addf with external tensor through AddDim
  // Input:  arith.addf in tensor<1x100x100x1xf16>
  // Expect: arith.addf in tensor<1x100x100xf16> using gemm directly
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_adddim
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x100x1xf16>
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x100x100xf16>
  // CHECK: rock.store %[[F]] to %{{.*}} {{.*}}: tensor<1x100x100xf16>
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x100x1xf16>
  func.func @test_gemm_adddim(%arg0: tensor<1x100x100xf16>, %arg1: tensor<1x100x100xf16>, %ext: tensor<1x100x100x1xf16>, %dest: tensor<1x100x100x1xf16>) -> tensor<1x100x100x1xf16> attributes {rock.kernel} {
    %gemm = rock.gemm %arg0 * %arg1 : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_4d = rock.transform %gemm by #tf_adddim : tensor<1x100x100xf16> to tensor<1x100x100x1xf16>
    %fused = arith.addf %gemm_4d, %ext : tensor<1x100x100x1xf16>
    %r = rock.store %fused to %dest by set : tensor<1x100x100x1xf16> -> tensor<1x100x100x1xf16> to tensor<1x100x100x1xf16>
    return %r : tensor<1x100x100x1xf16>
  }

  // ============================================================
  // GEMM + Unmerge: subf with splat constant (constant recreation)
  // Input:  arith.subf in tensor<1x100x10x10xf16>
  // Expect: arith.subf in tensor<1x100x100xf16> using gemm directly
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_subf_const_unmerge
  // CHECK-NOT: arith.subf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.subf %[[G]], %{{.*}} : tensor<1x100x100xf16>
  // CHECK: rock.store %[[F]] to %{{.*}} {{.*}}: tensor<1x100x100xf16>
  // CHECK-NOT: arith.subf {{.*}} : tensor<1x100x10x10xf16>
  func.func @test_gemm_subf_const_unmerge(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %dest: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.kernel} {
    %a = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_4d = rock.transform %gemm by #tf_unmerge_n : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    %cst = arith.constant dense<1.000000e+00> : tensor<1x100x10x10xf16>
    %fused = arith.subf %gemm_4d, %cst : tensor<1x100x10x10xf16>
    %d = rock.transform %dest by #tf_flat_4d : tensor<10000xf16> to tensor<1x100x10x10xf16>
    %r = rock.store %fused to %d by set : tensor<1x100x10x10xf16> -> tensor<10000xf16> to tensor<1x100x10x10xf16>
    return %r : tensor<10000xf16>
  }

  // ============================================================
  // GEMM + Unmerge: extf type change through transform
  // Input:  arith.extf in tensor<1x100x10x10xf16> to tensor<1x100x10x10xf32>
  // Expect: arith.extf in tensor<1x100x100xf16> to tensor<1x100x100xf32>
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_extf_unmerge
  // CHECK-NOT: arith.extf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[E:.*]] = arith.extf %[[G]] : tensor<1x100x100xf16> to tensor<1x100x100xf32>
  // CHECK: rock.store %[[E]] to %{{.*}} {{.*}}: tensor<1x100x100xf32>
  // CHECK-NOT: arith.extf {{.*}} : tensor<1x100x10x10xf16>
  func.func @test_gemm_extf_unmerge(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %dest: tensor<10000xf32>) -> tensor<10000xf32> attributes {rock.kernel} {
    %a = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_4d = rock.transform %gemm by #tf_unmerge_n : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    %ext = arith.extf %gemm_4d : tensor<1x100x10x10xf16> to tensor<1x100x10x10xf32>
    %d = rock.transform %dest by #tf_flat_4d : tensor<10000xf32> to tensor<1x100x10x10xf32>
    %r = rock.store %ext to %d by set : tensor<1x100x10x10xf32> -> tensor<10000xf32> to tensor<1x100x10x10xf32>
    return %r : tensor<10000xf32>
  }

  // ============================================================
  // GEMM + Unmerge: cascaded addf + extf through transform
  // Input:  arith.addf/extf in tensor<1x100x10x10x...>
  // Expect: both in tensor<1x100x100x...> using gemm directly
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_cascaded_unmerge
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK-NOT: arith.extf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[ADD:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x100x100xf16>
  // CHECK: %[[EXT:.*]] = arith.extf %[[ADD]] : tensor<1x100x100xf16> to tensor<1x100x100xf32>
  // CHECK: rock.store %[[EXT]] to %{{.*}} {{.*}}: tensor<1x100x100xf32>
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x10x10xf16>
  func.func @test_gemm_cascaded_unmerge(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %bias: tensor<10000xf16>, %dest: tensor<10000xf32>) -> tensor<10000xf32> attributes {rock.kernel} {
    %a = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_4d = rock.transform %gemm by #tf_unmerge_n : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    %bias_4d = rock.transform %bias by #tf_flat_4d : tensor<10000xf16> to tensor<1x100x10x10xf16>
    %fused = arith.addf %gemm_4d, %bias_4d : tensor<1x100x10x10xf16>
    %fused_f32 = arith.extf %fused : tensor<1x100x10x10xf16> to tensor<1x100x10x10xf32>
    %d = rock.transform %dest by #tf_flat_4d : tensor<10000xf32> to tensor<1x100x10x10xf32>
    %r = rock.store %fused_f32 to %d by set : tensor<1x100x10x10xf32> -> tensor<10000xf32> to tensor<1x100x10x10xf32>
    return %r : tensor<10000xf32>
  }

  // ============================================================
  // GEMM + Unmerge/Merge chain: interleaved transforms + fusions
  // Input:  first addf in 4D, second addf in 3D (already gemm space)
  // Expect: both addf in 3D gemm space, first uses gemm directly
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_interleaved
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK: %[[G:.*]] = rock.gemm
  // First addf: was in 4D, now uses gemm directly in 3D
  // CHECK: %[[F1:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x100x100xf16>
  // Second addf: was already 3D but now chains from regularized first
  // CHECK: %[[F2:.*]] = arith.addf %[[F1]], %{{.*}} : tensor<1x100x100xf16>
  // CHECK: rock.store %[[F2]] to %{{.*}} {{.*}}: tensor<1x100x100xf16>
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x10x10xf16>
  func.func @test_gemm_interleaved(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %ext1: tensor<10000xf16>, %ext2: tensor<10000xf16>, %dest: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.kernel} {
    %a = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_4d = rock.transform %gemm by #tf_unmerge_n : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    %e1 = rock.transform %ext1 by #tf_flat_4d : tensor<10000xf16> to tensor<1x100x10x10xf16>
    %f1 = arith.addf %gemm_4d, %e1 : tensor<1x100x10x10xf16>
    %f1_3d = rock.transform %f1 by #tf_merge_n : tensor<1x100x10x10xf16> to tensor<1x100x100xf16>
    %e2 = rock.transform %ext2 by #tf_c : tensor<10000xf16> to tensor<1x100x100xf16>
    %f2 = arith.addf %f1_3d, %e2 : tensor<1x100x100xf16>
    %d = rock.transform %dest by #tf_c : tensor<10000xf16> to tensor<1x100x100xf16>
    %r = rock.store %f2 to %d by set : tensor<1x100x100xf16> -> tensor<10000xf16> to tensor<1x100x100xf16>
    return %r : tensor<10000xf16>
  }

  // ============================================================
  // GEMM + self-referential DAG: gemm used multiple times
  // Input:  exp/mulf in 4D, final addf mixes 3D and 4D
  // Expect: all fusion ops in 3D gemm space, exact sequence
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_self_ref
  // CHECK-NOT: math.exp {{.*}} : tensor<1x100x10x10xf16>
  // CHECK-NOT: arith.mulf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK-NEXT: %[[EXP:.*]] = math.exp %[[G]] : tensor<1x100x100xf16>
  // CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[G]], %[[EXP]] : tensor<1x100x100xf16>
  // CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[MUL]], %[[G]] : tensor<1x100x100xf16>
  // CHECK: rock.store %[[ADD]] to %{{.*}} {{.*}}: tensor<1x100x100xf16>
  // CHECK-NOT: math.exp {{.*}} : tensor<1x100x10x10xf16>
  func.func @test_gemm_self_ref(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %dest: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.kernel} {
    %a = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %g4d = rock.transform %gemm by #tf_unmerge_n : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    %exp = math.exp %g4d : tensor<1x100x10x10xf16>
    %mul = arith.mulf %g4d, %exp : tensor<1x100x10x10xf16>
    %mul3d = rock.transform %mul by #tf_merge_n : tensor<1x100x10x10xf16> to tensor<1x100x100xf16>
    %add = arith.addf %mul3d, %gemm : tensor<1x100x100xf16>
    %d = rock.transform %dest by #tf_c : tensor<10000xf16> to tensor<1x100x100xf16>
    %r = rock.store %add to %d by set : tensor<1x100x100xf16> -> tensor<10000xf16> to tensor<1x100x100xf16>
    return %r : tensor<10000xf16>
  }

  // ============================================================
  // CONV + Unmerge: addf with external tensor through Unmerge(1o)
  // Input:  arith.addf in tensor<2x1x4x8x4x2xf16> (6D)
  // Expect: arith.addf in tensor<2x1x4x8x8xf16> (conv space)
  // ============================================================

  // CHECK-LABEL: func.func @test_conv_unmerge
  // CHECK-NOT: arith.addf {{.*}} : tensor<2x1x4x8x4x2xf16>
  // CHECK: %[[C:.*]] = rock.conv
  // CHECK: %[[F:.*]] = arith.addf %[[C]], %{{.*}} : tensor<2x1x4x8x8xf16>
  // CHECK: rock.store %[[F]] to %{{.*}} {{.*}}: tensor<2x1x4x8x8xf16>
  // CHECK-NOT: arith.addf {{.*}} : tensor<2x1x4x8x4x2xf16>
  func.func @test_conv_unmerge(%fil: tensor<288xf16>, %input: tensor<1024xf16>, %ext: tensor<2x1x4x8x4x2xf16>, %dest: tensor<2x1x4x8x4x2xf16>) -> tensor<2x1x4x8x4x2xf16> attributes {rock.kernel} {
    %f = rock.transform %fil by #tf_fil : tensor<288xf16> to tensor<1x4x3x3x8xf16>
    %i = rock.transform %input by #tf_in : tensor<1024xf16> to tensor<2x1x8x8x8xf16>
    %conv = rock.conv(%f, %i) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : tensor<1x4x3x3x8xf16>, tensor<2x1x8x8x8xf16> -> tensor<2x1x4x8x8xf16>
    %conv_6d = rock.transform %conv by #tf_conv_unmerge : tensor<2x1x4x8x8xf16> to tensor<2x1x4x8x4x2xf16>
    %fused = arith.addf %conv_6d, %ext : tensor<2x1x4x8x4x2xf16>
    %r = rock.store %fused to %dest by set : tensor<2x1x4x8x4x2xf16> -> tensor<2x1x4x8x4x2xf16> to tensor<2x1x4x8x4x2xf16>
    return %r : tensor<2x1x4x8x4x2xf16>
  }

  // ============================================================
  // ATTENTION + Unmerge: addf with external tensor through Unmerge(seq_q)
  // Input:  arith.addf in tensor<1x8x8x32xf16> (4D)
  // Expect: arith.addf in tensor<1x64x32xf16> (attention space)
  // ============================================================

  // CHECK-LABEL: func.func @test_attention_unmerge
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x8x8x32xf16>
  // CHECK: %[[A:.*]] = rock.attention
  // CHECK: %[[F:.*]] = arith.addf %[[A]], %{{.*}} : tensor<1x64x32xf16>
  // CHECK: rock.store %[[F]] to %{{.*}} {{.*}}: tensor<1x64x32xf16>
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x8x8x32xf16>
  func.func @test_attention_unmerge(%q_raw: tensor<2048xf16>, %k_raw: tensor<2048xf16>, %v_raw: tensor<2048xf16>, %ext: tensor<1x8x8x32xf16>, %dest: tensor<1x8x8x32xf16>) -> tensor<1x8x8x32xf16> attributes {rock.kernel} {
    %q = rock.transform %q_raw by #tf_attn_q : tensor<2048xf16> to tensor<1x64x32xf16>
    %k = rock.transform %k_raw by #tf_attn_k : tensor<2048xf16> to tensor<1x32x64xf16>
    %v = rock.transform %v_raw by #tf_attn_v : tensor<2048xf16> to tensor<1x64x32xf16>
    %attn = rock.attention{
      qk = %q * %k : tensor<1x64x32xf16>, tensor<1x32x64xf16>
      qk = elementwise {
      ^bb0(%qk: tensor<1x64x64xf16>):
        rock.yield %qk : tensor<1x64x64xf16>
      }
      softmax(qk) * %v : tensor<1x64x32xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32} -> tensor<1x64x32xf16>
    %attn_4d = rock.transform %attn by #tf_attn_unmerge : tensor<1x64x32xf16> to tensor<1x8x8x32xf16>
    %fused = arith.addf %attn_4d, %ext : tensor<1x8x8x32xf16>
    %r = rock.store %fused to %dest by set : tensor<1x8x8x32xf16> -> tensor<1x8x8x32xf16> to tensor<1x8x8x32xf16>
    return %r : tensor<1x8x8x32xf16>
  }

  // ============================================================
  // GEMM + Unmerge + reduce: rock-lower-reduce converts the
  // reduce to broadcast+atomic_add first, then regularize-output
  // moves the fusion to gemm space.
  // Input:  arith.addf in tensor<1x100x10x10xf16> + rock.reduce
  // Expect: arith.addf in tensor<1x100x100xf16>, no reduce, atomic_add
  // ============================================================

  // CHECK-LABEL: func.func @test_gemm_reduce_unmerge
  // CHECK-SAME: rock.prefill
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK-NOT: rock.reduce
  // CHECK: %[[G:.*]] = rock.gemm
  // CHECK: %[[F:.*]] = arith.addf %[[G]], %{{.*}} : tensor<1x100x100xf16>
  // CHECK: rock.store %[[F]] to %{{.*}} {{.*}}atomic_add : tensor<1x100x100xf16>
  // CHECK-NOT: arith.addf {{.*}} : tensor<1x100x10x10xf16>
  // CHECK-NOT: rock.reduce
  func.func @test_gemm_reduce_unmerge(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %ext: tensor<10000xf16>, %dest: tensor<1x100x10x1xf16>) -> tensor<1x100x10x1xf16> attributes {rock.kernel} {
    %a = rock.transform %arg0 by #tf_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg1 by #tf_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %g4d = rock.transform %gemm by #tf_unmerge_n : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    %ext4d = rock.transform %ext by #tf_flat_4d : tensor<10000xf16> to tensor<1x100x10x10xf16>
    %fused = arith.addf %g4d, %ext4d : tensor<1x100x10x10xf16>
    %reduced = rock.reduce sum %fused {axis = 3 : index} : tensor<1x100x10x10xf16> -> tensor<1x100x10x1xf16>
    %r = rock.store %reduced to %dest by set : tensor<1x100x10x1xf16> -> tensor<1x100x10x1xf16> to tensor<1x100x10x1xf16>
    return %r : tensor<1x100x10x1xf16>
  }
}
