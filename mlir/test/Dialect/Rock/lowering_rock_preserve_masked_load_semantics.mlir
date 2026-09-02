// RUN: rocmlir-opt -rock-preserve-masked-load-semantics -mlir-print-local-scope %s | FileCheck %s

// ============================================================
// addf with constant: non-zero-preserving (0 + 1 = 1).
// Should insert arith.select to re-zero OOB positions.
// ============================================================

// CHECK-LABEL: func.func @test_addf_constant
// CHECK-SAME: (%[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>,
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %[[PTRS]][%[[MASK]]]
// CHECK: %[[FUSED:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf16>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[ZERO]] : tensor<64x64xi1>, tensor<64x64xf16>
// CHECK: rock.blockwise_store_ptr %[[SAFE]]
func.func @test_addf_constant(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst = arith.constant dense<1.0> : tensor<64x64xf16>
  %fused = arith.addf %tile, %cst : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// A narrowed load is expanded and broadcast before a non-zero-preserving
// fusion. This is the shape produced by input broadcast narrowing. The
// validity mask must follow the same shape operations so padded reduction
// lanes are restored to zero after the fusion.
// ============================================================

// CHECK-LABEL: func.func @test_narrowed_broadcast_exp
// CHECK-SAME: (%[[PTRS:.*]]: tensor<64xi32>, %[[MASK:.*]]: tensor<64xi1>,
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %[[PTRS]][%[[MASK]]]
// CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[LOAD]] {axis = 1 : i32}
// CHECK: %[[MASK_EXPANDED:.*]] = tt.expand_dims %[[MASK]] {axis = 1 : i32}
// CHECK-SAME: tensor<64xi1> -> tensor<64x1xi1>
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[EXPANDED]]
// CHECK: %[[MASK_BROADCAST:.*]] = tt.broadcast %[[MASK_EXPANDED]]
// CHECK-SAME: tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK: %[[FUSED:.*]] = math.exp %[[BROADCAST]] : tensor<64x64xf16>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK_BROADCAST]], %[[FUSED]], %[[ZERO]]
// CHECK-SAME: tensor<64x64xi1>, tensor<64x64xf16>
// CHECK: rock.blockwise_store_ptr %[[SAFE]]
func.func @test_narrowed_broadcast_exp(
    %ptrs: tensor<64xi32>, %mask: tensor<64xi1>,
    %dest_ptrs: tensor<64x64xi32>,
    %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64xi32>, tensor<64xi1> -> tensor<64xf16>
  %expanded = tt.expand_dims %tile {axis = 1 : i32} : tensor<64xf16> -> tensor<64x1xf16>
  %broadcast = tt.broadcast %expanded : tensor<64x1xf16> -> tensor<64x64xf16>
  %fused = math.exp %broadcast : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// A zero-preserving fusion after a narrowed broadcast needs no re-masking.
// Analysis must not leave dead i1 expand/broadcast clones behind.
// ============================================================

// CHECK-LABEL: func.func @test_narrowed_broadcast_mulf
// CHECK-SAME: (%[[PTRS:.*]]: tensor<64xi32>, %[[MASK:.*]]: tensor<64xi1>,
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %[[PTRS]][%[[MASK]]]
// CHECK: %[[EXPANDED:.*]] = tt.expand_dims %[[LOAD]] {axis = 1 : i32}
// CHECK-NOT: tt.expand_dims %[[MASK]]
// CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[EXPANDED]]
// CHECK-NOT: tt.broadcast {{.*}} : tensor<64x1xi1> -> tensor<64x64xi1>
// CHECK: %[[FUSED:.*]] = arith.mulf %[[BROADCAST]], %{{.*}}
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[FUSED]]
func.func @test_narrowed_broadcast_mulf(
    %ptrs: tensor<64xi32>, %mask: tensor<64xi1>,
    %dest_ptrs: tensor<64x64xi32>,
    %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64xi32>, tensor<64xi1> -> tensor<64xf16>
  %expanded = tt.expand_dims %tile {axis = 1 : i32} : tensor<64xf16> -> tensor<64x1xf16>
  %broadcast = tt.broadcast %expanded : tensor<64x1xf16> -> tensor<64x64xf16>
  %cst = arith.constant dense<2.0> : tensor<64x64xf16>
  %fused = arith.mulf %broadcast, %cst : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// mulf with constant: zero-preserving (0 * 2 = 0).
// No arith.select should be inserted.
// ============================================================

// CHECK-LABEL: func.func @test_mulf_constant
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr
// CHECK: %[[FUSED:.*]] = arith.mulf %[[LOAD]], %{{.*}} : tensor<64x64xf16>
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[FUSED]]
func.func @test_mulf_constant(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst = arith.constant dense<2.0> : tensor<64x64xf16>
  %fused = arith.mulf %tile, %cst : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// Trivial mask (constant splat true): no Pad/Embed OOB concern.
// No arith.select should be inserted regardless of fusion type.
// ============================================================

// CHECK-LABEL: func.func @test_trivial_mask
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr
// CHECK: %[[FUSED:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf16>
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[FUSED]]
func.func @test_trivial_mask(
    %ptrs: tensor<64x64xi32>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %trivial_mask = arith.constant dense<true> : tensor<64x64xi1>
  %tile = rock.blockwise_load_ptr %ptrs[%trivial_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst = arith.constant dense<1.0> : tensor<64x64xf16>
  %fused = arith.addf %tile, %cst : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// No fusion: load goes directly to store.
// No arith.select should be inserted (no fusion chain leaves).
// ============================================================

// CHECK-LABEL: func.func @test_no_fusion
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[LOAD]]
func.func @test_no_fusion(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  rock.blockwise_store_ptr %tile -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// Fusion chain: mulf (zero-preserving) then addf constant
// (non-zero-preserving). Overall: 0*2+1=1, not zero-preserving.
// Should insert arith.select after the chain leaf (addf result).
// ============================================================

// CHECK-LABEL: func.func @test_chain_mulf_addf
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[MUL:.*]] = arith.mulf %[[LOAD]], %{{.*}} : tensor<64x64xf16>
// CHECK: %[[ADD:.*]] = arith.addf %[[MUL]], %{{.*}} : tensor<64x64xf16>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[ADD]], %[[ZERO]] : tensor<64x64xi1>, tensor<64x64xf16>
// CHECK: rock.blockwise_store_ptr %[[SAFE]]
func.func @test_chain_mulf_addf(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst2 = arith.constant dense<2.0> : tensor<64x64xf16>
  %cst1 = arith.constant dense<1.0> : tensor<64x64xf16>
  %step1 = arith.mulf %tile, %cst2 : tensor<64x64xf16>
  %step2 = arith.addf %step1, %cst1 : tensor<64x64xf16>
  rock.blockwise_store_ptr %step2 -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// Fusion chain: mulf then mulf (both zero-preserving).
// Overall: 0*2*3=0, zero-preserving.
// No arith.select should be inserted.
// ============================================================

// CHECK-LABEL: func.func @test_chain_mulf_mulf
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr
// CHECK: arith.mulf
// CHECK: %[[MUL2:.*]] = arith.mulf
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[MUL2]]
func.func @test_chain_mulf_mulf(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst2 = arith.constant dense<2.0> : tensor<64x64xf16>
  %cst3 = arith.constant dense<3.0> : tensor<64x64xf16>
  %step1 = arith.mulf %tile, %cst2 : tensor<64x64xf16>
  %step2 = arith.mulf %step1, %cst3 : tensor<64x64xf16>
  rock.blockwise_store_ptr %step2 -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// Two loads -> addf: zero-preserving (0+0=0).
// No arith.select should be inserted.
// ============================================================

// CHECK-LABEL: func.func @test_two_loads_addf
// CHECK: %[[A:.*]] = rock.blockwise_load_ptr
// CHECK: %[[B:.*]] = rock.blockwise_load_ptr
// CHECK: %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : tensor<64x64xf16>
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[SUM]]
func.func @test_two_loads_addf(
    %ptrs_a: tensor<64x64xi32>, %mask_a: tensor<64x64xi1>,
    %ptrs_b: tensor<64x64xi32>, %mask_b: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %ptrs_a[%mask_a] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %b = rock.blockwise_load_ptr %ptrs_b[%mask_b] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %fused = arith.addf %a, %b : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// Two loads -> addf -> addf constant: non-zero-preserving
// (0+0+1=1). Masks from both loads are combined with arith.andi.
// ============================================================

// CHECK-LABEL: func.func @test_two_loads_then_const
// CHECK-SAME: (%{{.*}}: tensor<64x64xi32>, %[[MASK_A:.*]]: tensor<64x64xi1>, %{{.*}}: tensor<64x64xi32>, %[[MASK_B:.*]]: tensor<64x64xi1>,
// CHECK: rock.blockwise_load_ptr
// CHECK: rock.blockwise_load_ptr
// CHECK: arith.addf
// CHECK: %[[LEAF:.*]] = arith.addf {{.*}}, %{{.*}} : tensor<64x64xf16>
// CHECK: %[[COMBINED:.*]] = arith.andi %[[MASK_A]], %[[MASK_B]] : tensor<64x64xi1>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
// CHECK: %[[SAFE:.*]] = arith.select %[[COMBINED]], %[[LEAF]], %[[ZERO]] : tensor<64x64xi1>, tensor<64x64xf16>
// CHECK: rock.blockwise_store_ptr %[[SAFE]]
func.func @test_two_loads_then_const(
    %ptrs_a: tensor<64x64xi32>, %mask_a: tensor<64x64xi1>,
    %ptrs_b: tensor<64x64xi32>, %mask_b: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %ptrs_a[%mask_a] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %b = rock.blockwise_load_ptr %ptrs_b[%mask_b] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %sum = arith.addf %a, %b : tensor<64x64xf16>
  %cst = arith.constant dense<1.0> : tensor<64x64xf16>
  %fused = arith.addf %sum, %cst : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// math.exp: non-zero-preserving (exp(0)=1).
// Should insert arith.select.
// ============================================================

// CHECK-LABEL: func.func @test_math_exp
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[EXP:.*]] = math.exp %[[LOAD]] : tensor<64x64xf32>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[EXP]], %[[ZERO]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: rock.blockwise_store_ptr %[[SAFE]]
func.func @test_math_exp(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %fused = math.exp %tile : tensor<64x64xf32>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// arith.extf: zero-preserving (extf(0.0 f16) = 0.0 f32).
// No arith.select should be inserted.
// ============================================================

// CHECK-LABEL: func.func @test_extf
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr
// CHECK: %[[EXT:.*]] = arith.extf %[[LOAD]] : tensor<64x64xf16> to tensor<64x64xf32>
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[EXT]]
func.func @test_extf(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %fused = arith.extf %tile : tensor<64x64xf16> to tensor<64x64xf32>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// arith.negf: zero-preserving (negf(0) = -0 ≈ 0).
// No arith.select should be inserted.
// ============================================================

// CHECK-LABEL: func.func @test_negf
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr
// CHECK: %[[NEG:.*]] = arith.negf %[[LOAD]] : tensor<64x64xf16>
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[NEG]]
func.func @test_negf(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %fused = arith.negf %tile : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// ============================================================
// Max reduction after GEMM: a non-zero-preserving epilogue load fusion still
// needs remasking, but the neutral value for max is -inf rather than zero.
// ============================================================

// CHECK-LABEL: func.func @test_gemm_output_add_load_max_reduce_uses_neg_inf
// CHECK: %[[GEMM:.*]] = rock.blockwise_gemm
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf32>
// CHECK: %[[FUSED:.*]] = arith.addf %[[GEMM]], %[[ADD]] : tensor<64x64xf32>
// CHECK: %[[NEG_INF:.*]] = arith.constant dense<0xFF800000> : tensor<64x64xf32>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[NEG_INF]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce max %[[SAFE]]
// CHECK: rock.blockwise_store_ptr %[[REDUCED]]
func.func @test_gemm_output_add_load_max_reduce_uses_neg_inf(
    %a_ptrs: tensor<64x64xi32>, %a_mask: tensor<64x64xi1>,
    %b_ptrs: tensor<64x64xi32>, %b_mask: tensor<64x64xi1>,
    %d_ptrs: tensor<64x64xi32>, %d_mask: tensor<64x64xi1>,
    %out_ptrs: tensor<64xi32>, %out_mask: tensor<64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %zero = arith.constant dense<0.0> : tensor<64x64xf32>
  %c = rock.blockwise_gemm(%a, %b, %zero) : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  %d = rock.blockwise_load_ptr %d_ptrs[%d_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %cst = arith.constant dense<1.0> : tensor<64x64xf32>
  %d2 = arith.addf %d, %cst : tensor<64x64xf32>
  %res = arith.addf %c, %d2 : tensor<64x64xf32>
  %reduced = rock.blockwise_reduce max %res {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  rock.blockwise_store_ptr %reduced -> %out_ptrs(%out_mask) by set : tensor<64xf32> -> tensor<64xi32>(tensor<64xi1>)
  return
}

// ============================================================
// Zero-preserving epilogue fusion before max reduction: zero-preservation is
// enough for zero-fill consumers, but max still needs -inf for masked-out lanes.
// ============================================================

// CHECK-LABEL: func.func @test_gemm_output_zero_preserving_chain_max_reduce_uses_neg_inf
// CHECK: %[[GEMM:.*]] = rock.blockwise_gemm
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[SCALED:.*]] = arith.mulf %[[LOAD]], %{{.*}} : tensor<64x64xf32>
// CHECK: %[[FUSED:.*]] = arith.mulf %[[GEMM]], %[[SCALED]] : tensor<64x64xf32>
// CHECK: %[[NEG_INF:.*]] = arith.constant dense<0xFF800000> : tensor<64x64xf32>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[NEG_INF]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce max %[[SAFE]]
// CHECK: rock.blockwise_store_ptr %[[REDUCED]]
func.func @test_gemm_output_zero_preserving_chain_max_reduce_uses_neg_inf(
    %a_ptrs: tensor<64x64xi32>, %a_mask: tensor<64x64xi1>,
    %b_ptrs: tensor<64x64xi32>, %b_mask: tensor<64x64xi1>,
    %d_ptrs: tensor<64x64xi32>, %d_mask: tensor<64x64xi1>,
    %out_ptrs: tensor<64xi32>, %out_mask: tensor<64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %zero = arith.constant dense<0.0> : tensor<64x64xf32>
  %c = rock.blockwise_gemm(%a, %b, %zero) : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  %d = rock.blockwise_load_ptr %d_ptrs[%d_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %cst = arith.constant dense<2.0> : tensor<64x64xf32>
  %scaled = arith.mulf %d, %cst : tensor<64x64xf32>
  %res = arith.mulf %c, %scaled : tensor<64x64xf32>
  %reduced = rock.blockwise_reduce max %res {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  rock.blockwise_store_ptr %reduced -> %out_ptrs(%out_mask) by set : tensor<64xf32> -> tensor<64xi32>(tensor<64xi1>)
  return
}

// ============================================================
// Zero-preserving epilogue fusion before a view-like op and max reduction:
// the max consumer is not a direct user of the fusion leaf, but it still needs
// -inf for masked-out lanes.
// ============================================================

// CHECK-LABEL: func.func @test_gemm_output_zero_preserving_transform_max_reduce_uses_neg_inf
// CHECK: %[[GEMM:.*]] = rock.blockwise_gemm
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[SCALED:.*]] = arith.mulf %[[LOAD]], %{{.*}} : tensor<64x64xf32>
// CHECK: %[[FUSED:.*]] = arith.mulf %[[GEMM]], %[[SCALED]] : tensor<64x64xf32>
// CHECK: %[[NEG_INF:.*]] = arith.constant dense<0xFF800000> : tensor<64x64xf32>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[NEG_INF]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: %[[VIEW:.*]] = rock.transform %[[SAFE]]
// CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce max %[[VIEW]]
// CHECK: rock.blockwise_store_ptr %[[REDUCED]]
func.func @test_gemm_output_zero_preserving_transform_max_reduce_uses_neg_inf(
    %a_ptrs: tensor<64x64xi32>, %a_mask: tensor<64x64xi1>,
    %b_ptrs: tensor<64x64xi32>, %b_mask: tensor<64x64xi1>,
    %d_ptrs: tensor<64x64xi32>, %d_mask: tensor<64x64xi1>,
    %out_ptrs: tensor<64xi32>, %out_mask: tensor<64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %zero = arith.constant dense<0.0> : tensor<64x64xf32>
  %c = rock.blockwise_gemm(%a, %b, %zero) : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  %d = rock.blockwise_load_ptr %d_ptrs[%d_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %cst = arith.constant dense<2.0> : tensor<64x64xf32>
  %scaled = arith.mulf %d, %cst : tensor<64x64xf32>
  %res = arith.mulf %c, %scaled : tensor<64x64xf32>
  %view = rock.transform %res by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["dim0", "dim1"] at [0, 1] -> ["dim0", "dim1"] at [0, 1]>] bounds = [64, 64] -> [64, 64]> : tensor<64x64xf32> to tensor<64x64xf32>
  %reduced = rock.blockwise_reduce max %view {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  rock.blockwise_store_ptr %reduced -> %out_ptrs(%out_mask) by set : tensor<64xf32> -> tensor<64xi32>(tensor<64xi1>)
  return
}

// ============================================================
// Mixed consumers after GEMM: one epilogue fusion leaf feeds both a
// rock.blockwise_reduce max (needs -inf fill) and a rock.blockwise_store_ptr
// (needs zero fill). Each consumer must get its own arith.select with the
// right neutral value.
// ============================================================

// CHECK-LABEL: func.func @test_gemm_output_mixed_consumers_max_and_store
// CHECK: %[[GEMM:.*]] = rock.blockwise_gemm
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf32>
// CHECK: %[[FUSED:.*]] = arith.addf %[[GEMM]], %[[ADD]] : tensor<64x64xf32>
// CHECK: %[[NEG_INF:.*]] = arith.constant dense<0xFF800000> : tensor<64x64xf32>
// CHECK: %[[SAFE_MAX:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[NEG_INF]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce max %[[SAFE_MAX]]
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
// CHECK: %[[SAFE_STORE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[ZERO]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: rock.blockwise_store_ptr %[[SAFE_STORE]]
// CHECK: rock.blockwise_store_ptr %[[REDUCED]]
func.func @test_gemm_output_mixed_consumers_max_and_store(
    %a_ptrs: tensor<64x64xi32>, %a_mask: tensor<64x64xi1>,
    %b_ptrs: tensor<64x64xi32>, %b_mask: tensor<64x64xi1>,
    %d_ptrs: tensor<64x64xi32>, %d_mask: tensor<64x64xi1>,
    %out_ptrs: tensor<64xi32>, %out_mask: tensor<64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %zero = arith.constant dense<0.0> : tensor<64x64xf32>
  %c = rock.blockwise_gemm(%a, %b, %zero) : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  %d = rock.blockwise_load_ptr %d_ptrs[%d_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %cst = arith.constant dense<1.0> : tensor<64x64xf32>
  %d2 = arith.addf %d, %cst : tensor<64x64xf32>
  %res = arith.addf %c, %d2 : tensor<64x64xf32>
  %reduced = rock.blockwise_reduce max %res {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  rock.blockwise_store_ptr %res -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>)
  rock.blockwise_store_ptr %reduced -> %out_ptrs(%out_mask) by set : tensor<64xf32> -> tensor<64xi32>(tensor<64xi1>)
  return
}

// ============================================================
// Sum reduction after GEMM: zero is already the neutral element for sum, so
// the fill stays at zero and must NOT switch to -inf.
// ============================================================

// CHECK-LABEL: func.func @test_gemm_output_add_load_sum_reduce_uses_zero
// CHECK: %[[GEMM:.*]] = rock.blockwise_gemm
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf32>
// CHECK: %[[FUSED:.*]] = arith.addf %[[GEMM]], %[[ADD]] : tensor<64x64xf32>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[ZERO]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce sum %[[SAFE]]
// CHECK: rock.blockwise_store_ptr %[[REDUCED]]
func.func @test_gemm_output_add_load_sum_reduce_uses_zero(
    %a_ptrs: tensor<64x64xi32>, %a_mask: tensor<64x64xi1>,
    %b_ptrs: tensor<64x64xi32>, %b_mask: tensor<64x64xi1>,
    %d_ptrs: tensor<64x64xi32>, %d_mask: tensor<64x64xi1>,
    %out_ptrs: tensor<64xi32>, %out_mask: tensor<64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %zero = arith.constant dense<0.0> : tensor<64x64xf32>
  %c = rock.blockwise_gemm(%a, %b, %zero) : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  %d = rock.blockwise_load_ptr %d_ptrs[%d_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %cst = arith.constant dense<1.0> : tensor<64x64xf32>
  %d2 = arith.addf %d, %cst : tensor<64x64xf32>
  %res = arith.addf %c, %d2 : tensor<64x64xf32>
  %reduced = rock.blockwise_reduce sum %res {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  rock.blockwise_store_ptr %reduced -> %out_ptrs(%out_mask) by set : tensor<64xf32> -> tensor<64xi32>(tensor<64xi1>)
  return
}

// ============================================================
// FP8 max reduction after GEMM (f8E4M3FN): the format has no infinity, so the
// fill must be the most negative finite value (-448.0, the largest
// representable magnitude in E4M3FN) rather than -inf. This exercises the
// `APFloat::semanticsHasInf` == false branch of `createMaskFillValue`.
// ============================================================

// CHECK-LABEL: func.func @test_gemm_output_fp8_max_reduce_uses_largest_finite
// CHECK: %[[GEMM:.*]] = rock.blockwise_gemm
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf8E4M3FN>
// CHECK: %[[FUSED:.*]] = arith.addf %[[GEMM]], %[[ADD]] : tensor<64x64xf8E4M3FN>
// CHECK: %[[NEG_MAX:.*]] = arith.constant dense<-4.480000e+02> : tensor<64x64xf8E4M3FN>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[NEG_MAX]] : tensor<64x64xi1>, tensor<64x64xf8E4M3FN>
// CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce max %[[SAFE]]
// CHECK: rock.blockwise_store_ptr %[[REDUCED]]
func.func @test_gemm_output_fp8_max_reduce_uses_largest_finite(
    %a_ptrs: tensor<64x64xi32>, %a_mask: tensor<64x64xi1>,
    %b_ptrs: tensor<64x64xi32>, %b_mask: tensor<64x64xi1>,
    %d_ptrs: tensor<64x64xi32>, %d_mask: tensor<64x64xi1>,
    %out_ptrs: tensor<64xi32>, %out_mask: tensor<64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf8E4M3FN>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf8E4M3FN>
  %zero = arith.constant dense<0.0> : tensor<64x64xf8E4M3FN>
  %c = rock.blockwise_gemm(%a, %b, %zero) : tensor<64x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>, tensor<64x64xf8E4M3FN> -> tensor<64x64xf8E4M3FN>
  %d = rock.blockwise_load_ptr %d_ptrs[%d_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf8E4M3FN>
  %cst = arith.constant dense<1.0> : tensor<64x64xf8E4M3FN>
  %d2 = arith.addf %d, %cst : tensor<64x64xf8E4M3FN>
  %res = arith.addf %c, %d2 : tensor<64x64xf8E4M3FN>
  %reduced = rock.blockwise_reduce max %res {axis = 1 : index} : tensor<64x64xf8E4M3FN> -> tensor<64xf8E4M3FN>
  rock.blockwise_store_ptr %reduced -> %out_ptrs(%out_mask) by set : tensor<64xf8E4M3FN> -> tensor<64xi32>(tensor<64xi1>)
  return
}

// ============================================================
// Integer max reduction after GEMM: zero is wrong because masked-out positive
// lanes would dominate negative real values. Signless integers are treated as
// signed, so the fill is the signed minimum of the bit width (INT_MIN).
// (Unsigned integer fusion chains aren't constructable through the
// rock.blockwise_load_ptr -> arith path, so they aren't tested here.)
// ============================================================

// CHECK-LABEL: func.func @test_gemm_output_int_max_reduce_uses_signed_min
// CHECK: %[[GEMM:.*]] = rock.blockwise_gemm
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[ADD:.*]] = arith.addi %[[LOAD]], %{{.*}} : tensor<64x64xi32>
// CHECK: %[[FUSED:.*]] = arith.addi %[[GEMM]], %[[ADD]] : tensor<64x64xi32>
// CHECK: %[[INT_MIN:.*]] = arith.constant dense<-2147483648> : tensor<64x64xi32>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[INT_MIN]] : tensor<64x64xi1>, tensor<64x64xi32>
// CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce max %[[SAFE]]
// CHECK: rock.blockwise_store_ptr %[[REDUCED]]
func.func @test_gemm_output_int_max_reduce_uses_signed_min(
    %a_ptrs: tensor<64x64xi32>, %a_mask: tensor<64x64xi1>,
    %b_ptrs: tensor<64x64xi32>, %b_mask: tensor<64x64xi1>,
    %d_ptrs: tensor<64x64xi32>, %d_mask: tensor<64x64xi1>,
    %out_ptrs: tensor<64xi32>, %out_mask: tensor<64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xi8>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xi8>
  %zero = arith.constant dense<0> : tensor<64x64xi32>
  %c = rock.blockwise_gemm(%a, %b, %zero) : tensor<64x64xi8>, tensor<64x64xi8>, tensor<64x64xi32> -> tensor<64x64xi32>
  %d = rock.blockwise_load_ptr %d_ptrs[%d_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xi32>
  %cst = arith.constant dense<1> : tensor<64x64xi32>
  %d2 = arith.addi %d, %cst : tensor<64x64xi32>
  %res = arith.addi %c, %d2 : tensor<64x64xi32>
  %reduced = rock.blockwise_reduce max %res {axis = 1 : index} : tensor<64x64xi32> -> tensor<64xi32>
  rock.blockwise_store_ptr %reduced -> %out_ptrs(%out_mask) by set : tensor<64xi32> -> tensor<64xi32>(tensor<64xi1>)
  return
}

// ============================================================
// Integer sum reduction after GEMM: zero is the neutral element for sum and
// the fill must NOT switch to INT_MIN.
// ============================================================

// CHECK-LABEL: func.func @test_gemm_output_int_sum_reduce_uses_zero
// CHECK: %[[GEMM:.*]] = rock.blockwise_gemm
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK: %[[ADD:.*]] = arith.addi %[[LOAD]], %{{.*}} : tensor<64x64xi32>
// CHECK: %[[FUSED:.*]] = arith.addi %[[GEMM]], %[[ADD]] : tensor<64x64xi32>
// CHECK: %[[ZERO:.*]] = arith.constant dense<0> : tensor<64x64xi32>
// CHECK: %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[ZERO]] : tensor<64x64xi1>, tensor<64x64xi32>
// CHECK: %[[REDUCED:.*]] = rock.blockwise_reduce sum %[[SAFE]]
// CHECK: rock.blockwise_store_ptr %[[REDUCED]]
func.func @test_gemm_output_int_sum_reduce_uses_zero(
    %a_ptrs: tensor<64x64xi32>, %a_mask: tensor<64x64xi1>,
    %b_ptrs: tensor<64x64xi32>, %b_mask: tensor<64x64xi1>,
    %d_ptrs: tensor<64x64xi32>, %d_mask: tensor<64x64xi1>,
    %out_ptrs: tensor<64xi32>, %out_mask: tensor<64xi1>) attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xi8>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xi8>
  %zero = arith.constant dense<0> : tensor<64x64xi32>
  %c = rock.blockwise_gemm(%a, %b, %zero) : tensor<64x64xi8>, tensor<64x64xi8>, tensor<64x64xi32> -> tensor<64x64xi32>
  %d = rock.blockwise_load_ptr %d_ptrs[%d_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xi32>
  %cst = arith.constant dense<1> : tensor<64x64xi32>
  %d2 = arith.addi %d, %cst : tensor<64x64xi32>
  %res = arith.addi %c, %d2 : tensor<64x64xi32>
  %reduced = rock.blockwise_reduce sum %res {axis = 1 : index} : tensor<64x64xi32> -> tensor<64xi32>
  rock.blockwise_store_ptr %reduced -> %out_ptrs(%out_mask) by set : tensor<64xi32> -> tensor<64xi32>(tensor<64xi1>)
  return
}

// ============================================================
// Fusion BETWEEN two GEMMs (gemm+gemm / conv+gemm), which is the shape a
// split-k inter-gemm fusion lowers to: gemm0's accumulator is combined with a
// padded elementwise input and the result feeds gemm1 as its A operand.
//
// A multiply is zero-preserving here, so no select may be inserted: the
// elementwise input reads zero in its padded lanes, and so does gemm0's
// output, because the padded columns of gemm0's B operand read as zero. The
// select would otherwise land in gemm1's K loop and cost an instruction per
// iteration.
// ============================================================

// CHECK-LABEL: func.func @test_inter_gemm_body_mulf_load_not_remasked
// CHECK: %[[A:.*]] = rock.blockwise_load_ptr
// CHECK: %[[B:.*]] = rock.blockwise_load_ptr
// CHECK: %[[QK:.*]] = rock.blockwise_gemm(%[[A]], %[[B]], %{{.*}}) :
// CHECK: %[[EW:.*]] = rock.blockwise_load_ptr
// CHECK: %[[SCALED:.*]] = arith.mulf %[[QK]], %[[EW]]
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_gemm(%[[SCALED]],
func.func @test_inter_gemm_body_mulf_load_not_remasked(
    %a_ptrs: tensor<32x32xi32>, %a_mask: tensor<32x32xi1>,
    %b_ptrs: tensor<32x32xi32>, %b_mask: tensor<32x32xi1>,
    %ew_ptrs: tensor<32x32xi32>, %ew_mask: tensor<32x32xi1>,
    %c_ptrs: tensor<32x4xi32>, %c_mask: tensor<32x4xi1>,
    %out_ptrs: tensor<32x4xi32>, %out_mask: tensor<32x4xi1>) attributes {rock.kernel} {
  %zero0 = arith.constant dense<0.0> : tensor<32x32xf32>
  %zero1 = arith.constant dense<0.0> : tensor<32x4xf32>
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %qk = rock.blockwise_gemm(%a, %b, %zero0) : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  %ew = rock.blockwise_load_ptr %ew_ptrs[%ew_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %scaled = arith.mulf %qk, %ew fastmath<nsz,contract> : tensor<32x32xf32>
  %c = rock.blockwise_load_ptr %c_ptrs[%c_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x4xi32>, tensor<32x4xi1> -> tensor<32x4xf32>
  %out = rock.blockwise_gemm(%scaled, %c, %zero1) {rock.o_transposed = #rock.o_transposed<false>} : tensor<32x32xf32>, tensor<32x4xf32>, tensor<32x4xf32> -> tensor<32x4xf32>
  rock.blockwise_store_ptr %out -> %out_ptrs(%out_mask) by atomic_add : tensor<32x4xf32> -> tensor<32x4xi32>(tensor<32x4xi1>)
  return
}

// ============================================================
// Same inter-GEMM shape, but the body applies exp, which turns the padded
// zeros into ones. Those lanes must be selected back to zero before they
// reach gemm1's accumulation.
// ============================================================

// CHECK-LABEL: func.func @test_inter_gemm_body_exp_remasked
// CHECK: %[[QK:.*]] = rock.blockwise_gemm
// CHECK: %[[EW:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[EW_MASK:.*]]]
// CHECK: %[[SCALED:.*]] = arith.mulf %[[QK]], %[[EW]]
// CHECK: %[[EXP:.*]] = math.exp %[[SCALED]]
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x32xf32>
// CHECK: %[[SAFE:.*]] = arith.select %[[EW_MASK]], %[[EXP]], %[[ZERO]] : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK: rock.blockwise_gemm(%[[SAFE]],
func.func @test_inter_gemm_body_exp_remasked(
    %a_ptrs: tensor<32x32xi32>, %a_mask: tensor<32x32xi1>,
    %b_ptrs: tensor<32x32xi32>, %b_mask: tensor<32x32xi1>,
    %ew_ptrs: tensor<32x32xi32>, %ew_mask: tensor<32x32xi1>,
    %c_ptrs: tensor<32x4xi32>, %c_mask: tensor<32x4xi1>,
    %out_ptrs: tensor<32x4xi32>, %out_mask: tensor<32x4xi1>) attributes {rock.kernel} {
  %zero0 = arith.constant dense<0.0> : tensor<32x32xf32>
  %zero1 = arith.constant dense<0.0> : tensor<32x4xf32>
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %qk = rock.blockwise_gemm(%a, %b, %zero0) : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  %ew = rock.blockwise_load_ptr %ew_ptrs[%ew_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %scaled = arith.mulf %qk, %ew fastmath<nsz,contract> : tensor<32x32xf32>
  %exp = math.exp %scaled fastmath<nsz,contract,afn> : tensor<32x32xf32>
  %c = rock.blockwise_load_ptr %c_ptrs[%c_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x4xi32>, tensor<32x4xi1> -> tensor<32x4xf32>
  %out = rock.blockwise_gemm(%exp, %c, %zero1) {rock.o_transposed = #rock.o_transposed<false>} : tensor<32x32xf32>, tensor<32x4xf32>, tensor<32x4xf32> -> tensor<32x4xf32>
  rock.blockwise_store_ptr %out -> %out_ptrs(%out_mask) by atomic_add : tensor<32x4xf32> -> tensor<32x4xi32>(tensor<32x4xi1>)
  return
}

// ============================================================
// When gemm1 has more than one K tile, its accumulator is carried around an
// scf.for and the body runs inside the loop. The accumulator is only zeroed
// at the loop's entry, so recognizing the multiply as zero-preserving means
// following the loop-carried value back to its initial value.
// ============================================================

// CHECK-LABEL: func.func @test_inter_gemm_body_in_loop_not_remasked
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}})
// CHECK: %[[QK:.*]] = rock.blockwise_gemm
// CHECK: %[[EW:.*]] = rock.blockwise_load_ptr
// CHECK: %[[SCALED:.*]] = arith.mulf %[[QK]], %[[EW]]
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_gemm(%[[SCALED]], %{{.*}}, %[[ACC]])
func.func @test_inter_gemm_body_in_loop_not_remasked(
    %a_ptrs: tensor<32x32xi32>, %a_mask: tensor<32x32xi1>,
    %b_ptrs: tensor<32x32xi32>, %b_mask: tensor<32x32xi1>,
    %ew_ptrs: tensor<32x32xi32>, %ew_mask: tensor<32x32xi1>,
    %c_ptrs: tensor<32x32xi32>, %c_mask: tensor<32x32xi1>,
    %out_ptrs: tensor<32x32xi32>, %out_mask: tensor<32x32xi1>) attributes {rock.kernel} {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %zero = arith.constant dense<0.0> : tensor<32x32xf32>
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %acc = scf.for %i = %c0 to %c2 step %c1 iter_args(%accIn = %zero) -> (tensor<32x32xf32>) : i32 {
    %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
    %qk = rock.blockwise_gemm(%a, %b, %zero) : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
    %ew = rock.blockwise_load_ptr %ew_ptrs[%ew_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
    %scaled = arith.mulf %qk, %ew fastmath<nsz,contract> : tensor<32x32xf32>
    %c = rock.blockwise_load_ptr %c_ptrs[%c_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
    %next = rock.blockwise_gemm(%scaled, %c, %accIn) {rock.o_transposed = #rock.o_transposed<false>} : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
    scf.yield %next : tensor<32x32xf32>
  }
  rock.blockwise_store_ptr %acc -> %out_ptrs(%out_mask) by atomic_add : tensor<32x32xf32> -> tensor<32x32xi32>(tensor<32x32xi1>)
  return
}

// ============================================================
// The padded region of gemm0's output is only known to be zero because both
// of its operands are masked loads. With B a constant, nothing zeroes the
// padded columns, so the multiply is no longer known to preserve zero.
// ============================================================

// CHECK-LABEL: func.func @test_inter_gemm_body_gemm_b_not_loaded_remasked
// CHECK: %[[QK:.*]] = rock.blockwise_gemm
// CHECK: %[[EW:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[EW_MASK:.*]]]
// CHECK: %[[SCALED:.*]] = arith.mulf %[[QK]], %[[EW]]
// CHECK: %[[SAFE:.*]] = arith.select %[[EW_MASK]], %[[SCALED]], %{{.*}} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK: rock.blockwise_gemm(%[[SAFE]],
func.func @test_inter_gemm_body_gemm_b_not_loaded_remasked(
    %a_ptrs: tensor<32x32xi32>, %a_mask: tensor<32x32xi1>,
    %ew_ptrs: tensor<32x32xi32>, %ew_mask: tensor<32x32xi1>,
    %c_ptrs: tensor<32x4xi32>, %c_mask: tensor<32x4xi1>,
    %out_ptrs: tensor<32x4xi32>, %out_mask: tensor<32x4xi1>) attributes {rock.kernel} {
  %zero0 = arith.constant dense<0.0> : tensor<32x32xf32>
  %zero1 = arith.constant dense<0.0> : tensor<32x4xf32>
  %b = arith.constant dense<3.0> : tensor<32x32xf32>
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %qk = rock.blockwise_gemm(%a, %b, %zero0) : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  %ew = rock.blockwise_load_ptr %ew_ptrs[%ew_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %scaled = arith.mulf %qk, %ew fastmath<nsz,contract> : tensor<32x32xf32>
  %c = rock.blockwise_load_ptr %c_ptrs[%c_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x4xi32>, tensor<32x4xi1> -> tensor<32x4xf32>
  %out = rock.blockwise_gemm(%scaled, %c, %zero1) {rock.o_transposed = #rock.o_transposed<false>} : tensor<32x32xf32>, tensor<32x4xf32>, tensor<32x4xf32> -> tensor<32x4xf32>
  rock.blockwise_store_ptr %out -> %out_ptrs(%out_mask) by atomic_add : tensor<32x4xf32> -> tensor<32x4xi32>(tensor<32x4xi1>)
  return
}

// ============================================================
// gemm0 accumulating onto a non-zero tile leaves that value in the padded
// lanes of its output, so the multiply does not preserve zero there either.
// ============================================================

// CHECK-LABEL: func.func @test_inter_gemm_body_gemm_acc_not_zeroed_remasked
// CHECK: %[[QK:.*]] = rock.blockwise_gemm
// CHECK: %[[EW:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[EW_MASK:.*]]]
// CHECK: %[[SCALED:.*]] = arith.mulf %[[QK]], %[[EW]]
// CHECK: %[[SAFE:.*]] = arith.select %[[EW_MASK]], %[[SCALED]], %{{.*}} : tensor<32x32xi1>, tensor<32x32xf32>
// CHECK: rock.blockwise_gemm(%[[SAFE]],
func.func @test_inter_gemm_body_gemm_acc_not_zeroed_remasked(
    %a_ptrs: tensor<32x32xi32>, %a_mask: tensor<32x32xi1>,
    %b_ptrs: tensor<32x32xi32>, %b_mask: tensor<32x32xi1>,
    %ew_ptrs: tensor<32x32xi32>, %ew_mask: tensor<32x32xi1>,
    %c_ptrs: tensor<32x4xi32>, %c_mask: tensor<32x4xi1>,
    %out_ptrs: tensor<32x4xi32>, %out_mask: tensor<32x4xi1>) attributes {rock.kernel} {
  %nonzero = arith.constant dense<1.0> : tensor<32x32xf32>
  %zero1 = arith.constant dense<0.0> : tensor<32x4xf32>
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %qk = rock.blockwise_gemm(%a, %b, %nonzero) : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  %ew = rock.blockwise_load_ptr %ew_ptrs[%ew_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %scaled = arith.mulf %qk, %ew fastmath<nsz,contract> : tensor<32x32xf32>
  %c = rock.blockwise_load_ptr %c_ptrs[%c_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x4xi32>, tensor<32x4xi1> -> tensor<32x4xf32>
  %out = rock.blockwise_gemm(%scaled, %c, %zero1) {rock.o_transposed = #rock.o_transposed<false>} : tensor<32x32xf32>, tensor<32x4xf32>, tensor<32x4xf32> -> tensor<32x4xf32>
  rock.blockwise_store_ptr %out -> %out_ptrs(%out_mask) by atomic_add : tensor<32x4xf32> -> tensor<32x4xi32>(tensor<32x4xi1>)
  return
}

// ============================================================
// Trivial mask on the elementwise input of an inter-GEMM fusion: there are no
// padded lanes to restore, so even exp needs no select. This isolates the
// trivial-mask early-out from the zero-preservation analysis.
// ============================================================

// CHECK-LABEL: func.func @test_inter_gemm_body_trivial_mask_not_remasked
// CHECK: %[[QK:.*]] = rock.blockwise_gemm
// CHECK: %[[EW:.*]] = rock.blockwise_load_ptr
// CHECK: %[[SCALED:.*]] = arith.mulf %[[QK]], %[[EW]]
// CHECK: %[[EXP:.*]] = math.exp %[[SCALED]]
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_gemm(%[[EXP]],
func.func @test_inter_gemm_body_trivial_mask_not_remasked(
    %a_ptrs: tensor<32x32xi32>, %a_mask: tensor<32x32xi1>,
    %b_ptrs: tensor<32x32xi32>, %b_mask: tensor<32x32xi1>,
    %ew_ptrs: tensor<32x32xi32>,
    %c_ptrs: tensor<32x4xi32>, %c_mask: tensor<32x4xi1>,
    %out_ptrs: tensor<32x4xi32>, %out_mask: tensor<32x4xi1>) attributes {rock.kernel} {
  %zero0 = arith.constant dense<0.0> : tensor<32x32xf32>
  %zero1 = arith.constant dense<0.0> : tensor<32x4xf32>
  %trivial = arith.constant dense<true> : tensor<32x32xi1>
  %a = rock.blockwise_load_ptr %a_ptrs[%a_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %b = rock.blockwise_load_ptr %b_ptrs[%b_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %qk = rock.blockwise_gemm(%a, %b, %zero0) : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32> -> tensor<32x32xf32>
  %ew = rock.blockwise_load_ptr %ew_ptrs[%trivial] {cacheModifier = #rock<CacheModifier none>} : tensor<32x32xi32>, tensor<32x32xi1> -> tensor<32x32xf32>
  %scaled = arith.mulf %qk, %ew fastmath<nsz,contract> : tensor<32x32xf32>
  %exp = math.exp %scaled fastmath<nsz,contract,afn> : tensor<32x32xf32>
  %c = rock.blockwise_load_ptr %c_ptrs[%c_mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x4xi32>, tensor<32x4xi1> -> tensor<32x4xf32>
  %out = rock.blockwise_gemm(%exp, %c, %zero1) {rock.o_transposed = #rock.o_transposed<false>} : tensor<32x32xf32>, tensor<32x4xf32>, tensor<32x4xf32> -> tensor<32x4xf32>
  rock.blockwise_store_ptr %out -> %out_ptrs(%out_mask) by atomic_add : tensor<32x4xf32> -> tensor<32x4xi32>(tensor<32x4xi1>)
  return
}

// ============================================================
// Non-kernel function: pass should skip entirely.
// Even with a non-zero-preserving fusion, no select is inserted.
// ============================================================

// CHECK-LABEL: func.func @test_non_kernel
// CHECK: %[[LOAD:.*]] = rock.blockwise_load_ptr
// CHECK: %[[FUSED:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf16>
// CHECK-NOT: arith.select
// CHECK: rock.blockwise_store_ptr %[[FUSED]]
func.func @test_non_kernel(
    %ptrs: tensor<64x64xi32>, %mask: tensor<64x64xi1>,
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst = arith.constant dense<1.0> : tensor<64x64xf16>
  %fused = arith.addf %tile, %cst : tensor<64x64xf16>
  rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}
