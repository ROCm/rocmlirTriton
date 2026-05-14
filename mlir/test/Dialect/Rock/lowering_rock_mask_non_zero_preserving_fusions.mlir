// RUN: rocmlir-opt -rock-mask-non-zero-preserving-fusions -mlir-print-local-scope %s | FileCheck %s

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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst = arith.constant dense<1.0> : tensor<64x64xf16>
  %fused = arith.addf %tile, %cst : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst = arith.constant dense<2.0> : tensor<64x64xf16>
  %fused = arith.mulf %tile, %cst : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %trivial_mask = arith.constant dense<true> : tensor<64x64xi1>
  %tile = rock.blockwise_load_ptr %ptrs[%trivial_mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst = arith.constant dense<1.0> : tensor<64x64xf16>
  %fused = arith.addf %tile, %cst : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %tile -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst2 = arith.constant dense<2.0> : tensor<64x64xf16>
  %cst1 = arith.constant dense<1.0> : tensor<64x64xf16>
  %step1 = arith.mulf %tile, %cst2 : tensor<64x64xf16>
  %step2 = arith.addf %step1, %cst1 : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %step2 -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst2 = arith.constant dense<2.0> : tensor<64x64xf16>
  %cst3 = arith.constant dense<3.0> : tensor<64x64xf16>
  %step1 = arith.mulf %tile, %cst2 : tensor<64x64xf16>
  %step2 = arith.mulf %step1, %cst3 : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %step2 -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %ptrs_a[%mask_a] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %b = rock.blockwise_load_ptr %ptrs_b[%mask_b] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %fused = arith.addf %a, %b : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %a = rock.blockwise_load_ptr %ptrs_a[%mask_a] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %b = rock.blockwise_load_ptr %ptrs_b[%mask_b] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %sum = arith.addf %a, %b : tensor<64x64xf16>
  %cst = arith.constant dense<1.0> : tensor<64x64xf16>
  %fused = arith.addf %sum, %cst : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf32> attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %fused = math.exp %tile : tensor<64x64xf32>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf32>
  return %r : tensor<4096xf32>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf32> attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %fused = arith.extf %tile : tensor<64x64xf16> to tensor<64x64xf32>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf32>
  return %r : tensor<4096xf32>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> attributes {rock.kernel} {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %fused = arith.negf %tile : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
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
    %dest_ptrs: tensor<64x64xi32>, %dest_mask: tensor<64x64xi1>) -> tensor<4096xf16> {
  %tile = rock.blockwise_load_ptr %ptrs[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %cst = arith.constant dense<1.0> : tensor<64x64xf16>
  %fused = arith.addf %tile, %cst : tensor<64x64xf16>
  %r = rock.blockwise_store_ptr %fused -> %dest_ptrs(%dest_mask) by set : tensor<64x64xf16> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<4096xf16>
  return %r : tensor<4096xf16>
}

// ============================================================
// Leaf consumed by `blockwise_reduce max`: filler must be -inf.
// (Regression: bug fixed when this used to be 0, breaking causal-
// attention softmax when the row max was negative.)
// ============================================================

// CHECK-LABEL: func.func @test_leaf_to_max_reduce_f32
// CHECK:         %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK:         %[[FUSED:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf32>
// CHECK:         %[[NEGINF:.*]] = arith.constant dense<0xFF800000> : tensor<64x64xf32>
// CHECK:         %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[NEGINF]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK:         rock.blockwise_reduce max %[[SAFE]]
func.func @test_leaf_to_max_reduce_f32(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi1>) -> tensor<64xf32> attributes {rock.kernel} {
  %cst = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
  %0 = rock.blockwise_load_ptr %arg0[%arg1] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %1 = arith.addf %0, %cst : tensor<64x64xf32>
  %2 = rock.blockwise_reduce max %1 {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  return %2 : tensor<64xf32>
}

// ============================================================
// Same as above but f16: -inf encoding for f16 is 0xFC00.
// ============================================================

// CHECK-LABEL: func.func @test_leaf_to_max_reduce_f16
// CHECK:         %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK:         %[[FUSED:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf16>
// CHECK:         %[[NEGINF:.*]] = arith.constant dense<0xFC00> : tensor<64x64xf16>
// CHECK:         %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[NEGINF]] : tensor<64x64xi1>, tensor<64x64xf16>
// CHECK:         rock.blockwise_reduce max %[[SAFE]]
func.func @test_leaf_to_max_reduce_f16(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi1>) -> tensor<64xf16> attributes {rock.kernel} {
  %cst = arith.constant dense<1.000000e+00> : tensor<64x64xf16>
  %0 = rock.blockwise_load_ptr %arg0[%arg1] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  %1 = arith.addf %0, %cst : tensor<64x64xf16>
  %2 = rock.blockwise_reduce max %1 {axis = 1 : index} : tensor<64x64xf16> -> tensor<64xf16>
  return %2 : tensor<64xf16>
}

// ============================================================
// Leaf consumed by `blockwise_reduce sum`: filler stays at 0.
// Sanity check that the new filler logic doesn't regress the
// sum-reduce (additive identity) path.
// ============================================================

// CHECK-LABEL: func.func @test_leaf_to_sum_reduce
// CHECK:         %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK:         %[[FUSED:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf32>
// CHECK:         %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
// CHECK:         %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[ZERO]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK:         rock.blockwise_reduce sum %[[SAFE]]
func.func @test_leaf_to_sum_reduce(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi1>) -> tensor<64xf32> attributes {rock.kernel} {
  %cst = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
  %0 = rock.blockwise_load_ptr %arg0[%arg1] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %1 = arith.addf %0, %cst : tensor<64x64xf32>
  %2 = rock.blockwise_reduce sum %1 {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  return %2 : tensor<64xf32>
}

// ============================================================
// Max-reduce reached through additional fusion ops downstream of
// the leaf. chooseFillerKind() walks fusion-op users transitively,
// so the filler must still be -inf even when the reduce isn't a
// direct user of the leaf.
// ============================================================

// CHECK-LABEL: func.func @test_max_reduce_through_fusion_chain
// CHECK:         %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK:         %[[ADD:.*]] = arith.addf %[[LOAD]], %{{.*}} : tensor<64x64xf32>
// CHECK:         %[[SCALED:.*]] = arith.mulf %[[ADD]], %{{.*}} : tensor<64x64xf32>
// CHECK:         %[[NEGINF:.*]] = arith.constant dense<0xFF800000> : tensor<64x64xf32>
// CHECK:         %[[SAFE:.*]] = arith.select %[[MASK]], %[[SCALED]], %[[NEGINF]] : tensor<64x64xi1>, tensor<64x64xf32>
// CHECK:         rock.blockwise_reduce max %[[SAFE]]
func.func @test_max_reduce_through_fusion_chain(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi1>) -> tensor<64xf32> attributes {rock.kernel} {
  %cst = arith.constant dense<2.000000e+00> : tensor<64x64xf32>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<64x64xf32>
  %0 = rock.blockwise_load_ptr %arg0[%arg1] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf32>
  %1 = arith.addf %0, %cst_0 : tensor<64x64xf32>
  %2 = arith.mulf %1, %cst : tensor<64x64xf32>
  %3 = rock.blockwise_reduce max %2 {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  return %3 : tensor<64xf32>
}

// ============================================================
// Integer leaf type with a max-reduce consumer: filler must be the
// signed minimum of the element type (matches the `arith.maxsi`
// lowering of `rock.blockwise_reduce max` on integers). Using zero
// here would silently corrupt results when valid lanes are negative.
// ============================================================

// CHECK-LABEL: func.func @test_max_reduce_int_uses_signed_min
// CHECK:         %[[LOAD:.*]] = rock.blockwise_load_ptr %{{.*}}[%[[MASK:.*]]]
// CHECK:         %[[FUSED:.*]] = arith.addi %[[LOAD]], %{{.*}} : tensor<64x64xi32>
// CHECK:         %[[SMIN:.*]] = arith.constant dense<-2147483648> : tensor<64x64xi32>
// CHECK:         %[[SAFE:.*]] = arith.select %[[MASK]], %[[FUSED]], %[[SMIN]] : tensor<64x64xi1>, tensor<64x64xi32>
// CHECK:         rock.blockwise_reduce max %[[SAFE]]
func.func @test_max_reduce_int_uses_signed_min(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi1>) -> tensor<64xi32> attributes {rock.kernel} {
  %cst = arith.constant dense<1> : tensor<64x64xi32>
  %0 = rock.blockwise_load_ptr %arg0[%arg1] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xi32>
  %1 = arith.addi %0, %cst : tensor<64x64xi32>
  %2 = rock.blockwise_reduce max %1 {axis = 1 : index} : tensor<64x64xi32> -> tensor<64xi32>
  return %2 : tensor<64xi32>
}
