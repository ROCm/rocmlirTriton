// Unit tests for rock-regularize-elementwise pass.
// Tests that inter-fusion rock.transform ops are pushed upward to inputs
// in standalone elementwise kernels.

// RUN: rocmlir-opt -rock-regularize-elementwise -mlir-print-local-scope %s | FileCheck %s

// --- Merge 3x4 -> 12 ---
#map_merge = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
#tf_merge = #rock.transform_map<#map_merge by [<Merge{3, 4} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [12] -> [3, 4]>

// --- Unmerge 12 -> 3x4 ---
#map_unmerge = affine_map<(d0, d1) -> (d0 * 4 + d1)>
#tf_unmerge = #rock.transform_map<#map_unmerge by [<Unmerge{3, 4} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>] bounds = [3, 4] -> [12]>

// --- Merge 12 -> 2x6 ---
#map_merge26 = affine_map<(d0) -> (d0 floordiv 6, d0 mod 6)>
#tf_merge26 = #rock.transform_map<#map_merge26 by [<Merge{2, 6} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [12] -> [2, 6]>

// --- Unmerge 12 -> 2x6 ---
#map_unmerge26 = affine_map<(d0, d1) -> (d0 * 6 + d1)>
#tf_unmerge26 = #rock.transform_map<#map_unmerge26 by [<Unmerge{2, 6} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>] bounds = [2, 6] -> [12]>

// --- Merge 4x3 -> 12 ---
#map_merge43 = affine_map<(d0) -> (d0 floordiv 3, d0 mod 3)>
#tf_merge43 = #rock.transform_map<#map_merge43 by [<Merge{4, 3} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [12] -> [4, 3]>

// --- Unmerge 12 -> 4x3 ---
#map_unmerge43 = affine_map<(d0, d1) -> (d0 * 3 + d1)>
#tf_unmerge43 = #rock.transform_map<#map_unmerge43 by [<Unmerge{4, 3} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>] bounds = [4, 3] -> [12]>

// --- Broadcast 3x4 -> 2x3x4 (AddDim{2} at dim0) ---
#map_bcast = affine_map<(d0, d1, d2) -> (d1, d2)>
#tf_bcast = #rock.transform_map<#map_bcast by [<AddDim{2} ["dim0"] at [0] -> [] at []>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [0]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [1]>] bounds = [2, 3, 4] -> [3, 4]>

// --- Slice dim0 [0,2): 3x4 -> 2x4 ---
#map_slice = affine_map<(d0, d1) -> (d0, d1)>
#tf_slice = #rock.transform_map<#map_slice by [<Slice{0, 2, 0, 4} ["dim0_sliced", "dim1_sliced"] at [0, 1] -> ["dim0", "dim1"] at [0, 1]>] bounds = [2, 4] -> [3, 4]>

module {

  // ============================================================
  // NO-OP: Not an elementwise kernel (has FusionRoot gemm).
  // The pass should not modify this function.
  // ============================================================

  // CHECK-LABEL: func.func @test_noop_gemm
  // CHECK: rock.gemm
  // CHECK-NEXT: arith.addf {{.*}} : tensor<1x3x3xf32>
  func.func @test_noop_gemm(%a: tensor<1x3x4xf32>, %b: tensor<1x4x3xf32>, %ext: tensor<1x3x3xf32>, %dest: tensor<1x3x3xf32>) -> tensor<1x3x3xf32> attributes {rock.kernel} {
    %gemm = rock.gemm %a * %b : tensor<1x3x4xf32> * tensor<1x4x3xf32> -> tensor<1x3x3xf32>
    %fused = arith.addf %gemm, %ext : tensor<1x3x3xf32>
    %r = rock.store %fused to %dest by set : tensor<1x3x3xf32> -> tensor<1x3x3xf32> to tensor<1x3x3xf32>
    return %r : tensor<1x3x3xf32>
  }

  // ============================================================
  // NO-OP: No inter-fusion transforms.
  // All transforms are already on inputs, nothing to push.
  // ============================================================

  // CHECK-LABEL: func.func @test_noop_no_inter_transform
  // CHECK: rock.transform %arg0
  // CHECK: rock.transform %arg1
  // CHECK-NEXT: arith.addf {{.*}} : tensor<12xf32>
  // CHECK-NEXT: arith.mulf {{.*}} : tensor<12xf32>
  func.func @test_noop_no_inter_transform(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel} {
    %t0 = rock.transform %arg0 by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %t1 = rock.transform %arg1 by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %add = arith.addf %t0, %t1 : tensor<12xf32>
    %mul = arith.mulf %add, %arg2 : tensor<12xf32>
    %r = rock.store %mul to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }

  // ============================================================
  // Single inter-fusion Merge transform: addf(3x4) -> Merge -> mulf(12)
  // Expect: Merge pushed to addf inputs, both fusions in 12.
  // ============================================================

  // CHECK-LABEL: func.func @test_single_merge
  // CHECK-NOT: arith.addf {{.*}} : tensor<3x4xf32>
  // CHECK: %[[T0:.*]] = rock.transform %arg0 by {{.*}}Merge
  // CHECK: %[[T1:.*]] = rock.transform %arg1 by {{.*}}Merge
  // CHECK: %[[ADD:.*]] = arith.addf %[[T0]], %[[T1]] : tensor<12xf32>
  // CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %arg2 : tensor<12xf32>
  // CHECK: rock.store %[[MUL]]
  func.func @test_single_merge(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %t = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %mul = arith.mulf %t, %arg2 : tensor<12xf32>
    %r = rock.store %mul to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }

  // ============================================================
  // Single inter-fusion Unmerge transform: addf(12) -> Unmerge -> mulf(3x4)
  // Expect: Unmerge pushed to addf inputs, both fusions in 3x4.
  // ============================================================

  // CHECK-LABEL: func.func @test_single_unmerge
  // CHECK-NOT: arith.addf {{.*}} : tensor<12xf32>
  // CHECK: %[[T0:.*]] = rock.transform %arg0 by {{.*}}Unmerge
  // CHECK: %[[T1:.*]] = rock.transform %arg1 by {{.*}}Unmerge
  // CHECK: %[[ADD:.*]] = arith.addf %[[T0]], %[[T1]] : tensor<3x4xf32>
  // CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %arg2 : tensor<3x4xf32>
  // CHECK: rock.store %[[MUL]]
  func.func @test_single_unmerge(%arg0: tensor<12xf32>, %arg1: tensor<12xf32>, %arg2: tensor<3x4xf32>, %dest: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {rock.kernel} {
    %add = arith.addf %arg0, %arg1 : tensor<12xf32>
    %t = rock.transform %add by #tf_unmerge : tensor<12xf32> to tensor<3x4xf32>
    %mul = arith.mulf %t, %arg2 : tensor<3x4xf32>
    %r = rock.store %mul to %dest by set : tensor<3x4xf32> -> tensor<3x4xf32> to tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }

  // ============================================================
  // Chained transforms: addf(3x4) -> Merge(12) -> Unmerge(2x6) -> mulf(2x6)
  // Two transforms in a row between fusions. Requires two iterations.
  // Expect: both pushed to inputs, fusions in 2x6.
  // ============================================================

  // CHECK-LABEL: func.func @test_chained_transforms
  // CHECK-NOT: arith.addf {{.*}} : tensor<3x4xf32>
  // CHECK-NOT: arith.addf {{.*}} : tensor<12xf32>
  // CHECK-DAG: rock.transform %arg0 by {{.*}} : tensor<3x4xf32> to tensor<12xf32>
  // CHECK-DAG: rock.transform %arg1 by {{.*}} : tensor<3x4xf32> to tensor<12xf32>
  // CHECK: rock.transform %{{.*}} by {{.*}} : tensor<12xf32> to tensor<2x6xf32>
  // CHECK: rock.transform %{{.*}} by {{.*}} : tensor<12xf32> to tensor<2x6xf32>
  // CHECK: arith.addf {{.*}} : tensor<2x6xf32>
  // CHECK: arith.mulf {{.*}} : tensor<2x6xf32>
  func.func @test_chained_transforms(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x6xf32>, %dest: tensor<2x6xf32>) -> tensor<2x6xf32> attributes {rock.kernel} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %t1 = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %t2 = rock.transform %t1 by #tf_unmerge26 : tensor<12xf32> to tensor<2x6xf32>
    %mul = arith.mulf %t2, %arg2 : tensor<2x6xf32>
    %r = rock.store %mul to %dest by set : tensor<2x6xf32> -> tensor<2x6xf32> to tensor<2x6xf32>
    return %r : tensor<2x6xf32>
  }

  // ============================================================
  // Long chain: addf(3x4) -> t1(12) -> mulf(12) -> t2(4x3) -> addf(4x3)
  // Three fusions, two inter-fusion transforms. Requires two iterations.
  // ============================================================

  // CHECK-LABEL: func.func @test_long_chain
  // CHECK-NOT: arith.addf {{.*}} : tensor<3x4xf32>
  // CHECK-NOT: arith.mulf {{.*}} : tensor<12xf32>
  // After full convergence, all fusions should operate in 4x3.
  // CHECK: arith.addf {{.*}} : tensor<4x3xf32>
  // CHECK: arith.mulf {{.*}} : tensor<4x3xf32>
  // CHECK: arith.addf {{.*}} : tensor<4x3xf32>
  // CHECK: rock.store
  func.func @test_long_chain(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<12xf32>, %arg3: tensor<4x3xf32>, %dest: tensor<4x3xf32>) -> tensor<4x3xf32> attributes {rock.kernel} {
    %f1 = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %t1 = rock.transform %f1 by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %f2 = arith.mulf %t1, %arg2 : tensor<12xf32>
    %t2 = rock.transform %f2 by #tf_unmerge43 : tensor<12xf32> to tensor<4x3xf32>
    %f3 = arith.addf %t2, %arg3 : tensor<4x3xf32>
    %r = rock.store %f3 to %dest by set : tensor<4x3xf32> -> tensor<4x3xf32> to tensor<4x3xf32>
    return %r : tensor<4x3xf32>
  }

  // ============================================================
  // Mixed direct + transform: addf result used directly by mulf AND
  // through a Merge by another addf. The pass must clone the upstream
  // addf for the transform branch, leaving the original for mulf.
  // ============================================================

  // CHECK-LABEL: func.func @test_mixed_direct_transform
  // Cloned addf in 12 for the transform branch (inserted before original).
  // CHECK: %[[T0:.*]] = rock.transform %arg0 by {{.*}}Merge
  // CHECK: %[[T1:.*]] = rock.transform %arg1 by {{.*}}Merge
  // CHECK: %[[ADD_CLONE:.*]] = arith.addf %[[T0]], %[[T1]] : tensor<12xf32>
  // The original addf in 3x4 survives for the direct mulf user.
  // CHECK: %[[ADD_ORIG:.*]] = arith.addf %arg0, %arg1 : tensor<3x4xf32>
  // CHECK: arith.mulf %[[ADD_ORIG]], %arg2 : tensor<3x4xf32>
  // CHECK: arith.addf %[[ADD_CLONE]], %arg3 : tensor<12xf32>
  func.func @test_mixed_direct_transform(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>, %arg3: tensor<12xf32>, %dest0: tensor<3x4xf32>, %dest1: tensor<12xf32>) -> (tensor<3x4xf32>, tensor<12xf32>) attributes {rock.kernel} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %direct = arith.mulf %add, %arg2 : tensor<3x4xf32>
    %t = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %reshaped = arith.addf %t, %arg3 : tensor<12xf32>
    %r0 = rock.store %direct to %dest0 by set : tensor<3x4xf32> -> tensor<3x4xf32> to tensor<3x4xf32>
    %r1 = rock.store %reshaped to %dest1 by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r0, %r1 : tensor<3x4xf32>, tensor<12xf32>
  }

  // ============================================================
  // Fan-out: one fusion fans out to two incompatible transforms
  // (broadcast vs slice), each feeding a different downstream fusion.
  // The pass must clone the upstream fusion for each branch.
  // ============================================================

  // CHECK-LABEL: func.func @test_fanout_incompatible
  // No fusion in the original 3x4 space should remain.
  // CHECK-NOT: arith.addf %arg0, %arg1 : tensor<3x4xf32>
  // Branch A: broadcast transforms pushed to inputs, fusion in 2x3x4.
  // CHECK: rock.transform %arg0 by {{.*}}AddDim{{.*}} : tensor<3x4xf32> to tensor<2x3x4xf32>
  // CHECK: rock.transform %arg1 by {{.*}}AddDim{{.*}} : tensor<3x4xf32> to tensor<2x3x4xf32>
  // CHECK: arith.addf {{.*}} : tensor<2x3x4xf32>
  // Branch B: slice transforms pushed to inputs, fusion in 2x4.
  // CHECK: rock.transform %arg0 by {{.*}}Slice{{.*}} : tensor<3x4xf32> to tensor<2x4xf32>
  // CHECK: rock.transform %arg1 by {{.*}}Slice{{.*}} : tensor<3x4xf32> to tensor<2x4xf32>
  // CHECK: arith.addf {{.*}} : tensor<2x4xf32>
  // Downstream fusions.
  // CHECK: arith.addf {{.*}} : tensor<2x3x4xf32>
  // CHECK: arith.mulf {{.*}} : tensor<2x4xf32>
  func.func @test_fanout_incompatible(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x3x4xf32>, %arg3: tensor<2x4xf32>, %dest0: tensor<2x3x4xf32>, %dest1: tensor<2x4xf32>) -> (tensor<2x3x4xf32>, tensor<2x4xf32>) attributes {rock.kernel} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %t0 = rock.transform %add by #tf_bcast : tensor<3x4xf32> to tensor<2x3x4xf32>
    %result0 = arith.addf %t0, %arg2 : tensor<2x3x4xf32>
    %t1 = rock.transform %add by #tf_slice : tensor<3x4xf32> to tensor<2x4xf32>
    %result1 = arith.mulf %t1, %arg3 : tensor<2x4xf32>
    %r0 = rock.store %result0 to %dest0 by set : tensor<2x3x4xf32> -> tensor<2x3x4xf32> to tensor<2x3x4xf32>
    %r1 = rock.store %result1 to %dest1 by set : tensor<2x4xf32> -> tensor<2x4xf32> to tensor<2x4xf32>
    return %r0, %r1 : tensor<2x3x4xf32>, tensor<2x4xf32>
  }

  // ============================================================
  // Self-use: addf(%x, %x) -> transform -> mulf. Both operands
  // are the same value and should each get the transform applied.
  // ============================================================

  // CHECK-LABEL: func.func @test_self_use
  // CHECK: %[[T0:.*]] = rock.transform %arg0 by {{.*}}Merge
  // CHECK: %[[T1:.*]] = rock.transform %arg0 by {{.*}}Merge
  // CHECK: %[[ADD:.*]] = arith.addf %[[T0]], %[[T1]] : tensor<12xf32>
  // CHECK: arith.mulf %[[ADD]], %arg1 : tensor<12xf32>
  func.func @test_self_use(%arg0: tensor<3x4xf32>, %arg1: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel} {
    %add = arith.addf %arg0, %arg0 : tensor<3x4xf32>
    %t = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %mul = arith.mulf %t, %arg1 : tensor<12xf32>
    %r = rock.store %mul to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }

  // ============================================================
  // Unary fusion: math.exp(x) -> transform -> mulf.
  // Single operand should get the transform.
  // ============================================================

  // CHECK-LABEL: func.func @test_unary
  // CHECK: %[[T:.*]] = rock.transform %arg0 by {{.*}}Merge
  // CHECK: %[[EXP:.*]] = math.exp %[[T]] : tensor<12xf32>
  // CHECK: arith.mulf %[[EXP]], %arg1 : tensor<12xf32>
  func.func @test_unary(%arg0: tensor<3x4xf32>, %arg1: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel} {
    %exp = math.exp %arg0 : tensor<3x4xf32>
    %t = rock.transform %exp by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %mul = arith.mulf %t, %arg1 : tensor<12xf32>
    %r = rock.store %mul to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }

  // ============================================================
  // Type-changing fusion: arith.extf(f16->f32) -> transform -> mulf.
  // The transform must preserve the element type of each operand.
  // ============================================================

  // CHECK-LABEL: func.func @test_type_change
  // CHECK: %[[T:.*]] = rock.transform %arg0 by {{.*}}Merge{{.*}} : tensor<3x4xf16> to tensor<12xf16>
  // CHECK: %[[EXT:.*]] = arith.extf %[[T]] : tensor<12xf16> to tensor<12xf32>
  // CHECK: arith.mulf %[[EXT]], %arg1 : tensor<12xf32>
  func.func @test_type_change(%arg0: tensor<3x4xf16>, %arg1: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel} {
    %ext = arith.extf %arg0 : tensor<3x4xf16> to tensor<3x4xf32>
    %t = rock.transform %ext by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %mul = arith.mulf %t, %arg1 : tensor<12xf32>
    %r = rock.store %mul to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }
}
