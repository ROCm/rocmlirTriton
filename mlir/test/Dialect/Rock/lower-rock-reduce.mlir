// Unit tests for rock-lower-reduce pass
// Verifies that rock.reduce + rock.store is lowered to
// broadcast transform + atomic store + prefill attribute.

// RUN: rocmlir-opt -rock-lower-reduce -mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: func.func @test_reduce_sum
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x12x12xf32>, %[[OUTPUT:.*]]: tensor<2x12x1xf32> {rock.prefill = 0.000000e+00 : f32})
// CHECK-NOT: rock.reduce
// CHECK: %[[BC:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Broadcast{{.*}} : tensor<2x12x1xf32> to tensor<2x12x12xf32>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_add : tensor<2x12x12xf32> -> tensor<2x12x1xf32> to tensor<2x12x12xf32>
func.func @test_reduce_sum(%arg0: tensor<2x12x12xf32>, %arg1: tensor<2x12x1xf32>) -> tensor<2x12x1xf32> {
  %reduced = rock.reduce sum %arg0 {axis = 2 : index} : tensor<2x12x12xf32> -> tensor<2x12x1xf32>
  %result = rock.store %reduced to %arg1 by set : tensor<2x12x1xf32> -> tensor<2x12x1xf32> to tensor<2x12x1xf32>
  return %result : tensor<2x12x1xf32>
}

// CHECK-LABEL: func.func @test_reduce_max
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x12x12xf32>, %[[OUTPUT:.*]]: tensor<2x12x1xf32> {rock.prefill = 0xFF800000 : f32})
// CHECK-NOT: rock.reduce
// CHECK: %[[BC:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Broadcast{{.*}} : tensor<2x12x1xf32> to tensor<2x12x12xf32>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_max : tensor<2x12x12xf32> -> tensor<2x12x1xf32> to tensor<2x12x12xf32>
func.func @test_reduce_max(%arg0: tensor<2x12x12xf32>, %arg1: tensor<2x12x1xf32>) -> tensor<2x12x1xf32> {
  %reduced = rock.reduce max %arg0 {axis = 2 : index} : tensor<2x12x12xf32> -> tensor<2x12x1xf32>
  %result = rock.store %reduced to %arg1 by set : tensor<2x12x1xf32> -> tensor<2x12x1xf32> to tensor<2x12x1xf32>
  return %result : tensor<2x12x1xf32>
}

// CHECK-LABEL: func.func @test_reduce_sum_f16
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x12x12xf16>, %[[OUTPUT:.*]]: tensor<2x12x1xf16> {rock.prefill = 0.000000e+00 : f16})
// CHECK-NOT: rock.reduce
// CHECK: %[[BC:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Broadcast{{.*}} : tensor<2x12x1xf16> to tensor<2x12x12xf16>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_add : tensor<2x12x12xf16> -> tensor<2x12x1xf16> to tensor<2x12x12xf16>
func.func @test_reduce_sum_f16(%arg0: tensor<2x12x12xf16>, %arg1: tensor<2x12x1xf16>) -> tensor<2x12x1xf16> {
  %reduced = rock.reduce sum %arg0 {axis = 2 : index} : tensor<2x12x12xf16> -> tensor<2x12x1xf16>
  %result = rock.store %reduced to %arg1 by set : tensor<2x12x1xf16> -> tensor<2x12x1xf16> to tensor<2x12x1xf16>
  return %result : tensor<2x12x1xf16>
}

// CHECK-LABEL: func.func @test_reduce_sum_bf16
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x12x12xbf16>, %[[OUTPUT:.*]]: tensor<2x12x1xbf16> {rock.prefill = 0.000000e+00 : bf16})
// CHECK-NOT: rock.reduce
// CHECK: %[[BC:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Broadcast{{.*}} : tensor<2x12x1xbf16> to tensor<2x12x12xbf16>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_add : tensor<2x12x12xbf16> -> tensor<2x12x1xbf16> to tensor<2x12x12xbf16>
func.func @test_reduce_sum_bf16(%arg0: tensor<2x12x12xbf16>, %arg1: tensor<2x12x1xbf16>) -> tensor<2x12x1xbf16> {
  %reduced = rock.reduce sum %arg0 {axis = 2 : index} : tensor<2x12x12xbf16> -> tensor<2x12x1xbf16>
  %result = rock.store %reduced to %arg1 by set : tensor<2x12x1xbf16> -> tensor<2x12x1xbf16> to tensor<2x12x1xbf16>
  return %result : tensor<2x12x1xbf16>
}

// Verify that an intermediate Merge transform between reduce and store
// is handled: the dest gets an Unmerge (inverse of Merge) followed by Broadcast.

// CHECK-LABEL: func.func @test_reduce_sum_with_intermediate_merge
// CHECK-SAME: (%[[INPUT:.*]]: tensor<1x5x3xf32>, %[[OUTPUT:.*]]: tensor<5xf32> {rock.prefill = 0.000000e+00 : f32})
// CHECK-NOT: rock.reduce
// CHECK: %[[UNMERGE:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Unmerge{{.*}} : tensor<5xf32> to tensor<1x5x1xf32>
// CHECK: %[[BC:.*]] = rock.transform %[[UNMERGE]] by {{.*}}Broadcast{{.*}} : tensor<1x5x1xf32> to tensor<1x5x3xf32>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_add
func.func @test_reduce_sum_with_intermediate_merge(%arg0: tensor<1x5x3xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %reduced = rock.reduce sum %arg0 {axis = 2 : index} : tensor<1x5x3xf32> -> tensor<1x5x1xf32>
  %flat = rock.transform %reduced by <affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 5, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [5] -> [1, 5, 1]> : tensor<1x5x1xf32> to tensor<5xf32>
  %result = rock.store %flat to %arg1 by set : tensor<5xf32> -> tensor<5xf32> to tensor<5xf32>
  return %result : tensor<5xf32>
}
