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
