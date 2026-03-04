// RUN: rocmlir-opt -rock-affix-params -rock-lower-reduce -mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: test_gemm_reduce_last_axis_fusion
// CHECK-SAME: %arg2: tensor<1x128x1xf32> {rock.prefill = 0.000000e+00 : f32}
// CHECK-NOT: rock.reduce
// CHECK: rock.transform %arg2 by {{.*}}Broadcast{{.*}} : tensor<1x128x1xf32> to tensor<1x128x256xf32>
// CHECK: rock.store %{{.*}} by atomic_add
func.func @test_gemm_reduce_last_axis_fusion(%arg0: tensor<1x128x64xf32>, %arg1: tensor<1x64x256xf32>, %arg2: tensor<1x128x1xf32>) -> tensor<1x128x1xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-", rock.kernel} {
  %gemm = rock.gemm %arg0 * %arg1 : tensor<1x128x64xf32> * tensor<1x64x256xf32> -> tensor<1x128x256xf32>
  %reduced = rock.reduce sum %gemm {axis = 2 : index} : tensor<1x128x256xf32> -> tensor<1x128x1xf32>
  %out = rock.store %reduced to %arg2 by set : tensor<1x128x1xf32> -> tensor<1x128x1xf32> to tensor<1x128x1xf32>
  return %out : tensor<1x128x1xf32>
}


// CHECK-LABEL: test_gemm_reduce_middle_axis_fusion
// CHECK-SAME: %arg2: tensor<1x1x256xf32> {rock.prefill = 0.000000e+00 : f32}
// CHECK-NOT: rock.reduce
// CHECK: rock.transform %arg2 by {{.*}}Broadcast{{.*}} : tensor<1x1x256xf32> to tensor<1x128x256xf32>
// CHECK: rock.store %{{.*}} by atomic_add
func.func @test_gemm_reduce_middle_axis_fusion(%arg0: tensor<1x128x64xf32>, %arg1: tensor<1x64x256xf32>, %arg2: tensor<1x1x256xf32>) -> tensor<1x1x256xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-", rock.kernel} {
  %gemm = rock.gemm %arg0 * %arg1 : tensor<1x128x64xf32> * tensor<1x64x256xf32> -> tensor<1x128x256xf32>
  %reduced = rock.reduce sum %gemm {axis = 1 : index} : tensor<1x128x256xf32> -> tensor<1x1x256xf32>
  %out = rock.store %reduced to %arg2 by set : tensor<1x1x256xf32> -> tensor<1x1x256xf32> to tensor<1x1x256xf32>
  return %out : tensor<1x1x256xf32>
}

// CHECK-LABEL: test_gemm_add_reduce_fusion
// CHECK-SAME: %arg3: tensor<1x128x1xf32> {rock.prefill = 0.000000e+00 : f32}
// CHECK-NOT: rock.reduce
// CHECK: rock.transform %arg3 by {{.*}}Broadcast{{.*}} : tensor<1x128x1xf32> to tensor<1x128x256xf32>
// CHECK: rock.store %{{.*}} by atomic_add
func.func @test_gemm_add_reduce_fusion(%arg0: tensor<1x128x64xf32>, %arg1: tensor<1x64x256xf32>, %arg2: tensor<1x128x256xf32>, %arg3: tensor<1x128x1xf32>) -> tensor<1x128x1xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-", rock.kernel} {
  %gemm = rock.gemm %arg0 * %arg1 : tensor<1x128x64xf32> * tensor<1x64x256xf32> -> tensor<1x128x256xf32>
  %fused_add = arith.addf %gemm, %arg2 : tensor<1x128x256xf32>
  %reduced = rock.reduce sum %fused_add {axis = 2 : index} : tensor<1x128x256xf32> -> tensor<1x128x1xf32>
  %out = rock.store %reduced to %arg3 by set : tensor<1x128x1xf32> -> tensor<1x128x1xf32> to tensor<1x128x1xf32>
  return %out : tensor<1x128x1xf32>
}

// CHECK-LABEL: test_gemm_reduce_max
// CHECK-SAME: %arg2: tensor<1x128x1xf32> {rock.prefill = 0xFF800000 : f32}
// CHECK-NOT: rock.reduce
// CHECK: rock.transform %arg2 by {{.*}}Broadcast{{.*}} : tensor<1x128x1xf32> to tensor<1x128x256xf32>
// CHECK: rock.store %{{.*}} by atomic_max
func.func @test_gemm_reduce_max(%arg0: tensor<1x128x64xf32>, %arg1: tensor<1x64x256xf32>, %arg2: tensor<1x128x1xf32>) -> tensor<1x128x1xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-", rock.kernel} {
  %gemm = rock.gemm %arg0 * %arg1 : tensor<1x128x64xf32> * tensor<1x64x256xf32> -> tensor<1x128x256xf32>
  %reduced = rock.reduce max %gemm {axis = 2 : index} : tensor<1x128x256xf32> -> tensor<1x128x1xf32>
  %out = rock.store %reduced to %arg2 by set : tensor<1x128x1xf32> -> tensor<1x128x1xf32> to tensor<1x128x1xf32>
  return %out : tensor<1x128x1xf32>
}
