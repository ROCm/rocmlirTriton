// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-to-ttir --split-input-file | FileCheck %s

// CHECK-LABEL: @test_load_conversion
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[ARG0]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:   %[[RESULT:.*]] = tt.load %[[PTR_TENSOR]], %[[MASK]] : tensor<64x64x!tt.ptr<f16>>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_load_tile_ptr
func.func @test_load_conversion(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi1>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_load_tile_ptr %arg0[%arg1] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// CHECK-LABEL: @test_store_conversion
// CHECK-SAME: (%[[VALUE:.*]]: tensor<64x64xf32>, %[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
//      CHECK:   tt.store %[[PTR_TENSOR]], %[[VALUE]], %[[MASK]] : tensor<64x64x!tt.ptr<f32>>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_store_tile_ptr
func.func @test_store_conversion(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) -> tensor<64x64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_store_tile_ptr %arg0 -> %arg1(%arg2) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: @test_gemm_conversion
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xf16>, %[[B:.*]]: tensor<64x64xf16>, %[[C:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.dot %[[A]], %[[B]], %[[C]] : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_conversion(%arg0: tensor<64x64xf16>, %arg1: tensor<64x64xf16>, %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_gemm(%arg0, %arg1, %arg2) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: @test_return_conversion
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xf32>, %[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   tt.store
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_store_tile_ptr
func.func @test_return_conversion(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) -> tensor<64x64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_store_tile_ptr %arg0 -> %arg1(%arg2) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: @test_atomic_add_store
// CHECK-SAME: (%[[VALUE:.*]]: tensor<64x64xf32>, %[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
//      CHECK:   tt.atomic_rmw fadd, relaxed, gpu, %[[PTR_TENSOR]], %[[VALUE]], %[[MASK]]
//  CHECK-NOT:   rock.blockwise_store_tile_ptr
func.func @test_atomic_add_store(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) -> tensor<64x64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_store_tile_ptr %arg0 -> %arg1(%arg2) by atomic_add : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: @test_atomic_max_store
// CHECK-SAME: (%[[VALUE:.*]]: tensor<64x64xi32>, %[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<i32>>
//      CHECK:   tt.atomic_rmw max, relaxed, gpu, %[[PTR_TENSOR]], %[[VALUE]], %[[MASK]]
//  CHECK-NOT:   rock.blockwise_store_tile_ptr
func.func @test_atomic_max_store(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) -> tensor<64x64xi32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_store_tile_ptr %arg0 -> %arg1(%arg2) by atomic_max : tensor<64x64xi32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64x64xi32>
  return %0 : tensor<64x64xi32>
}

// -----

// CHECK-LABEL: @test_reduce_sum
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = "tt.reduce"(%[[INPUT]]) <{axis = 1 : i32}>
//      CHECK:   ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//      CHECK:     %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
//      CHECK:     tt.reduce.return %[[SUM]] : f32
//      CHECK:   }) : (tensor<64x64xf32>) -> tensor<64xf32>
//  CHECK-NOT:   rock.blockwise_reduce
func.func @test_reduce_sum(%arg0: tensor<64x64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_reduce sum %arg0 {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @test_reduce_max
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = "tt.reduce"(%[[INPUT]]) <{axis = 0 : i32}>
//      CHECK:   ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//      CHECK:     %[[MAX:.*]] = arith.maximumf %[[LHS]], %[[RHS]] : f32
//      CHECK:     tt.reduce.return %[[MAX]] : f32
//      CHECK:   }) : (tensor<64x64xf32>) -> tensor<64xf32>
//  CHECK-NOT:   rock.blockwise_reduce
func.func @test_reduce_max(%arg0: tensor<64x64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_reduce max %arg0 {axis = 0 : index} : tensor<64x64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// CHECK-LABEL: @test_reduce_sum_int
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64x64xi32>)
//      CHECK:   "tt.reduce"(%[[INPUT]]) <{axis = 1 : i32}>
//      CHECK:   ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32):
//      CHECK:     %[[SUM:.*]] = arith.addi %[[LHS]], %[[RHS]] : i32
//      CHECK:     tt.reduce.return %[[SUM]] : i32
//  CHECK-NOT:   rock.blockwise_reduce
func.func @test_reduce_sum_int(%arg0: tensor<64x64xi32>) -> tensor<64xi32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_reduce sum %arg0 {axis = 1 : index} : tensor<64x64xi32> -> tensor<64xi32>
  return %0 : tensor<64xi32>
}

// -----

// CHECK-LABEL: @test_reduce_max_int
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64x64xi32>)
//      CHECK:   "tt.reduce"(%[[INPUT]]) <{axis = 1 : i32}>
//      CHECK:   ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32):
//      CHECK:     %[[MAX:.*]] = arith.maxsi %[[LHS]], %[[RHS]] : i32
//      CHECK:     tt.reduce.return %[[MAX]] : i32
//  CHECK-NOT:   rock.blockwise_reduce
func.func @test_reduce_max_int(%arg0: tensor<64x64xi32>) -> tensor<64xi32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_reduce max %arg0 {axis = 1 : index} : tensor<64x64xi32> -> tensor<64xi32>
  return %0 : tensor<64xi32>
}

// -----

// CHECK-LABEL: @test_load_f32
// CHECK-SAME: (%[[PTRS:.*]]: tensor<32x128xi32>, %[[MASK:.*]]: tensor<32x128xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<32x128xi32> -> tensor<32x128x!tt.ptr<f32>>
//      CHECK:   tt.load %[[PTR_TENSOR]], %[[MASK]] : tensor<32x128x!tt.ptr<f32>>
//  CHECK-NOT:   rock.blockwise_load_tile_ptr
func.func @test_load_f32(%arg0: tensor<32x128xi32>, %arg1: tensor<32x128xi1>) -> tensor<32x128xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_load_tile_ptr %arg0[%arg1] : tensor<32x128xi32>, tensor<32x128xi1> -> tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}

// -----

// CHECK-LABEL: @test_inside_scf_for
// CHECK-SAME: (%[[PTRS_A:.*]]: tensor<64x64xi32>, %[[PTRS_B:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
//      CHECK:   scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<64x64xf32>)
//      CHECK:     rock.cast_to_ptr {{.*}} -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>>
//      CHECK:     rock.cast_to_ptr {{.*}} -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.dot {{.*}} : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
//      CHECK:     scf.yield
//  CHECK-NOT:   rock.blockwise_load_tile_ptr
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_inside_scf_for(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) -> tensor<64x64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>

  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> tensor<64x64xf32> {
    %a = rock.blockwise_load_tile_ptr %arg0[%arg2] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    %b = rock.blockwise_load_tile_ptr %arg1[%arg2] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    %c = rock.blockwise_gemm(%a, %b, %acc) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
    scf.yield %c : tensor<64x64xf32>
  }

  return %result : tensor<64x64xf32>
}