// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-to-ttir --split-input-file | FileCheck %s

// CHECK-LABEL: @test_load_conversion
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[ARG0]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:   %[[RESULT:.*]] = tt.load %[[PTR_TENSOR]], %[[MASK]] : tensor<64x64x!tt.ptr<f16>>
//      CHECK:   return
//      CHECK:   }
//  CHECK-NOT:   rock.blockwise_load_tile_ptr
//  CHECK-NOT:   return %{{.*}}
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

// -----

// Verifies a real GEMM test case with scf.for loop, loads, gemm, and store
// CHECK-LABEL: @rock_gemm
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xf16>, %[[ARG1:.*]]: tensor<1024xf16>, %[[ARG2:.*]]: tensor<64xf32>)
//      CHECK:   %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
//      CHECK:   %[[PTR_C:.*]] = rock.extract_ptr %[[ARG2]]
//      CHECK:   %[[PTR_A:.*]] = rock.extract_ptr %[[ARG0]]
//      CHECK:   %[[PTR_B:.*]] = rock.extract_ptr %[[ARG1]]
//      CHECK:   scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<64x64xf32>)
//      CHECK:     rock.cast_to_ptr {{.*}} -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>>
//      CHECK:     rock.cast_to_ptr {{.*}} -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.dot {{.*}} : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
//      CHECK:     scf.yield
//      CHECK:   rock.cast_to_ptr {{.*}} -> tensor<64x64x!tt.ptr<f32>>
//      CHECK:   tt.store {{.*}} : tensor<64x64x!tt.ptr<f32>>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_load_ptr
//  CHECK-NOT:   rock.blockwise_gemm
//  CHECK-NOT:   rock.blockwise_store_ptr
func.func @rock_gemm(%arg0: tensor<1024xf16>, %arg1: tensor<1024xf16>, %arg2: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.block_size = 256 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 1 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
  %cst = arith.constant dense<128> : tensor<64x1xi32>
  %cst_0 = arith.constant dense<8> : tensor<64x1xi32>
  %cst_1 = arith.constant dense<8> : tensor<1x64xi32>
  %c64_i32 = arith.constant 64 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = rock.extract_ptr %arg2 : tensor<64xf32> -> i32
  %1 = rock.extract_ptr %arg0 : tensor<1024xf16> -> i32
  %2 = rock.extract_ptr %arg1 : tensor<1024xf16> -> i32
  %3 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %cst_2) -> (tensor<64x64xf32>)  : i32 {
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %21 = tt.expand_dims %20 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %22 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %24 = arith.muli %arg3, %c64_i32 : i32
    %25 = tt.splat %24 : i32 -> tensor<64x1xi32>
    %26 = arith.addi %25, %21 : tensor<64x1xi32>
    %27 = arith.cmpi ult, %23, %cst_1 : tensor<1x64xi32>
    %28 = arith.muli %26, %cst_0 : tensor<64x1xi32>
    %29 = tt.broadcast %28 : tensor<64x1xi32> -> tensor<64x64xi32>
    %30 = tt.broadcast %23 : tensor<1x64xi32> -> tensor<64x64xi32>
    %31 = arith.addi %29, %30 : tensor<64x64xi32>
    %32 = tt.splat %2 : i32 -> tensor<64x64xi32>
    %33 = arith.addi %32, %31 : tensor<64x64xi32>
    %34 = tt.broadcast %27 : tensor<1x64xi1> -> tensor<64x64xi1>
    %35 = rock.blockwise_load_ptr %33[%34] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    %36 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %37 = tt.expand_dims %36 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %38 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %39 = tt.expand_dims %38 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %40 = arith.muli %arg3, %c64_i32 : i32
    %41 = tt.splat %40 : i32 -> tensor<1x64xi32>
    %42 = arith.addi %41, %39 : tensor<1x64xi32>
    %43 = arith.cmpi ult, %37, %cst_0 : tensor<64x1xi32>
    %44 = arith.muli %37, %cst : tensor<64x1xi32>
    %45 = tt.broadcast %44 : tensor<64x1xi32> -> tensor<64x64xi32>
    %46 = tt.broadcast %42 : tensor<1x64xi32> -> tensor<64x64xi32>
    %47 = arith.addi %45, %46 : tensor<64x64xi32>
    %48 = tt.splat %1 : i32 -> tensor<64x64xi32>
    %49 = arith.addi %48, %47 : tensor<64x64xi32>
    %50 = tt.broadcast %43 : tensor<64x1xi1> -> tensor<64x64xi1>
    %51 = rock.blockwise_load_ptr %49[%50] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    %52 = rock.blockwise_gemm(%51, %35, %arg4) : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
    scf.yield %52 : tensor<64x64xf32>
  }
  %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %8 = arith.cmpi ult, %5, %cst_0 : tensor<64x1xi32>
  %9 = arith.cmpi ult, %7, %cst_1 : tensor<1x64xi32>
  %10 = tt.broadcast %9 : tensor<1x64xi1> -> tensor<64x64xi1>
  %11 = tt.broadcast %8 : tensor<64x1xi1> -> tensor<64x64xi1>
  %12 = arith.andi %10, %11 : tensor<64x64xi1>
  %13 = arith.muli %5, %cst_0 : tensor<64x1xi32>
  %14 = tt.broadcast %13 : tensor<64x1xi32> -> tensor<64x64xi32>
  %15 = tt.broadcast %7 : tensor<1x64xi32> -> tensor<64x64xi32>
  %16 = arith.addi %14, %15 : tensor<64x64xi32>
  %17 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %18 = arith.addi %17, %16 : tensor<64x64xi32>
  %19 = rock.blockwise_store_ptr %3 -> %18(%12) by  set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>) -> tensor<64xf32>
  return %19 : tensor<64xf32>
}