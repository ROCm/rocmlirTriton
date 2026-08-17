// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-to-ttir --split-input-file | FileCheck %s
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-to-ttir --split-input-file -mlir-print-op-generic | FileCheck %s --check-prefix=IEEE

// CHECK-LABEL: @test_load_conversion
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[ARG0]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:   %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
//      CHECK:   %[[RESULT:.*]] = tt.load %[[PTR_TENSOR]], %[[MASK]], %[[ZERO]] : tensor<64x64x!tt.ptr<f16>>
//      CHECK:   return %[[RESULT]] : tensor<64x64xf16>
//  CHECK-NOT:   rock.blockwise_load_ptr
func.func @test_load_conversion(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi1>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_load_ptr %arg0[%arg1] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// Cache modifier propagation: a blockwise_load_ptr with a non-default cache
// modifier (cs) lowers to a tt.load carrying the corresponding triton cache
// modifier.
// CHECK-LABEL: @test_load_cache_modifier
// CHECK: tt.load
// CHECK-SAME: cacheModifier = cs
func.func @test_load_cache_modifier(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi1>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_load_ptr %arg0[%arg1] {cacheModifier = #rock<CacheModifier cs>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// CHECK-LABEL: @test_store_conversion
// CHECK-SAME: (%[[VALUE:.*]]: tensor<64x64xf32>, %[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
//      CHECK:   tt.store %[[PTR_TENSOR]], %[[VALUE]], %[[MASK]] : tensor<64x64x!tt.ptr<f32>>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_store_ptr
func.func @test_store_conversion(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  rock.blockwise_store_ptr %arg0 -> %arg1(%arg2) by set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// -----

// A blockwise_load_ptr with a 64-bit offset tensor (emitted for large/padded
// index domains) lowers just like the i32 case: the i64 offset tensor is cast
// to a pointer tensor and consumed by tt.load.
// CHECK-LABEL: @test_load_conversion_i64
// CHECK-SAME: (%[[ARG0:.*]]: tensor<64x64xi64>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[ARG0]] : tensor<64x64xi64> -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:   %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf16>
//      CHECK:   %[[RESULT:.*]] = tt.load %[[PTR_TENSOR]], %[[MASK]], %[[ZERO]] : tensor<64x64x!tt.ptr<f16>>
//      CHECK:   return %[[RESULT]] : tensor<64x64xf16>
//  CHECK-NOT:   rock.blockwise_load_ptr
func.func @test_load_conversion_i64(%arg0: tensor<64x64xi64>, %arg1: tensor<64x64xi1>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_load_ptr %arg0[%arg1] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi64>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// As above, but for stores: a 64-bit offset tensor lowers to a pointer-tensor
// cast feeding tt.store.
// CHECK-LABEL: @test_store_conversion_i64
// CHECK-SAME: (%[[VALUE:.*]]: tensor<64x64xf32>, %[[PTRS:.*]]: tensor<64x64xi64>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<64x64xi64> -> tensor<64x64x!tt.ptr<f32>>
//      CHECK:   tt.store %[[PTR_TENSOR]], %[[VALUE]], %[[MASK]] : tensor<64x64x!tt.ptr<f32>>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_store_ptr
func.func @test_store_conversion_i64(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xi64>, %arg2: tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  rock.blockwise_store_ptr %arg0 -> %arg1(%arg2) by set : tensor<64x64xf32> -> tensor<64x64xi64>(tensor<64x64xi1>)
  return
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

// CHECK-LABEL: @test_atomic_add_store
// CHECK-SAME: (%[[VALUE:.*]]: tensor<64x64xf32>, %[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
//      CHECK:   tt.atomic_rmw fadd, relaxed, gpu, %[[PTR_TENSOR]], %[[VALUE]], %[[MASK]]
//  CHECK-NOT:   rock.blockwise_store_ptr
func.func @test_atomic_add_store(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  rock.blockwise_store_ptr %arg0 -> %arg1(%arg2) by atomic_add : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// -----

// A rank-1 reduction produces a scalar tt.reduce result. It is wrapped in a
// rank-0 tensor so it can feed the corresponding rank-0 atomic store.
// CHECK-LABEL: @test_reduce_rank1_atomic_add_store
// CHECK-SAME: (%[[INPUT:.*]]: tensor<256xf32>, %[[PTRS:.*]]: tensor<i32>, %[[MASK:.*]]: tensor<i1>)
//      CHECK:   %[[SCALAR:.*]] = "tt.reduce"(%[[INPUT]]) <{axis = 0 : i32}>
//      CHECK:   ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
//      CHECK:     %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
//      CHECK:     tt.reduce.return %[[SUM]] : f32
//      CHECK:   }) : (tensor<256xf32>) -> f32
//      CHECK:   %[[REDUCED:.*]] = tt.splat %[[SCALAR]] : f32 -> tensor<f32>
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<i32> -> tensor<!tt.ptr<f32>>
//      CHECK:   tt.atomic_rmw fadd, relaxed, gpu, %[[PTR_TENSOR]], %[[REDUCED]], %[[MASK]]
//  CHECK-NOT:   rock.blockwise_reduce
//  CHECK-NOT:   rock.blockwise_store_ptr
func.func @test_reduce_rank1_atomic_add_store(%arg0: tensor<256xf32>, %arg1: tensor<i32>, %arg2: tensor<i1>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_reduce sum %arg0 {axis = 0 : index} : tensor<256xf32> -> tensor<f32>
  rock.blockwise_store_ptr %0 -> %arg1(%arg2) by atomic_add : tensor<f32> -> tensor<i32>(tensor<i1>)
  return
}

// -----

// CHECK-LABEL: @test_atomic_max_store
// CHECK-SAME: (%[[VALUE:.*]]: tensor<64x64xi32>, %[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<i32>>
//      CHECK:   tt.atomic_rmw max, relaxed, gpu, %[[PTR_TENSOR]], %[[VALUE]], %[[MASK]]
//  CHECK-NOT:   rock.blockwise_store_ptr
func.func @test_atomic_max_store(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  rock.blockwise_store_ptr %arg0 -> %arg1(%arg2) by atomic_max : tensor<64x64xi32> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
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
//      CHECK:   %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<32x128xf32>
//      CHECK:   tt.load %[[PTR_TENSOR]], %[[MASK]], %[[ZERO]] : tensor<32x128x!tt.ptr<f32>>
//  CHECK-NOT:   rock.blockwise_load_ptr
func.func @test_load_f32(%arg0: tensor<32x128xi32>, %arg1: tensor<32x128xi1>) -> tensor<32x128xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_load_ptr %arg0[%arg1] {cacheModifier = #rock<CacheModifier none>} : tensor<32x128xi32>, tensor<32x128xi1> -> tensor<32x128xf32>
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
//  CHECK-NOT:   rock.blockwise_load_ptr
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_inside_scf_for(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) -> tensor<64x64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>

  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> tensor<64x64xf32> {
    %a = rock.blockwise_load_ptr %arg0[%arg2] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    %b = rock.blockwise_load_ptr %arg1[%arg2] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
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
func.func @rock_gemm(%arg0: tensor<1024xf16>, %arg1: tensor<1024xf16>, %arg2: tensor<64xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.block_size = 256 : i32, rock.enable_splitk_for_tuning, rock.grid_size = 1 : i32, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 256 : i64} {
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
    %35 = rock.blockwise_load_ptr %33[%34] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
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
    %51 = rock.blockwise_load_ptr %49[%50] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
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
  rock.blockwise_store_ptr %3 -> %18(%12) by  set : tensor<64x64xf32> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// -----

// Verifies integer atomic add uses ADD (not FADD)
// CHECK-LABEL: @test_atomic_add_int
// CHECK-SAME: (%[[VALUE:.*]]: tensor<64x64xi32>, %[[PTRS:.*]]: tensor<64x64xi32>, %[[MASK:.*]]: tensor<64x64xi1>)
//      CHECK:   %[[PTR_TENSOR:.*]] = rock.cast_to_ptr %[[PTRS]] : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<i32>>
//      CHECK:   tt.atomic_rmw add, relaxed, gpu, %[[PTR_TENSOR]], %[[VALUE]], %[[MASK]]
//  CHECK-NOT:   rock.blockwise_store_ptr
func.func @test_atomic_add_int(%arg0: tensor<64x64xi32>, %arg1: tensor<64x64xi32>, %arg2: tensor<64x64xi1>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  rock.blockwise_store_ptr %arg0 -> %arg1(%arg2) by atomic_add : tensor<64x64xi32> -> tensor<64x64xi32>(tensor<64x64xi1>)
  return
}

// -----

// Verifies functions without rock.kernel attribute are left unchanged
// CHECK-LABEL: @test_non_kernel_skipped
// CHECK-SAME: -> tensor<64xf32>
//      CHECK:   rock.blockwise_reduce sum
//      CHECK:   return %{{.*}} : tensor<64xf32>
func.func @test_non_kernel_skipped(%arg0: tensor<64x64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %0 = rock.blockwise_reduce sum %arg0 {axis = 1 : index} : tensor<64x64xf32> -> tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// Verifies f16 reduce uses addf with f16 element type
// CHECK-LABEL: @test_reduce_sum_f16
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64x64xf16>)
//      CHECK:   "tt.reduce"(%[[INPUT]]) <{axis = 1 : i32}>
//      CHECK:   ^bb0(%[[LHS:.*]]: f16, %[[RHS:.*]]: f16):
//      CHECK:     %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f16
//      CHECK:     tt.reduce.return %[[SUM]] : f16
//  CHECK-NOT:   rock.blockwise_reduce
func.func @test_reduce_sum_f16(%arg0: tensor<64x64xf16>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = rock.blockwise_reduce sum %arg0 {axis = 1 : index} : tensor<64x64xf16> -> tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// Test: f8 GEMM without scales lowers to tt.dot (not tt.dot_scaled).
// f8E4M3FN is a TT_Float type, so LegalizeFloatTypes does not set
// matrixAOrigElemType/matrixBOrigElemType and no scales are present.

// CHECK-LABEL: @test_unscaled_gemm_f8
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xf8E4M3FN>, %[[B:.*]]: tensor<64x64xf8E4M3FN>, %[[C:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.dot %[[A]], %[[B]], %[[C]] : tensor<64x64xf8E4M3FN> * tensor<64x64xf8E4M3FN> -> tensor<64x64xf32>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_unscaled_gemm_f8(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: f32 GEMM lowers to tt.dot with the default IEEE input precision on an
// arch that does not prefer the 3xBF16 decomposition.

// CHECK-LABEL: @test_gemm_f32
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xf32>, %[[B:.*]]: tensor<64x64xf32>, %[[C:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.dot %[[A]], %[[B]], %[[C]] : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_f32(
    %a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf32>, tensor<64x64xf32>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: on gfx950 an f32 GEMM asks for the 3xBF16 decomposition, which
// TritonGPUF32DotTC later expands into dense bf16 MFMAs. Absent
// `rock.use_bf16x3_for_f32`, the arch default decides.

// CHECK-LABEL: @test_gemm_f32_gfx950
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xf32>, %[[B:.*]]: tensor<64x64xf32>, %[[C:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.dot %[[A]], %[[B]], %[[C]], inputPrecision = bf16x3 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_f32_gfx950(
    %a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf32>, tensor<64x64xf32>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: `useBf16x3ForF32 = 0` in the perfConfig forces the IEEE path even on gfx950,
// where the arch default would decompose. The tri-state is consumed here, so
// the bridge attribute must not survive into Triton IR.

// CHECK-LABEL: @test_gemm_f32_gfx950_bf16x3_off
//  CHECK-NOT:   rock.use_bf16x3_for_f32
//      CHECK:   %[[RESULT:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
//  CHECK-NOT:   rock.blockwise_gemm
// IEEE-LABEL:   sym_name = "test_gemm_f32_gfx950_bf16x3_off"
// IEEE:         "tt.dot"({{.*}}) <{inputPrecision = 2 : i32,
func.func @test_gemm_f32_gfx950_bf16x3_off(
    %a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel,
                rock.use_bf16x3_for_f32 = 0 : i64} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf32>, tensor<64x64xf32>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: `useBf16x3ForF32 = 1` forces the decomposition on an arch whose default is
// the IEEE path.

// CHECK-LABEL: @test_gemm_f32_gfx942_bf16x3_on
//      CHECK:   %[[RESULT:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}}, inputPrecision = bf16x3 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_f32_gfx942_bf16x3_on(
    %a: tensor<64x64xf32>, %b: tensor<64x64xf32>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.kernel,
                rock.use_bf16x3_for_f32 = 1 : i64} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf32>, tensor<64x64xf32>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: the 3xBF16 decomposition is f32-only -- an f16 GEMM on gfx950 keeps the
// default IEEE input precision (which tt.dot ignores for non-f32 operands).

// CHECK-LABEL: @test_gemm_f16_gfx950
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xf16>, %[[B:.*]]: tensor<64x64xf16>, %[[C:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.dot %[[A]], %[[B]], %[[C]] : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_f16_gfx950(
    %a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: i8 GEMM lowers to tt.dot with i32 accumulator.

// CHECK-LABEL: @test_gemm_i8
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xi8>, %[[B:.*]]: tensor<64x64xi8>, %[[C:.*]]: tensor<64x64xi32>)
//      CHECK:   %[[RESULT:.*]] = tt.dot %[[A]], %[[B]], %[[C]] : tensor<64x64xi8> * tensor<64x64xi8> -> tensor<64x64xi32>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_i8(
    %a: tensor<64x64xi8>, %b: tensor<64x64xi8>,
    %c: tensor<64x64xi32>) -> tensor<64x64xi32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xi8>, tensor<64x64xi8>,
      tensor<64x64xi32> -> tensor<64x64xi32>
  return %result : tensor<64x64xi32>
}

// -----

// Test: FNUZ fp8 mixed GEMM (f8E4M3FNUZ x f8E5M2FNUZ) lowers to tt.dot.

// CHECK-LABEL: @test_gemm_f8_fnuz
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xf8E4M3FNUZ>, %[[B:.*]]: tensor<64x64xf8E5M2FNUZ>, %[[C:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.dot %[[A]], %[[B]], %[[C]] : tensor<64x64xf8E4M3FNUZ> * tensor<64x64xf8E5M2FNUZ> -> tensor<64x64xf32>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_f8_fnuz(
    %a: tensor<64x64xf8E4M3FNUZ>, %b: tensor<64x64xf8E5M2FNUZ>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf8E4M3FNUZ>, tensor<64x64xf8E5M2FNUZ>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: Mixed OCP fp8 GEMM (f8E4M3FN x f8E5M2) lowers to tt.dot.

// CHECK-LABEL: @test_gemm_f8_ocp_mixed
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xf8E4M3FN>, %[[B:.*]]: tensor<64x64xf8E5M2>, %[[C:.*]]: tensor<64x64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.dot %[[A]], %[[B]], %[[C]] : tensor<64x64xf8E4M3FN> * tensor<64x64xf8E5M2> -> tensor<64x64xf32>
//      CHECK:   return
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_f8_ocp_mixed(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E5M2>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf8E4M3FN>, tensor<64x64xf8E5M2>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: Scaled GEMM with f8 data and i8 scales lowers to tt.dot_scaled.

// CHECK-LABEL: @test_scaled_gemm_f8
// CHECK-SAME: (%[[A:.*]]: tensor<64x64xf8E4M3FN>, %[[B:.*]]: tensor<64x64xf8E4M3FN>, %[[C:.*]]: tensor<64x64xf32>, %[[SA:.*]]: tensor<64x2xi8>, %[[SB:.*]]: tensor<64x2xi8>)
//      CHECK:   tt.dot_scaled %[[A]] scale %[[SA]], %[[B]] scale %[[SB]], %[[C]]
// CHECK-SAME:     lhs = e4m3 rhs = e4m3
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_scaled_gemm_f8(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %c: tensor<64x64xf32>,
    %scaleA: tensor<64x2xi8>, %scaleB: tensor<64x2xi8>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: Scaled GEMM with i8 data (packed f4E2M1FN) and matrixAOrigElemType/matrixBOrigElemType.
// The original f4E2M1FN type maps to e2m1 in tt.dot_scaled.
// matrixAKPack=true, matrixBKPack=false tests asymmetric packing.

// CHECK-LABEL: @test_scaled_gemm_f4_packed
// CHECK-SAME: (%[[A:.*]]: tensor<64x32xi8>, %[[B:.*]]: tensor<64x32xi8>, %[[C:.*]]: tensor<64x64xf32>, %[[SA:.*]]: tensor<64x2xi8>, %[[SB:.*]]: tensor<64x2xi8>)
//      CHECK:   tt.dot_scaled %[[A]] scale %[[SA]], %[[B]] scale %[[SB]], %[[C]]
// CHECK-SAME:     lhs = e2m1 rhs = e2m1
// CHECK-SAME:     rhs_k_pack = false
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_scaled_gemm_f4_packed(
    %a: tensor<64x32xi8>, %b: tensor<64x32xi8>,
    %c: tensor<64x64xf32>,
    %scaleA: tensor<64x2xi8>, %scaleB: tensor<64x2xi8>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  // matrixA is MxK: K halved from 64 to 32 (matrixAKPack=true)
  // matrixB is KxN: N halved from 64 to 32 (matrixBKPack=false, D packed)
  %result = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64,
     matrixAOrigElemType = f4E2M1FN,
     matrixBOrigElemType = f4E2M1FN,
     matrixAKPack = true,
     matrixBKPack = false}
    : tensor<64x32xi8> scaled by tensor<64x2xi8>,
      tensor<64x32xi8> scaled by tensor<64x2xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: Mixed-type scaled GEMM. A is packed f4 (i8 with matrixAOrigElemType),
// B is f8E4M3FN (no matrixBOrigElemType since it's a TT_Float).
// Only matrixAOrigElemType is set; B falls back to its tensor element type
// via .value_or() in RockToTTIR.

// CHECK-LABEL: @test_scaled_gemm_mixed_f4_f8
// CHECK-SAME: (%[[A:.*]]: tensor<64x32xi8>, %[[B:.*]]: tensor<64x64xf8E4M3FN>, %[[C:.*]]: tensor<64x64xf32>, %[[SA:.*]]: tensor<64x2xi8>, %[[SB:.*]]: tensor<64x2xi8>)
//      CHECK:   tt.dot_scaled %[[A]] scale %[[SA]], %[[B]] scale %[[SB]], %[[C]]
// CHECK-SAME:     lhs = e2m1 rhs = e4m3
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_scaled_gemm_mixed_f4_f8(
    %a: tensor<64x32xi8>, %b: tensor<64x64xf8E4M3FN>,
    %c: tensor<64x64xf32>,
    %scaleA: tensor<64x2xi8>, %scaleB: tensor<64x2xi8>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64,
     matrixAOrigElemType = f4E2M1FN,
     matrixAKPack = true}
    : tensor<64x32xi8> scaled by tensor<64x2xi8>,
      tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: arith.truncf with FP8 result is converted to tt.fp_to_fp (RTNE).
// Triton's LLVM lowering cannot handle arith.truncf with FP8 output directly
// (it creates an illegal llvm.fptrunc to i8).

// CHECK-LABEL: @test_truncf_f32_to_f8E4M3FN
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x2xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.fp_to_fp %[[INPUT]], rounding = rtne : tensor<2x2xf32> -> tensor<2x2xf8E4M3FN>
//  CHECK-NOT:   arith.truncf
func.func @test_truncf_f32_to_f8E4M3FN(%arg0: tensor<2x2xf32>) -> tensor<2x2xf8E4M3FN> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<2x2xf32> to tensor<2x2xf8E4M3FN>
  return %0 : tensor<2x2xf8E4M3FN>
}

// -----

// Test: arith.truncf with FP8 E5M2 result is also converted to tt.fp_to_fp.

// CHECK-LABEL: @test_truncf_f32_to_f8E5M2
// CHECK-SAME: (%[[INPUT:.*]]: tensor<4xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.fp_to_fp %[[INPUT]], rounding = rtne : tensor<4xf32> -> tensor<4xf8E5M2>
//  CHECK-NOT:   arith.truncf
func.func @test_truncf_f32_to_f8E5M2(%arg0: tensor<4xf32>) -> tensor<4xf8E5M2> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<4xf32> to tensor<4xf8E5M2>
  return %0 : tensor<4xf8E5M2>
}

// -----

// Test: arith.extf with FP8 input is converted to tt.fp_to_fp (no rounding).

// CHECK-LABEL: @test_extf_f8E4M3FN_to_f32
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x2xf8E4M3FN>)
//      CHECK:   %[[RESULT:.*]] = tt.fp_to_fp %[[INPUT]] : tensor<2x2xf8E4M3FN> -> tensor<2x2xf32>
//  CHECK-NOT:   arith.extf
func.func @test_extf_f8E4M3FN_to_f32(%arg0: tensor<2x2xf8E4M3FN>) -> tensor<2x2xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.extf %arg0 : tensor<2x2xf8E4M3FN> to tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// Test: Non-FP8 arith.truncf (f32 → f16) remains unchanged.

// CHECK-LABEL: @test_truncf_f32_to_f16_unchanged
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64xf32>)
//      CHECK:   %[[RESULT:.*]] = arith.truncf %[[INPUT]] : tensor<64xf32> to tensor<64xf16>
//  CHECK-NOT:   tt.fp_to_fp
func.func @test_truncf_f32_to_f16_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.truncf %arg0 : tensor<64xf32> to tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// Test: Non-FP8 arith.extf (f16 → f32) remains unchanged.

// CHECK-LABEL: @test_extf_f16_to_f32_unchanged
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64xf16>)
//      CHECK:   %[[RESULT:.*]] = arith.extf %[[INPUT]] : tensor<64xf16> to tensor<64xf32>
//  CHECK-NOT:   tt.fp_to_fp
func.func @test_extf_f16_to_f32_unchanged(%arg0: tensor<64xf16>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.extf %arg0 : tensor<64xf16> to tensor<64xf32>
  return %0 : tensor<64xf32>
}

// -----

// Test: an explicit to_nearest_even on a non-FP8 truncf keeps it in arith.
// RTNE is already the default, so there is nothing for tt.fp_to_fp to add.
// This matches Triton's semantic.cast, where `use_custom_rounding` is only set
// for a requested mode that differs from RTNE.

// CHECK-LABEL: @test_truncf_f32_to_f16_rtne_unchanged
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64xf32>)
//      CHECK:   %[[RESULT:.*]] = arith.truncf %[[INPUT]] to_nearest_even : tensor<64xf32> to tensor<64xf16>
//  CHECK-NOT:   tt.fp_to_fp
func.func @test_truncf_f32_to_f16_rtne_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.truncf %arg0 to_nearest_even : tensor<64xf32> to tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// Test: a non-FP8 truncf asking for toward_zero becomes tt.fp_to_fp with RTZ.
// Triton's AMD arith.truncf lowering ignores the rounding attribute and always
// uses RTNE for f32 -> f16/bf16, so RTZ has to leave arith to be honoured.

// CHECK-LABEL: @test_truncf_f32_to_f16_rtz
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.fp_to_fp %[[INPUT]], rounding = rtz : tensor<64xf32> -> tensor<64xf16>
//  CHECK-NOT:   arith.truncf
func.func @test_truncf_f32_to_f16_rtz(%arg0: tensor<64xf32>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.truncf %arg0 toward_zero : tensor<64xf32> to tensor<64xf16>
  return %0 : tensor<64xf16>
}

// -----

// Test: the same holds for a bf16 destination.

// CHECK-LABEL: @test_truncf_f32_to_bf16_rtz
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x64xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.fp_to_fp %[[INPUT]], rounding = rtz : tensor<2x64xf32> -> tensor<2x64xbf16>
//  CHECK-NOT:   arith.truncf
func.func @test_truncf_f32_to_bf16_rtz(%arg0: tensor<2x64xf32>) -> tensor<2x64xbf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.truncf %arg0 toward_zero : tensor<2x64xf32> to tensor<2x64xbf16>
  return %0 : tensor<2x64xbf16>
}

// -----

// Test: an FP8 destination carries the requested RTZ through rather than
// falling back to the RTNE default used when no rounding mode is present.

// CHECK-LABEL: @test_truncf_f32_to_f8E5M2_rtz
// CHECK-SAME: (%[[INPUT:.*]]: tensor<4xf32>)
//      CHECK:   %[[RESULT:.*]] = tt.fp_to_fp %[[INPUT]], rounding = rtz : tensor<4xf32> -> tensor<4xf8E5M2>
//  CHECK-NOT:   arith.truncf
func.func @test_truncf_f32_to_f8E5M2_rtz(%arg0: tensor<4xf32>) -> tensor<4xf8E5M2> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %0 = arith.truncf %arg0 toward_zero : tensor<4xf32> to tensor<4xf8E5M2>
  return %0 : tensor<4xf8E5M2>
}

// -----

// Test: discardable rock metadata (rock.o_transposed) set on rock.blockwise_gemm
// is carried onto the lowered tt.dot so it survives into the Triton pipeline.

// CHECK-LABEL: @test_gemm_carries_otransposed
//      CHECK:   tt.dot
// CHECK-SAME:   rock.o_transposed = #rock.o_transposed<true>
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_gemm_carries_otransposed(
    %a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a, %b, %c)
    {rock.o_transposed = #rock.o_transposed<true>}
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: discardable rock metadata is also carried onto the lowered
// tt.dot_scaled (scaled GEMM path).

// CHECK-LABEL: @test_scaled_gemm_carries_otransposed
//      CHECK:   tt.dot_scaled
// CHECK-SAME:   rock.o_transposed = #rock.o_transposed<false>
//  CHECK-NOT:   rock.blockwise_gemm
func.func @test_scaled_gemm_carries_otransposed(
    %a: tensor<64x64xf8E4M3FN>, %b: tensor<64x64xf8E4M3FN>,
    %c: tensor<64x64xf32>,
    %scaleA: tensor<64x2xi8>, %scaleB: tensor<64x2xi8>) -> tensor<64x64xf32>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %result = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64, rock.o_transposed = #rock.o_transposed<false>}
    : tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf8E4M3FN> scaled by tensor<64x2xi8>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}

// -----

// Test: a `maxnumf(minnumf(x, hi), lo)` clamp whose operands may ignore NaNs
// becomes a single tt.clampf, which AMD lowers to one v_med3.

// CHECK-LABEL: @test_clamp_nnan_to_clampf
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64xf32>)
//  CHECK-DAG:   %[[LO:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
//  CHECK-DAG:   %[[HI:.*]] = arith.constant dense<6.000000e+00> : tensor<64xf32>
//      CHECK:   %[[RESULT:.*]] = tt.clampf %[[INPUT]], %[[LO]], %[[HI]], propagateNan = none : tensor<64xf32>
//  CHECK-NOT:   arith.minnumf
//  CHECK-NOT:   arith.maxnumf
func.func @test_clamp_nnan_to_clampf(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: the same idiom on f16, the other element type AMD contracts to v_med3.

// CHECK-LABEL: @test_clamp_f16_to_clampf
//      CHECK:   tt.clampf {{.*}}propagateNan = none : tensor<64xf16>
//  CHECK-NOT:   arith.minnumf
func.func @test_clamp_f16_to_clampf(%arg0: tensor<64xf16>) -> tensor<64xf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf16>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf16>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf16>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf16>
  return %1 : tensor<64xf16>
}

// -----

// Test: without `nnan` the pair stays put. minnumf/maxnumf send a NaN input to
// `hi`, whereas v_med3 would return `lo`, so the rewrite is not equivalent.

// CHECK-LABEL: @test_clamp_no_nnan_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_no_nnan_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: `nnan` on only one of the two ops is not enough.

// CHECK-LABEL: @test_clamp_partial_nnan_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_partial_nnan_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: the NaN-propagating spelling is left alone. A tt.clampf with
// propagateNan = none would not be a faithful replacement for it.

// CHECK-LABEL: @test_clamp_propagating_ops_unchanged
//      CHECK:   arith.minimumf
//      CHECK:   arith.maximumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_propagating_ops_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minimumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maximumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: unordered bounds are rejected. For lo > hi the pair saturates to `lo`
// while v_med3 would return `hi`.

// CHECK-LABEL: @test_clamp_unordered_bounds_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_unordered_bounds_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: non-constant bounds cannot be proven ordered, so the pair stays put.

// CHECK-LABEL: @test_clamp_dynamic_bound_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_dynamic_bound_unchanged(%arg0: tensor<64xf32>, %hi: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: bf16 converts too. AMD's v_med3 path skips it, but Triton's generic
// lowering emits the min-outer form that LLVM can contract into fmed3.

// CHECK-LABEL: @test_clamp_bf16_to_clampf
//      CHECK:   tt.clampf {{.*}}propagateNan = none : tensor<64xbf16>
//  CHECK-NOT:   arith.minnumf
func.func @test_clamp_bf16_to_clampf(%arg0: tensor<64xbf16>) -> tensor<64xbf16> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xbf16>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xbf16>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xbf16>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xbf16>
  return %1 : tensor<64xbf16>
}

// -----

// Test: f64 converts as well, for the same reason as bf16.

// CHECK-LABEL: @test_clamp_f64_to_clampf
//      CHECK:   tt.clampf {{.*}}propagateNan = none : tensor<64xf64>
//  CHECK-NOT:   arith.minnumf
func.func @test_clamp_f64_to_clampf(%arg0: tensor<64xf64>) -> tensor<64xf64> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf64>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf64>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf64>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf64>
  return %1 : tensor<64xf64>
}

// -----

// Test: FP8 is rejected. tt.clampf would accept it, but Triton maps FP8 to i8
// and the lowering would end up comparing integers.

// CHECK-LABEL: @test_clamp_f8_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_f8_unchanged(%arg0: tensor<64xf8E4M3FN>) -> tensor<64xf8E4M3FN> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf8E4M3FN>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf8E4M3FN>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf8E4M3FN>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf8E4M3FN>
  return %1 : tensor<64xf8E4M3FN>
}

// -----

// Test: a min feeding a second consumer is left alone, since folding it into
// the clamp would duplicate it rather than remove it.

// CHECK-LABEL: @test_clamp_multi_use_min_unchanged
//      CHECK:   arith.minnumf
//      CHECK:   arith.maxnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_multi_use_min_unchanged(%arg0: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.minnumf %arg0, %hi fastmath<nnan> : tensor<64xf32>
  %1 = arith.maxnumf %0, %lo fastmath<nnan> : tensor<64xf32>
  return %1, %0 : tensor<64xf32>, tensor<64xf32>
}

// -----

// Test: the min-outer spelling `minnumf(maxnumf(x, lo), hi)`, which is what
// migraphx.clip lowers to, folds just like the max-outer one. Under nnan the
// two orderings agree.

// CHECK-LABEL: @test_clamp_min_outer_to_clampf
// CHECK-SAME: (%[[INPUT:.*]]: tensor<64xf32>)
//  CHECK-DAG:   %[[LO:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
//  CHECK-DAG:   %[[HI:.*]] = arith.constant dense<6.000000e+00> : tensor<64xf32>
//      CHECK:   %[[RESULT:.*]] = tt.clampf %[[INPUT]], %[[LO]], %[[HI]], propagateNan = none : tensor<64xf32>
//  CHECK-NOT:   arith.minnumf
//  CHECK-NOT:   arith.maxnumf
func.func @test_clamp_min_outer_to_clampf(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.maxnumf %arg0, %lo fastmath<nnan> : tensor<64xf32>
  %1 = arith.minnumf %0, %hi fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: the min-outer spelling still needs nnan on both operations.

// CHECK-LABEL: @test_clamp_min_outer_no_nnan_unchanged
//      CHECK:   arith.maxnumf
//      CHECK:   arith.minnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_min_outer_no_nnan_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %0 = arith.maxnumf %arg0, %lo : tensor<64xf32>
  %1 = arith.minnumf %0, %hi : tensor<64xf32>
  return %1 : tensor<64xf32>
}

// -----

// Test: unordered bounds are rejected in the min-outer spelling too, where the
// original saturates to hi while v_med3 would return lo.

// CHECK-LABEL: @test_clamp_min_outer_unordered_bounds_unchanged
//      CHECK:   arith.maxnumf
//      CHECK:   arith.minnumf
//  CHECK-NOT:   tt.clampf
func.func @test_clamp_min_outer_unordered_bounds_unchanged(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
  %lo = arith.constant dense<6.000000e+00> : tensor<64xf32>
  %hi = arith.constant dense<0.000000e+00> : tensor<64xf32>
  %0 = arith.maxnumf %arg0, %lo fastmath<nnan> : tensor<64xf32>
  %1 = arith.minnumf %0, %hi fastmath<nnan> : tensor<64xf32>
  return %1 : tensor<64xf32>
}

