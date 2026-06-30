// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-tensor-to-triton-ptr --split-input-file | FileCheck %s

// Verifies func.func with rock.kernel is converted to tt.func with pointer arguments
// CHECK: module attributes {{{.*}}rock.grid_size.test_basic_conversion = 2 : i32
// CHECK-LABEL: tt.func @test_basic_conversion
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>) attributes {noinline = true, rock.arch = "{{.*}}"
//      CHECK:   tt.splat %[[ARG0]] : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:   tt.load
//      CHECK:   tt.return
//  CHECK-NOT:   rock.extract_ptr
//  CHECK-NOT:   rock.cast_to_ptr
//  CHECK-NOT:   func.func
func.func @test_basic_conversion(%arg0: tensor<4096xf16>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 2 : i32, rock.block_size = 256 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %2 = rock.cast_to_ptr %1 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %3 = tt.load %2, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  return
}

// -----

// Verifies arith.addi on pointer tensor is converted to tt.addptr
// CHECK-LABEL: tt.func @test_addi_to_addptr
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>)
//      CHECK:   %[[SPLAT:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
//      CHECK:   %[[OFFSET:.*]] = tt.make_range
//      CHECK:   %[[ADDPTR:.*]] = tt.addptr %[[SPLAT]], %[[OFFSET]] : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
//  CHECK-NOT:   rock.extract_ptr
//  CHECK-NOT:   rock.cast_to_ptr
func.func @test_addi_to_addptr(%arg0: tensor<4096xf16>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64xi32>
  %offset = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %2 = arith.addi %1, %offset : tensor<64xi32>
  %3 = rock.cast_to_ptr %2 : tensor<64xi32> -> tensor<64x!tt.ptr<f16>>
  %4 = tt.load %3, %cst_mask : tensor<64x!tt.ptr<f16>>
  return
}

// -----

// Verifies multiple pointer arguments are all converted
// CHECK-LABEL: tt.func @test_multiple_args
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>, %[[ARG1:.*]]: !tt.ptr<f16>, %[[ARG2:.*]]: !tt.ptr<f32>)
//      CHECK:   tt.splat %[[ARG0]]
//      CHECK:   tt.splat %[[ARG1]]
//      CHECK:   tt.splat %[[ARG2]]
//  CHECK-NOT:   rock.extract_ptr
func.func @test_multiple_args(%arg0: tensor<4096xf16>, %arg1: tensor<4096xf16>, %arg2: tensor<4096xf32>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %1 = rock.extract_ptr %arg1 : tensor<4096xf16> -> i32
  %2 = rock.extract_ptr %arg2 : tensor<4096xf32> -> i32
  %s0 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %s1 = tt.splat %1 : i32 -> tensor<64x64xi32>
  %s2 = tt.splat %2 : i32 -> tensor<64x64xi32>
  %p0 = rock.cast_to_ptr %s0 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %p1 = rock.cast_to_ptr %s1 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %p2 = rock.cast_to_ptr %s2 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %3 = tt.load %p0, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  %4 = tt.load %p1, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  %5 = arith.extf %3 : tensor<64x64xf16> to tensor<64x64xf32>
  tt.store %p2, %5, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  return
}

// -----

// Verifies func.return is converted to tt.return
// CHECK-LABEL: tt.func @test_return_conversion
//      CHECK:   tt.return
//  CHECK-NOT:   return
func.func @test_return_conversion(%arg0: tensor<64xf16>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<64xf16> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64xi32>
  %2 = rock.cast_to_ptr %1 : tensor<64xi32> -> tensor<64x!tt.ptr<f16>>
  %3 = tt.load %2, %cst_mask : tensor<64x!tt.ptr<f16>>
  return
}

// -----

// Verifies chained arith.addi operations are all converted to tt.addptr
// CHECK-LABEL: tt.func @test_chained_addptr
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>)
//      CHECK:   %[[SPLAT:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:   tt.addptr %[[SPLAT]]
//      CHECK:   tt.addptr
//  CHECK-NOT:   rock.cast_to_ptr
func.func @test_chained_addptr(%arg0: tensor<8192xf16>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %c128 = arith.constant dense<128> : tensor<64x64xi32>
  %0 = rock.extract_ptr %arg0 : tensor<8192xf16> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %row_offset = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %row_exp = tt.expand_dims %row_offset {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %row_bcast = tt.broadcast %row_exp : tensor<64x1xi32> -> tensor<64x64xi32>
  %row_scaled = arith.muli %row_bcast, %c128 : tensor<64x64xi32>
  %2 = arith.addi %1, %row_scaled : tensor<64x64xi32>
  %col_offset = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %col_exp = tt.expand_dims %col_offset {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %col_bcast = tt.broadcast %col_exp : tensor<1x64xi32> -> tensor<64x64xi32>
  %3 = arith.addi %2, %col_bcast : tensor<64x64xi32>
  %4 = rock.cast_to_ptr %3 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %5 = tt.load %4, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  return
}

// -----

// Verifies tt.store with pointer tensor works correctly
// CHECK-LABEL: tt.func @test_store_with_addptr
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>)
//      CHECK:   %[[SPLAT:.*]] = tt.splat %[[ARG0]] : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
//      CHECK:   %[[PTRS:.*]] = tt.addptr %[[SPLAT]]
//      CHECK:   tt.store %[[PTRS]]
//  CHECK-NOT:   rock.cast_to_ptr
func.func @test_store_with_addptr(%arg0: tensor<4096xf32>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %cst_val = arith.constant dense<1.0> : tensor<64x64xf32>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf32> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %offset = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %offset_exp = tt.expand_dims %offset {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
  %offset_bcast = tt.broadcast %offset_exp : tensor<64x1xi32> -> tensor<64x64xi32>
  %2 = arith.addi %1, %offset_bcast : tensor<64x64xi32>
  %3 = rock.cast_to_ptr %2 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  tt.store %3, %cst_val, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  return
}

// -----

// Verifies bf16 element type
// CHECK-LABEL: tt.func @test_bf16_type
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<bf16>)
//      CHECK:   tt.splat %[[ARG0]] : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>>
func.func @test_bf16_type(%arg0: tensor<64xbf16>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<64xbf16> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64xi32>
  %2 = rock.cast_to_ptr %1 : tensor<64xi32> -> tensor<64x!tt.ptr<bf16>>
  %3 = tt.load %2, %cst_mask : tensor<64x!tt.ptr<bf16>>
  return
}

// -----

// Verifies operations inside scf.for with iter_args are converted
// CHECK-LABEL: tt.func @test_inside_scf_for
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>)
//      CHECK:   %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<64xf32>
//      CHECK:   scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (tensor<64xf32>)
//      CHECK:     tt.splat %[[ARG0]] : !tt.ptr<f16>
//      CHECK:     tt.addptr
//      CHECK:     tt.load
//      CHECK:     scf.yield {{.*}} : tensor<64xf32>
//  CHECK-NOT:   rock.extract_ptr
//  CHECK-NOT:   rock.cast_to_ptr
func.func @test_inside_scf_for(%arg0: tensor<4096xf16>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst_mask = arith.constant dense<true> : tensor<64xi1>
  %cst_init = arith.constant dense<0.0> : tensor<64xf32>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32

  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst_init) -> tensor<64xf32> {
    %idx = arith.index_cast %i : index to i32
    %c64 = arith.constant 64 : i32
    %offset_scalar = arith.muli %idx, %c64 : i32
    %offset = tt.splat %offset_scalar : i32 -> tensor<64xi32>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %full_offset = arith.addi %offset, %range : tensor<64xi32>
    %1 = tt.splat %0 : i32 -> tensor<64xi32>
    %2 = arith.addi %1, %full_offset : tensor<64xi32>
    %3 = rock.cast_to_ptr %2 : tensor<64xi32> -> tensor<64x!tt.ptr<f16>>
    %4 = tt.load %3, %cst_mask : tensor<64x!tt.ptr<f16>>
    %5 = arith.extf %4 : tensor<64xf16> to tensor<64xf32>
    %6 = arith.addf %acc, %5 : tensor<64xf32>
    scf.yield %6 : tensor<64xf32>
  }
  return
}

// -----

// Verifies a real GEMM test case with scf.for loop, loads, dot, and store
// CHECK: module attributes {{{.*}}rock.grid_size.rock_gemm = 1 : i32
// CHECK-LABEL: tt.func @rock_gemm
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>, %[[ARG1:.*]]: !tt.ptr<f16>, %[[ARG2:.*]]: !tt.ptr<f32>)
//      CHECK:   %[[INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
//      CHECK:   scf.for {{.*}} iter_args({{.*}} = %[[INIT]]) -> (tensor<64x64xf32>)
//      CHECK:     tt.splat %[[ARG1]] : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.addptr {{.*}} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
//      CHECK:     tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.splat %[[ARG0]] : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.addptr {{.*}} : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
//      CHECK:     tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>>
//      CHECK:     tt.dot {{.*}} : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
//      CHECK:     scf.yield
//      CHECK:   tt.splat %[[ARG2]] : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
//      CHECK:   tt.addptr {{.*}} : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
//      CHECK:   tt.store {{.*}} : tensor<64x64x!tt.ptr<f32>>
//      CHECK:   tt.return
//  CHECK-NOT:   rock.extract_ptr
//  CHECK-NOT:   rock.cast_to_ptr
//  CHECK-NOT:   func.func
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
    %35 = rock.cast_to_ptr %33 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
    %36 = tt.load %35, %34 : tensor<64x64x!tt.ptr<f16>>
    %37 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %38 = tt.expand_dims %37 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %39 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %40 = tt.expand_dims %39 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %41 = arith.muli %arg3, %c64_i32 : i32
    %42 = tt.splat %41 : i32 -> tensor<1x64xi32>
    %43 = arith.addi %42, %40 : tensor<1x64xi32>
    %44 = arith.cmpi ult, %38, %cst_0 : tensor<64x1xi32>
    %45 = arith.muli %38, %cst : tensor<64x1xi32>
    %46 = tt.broadcast %45 : tensor<64x1xi32> -> tensor<64x64xi32>
    %47 = tt.broadcast %43 : tensor<1x64xi32> -> tensor<64x64xi32>
    %48 = arith.addi %46, %47 : tensor<64x64xi32>
    %49 = tt.splat %1 : i32 -> tensor<64x64xi32>
    %50 = arith.addi %49, %48 : tensor<64x64xi32>
    %51 = tt.broadcast %44 : tensor<64x1xi1> -> tensor<64x64xi1>
    %52 = rock.cast_to_ptr %50 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
    %53 = tt.load %52, %51 : tensor<64x64x!tt.ptr<f16>>
    %54 = tt.dot %53, %36, %arg4 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
    scf.yield %54 : tensor<64x64xf32>
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
  %19 = rock.cast_to_ptr %18 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  tt.store %19, %3, %12 : tensor<64x64x!tt.ptr<f32>>
  return
}

// -----

// Verifies a single prefill arg is serialized as a module attribute
// CHECK: module attributes {{{.*}}rock.prefill_args.test_single_prefill = [{index = 2 : i64, value = 0.000000e+00 : f32}]
// CHECK-LABEL: tt.func @test_single_prefill
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>, %[[ARG1:.*]]: !tt.ptr<f16>, %[[ARG2:.*]]: !tt.ptr<f32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_single_prefill(%arg0: tensor<4096xf16>, %arg1: tensor<4096xf16>, %arg2: tensor<4096xf32> {rock.prefill = 0.000000e+00 : f32}) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 256 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %1 = rock.extract_ptr %arg1 : tensor<4096xf16> -> i32
  %2 = rock.extract_ptr %arg2 : tensor<4096xf32> -> i32
  %s0 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %s1 = tt.splat %1 : i32 -> tensor<64x64xi32>
  %s2 = tt.splat %2 : i32 -> tensor<64x64xi32>
  %p0 = rock.cast_to_ptr %s0 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %p1 = rock.cast_to_ptr %s1 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %p2 = rock.cast_to_ptr %s2 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %3 = tt.load %p0, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  %4 = tt.load %p1, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  %5 = arith.extf %3 : tensor<64x64xf16> to tensor<64x64xf32>
  tt.store %p2, %5, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  return
}

// -----

// Verifies multiple prefill args of the same type are both serialized
// CHECK: module attributes {{{.*}}rock.prefill_args.test_multi_prefill_same_type = [{index = 2 : i64, value = 0.000000e+00 : f32}, {index = 3 : i64, value = 0.000000e+00 : f32}]
// CHECK-LABEL: tt.func @test_multi_prefill_same_type
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: !tt.ptr<f32> {rock.prefill = 0.000000e+00 : f32}, %[[ARG3:.*]]: !tt.ptr<f32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_multi_prefill_same_type(
    %arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>,
    %arg2: tensor<4096xf32> {rock.prefill = 0.000000e+00 : f32},
    %arg3: tensor<4096xf32> {rock.prefill = 0.000000e+00 : f32})
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 256 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf32> -> i32
  %1 = rock.extract_ptr %arg1 : tensor<4096xf32> -> i32
  %2 = rock.extract_ptr %arg2 : tensor<4096xf32> -> i32
  %3 = rock.extract_ptr %arg3 : tensor<4096xf32> -> i32
  %s0 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %s1 = tt.splat %1 : i32 -> tensor<64x64xi32>
  %s2 = tt.splat %2 : i32 -> tensor<64x64xi32>
  %s3 = tt.splat %3 : i32 -> tensor<64x64xi32>
  %p0 = rock.cast_to_ptr %s0 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %p1 = rock.cast_to_ptr %s1 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %p2 = rock.cast_to_ptr %s2 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %p3 = rock.cast_to_ptr %s3 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %a = tt.load %p0, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  %b = tt.load %p1, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  %c = arith.addf %a, %b : tensor<64x64xf32>
  tt.store %p2, %c, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  tt.store %p3, %c, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  return
}

// -----

// Verifies multiple prefill args with different element types (f16 and f32)
// CHECK: module attributes {{{.*}}rock.prefill_args.test_multi_prefill_mixed_types = [{index = 2 : i64, value = 0.000000e+00 : f32}, {index = 3 : i64, value = 0.000000e+00 : f16}]
// CHECK-LABEL: tt.func @test_multi_prefill_mixed_types
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>, %[[ARG1:.*]]: !tt.ptr<f16>, %[[ARG2:.*]]: !tt.ptr<f32> {rock.prefill = 0.000000e+00 : f32}, %[[ARG3:.*]]: !tt.ptr<f16> {rock.prefill = 0.000000e+00 : f16})
func.func @test_multi_prefill_mixed_types(
    %arg0: tensor<4096xf16>, %arg1: tensor<4096xf16>,
    %arg2: tensor<4096xf32> {rock.prefill = 0.000000e+00 : f32},
    %arg3: tensor<4096xf16> {rock.prefill = 0.000000e+00 : f16})
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 256 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %1 = rock.extract_ptr %arg1 : tensor<4096xf16> -> i32
  %2 = rock.extract_ptr %arg2 : tensor<4096xf32> -> i32
  %3 = rock.extract_ptr %arg3 : tensor<4096xf16> -> i32
  %s0 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %s1 = tt.splat %1 : i32 -> tensor<64x64xi32>
  %s2 = tt.splat %2 : i32 -> tensor<64x64xi32>
  %s3 = tt.splat %3 : i32 -> tensor<64x64xi32>
  %p0 = rock.cast_to_ptr %s0 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %p1 = rock.cast_to_ptr %s1 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %p2 = rock.cast_to_ptr %s2 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %p3 = rock.cast_to_ptr %s3 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %a = tt.load %p0, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  %b = tt.load %p1, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  %a_f32 = arith.extf %a : tensor<64x64xf16> to tensor<64x64xf32>
  %b_f32 = arith.extf %b : tensor<64x64xf16> to tensor<64x64xf32>
  %c_f32 = arith.addf %a_f32, %b_f32 : tensor<64x64xf32>
  %c_f16 = arith.truncf %c_f32 : tensor<64x64xf32> to tensor<64x64xf16>
  tt.store %p2, %c_f32, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  tt.store %p3, %c_f16, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  return
}

// -----

// Verifies non-kernel func.func is preserved alongside a converted kernel
// CHECK: func.func @helper_function
// CHECK-LABEL: tt.func @test_non_kernel_preserved
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>)
//  CHECK-NOT:   rock.extract_ptr
func.func @helper_function(%x: f32, %y: f32) -> f32 {
  %r = arith.addf %x, %y : f32
  return %r : f32
}
func.func @test_non_kernel_preserved(%arg0: tensor<64xf16>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<64xf16> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64xi32>
  %2 = rock.cast_to_ptr %1 : tensor<64xi32> -> tensor<64x!tt.ptr<f16>>
  %3 = tt.load %2, %cst_mask : tensor<64x!tt.ptr<f16>>
  return
}

// -----

// Verifies scalar (non-tensor) arguments pass through unchanged
// CHECK-LABEL: tt.func @test_mixed_tensor_scalar
// CHECK-SAME: (%[[PTR:.*]]: !tt.ptr<f16>, %[[SCALAR:.*]]: i32)
//      CHECK:   tt.splat %[[PTR]] : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
//      CHECK:   tt.splat %[[SCALAR]] : i32 -> tensor<64xi32>
//      CHECK:   tt.addptr
//      CHECK:   tt.load
//  CHECK-NOT:   rock.extract_ptr
//  CHECK-NOT:   rock.cast_to_ptr
func.func @test_mixed_tensor_scalar(%arg0: tensor<4096xf16>, %arg1: i32) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64xi32>
  %offset = tt.splat %arg1 : i32 -> tensor<64xi32>
  %2 = arith.addi %1, %offset : tensor<64xi32>
  %3 = rock.cast_to_ptr %2 : tensor<64xi32> -> tensor<64x!tt.ptr<f16>>
  %4 = tt.load %3, %cst_mask : tensor<64x!tt.ptr<f16>>
  return
}

// -----

// Verifies a dead tensor input (no extract_ptr) is still converted to tt.ptr,
// and a prefill arg on another argument is preserved
// CHECK: module attributes {{{.*}}rock.prefill_args.test_dead_input_with_prefill = [{index = 2 : i64, value = 0.000000e+00 : f32}]
// CHECK-LABEL: tt.func @test_dead_input_with_prefill
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>, %[[DEAD:.*]]: !tt.ptr<f16>, %[[ARG2:.*]]: !tt.ptr<f32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_dead_input_with_prefill(
    %arg0: tensor<4096xf16>,
    %arg1: tensor<4096xf16>,
    %arg2: tensor<4096xf32> {rock.prefill = 0.000000e+00 : f32})
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 256 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %2 = rock.extract_ptr %arg2 : tensor<4096xf32> -> i32
  %s0 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %s2 = tt.splat %2 : i32 -> tensor<64x64xi32>
  %p0 = rock.cast_to_ptr %s0 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %p2 = rock.cast_to_ptr %s2 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %a = tt.load %p0, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  %a_f32 = arith.extf %a : tensor<64x64xf16> to tensor<64x64xf32>
  tt.store %p2, %a_f32, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  return
}

// -----

// Verifies that pre-existing arg attributes set by RockAnalyzeMemoryUse
// (LLVM kernel-arg attrs and Triton metadata) survive the conversion from
// `func.func` to `tt.func`. If `setAllArgAttrs` were ever dropped, this
// would catch it.
// CHECK-LABEL: tt.func @test_arg_attrs_preserved
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16>
// CHECK-SAME: llvm.align = 16 : i64
// CHECK-SAME: llvm.dereferenceable = 8192 : i64
// CHECK-SAME: llvm.noalias
// CHECK-SAME: llvm.nocapture
// CHECK-SAME: llvm.nofree
// CHECK-SAME: llvm.nonnull
// CHECK-SAME: llvm.noundef
// CHECK-SAME: llvm.readonly
// CHECK-SAME: tt.divisibility = 16 : i32
// CHECK-SAME: tt.pointer_range = 32 : i32
// CHECK-SAME: %[[ARG1:.*]]: !tt.ptr<f32>
// CHECK-SAME: llvm.align = 16 : i64
// CHECK-SAME: llvm.dereferenceable = 16384 : i64
// CHECK-SAME: llvm.noalias
// CHECK-SAME: llvm.nocapture
// CHECK-SAME: llvm.nofree
// CHECK-SAME: llvm.nonnull
// CHECK-SAME: llvm.noundef
// CHECK-SAME: llvm.writeonly
// CHECK-SAME: tt.divisibility = 16 : i32
// CHECK-SAME: tt.pointer_range = 32 : i32
// CHECK-SAME: %[[ARG2:.*]]: i32 {tt.divisibility = 16 : i32})
func.func @test_arg_attrs_preserved(
    %arg0: tensor<4096xf16> {llvm.align = 16 : i64,
                              llvm.dereferenceable = 8192 : i64,
                              llvm.noalias,
                              llvm.nocapture,
                              llvm.nofree,
                              llvm.nonnull,
                              llvm.noundef,
                              llvm.readonly,
                              tt.divisibility = 16 : i32,
                              tt.pointer_range = 32 : i32},
    %arg1: tensor<4096xf32> {llvm.align = 16 : i64,
                              llvm.dereferenceable = 16384 : i64,
                              llvm.noalias,
                              llvm.nocapture,
                              llvm.nofree,
                              llvm.nonnull,
                              llvm.noundef,
                              llvm.writeonly,
                              tt.divisibility = 16 : i32,
                              tt.pointer_range = 32 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32})
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 256 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64x64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %1 = rock.extract_ptr %arg1 : tensor<4096xf32> -> i32
  %s0 = tt.splat %0 : i32 -> tensor<64x64xi32>
  %s1 = tt.splat %1 : i32 -> tensor<64x64xi32>
  %p0 = rock.cast_to_ptr %s0 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  %p1 = rock.cast_to_ptr %s1 : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f32>>
  %a = tt.load %p0, %cst_mask : tensor<64x64x!tt.ptr<f16>>
  %a_f32 = arith.extf %a : tensor<64x64xf16> to tensor<64x64xf32>
  tt.store %p1, %a_f32, %cst_mask : tensor<64x64x!tt.ptr<f32>>
  return
}
