// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-func-to-triton-func --split-input-file | FileCheck %s

// Verifies func.func with rock.kernel is converted to tt.func with pointer arguments
// CHECK: module attributes {{{.*}}rock.grid_size.test_basic_conversion = 2 : i32
// CHECK-LABEL: tt.func @test_basic_conversion
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
//      CHECK: attributes {noinline = true, rock.arch = "{{.*}}"
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
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
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
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %[[ARG1:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %[[ARG2:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
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
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
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
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
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
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
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
// CHECK-SAME: (%[[ARG0:.*]]: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32})
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
