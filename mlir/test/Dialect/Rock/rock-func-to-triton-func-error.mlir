// RUN: not rocmlir-opt -rock-func-to-triton-func --split-input-file %s 2>&1 | FileCheck %s

// Verifies that a rock.cast_to_ptr not in the extract_ptr chain triggers an error
// CHECK: error: unexpected Rock op remaining after FuncToTritonFunc
func.func @test_remaining_cast_to_ptr(%arg0: tensor<4096xf16>) attributes {rock.arch = "gfx90a", rock.kernel, rock.grid_size = 1 : i32, rock.block_size = 64 : i32} {
  %cst_mask = arith.constant dense<true> : tensor<64xi1>
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  %1 = tt.splat %0 : i32 -> tensor<64xi32>
  %2 = rock.cast_to_ptr %1 : tensor<64xi32> -> tensor<64x!tt.ptr<f16>>
  %3 = tt.load %2, %cst_mask : tensor<64x!tt.ptr<f16>>
  %cst_ptrs = arith.constant dense<42> : tensor<64xi32>
  %stray = rock.cast_to_ptr %cst_ptrs : tensor<64xi32> -> tensor<64x!tt.ptr<f16>>
  %loaded = tt.load %stray, %cst_mask : tensor<64x!tt.ptr<f16>>
  return
}

// -----

// Verifies that Rock ops in a non-kernel function trigger the verification error
// CHECK: error: unexpected Rock op remaining after FuncToTritonFunc
func.func @non_kernel_with_rock_op(%arg0: tensor<4096xf16>) -> i32 {
  %0 = rock.extract_ptr %arg0 : tensor<4096xf16> -> i32
  return %0 : i32
}
