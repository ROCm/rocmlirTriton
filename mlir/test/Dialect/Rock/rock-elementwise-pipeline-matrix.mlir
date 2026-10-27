// RUN: rocmlir-opt --rock-elementwise-pipeline -mlir-print-local-scope %s | FileCheck %s

module {
  // CHECK-LABEL: module attributes {
  // CHECK-DAG: rock.grid_size.ew_add = 1 : i32
  // CHECK-DAG: rock.grid_size.ew_mul_relu = 1 : i32
  // CHECK-DAG: rock.grid_size.ew_tanh = 1 : i32
  // CHECK-DAG: rock.grid_size.ew_add_i32 = 1 : i32

  // CHECK-LABEL: tt.func @ew_add(
  // CHECK: tt.load
  // CHECK: arith.addf
  // CHECK: tt.store
  // CHECK: tt.return
  func.func @ew_add(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, rock.elementwise, rock.block_size = 256 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
    %1 = rock.store %0 to %arg2 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
    return %1 : tensor<256xf32>
  }

  // CHECK-LABEL: tt.func @ew_mul_relu(
  // CHECK: arith.mulf
  // CHECK: arith.maximumf
  // CHECK: tt.store
  func.func @ew_mul_relu(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, rock.elementwise, rock.block_size = 256 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %c0 = arith.constant dense<0.0> : tensor<256xf32>
    %0 = arith.mulf %arg0, %arg1 : tensor<256xf32>
    %1 = arith.maximumf %0, %c0 : tensor<256xf32>
    %2 = rock.store %1 to %arg2 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
    return %2 : tensor<256xf32>
  }

  // CHECK-LABEL: tt.func @ew_tanh(
  // CHECK: math.exp
  // CHECK: tt.store
  func.func @ew_tanh(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, rock.elementwise, rock.block_size = 256 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %0 = math.tanh %arg0 : tensor<256xf32>
    %1 = rock.store %0 to %arg1 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
    return %1 : tensor<256xf32>
  }

  // CHECK-LABEL: tt.func @ew_add_i32(
  // CHECK: arith.addi
  // CHECK: tt.store
  func.func @ew_add_i32(%arg0: tensor<256xi32>, %arg1: tensor<256xi32>, %arg2: tensor<256xi32>) -> tensor<256xi32> attributes {rock.kernel, rock.elementwise, rock.block_size = 256 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %0 = arith.addi %arg0, %arg1 : tensor<256xi32>
    %1 = rock.store %0 to %arg2 by set : tensor<256xi32> -> tensor<256xi32> to tensor<256xi32>
    return %1 : tensor<256xi32>
  }

  // CHECK-NOT: rock.load_marker
  // CHECK-NOT: rock.store_marker
  // CHECK-NOT: rock.blockwise_
}
