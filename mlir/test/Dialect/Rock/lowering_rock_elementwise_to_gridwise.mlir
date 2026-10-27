// RUN: rocmlir-opt -rock-elementwise-to-gridwise -mlir-print-local-scope %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @pad_and_set_grid_size
  // CHECK-SAME: rock.grid_size = 1 : i32
  // CHECK-DAG: rock.transform %arg0 by {{.*}}Pad{0, 244}
  // CHECK-DAG: rock.transform %arg1 by {{.*}}Pad{0, 244}
  // CHECK: arith.addf {{.*}} : tensor<256xf32>
  // CHECK: rock.store {{.*}} : tensor<256xf32> -> tensor<12xf32> to tensor<256xf32>
  func.func @pad_and_set_grid_size(%arg0: tensor<12xf32>, %arg1: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel, rock.elementwise, rock.block_size = 256 : i32} {
    %0 = arith.addf %arg0, %arg1 : tensor<12xf32>
    %1 = rock.store %0 to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %1 : tensor<12xf32>
  }

  // CHECK-LABEL: func.func @skip_unmarked_kernel
  // CHECK-NOT: rock.grid_size
  func.func @skip_unmarked_kernel(%arg0: tensor<12xf32>, %arg1: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel, rock.block_size = 256 : i32} {
    %0 = arith.addf %arg0, %arg1 : tensor<12xf32>
    %1 = rock.store %0 to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %1 : tensor<12xf32>
  }
}
