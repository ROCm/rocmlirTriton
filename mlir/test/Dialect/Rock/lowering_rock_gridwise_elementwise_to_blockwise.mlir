// RUN: rocmlir-opt -rock-gridwise-elementwise-to-blockwise -mlir-print-local-scope %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @insert_markers
  // CHECK: %[[BID:.*]] = tt.get_program_id x
  // CHECK: %[[LM1:.*]] = rock.load_marker %arg1 views
  // CHECK: %[[LM0:.*]] = rock.load_marker %arg0 views
  // CHECK: %[[ADD:.*]] = arith.addf %[[LM0]], %[[LM1]] : tensor<256xf32>
  // CHECK: %[[SM:.*]] = rock.store_marker %[[ADD]] views
  // CHECK: rock.store %[[SM]] to %dest
  func.func @insert_markers(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %dest: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, rock.elementwise, rock.block_size = 256 : i32, rock.grid_size = 1 : i32} {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
    %1 = rock.store %0 to %dest by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
    return %1 : tensor<256xf32>
  }

  // CHECK-LABEL: func.func @skip_unmarked_kernel
  // CHECK-NOT: tt.get_program_id
  func.func @skip_unmarked_kernel(%arg0: tensor<256xf32>, %arg1: tensor<256xf32>, %dest: tensor<256xf32>) -> tensor<256xf32> attributes {rock.kernel, rock.grid_size = 1 : i32} {
    %0 = arith.addf %arg0, %arg1 : tensor<256xf32>
    %1 = rock.store %0 to %dest by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
    return %1 : tensor<256xf32>
  }
}
