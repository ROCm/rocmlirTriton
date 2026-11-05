// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End to end through the standalone reduce pipeline. The kernel a reduction
// should compile to is just a load, a `tt.reduce` and a store, so that is what
// these check for.

// RUN: rocmlir-opt --rock-reduce-pipeline='arch=gfx942' -mlir-print-local-scope %s | FileCheck %s

#view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// One workgroup per row, so the grid is the row count.
// CHECK-LABEL: module attributes {rock.grid_size.reduce_add = 8 : i32}
// CHECK: tt.func @reduce_add(
// CHECK: tt.get_program_id x
// CHECK: %[[A:.*]] = tt.load
// CHECK: %[[B:.*]] = tt.load
// CHECK: %[[SUM:.*]] = arith.addf %[[A]], %[[B]]{{.*}} : tensor<1x1024xf32>
// CHECK: %[[RED:.*]] = "tt.reduce"(%[[SUM]]) <{axis = 1 : i32}>
// CHECK:   arith.addf {{.*}} : f32
// CHECK:   tt.reduce.return
// CHECK: tt.store {{.*}}, %[[RED]]
// CHECK: tt.return
// CHECK-NOT: rock.
module {
  func.func @reduce_add(%arg0: tensor<8192xf32>, %arg1: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
      attributes {rock.kernel, rock.reduce, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
    %b = rock.transform %arg1 by #view : tensor<8192xf32> to tensor<8x1024xf32>
    %s = arith.addf %a, %b : tensor<8x1024xf32>
    %0 = rock.reduce sum %s {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
    %1 = rock.transform %0 by #drop : tensor<8x1xf32> to tensor<8xf32>
    %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}
