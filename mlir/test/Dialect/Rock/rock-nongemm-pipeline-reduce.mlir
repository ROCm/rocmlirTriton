// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End to end through the nongemm pipeline, for the kernels that end in a
// reduction. The kernel one should compile to is just a load, a `tt.reduce`
// and a store, so that is what these check for.

// RUN: rocmlir-opt -split-input-file --rock-nongemm-pipeline='arch=gfx942' -mlir-print-local-scope %s | FileCheck %s

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
// CHECK-NOT: rock.load_marker
// CHECK-NOT: rock.store_marker
module {
  func.func @reduce_add(%arg0: tensor<8192xf32>, %arg1: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
      attributes {rock.kernel, rock.nongemm, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
    %b = rock.transform %arg1 by #view : tensor<8192xf32> to tensor<8x1024xf32>
    %s = arith.addf %a, %b : tensor<8x1024xf32>
    %0 = rock.reduce sum %s {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
    %1 = rock.transform %0 by #drop : tensor<8x1xf32> to tensor<8xf32>
    %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}

// -----

// A reduction too wide for one tile is summed a tile at a time. Only the loop
// trip count grows with the reduced extent: there is still one `tt.reduce`,
// and the loop body holds nothing but the loads and the running sum.
#wide = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 8192 + d1)> by [<Unmerge{4, 8192} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [4, 8192] -> [32768]>
#wide_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{4, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [4] -> [4, 1]>

// CHECK-LABEL: tt.func @reduce_wide(
// CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<1x2048xf32>
// CHECK: %[[LOOP:.*]] = scf.for %{{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (tensor<1x2048xf32>)
// CHECK:   tt.load
// CHECK:   %[[NEXT:.*]] = arith.addf %[[ACC]], %{{.*}} : tensor<1x2048xf32>
// CHECK:   scf.yield %[[NEXT]]
// CHECK: "tt.reduce"(%[[LOOP]]) <{axis = 1 : i32}>
// CHECK: tt.store
// CHECK: tt.return
module {
  func.func @reduce_wide(%arg0: tensor<32768xf32>, %out: tensor<4xf32>) -> tensor<4xf32>
      attributes {rock.kernel, rock.nongemm, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %a = rock.transform %arg0 by #wide : tensor<32768xf32> to tensor<4x8192xf32>
    %0 = rock.reduce sum %a {axis = 1 : index} : tensor<4x8192xf32> -> tensor<4x1xf32>
    %1 = rock.transform %0 by #wide_drop : tensor<4x1xf32> to tensor<4xf32>
    %2 = rock.store %1 to %out by set : tensor<4xf32> -> tensor<4xf32> to tensor<4xf32>
    return %2 : tensor<4xf32>
  }
}

// -----

// The whole point of the nongemm kernel type: views and elementwise ops mixed
// freely, with the reduction last. `rock-regularize-nongemm` pushes the view
// between the two elementwise ops down to the leaf it came from, so that
// nothing but loads, arithmetic and the reduction is left by the time this
// reaches Triton.
#mixed_in = #rock.transform_map<affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 512 + d2)> by [<Unmerge{2, 4, 512} ["b", "row", "col"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [2, 4, 512] -> [4096]>
#mixed_flat = #rock.transform_map<affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["row"] at [0] -> ["b", "r"] at [0, 1]>, <PassThrough ["col"] at [1] -> ["col"] at [2]>] bounds = [8, 512] -> [2, 4, 512]>
#mixed_2d = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 512 + d1)> by [<Unmerge{8, 512} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 512] -> [4096]>
#mixed_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// CHECK-LABEL: module attributes {rock.grid_size.mixed_views_and_elementwise = 4 : i32}
// CHECK: tt.func @mixed_views_and_elementwise(
// CHECK: tt.load
// CHECK: math.absf {{.*}} : tensor<2x512xf32>
// CHECK: tt.load
// CHECK: arith.mulf {{.*}} : tensor<2x512xf32>
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
// CHECK: tt.store
// CHECK-NOT: rock.transform
module {
  func.func @mixed_views_and_elementwise(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
      attributes {rock.kernel, rock.nongemm, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
    %a3 = rock.transform %arg0 by #mixed_in : tensor<4096xf32> to tensor<2x4x512xf32>
    %abs = math.absf %a3 : tensor<2x4x512xf32>
    %flat = rock.transform %abs by #mixed_flat : tensor<2x4x512xf32> to tensor<8x512xf32>
    %b = rock.transform %arg1 by #mixed_2d : tensor<4096xf32> to tensor<8x512xf32>
    %m = arith.mulf %flat, %b : tensor<8x512xf32>
    %0 = rock.reduce sum %m {axis = 1 : index} : tensor<8x512xf32> -> tensor<8x1xf32>
    %1 = rock.transform %0 by #mixed_drop : tensor<8x1xf32> to tensor<8xf32>
    %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
    return %2 : tensor<8xf32>
  }
}
