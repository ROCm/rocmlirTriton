// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The tile shape this pass picks is the whole schedule: `tritongpu-coalesce`
// derives the blocked layout of the load from it, and that layout is what
// decides whether the `tt.reduce` it feeds combines in registers, across the
// lanes of a wave, or through LDS. These tests therefore check the tile, which
// shows up as the result type of the `rock.load_marker`, and the grid size.
//
// With the default 4 waves of 64 lanes, a thread wants `128 / elementBits`
// elements, so the tile is `colsPerBlock = nextPow2(N)` by
// `rowsPerBlock = 256 * vecWidth / colsPerBlock`.

// RUN: rocmlir-opt -split-input-file -rock-lower-reduce-to-blockwise=arch=gfx942 -mlir-print-local-scope %s | FileCheck %s

// A row long enough to fill the workgroup on its own: one row per workgroup,
// and the reduction has to cross all four waves through LDS.
#long_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#long_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// CHECK-LABEL: func.func @long_rows
// CHECK-SAME: rock.grid_size = 8 : i32
// CHECK: %[[BID:.*]] = tt.get_program_id x
// CHECK: %[[TILE:.*]] = rock.load_marker {{.*}}[%[[BID]]] {{.*}} -> tensor<1x1024xf32>
// CHECK: %[[RED:.*]] = rock.blockwise_reduce sum %[[TILE]] {axis = 1 : index} : tensor<1x1024xf32> -> tensor<1xf32>
// CHECK: rock.store_marker %[[RED]] {{.*}}[%[[BID]]] : tensor<1xf32> -> tensor<8xf32>
// CHECK-NOT: rock.reduce
func.func @long_rows(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #long_view : tensor<8192xf32> to tensor<8x1024xf32>
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #long_drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// A short row leaves most of a workgroup idle, so the tile takes 64 rows at a
// time instead. Only four lanes then share a reduction, which Triton can do
// with a pair of DPP steps and no LDS.
#short_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 16 + d1)> by [<Unmerge{4096, 16} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [4096, 16] -> [65536]>
#short_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{4096, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [4096] -> [4096, 1]>

// CHECK-LABEL: func.func @short_rows
// CHECK-SAME: rock.grid_size = 64 : i32
// CHECK: rock.load_marker {{.*}} -> tensor<64x16xf32>
// CHECK: rock.blockwise_reduce sum {{.*}} : tensor<64x16xf32> -> tensor<64xf32>
func.func @short_rows(%arg0: tensor<65536xf32>, %out: tensor<4096xf32>) -> tensor<4096xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #short_view : tensor<65536xf32> to tensor<4096x16xf32>
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<4096x16xf32> -> tensor<4096x1xf32>
  %1 = rock.transform %0 by #short_drop : tensor<4096x1xf32> to tensor<4096xf32>
  %2 = rock.store %1 to %out by set : tensor<4096xf32> -> tensor<4096xf32> to tensor<4096xf32>
  return %2 : tensor<4096xf32>
}

// -----

// Twice as many f16 values fit in the 128-bit load a lane wants, so the same
// row length takes twice the rows per workgroup.
#f16_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#f16_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// CHECK-LABEL: func.func @half_precision
// CHECK-SAME: rock.grid_size = 4 : i32
// CHECK: rock.load_marker {{.*}} -> tensor<2x1024xf16>
func.func @half_precision(%arg0: tensor<8192xf16>, %out: tensor<8xf16>) -> tensor<8xf16>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #f16_view : tensor<8192xf16> to tensor<8x1024xf16>
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf16> -> tensor<8x1xf16>
  %1 = rock.transform %0 by #f16_drop : tensor<8x1xf16> to tensor<8xf16>
  %2 = rock.store %1 to %out by set : tensor<8xf16> -> tensor<8xf16> to tensor<8xf16>
  return %2 : tensor<8xf16>
}

// -----

// Neither extent is a power of two, so both axes of the tile are padded. The
// padded lanes read as zero, the identity of the sum, and the padded rows are
// masked off on the store side.
#pad_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 300 + d1)> by [<Unmerge{100, 300} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [100, 300] -> [30000]>
#pad_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{100, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [100] -> [100, 1]>

// CHECK-LABEL: func.func @padded_both_axes
// CHECK-SAME: rock.grid_size = 50 : i32
// CHECK: rock.transform {{.*}}Pad{0, 0, 0, 212}
// CHECK: rock.load_marker {{.*}} -> tensor<2x512xf32>
// CHECK: rock.blockwise_reduce sum {{.*}} : tensor<2x512xf32> -> tensor<2xf32>
func.func @padded_both_axes(%arg0: tensor<30000xf32>, %out: tensor<100xf32>) -> tensor<100xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #pad_view : tensor<30000xf32> to tensor<100x300xf32>
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<100x300xf32> -> tensor<100x1xf32>
  %1 = rock.transform %0 by #pad_drop : tensor<100x1xf32> to tensor<100xf32>
  %2 = rock.store %1 to %out by set : tensor<100xf32> -> tensor<100xf32> to tensor<100xf32>
  return %2 : tensor<100xf32>
}

// -----

// A rank-3 input reduces over its last axis, with the leading axes merged into
// the rows the grid is spread over.
#rank3_view = #rock.transform_map<affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 512 + d2)> by [<Unmerge{2, 4, 512} ["b", "row", "col"] at [0, 1, 2] -> ["flat"] at [0]>] bounds = [2, 4, 512] -> [4096]>
#rank3_drop = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 4, d0 mod 4, 0)> by [<Merge{2, 4, 1} ["row"] at [0] -> ["d0", "d1", "d2"] at [0, 1, 2]>] bounds = [8] -> [2, 4, 1]>

// CHECK-LABEL: func.func @rank3
// CHECK-SAME: rock.grid_size = 4 : i32
// CHECK: rock.transform {{.*}}Merge{2, 4}
// CHECK: rock.load_marker {{.*}} -> tensor<2x512xf32>
func.func @rank3(%arg0: tensor<4096xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #rank3_view : tensor<4096xf32> to tensor<2x4x512xf32>
  %0 = rock.reduce sum %a {axis = 2 : index} : tensor<2x4x512xf32> -> tensor<2x4x1xf32>
  %1 = rock.transform %0 by #rank3_drop : tensor<2x4x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// A schedule already on the op is honoured rather than recomputed, so a test
// (or a future tuner) can pin a tile the heuristic would not have chosen.
#pinned_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#pinned_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// CHECK-LABEL: func.func @pinned_params
// CHECK-SAME: rock.grid_size = 2 : i32
// CHECK: rock.load_marker {{.*}} -> tensor<4x1024xf32>
func.func @pinned_params(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #pinned_view : tensor<8192xf32> to tensor<8x1024xf32>
  %0 = rock.reduce sum %a {axis = 1 : index, params = #rock.reduce_params<rowsPerBlock = 4, colsPerBlock = 1024, numWaves = 4>} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #pinned_drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// The elementwise ops feeding the reduction are retyped onto the tile, with
// one load per leaf input.
#fused_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#fused_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// CHECK-LABEL: func.func @pre_reduce_fusion
// CHECK: %[[A:.*]] = rock.load_marker {{.*}} -> tensor<1x1024xf32>
// CHECK: %[[B:.*]] = rock.load_marker {{.*}} -> tensor<1x1024xf32>
// CHECK: %[[SUM:.*]] = arith.addf {{.*}} : tensor<1x1024xf32>
// CHECK: rock.blockwise_reduce sum %[[SUM]]
func.func @pre_reduce_fusion(%arg0: tensor<8192xf32>, %arg1: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #fused_view : tensor<8192xf32> to tensor<8x1024xf32>
  %b = rock.transform %arg1 by #fused_view : tensor<8192xf32> to tensor<8x1024xf32>
  %s = arith.addf %a, %b : tensor<8x1024xf32>
  %0 = rock.reduce sum %s {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #fused_drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// A function without the reduce marker belongs to another pipeline and is
// left alone.
// CHECK-LABEL: func.func @not_a_reduce_kernel
// CHECK: rock.reduce
// CHECK-NOT: rock.blockwise_reduce
func.func @not_a_reduce_kernel(%arg0: tensor<8x1024xf32>) -> tensor<8x1xf32> {
  %0 = rock.reduce sum %arg0 {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  return %0 : tensor<8x1xf32>
}
