// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `rock-lower-reduce-to-blockwise` only accepts what it can lower correctly
// and diagnoses the rest, rather than silently leaving a `rock.reduce` behind
// for a later pass to trip over.

// RUN: rocmlir-opt -split-input-file -verify-diagnostics -rock-lower-reduce-to-blockwise=arch=gfx942 %s

#view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// A reduction over anything but the last axis would need a tile whose
// contiguous axis is not the reduced one.
func.func @not_last_axis(%arg0: tensor<8192xf32>, %out: tensor<1024xf32>) -> tensor<1024xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{only last-axis reductions are supported, got axis 0 of a rank-2 input}}
  %0 = rock.reduce sum %a {axis = 0 : index} : tensor<8x1024xf32> -> tensor<1x1024xf32>
  %1 = rock.transform %0 by <affine_map<(d0) -> (0, d0)> by [<Merge{1, 1024} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [1024] -> [1, 1024]> : tensor<1x1024xf32> to tensor<1024xf32>
  %2 = rock.store %1 to %out by set : tensor<1024xf32> -> tensor<1024xf32> to tensor<1024xf32>
  return %2 : tensor<1024xf32>
}

// -----

#view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// Only sum is in scope. A max reduction needs the neutral fill and the NaN
// reasoning that go with it.
func.func @max_reduce(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{only sum reductions are supported by the reduce pipeline}}
  %0 = rock.reduce max %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

#view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// Fusing an elementwise op after the reduction would have to run on one
// element per workgroup, which this pipeline does not set up.
func.func @post_reduce_fusion(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %cst = arith.constant dense<2.0> : tensor<8x1xf32>
  %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{post-reduce fusion is not supported}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %scaled = arith.mulf %0, %cst : tensor<8x1xf32>
  %1 = rock.transform %scaled by #drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

#wide = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 8192 + d1)> by [<Unmerge{2, 8192} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [2, 8192] -> [16384]>
#drop2 = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{2, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [2] -> [2, 1]>

// A reduction wider than a workgroup can hold in registers needs a loop over
// the reduced axis, which is not implemented yet.
func.func @too_wide(%arg0: tensor<16384xf32>, %out: tensor<2xf32>) -> tensor<2xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #wide : tensor<16384xf32> to tensor<2x8192xf32>
  // expected-error @+1 {{reduction over 8192 elements needs a tile wider than the 2048-element limit}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<2x8192xf32> -> tensor<2x1xf32>
  %1 = rock.transform %0 by #drop2 : tensor<2x1xf32> to tensor<2xf32>
  %2 = rock.store %1 to %out by set : tensor<2xf32> -> tensor<2xf32> to tensor<2xf32>
  return %2 : tensor<2xf32>
}

// -----

#view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// A pinned tile narrower than the reduction implies the loop that Phase B
// will add, so it is rejected rather than quietly dropping elements.
func.func @pinned_params_too_narrow(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{rock.reduce_params has colsPerBlock 256, which is short of the 1024-element reduction}}
  %0 = rock.reduce sum %a {axis = 1 : index, params = #rock.reduce_params<rowsPerBlock = 1, colsPerBlock = 256, numWaves = 4>} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

#view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// `numWaves` also fixes the launch block size, so a schedule chosen for a
// different one cannot be honoured.
func.func @pinned_params_wrong_waves(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{rock.reduce_params has numWaves 8 but the pass was given 4}}
  %0 = rock.reduce sum %a {axis = 1 : index, params = #rock.reduce_params<rowsPerBlock = 1, colsPerBlock = 1024, numWaves = 8>} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// A transposed view makes the reduced axis the strided one, so the loads a
// tile shaped for a contiguous reduction would issue are the wrong ones.
#transposed = #rock.transform_map<affine_map<(d0, d1) -> (d1 * 8 + d0)> by [<Unmerge{1024, 8} ["col", "row"] at [1, 0] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

func.func @strided_reduce_axis(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.reduce} {
  %a = rock.transform %arg0 by #transposed : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{reduction input is strided by 8 along the reduced axis}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}
