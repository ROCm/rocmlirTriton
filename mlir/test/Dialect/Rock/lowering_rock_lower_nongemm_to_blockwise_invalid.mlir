// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// `rock-lower-reduce-to-blockwise` only accepts what it can lower correctly
// and diagnoses the rest, rather than silently leaving a `rock.reduce` behind
// for a later pass to trip over.

// RUN: rocmlir-opt -split-input-file -verify-diagnostics -rock-lower-nongemm-to-blockwise=arch=gfx942 %s

#view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// A reduction over anything but the last axis would need a tile whose
// contiguous axis is not the reduced one.
func.func @not_last_axis(%arg0: tensor<8192xf32>, %out: tensor<1024xf32>) -> tensor<1024xf32>
    attributes {rock.kernel, rock.nongemm} {
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
    attributes {rock.kernel, rock.nongemm} {
  %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{only sum reductions are supported by the nongemm pipeline}}
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
    attributes {rock.kernel, rock.nongemm} {
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

// A pinned tile wider than the register budget is rejected: the limit is what
// the loop over the reduced axis exists to respect, so overriding it defeats
// the point.
func.func @pinned_params_too_wide(%arg0: tensor<16384xf32>, %out: tensor<2xf32>) -> tensor<2xf32>
    attributes {rock.kernel, rock.nongemm,
                rock.nongemm_params = #rock.nongemm_params<rowsPerBlock = 1, colsPerBlock = 8192, numWaves = 4, numStages = 2>} {
  %a = rock.transform %arg0 by #wide : tensor<16384xf32> to tensor<2x8192xf32>
  // expected-error @+1 {{rock.nongemm_params has colsPerBlock 8192, wider than the 2048-element limit}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<2x8192xf32> -> tensor<2x1xf32>
  %1 = rock.transform %0 by #drop2 : tensor<2x1xf32> to tensor<2xf32>
  %2 = rock.store %1 to %out by set : tensor<2xf32> -> tensor<2xf32> to tensor<2xf32>
  return %2 : tensor<2xf32>
}

// -----

#view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// `numWaves` is not a knob: it has to be the number of waves the kernel is
// actually launched with, which this pipeline does not get to choose.
func.func @pinned_params_wrong_waves(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.nongemm,
                rock.nongemm_params = #rock.nongemm_params<rowsPerBlock = 1, colsPerBlock = 1024, numWaves = 8, numStages = 2>} {
  %a = rock.transform %arg0 by #view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{rock.nongemm_params has numWaves 8, but a nongemm kernel is launched 4 waves to a workgroup}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
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
    attributes {rock.kernel, rock.nongemm} {
  %a = rock.transform %arg0 by #transposed : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{reduction input is strided by 8 along the reduced axis}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

#stage_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#stage_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>

// `numStages` is no more a knob than `numWaves`: nothing carries it from this
// attribute to the Triton pipeline yet, so a pinned value that disagrees with
// the `TritonOptions` default would describe a schedule that never runs.
func.func @pinned_params_wrong_stages(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.nongemm,
                rock.nongemm_params = #rock.nongemm_params<rowsPerBlock = 1, colsPerBlock = 1024, numWaves = 4, numStages = 1>} {
  %a = rock.transform %arg0 by #stage_view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{rock.nongemm_params has numStages 1, but nothing propagates it to the Triton pipeline yet}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #stage_drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.store %1 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// Two outputs would want two grids: nothing says two results have the same
// shape, and the tile is chosen for one of them.
// expected-error @+1 {{a nongemm kernel must hold exactly one rock.store, found 2}}
func.func @two_stores(%arg0: tensor<256xf32>, %d0: tensor<256xf32>, %d1: tensor<256xf32>)
    -> (tensor<256xf32>, tensor<256xf32>) attributes {rock.kernel, rock.nongemm} {
  %0 = math.absf %arg0 : tensor<256xf32>
  %1 = math.exp %arg0 : tensor<256xf32>
  %2 = rock.store %0 to %d0 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  %3 = rock.store %1 to %d1 by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  return %2, %3 : tensor<256xf32>, tensor<256xf32>
}

// -----

// A map with no elementwise op at all is a copy: there is no fusion op to
// retype onto a tile, and nothing for this pipeline to do with it.
#copy_view = #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["a"] at [0] -> ["a"] at [0]>] bounds = [256] -> [256]>

func.func @copy_only(%arg0: tensor<256xf32>, %dest: tensor<256xf32>) -> tensor<256xf32>
    attributes {rock.kernel, rock.nongemm} {
  %v = rock.transform %arg0 by #copy_view : tensor<256xf32> to tensor<256xf32>
  // expected-error @+1 {{must compute its stored value with at least one elementwise op}}
  %0 = rock.store %v to %dest by set : tensor<256xf32> -> tensor<256xf32> to tensor<256xf32>
  return %0 : tensor<256xf32>
}

// -----

// Rows exist to make a cross-lane combine cheap, and a map has none; a pinned
// `rowsPerBlock` would only cap the vector width each thread loads with.
func.func @map_with_pinned_rows(%arg0: tensor<1024xf32>, %dest: tensor<1024xf32>) -> tensor<1024xf32>
    attributes {rock.kernel, rock.nongemm,
                rock.nongemm_params = #rock.nongemm_params<rowsPerBlock = 4, colsPerBlock = 256>} {
  %0 = math.absf %arg0 : tensor<1024xf32>
  // expected-error @+1 {{rock.nongemm_params has rowsPerBlock 4, but a nongemm kernel with no reduce is tiled over its flattened elements}}
  %1 = rock.store %0 to %dest by set : tensor<1024xf32> -> tensor<1024xf32> to tensor<1024xf32>
  return %1 : tensor<1024xf32>
}

// -----

// The views below the reduction are dropped, with both sides of the store
// re-expressed in one flat space, so a view that moves an element cannot be
// among them. This one swaps the two dimensions, which is what a
// `tosa.transpose` after the reduction would become: the types still line up
// after dropping it, and every output would go to the wrong address.
#swap_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#swap = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["a", "b"] at [0, 1] -> ["b", "a"] at [1, 0]>] bounds = [1, 8] -> [8, 1]>
#swap_drop = #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{1, 8} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [1, 8]>

func.func @transpose_after_reduce(%arg0: tensor<8192xf32>, %out: tensor<8xf32>) -> tensor<8xf32>
    attributes {rock.kernel, rock.nongemm} {
  %a = rock.transform %arg0 by #swap_view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{reorder the result: dimension 1 of 'tensor<8x1xf32>' is read out of order}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #swap : tensor<8x1xf32> to tensor<1x8xf32>
  %2 = rock.transform %1 by #swap_drop : tensor<1x8xf32> to tensor<8xf32>
  %3 = rock.store %2 to %out by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// Nor can a view below the reduction add elements that were never reduced:
// a `Pad` is not a reshape, so dropping it would change what is stored.
#pad_view = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 1024 + d1)> by [<Unmerge{8, 1024} ["row", "col"] at [0, 1] -> ["flat"] at [0]>] bounds = [8, 1024] -> [8192]>
#pad_drop = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{8, 1} ["row"] at [0] -> ["d0", "d1"] at [0, 1]>] bounds = [8] -> [8, 1]>
#pad_out = #rock.transform_map<affine_map<(d0) -> (d0)> by [<Pad{0, 8} ["padded"] at [0] -> ["row"] at [0]>] bounds = [16] -> [8]>

func.func @pad_after_reduce(%arg0: tensor<8192xf32>, %out: tensor<16xf32>) -> tensor<16xf32>
    attributes {rock.kernel, rock.nongemm} {
  %a = rock.transform %arg0 by #pad_view : tensor<8192xf32> to tensor<8x1024xf32>
  // expected-error @+1 {{must only reshape the result, but one of them is a Pad}}
  %0 = rock.reduce sum %a {axis = 1 : index} : tensor<8x1024xf32> -> tensor<8x1xf32>
  %1 = rock.transform %0 by #pad_drop : tensor<8x1xf32> to tensor<8xf32>
  %2 = rock.transform %1 by #pad_out : tensor<8xf32> to tensor<16xf32>
  %3 = rock.store %2 to %out by set : tensor<16xf32> -> tensor<16xf32> to tensor<16xf32>
  return %3 : tensor<16xf32>
}
