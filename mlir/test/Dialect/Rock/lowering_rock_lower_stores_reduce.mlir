// Unit tests for the repeated-address atomic reduction that rock-lower-stores
// performs while creating each rock.blockwise_store.
//
// A reduction fused into a GEMM arrives here as a Broadcast on the store
// destination plus an atomic_add store, because that is what rock-lower-reduce
// emits. When a whole tile axis maps to one address, the pass sums that axis in
// a blockwise reduction and pins the matching destination dimension to zero, so
// one atomic replaces a tile axis worth of them.

// RUN: rocmlir-opt -split-input-file -rock-lower-stores -canonicalize -mlir-print-local-scope %s | FileCheck %s

// A tile axis that the destination broadcasts away is reduced blockwise and
// pinned to zero in the destination view.
// CHECK-LABEL: @reduce_broadcast_axis
// CHECK-SAME: (%[[TILE:.*]]: tensor<64x256xf32>, %[[OUT:.*]]: tensor<64xf32>)
//      CHECK: %[[VIEW:.*]] = rock.transform %{{.*}} by {{.*}}Broadcast
//      CHECK: %[[RED:.*]] = rock.blockwise_reduce sum %[[TILE]] {axis = 1 : index} : tensor<64x256xf32> -> tensor<64xf32>
//      CHECK: %[[PINNED:.*]] = rock.transform %[[VIEW]] by <affine_map<(d0) -> (d0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <ConstDim{0, 256} [] at [] -> ["dim1"] at [1]>] bounds = [64] -> [64, 256]> : tensor<64x256xf32> to tensor<64xf32>
//      CHECK: rock.blockwise_store %[[RED]] -> %[[PINNED]] by atomic_add : tensor<64xf32> -> tensor<64xf32> -> tensor<64xf32>
func.func @reduce_broadcast_axis(%tile: tensor<64x256xf32>, %out: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<64x256xf32> -> tensor<64x256xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <AddDim{1} ["dim1"] at [1] -> [] at []>] bounds = [64, 1] -> [64]> : tensor<64xf32> to tensor<64x1xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 256] -> [64, 1]> : tensor<64x1xf32> to tensor<64x256xf32>
  %2 = rock.store %sm to %1 by atomic_add : tensor<64x256xf32> -> tensor<64xf32> to tensor<64x256xf32>
  return %2 : tensor<64xf32>
}

// -----

// A rank-1 tile can be reduced to a rank-0 tensor. RockToTTIR wraps the scalar
// result produced by tt.reduce back into this tensor representation.
// CHECK-LABEL: @reduce_rank1_broadcast_axis
// CHECK-SAME: (%[[TILE:.*]]: tensor<256xf32>, %[[OUT:.*]]: tensor<1xf32>)
//      CHECK: %[[VIEW:.*]] = rock.transform %[[OUT]] by {{.*}}Broadcast
//      CHECK: %[[RED:.*]] = rock.blockwise_reduce sum %[[TILE]] {axis = 0 : index} : tensor<256xf32> -> tensor<f32>
//      CHECK: %[[PINNED:.*]] = rock.transform %[[VIEW]] by <affine_map<() -> (0)> by [<ConstDim{0, 256} [] at [] -> ["dim0"] at [0]>] bounds = [] -> [256]> : tensor<256xf32> to tensor<f32>
//      CHECK: rock.blockwise_store %[[RED]] -> %[[PINNED]] by atomic_add : tensor<f32> -> tensor<f32> -> tensor<1xf32>
func.func @reduce_rank1_broadcast_axis(%tile: tensor<256xf32>, %out: tensor<1xf32>) -> tensor<1xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<256xf32> -> tensor<256xf32>
  %0 = rock.transform %out by <affine_map<(d0) -> (0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>] bounds = [256] -> [1]> : tensor<1xf32> to tensor<256xf32>
  %1 = rock.store %sm to %0 by atomic_add : tensor<256xf32> -> tensor<1xf32> to tensor<256xf32>
  return %1 : tensor<1xf32>
}

// -----

// When both tile axes collapse to the same address, the wider one is reduced.
// Only one axis is ever reduced, so the store keeps a rank-1 source.
// CHECK-LABEL: @reduce_largest_invariant_axis
//      CHECK: rock.blockwise_reduce sum %{{.*}} {axis = 1 : index} : tensor<64x256xf32> -> tensor<64xf32>
//  CHECK-NOT: rock.blockwise_reduce
//      CHECK: rock.blockwise_store %{{.*}} by atomic_add : tensor<64xf32> -> tensor<64xf32> -> tensor<1xf32>
func.func @reduce_largest_invariant_axis(%tile: tensor<64x256xf32>, %out: tensor<1xf32>) -> tensor<1xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<64x256xf32> -> tensor<64x256xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1) -> (0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <AddDim{1} ["dim1"] at [1] -> [] at []>] bounds = [64, 1] -> [1]> : tensor<1xf32> to tensor<64x1xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 256] -> [64, 1]> : tensor<64x1xf32> to tensor<64x256xf32>
  %2 = rock.store %sm to %1 by atomic_add : tensor<64x256xf32> -> tensor<1xf32> to tensor<64x256xf32>
  return %2 : tensor<1xf32>
}

// -----

// Regression case from a fused GEMM + reduction: the reduction output is
// broadcast back over the whole N range, so the 256-wide tile axis maps to one
// address. The block/iteration split carried by the store_marker view must not
// hide that.
#tile_view = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d3, d2 * 256 + d4)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{5, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{128, 256} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 5, 128, 64, 256] -> [1, 320, 32768]>

// CHECK-LABEL: @reduce_fused_gemm_reduction
//      CHECK: %[[RED:.*]] = rock.blockwise_reduce sum %{{.*}} {axis = 1 : index} : tensor<64x256xf32> -> tensor<64xf32>
//      CHECK: %[[PINNED:.*]] = rock.transform %{{.*}} by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3, 0)> by [<PassThrough ["g_block", "m_block", "n_block", "m_iter"] at [0, 1, 2, 3] -> ["g_block", "m_block", "n_block", "m_iter"] at [0, 1, 2, 3]>, <ConstDim{0, 256} [] at [] -> ["n_iter"] at [4]>] bounds = [1, 5, 128, 64] -> [1, 5, 128, 64, 256]> : tensor<1x5x128x64x256xf32> to tensor<1x5x128x64xf32>
//      CHECK: rock.blockwise_store %[[RED]] -> %[[PINNED]][%{{.*}}, %{{.*}}, %{{.*}}] by atomic_add : tensor<64xf32> -> tensor<1x5x128x64xf32> -> tensor<64xf32>
func.func @reduce_fused_gemm_reduction(%tile: tensor<64x256xf32>, %out: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %c0 = arith.constant 0 : i32
  %sm = rock.store_marker %tile views [#tile_view] [%c0, %c0, %c0] : tensor<64x256xf32> -> tensor<1x320x32768xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1, d2) -> (d0 * 32 + d1 + d2)> by [<Unmerge{2, 32, 1} ["col0", "col1", "col2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 32, 1] -> [64]> : tensor<64xf32> to tensor<2x32x1xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 32, 163840] -> [2, 32, 1]> : tensor<2x32x1xf32> to tensor<2x32x163840xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, (d2 * 128 + d3) * 128 + d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Unmerge{10, 128, 128} ["col2", "col3", "col4"] at [2, 3, 4] -> ["dim2"] at [2]>] bounds = [2, 32, 10, 128, 128] -> [2, 32, 163840]> : tensor<2x32x163840xf32> to tensor<2x32x10x128x128xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 10, d1 mod 10, d2, d3)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Merge{32, 10} ["dim1"] at [1] -> ["exp1", "exp2"] at [1, 2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [4]>] bounds = [2, 320, 128, 128] -> [2, 32, 10, 128, 128]> : tensor<2x32x10x128x128xf32> to tensor<2x320x128x128xf32>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 320 + d2, d3, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{1, 320} ["col1", "col2"] at [1, 2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>] bounds = [2, 1, 320, 128, 128] -> [2, 320, 128, 128]> : tensor<2x320x128x128xf32> to tensor<2x1x320x128x128xf32>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d2 floordiv 16384, d0, d1, (d2 floordiv 128) mod 128, d2 mod 128)> by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, <Merge{2, 128, 128} ["gemmN"] at [2] -> ["no", "ho", "wo"] at [0, 3, 4]>] bounds = [1, 320, 32768] -> [2, 1, 320, 128, 128]> : tensor<2x1x320x128x128xf32> to tensor<1x320x32768xf32>
  %6 = rock.store %sm to %5 by atomic_add : tensor<1x320x32768xf32> -> tensor<64xf32> to tensor<1x320x32768xf32>
  return %6 : tensor<64xf32>
}

// -----

// A plain GEMM output store: every tile element has its own address, so there
// is nothing to pre-reduce.
// CHECK-LABEL: @no_reduce_when_address_varies
//  CHECK-NOT: rock.blockwise_reduce
//      CHECK: rock.blockwise_store %{{.*}} by atomic_add : tensor<64x256xf32> -> tensor<64x256xf32> -> tensor<16384xf32>
func.func @no_reduce_when_address_varies(%tile: tensor<64x256xf32>, %out: tensor<16384xf32>) -> tensor<16384xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<64x256xf32> -> tensor<64x256xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1) -> (d0 * 256 + d1)> by [<Unmerge{64, 256} ["dim0", "dim1"] at [0, 1] -> ["raw"] at [0]>] bounds = [64, 256] -> [16384]> : tensor<16384xf32> to tensor<64x256xf32>
  %1 = rock.store %sm to %0 by atomic_add : tensor<64x256xf32> -> tensor<16384xf32> to tensor<64x256xf32>
  return %1 : tensor<16384xf32>
}

// -----

// AddDim removes its coordinate from the destination address, but a unit source
// axis has only one contributor and therefore needs no blockwise reduction.
// CHECK-LABEL: @no_reduce_for_unit_add_dim
//  CHECK-NOT: rock.blockwise_reduce
//      CHECK: rock.blockwise_store %{{.*}} by atomic_add : tensor<64x1xf32> -> tensor<64x1xf32> -> tensor<64xf32>
func.func @no_reduce_for_unit_add_dim(%tile: tensor<64x1xf32>, %out: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<64x1xf32> -> tensor<64x1xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["row"] at [0] -> ["row"] at [0]>, <AddDim{1} ["unit"] at [1] -> [] at []>] bounds = [64, 1] -> [64]> : tensor<64xf32> to tensor<64x1xf32>
  %1 = rock.store %sm to %0 by atomic_add : tensor<64x1xf32> -> tensor<64xf32> to tensor<64x1xf32>
  return %1 : tensor<64xf32>
}

// -----

// The standalone pass may encounter shapes that the production pipeline would
// first decompose. Leave their atomics intact rather than create a tt.reduce
// whose tensor has a non-power-of-two element count.
// CHECK-LABEL: @no_reduce_for_non_power_of_two_tile
//  CHECK-NOT: rock.blockwise_reduce
//      CHECK: rock.blockwise_store %{{.*}} by atomic_add : tensor<64x3xf32> -> tensor<64x3xf32> -> tensor<64xf32>
func.func @no_reduce_for_non_power_of_two_tile(%tile: tensor<64x3xf32>, %out: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<64x3xf32> -> tensor<64x3xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["row"] at [0] -> ["row"] at [0]>, <AddDim{3} ["reduction"] at [1] -> [] at []>] bounds = [64, 3] -> [64]> : tensor<64xf32> to tensor<64x3xf32>
  %1 = rock.store %sm to %0 by atomic_add : tensor<64x3xf32> -> tensor<64xf32> to tensor<64x3xf32>
  return %1 : tensor<64xf32>
}

// -----

// Only atomic_add accumulates, so a `set` store over the same broadcast view
// keeps its last-writer-wins semantics.
// CHECK-LABEL: @no_reduce_for_set_store
//  CHECK-NOT: rock.blockwise_reduce
//      CHECK: rock.blockwise_store %{{.*}} by set
func.func @no_reduce_for_set_store(%tile: tensor<64x256xf32>, %out: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<64x256xf32> -> tensor<64x256xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <AddDim{1} ["dim1"] at [1] -> [] at []>] bounds = [64, 1] -> [64]> : tensor<64xf32> to tensor<64x1xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 256] -> [64, 1]> : tensor<64x1xf32> to tensor<64x256xf32>
  %2 = rock.store %sm to %1 by set : tensor<64x256xf32> -> tensor<64xf32> to tensor<64x256xf32>
  return %2 : tensor<64xf32>
}

// -----

// atomic_max is left alone: arith.maximumf and the hardware atomic disagree
// about NaN, so folding lanes together is not obviously value-preserving.
// CHECK-LABEL: @no_reduce_for_atomic_max
//  CHECK-NOT: rock.blockwise_reduce
//      CHECK: rock.blockwise_store %{{.*}} by atomic_max
func.func @no_reduce_for_atomic_max(%tile: tensor<64x256xf32>, %out: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<64x256xf32> -> tensor<64x256xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <AddDim{1} ["dim1"] at [1] -> [] at []>] bounds = [64, 1] -> [64]> : tensor<64xf32> to tensor<64x1xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 256] -> [64, 1]> : tensor<64x1xf32> to tensor<64x256xf32>
  %2 = rock.store %sm to %1 by atomic_max : tensor<64x256xf32> -> tensor<64xf32> to tensor<64x256xf32>
  return %2 : tensor<64xf32>
}

// -----

// The broadcast axis is padded, so the last 56 lanes along it are masked off.
// Folding them into the sum the surviving lanes store would add values that
// never reached memory, and the pass declines the rewrite rather than prove
// the padding harmless. This is also what keeps a GEMM whose M or N is not a
// multiple of the tile out of the rewrite, since gemm-to-gridwise pads the
// output views in that case.
// CHECK-LABEL: @no_reduce_when_padded
//  CHECK-NOT: rock.blockwise_reduce
//      CHECK: rock.blockwise_store %{{.*}} by atomic_add
func.func @no_reduce_when_padded(%tile: tensor<64x256xf32>, %out: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel} {
  %sm = rock.store_marker %tile views [] : tensor<64x256xf32> -> tensor<64x256xf32>
  %0 = rock.transform %out by <affine_map<(d0, d1) -> (d0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <AddDim{1} ["dim1"] at [1] -> [] at []>] bounds = [64, 1] -> [64]> : tensor<64xf32> to tensor<64x1xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1) -> (d0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 200] -> [64, 1]> : tensor<64x1xf32> to tensor<64x200xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Pad{0, 56} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 256] -> [64, 200]> : tensor<64x200xf32> to tensor<64x256xf32>
  %3 = rock.store %sm to %2 by atomic_add : tensor<64x256xf32> -> tensor<64xf32> to tensor<64x256xf32>
  return %3 : tensor<64xf32>
}
