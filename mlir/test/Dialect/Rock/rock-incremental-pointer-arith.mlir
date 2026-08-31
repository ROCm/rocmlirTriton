// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-incremental-pointer-arith --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @affine_linear_load
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<32768xf16>, %[[INIT:.*]]: tensor<64x64xf16>)
// Affine path: no new iter_arg, the loop carries only the original one.
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<64x64xf16>)
// Base pointer rebuilt in the loop with the iv pinned to the lower bound %c0_i32:
//       CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c1_i32]
//       CHECK:     %[[D:.*]] = arith.subi %[[IV]], %{{.*}} : i32
//       CHECK:     %[[OFF:.*]] = arith.muli %[[D]], %{{.*}} : i32
//       CHECK:     %[[OFFT:.*]] = tt.splat %[[OFF]] : i32 -> tensor<64x64xi32>
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[OFFT]] : tensor<64x64xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//       CHECK:     scf.yield %{{.*}} : tensor<64x64xf16>
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
func.func @affine_linear_load(%arg0: tensor<32768xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = rock.transform %arg0 by #transform_map : tensor<32768xf16> to tensor<1x256x128xf16>
  %1 = scf.for %arg2 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<64x64xf16>)  : i32 {
    %2 = rock.transform %0 by #transform_map1 : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %2[%arg2, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %3 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    scf.yield %3 : tensor<64x64xf16>
  }
  return %1 : tensor<64x64xf16>
}

// -----

// Same linear load, but the loop induction variable is `index` instead of i32.
//
// CHECK-LABEL: func @affine_index_iv
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<32768xf16>, %[[INIT:.*]]: tensor<64x64xf16>)
// Affine path: only the original iter_arg is carried:
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<64x64xf16>) {
// Base index uses the loop lower bound (index_cast of %c0), not the iv:
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%{{.*}}, %c0_i32, %c0_i32, %c1_i32]
//       CHECK:     %[[D:.*]] = arith.subi %[[IV]], %{{.*}} : index
//       CHECK:     %[[DC:.*]] = arith.index_cast %[[D]] : index to i32
//       CHECK:     %[[OFF:.*]] = arith.muli %[[DC]], %{{.*}} : i32
//       CHECK:     %[[OFFT:.*]] = tt.splat %[[OFF]] : i32 -> tensor<64x64xi32>
//       CHECK:     %[[PTR:.*]] = arith.addi %{{.*}}, %[[OFFT]] : tensor<64x64xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%{{.*}}]
//       CHECK:     scf.yield
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
func.func @affine_index_iv(%arg0: tensor<32768xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by #transform_map : tensor<32768xf16> to tensor<1x256x128xf16>
  %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %arg1) -> (tensor<64x64xf16>) {
    %iv = arith.index_cast %arg2 : index to i32
    %2 = rock.transform %0 by #transform_map1 : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %2[%iv, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %3 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    scf.yield %3 : tensor<64x64xf16>
  }
  return %1 : tensor<64x64xf16>
}

// -----

// Negative case: the transform chain contains a Pad on the K dimension, so the
// validity mask is coordinate-dependent (and here even iv-dependent). The pass
// must conservatively leave the loop untouched: no state is added before the
// loop and the transforms_to_ptr stays in the body.
//
// CHECK-LABEL: func @no_simplify_with_pad
//   CHECK-NOT:   rock.transforms_to_ptr
//       CHECK:   scf.for
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%{{.*}}, %c0_i32, %c0_i32, %c1_i32]
//       CHECK:     rock.blockwise_load_ptr
//       CHECK:     scf.yield
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{254, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 254, 128] -> [32512]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Pad{0, 2} ["kpad"] at [1] -> ["k"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>] bounds = [1, 256, 128] -> [1, 254, 128]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
func.func @no_simplify_with_pad(%arg0: tensor<32512xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = rock.transform %arg0 by #transform_map : tensor<32512xf16> to tensor<1x254x128xf16>
  %1 = scf.for %arg2 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<64x64xf16>)  : i32 {
    %2 = rock.transform %0 by #transform_map1 : tensor<1x254x128xf16> to tensor<1x256x128xf16>
    %3 = rock.transform %2 by #transform_map2 : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %3[%arg2, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %4 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    scf.yield %4 : tensor<64x64xf16>
  }
  return %1 : tensor<64x64xf16>
}

// -----

// 1x1-convolution-style loop with two operands sharing one loop:
//   - the filter chain has no validity-impacting maps and a linear offset, so
//     it IS simplified in place (base op pinned to the lower bound + scalar
//     affine tail, left in the loop),
//   - the input chain has a Pad on K, so it is NOT simplified and its
//     transforms_to_ptr stays inside the loop, still indexed by the iv.
//
// CHECK-LABEL: func @affine_one_of_two
//  CHECK-SAME: (%[[FILTER:.*]]: tensor<32768xf16>, %[[INPUT:.*]]: tensor<32512xf16>, %[[INIT:.*]]: tensor<64x64xf16>)
// Affine path: no offset accumulator, only the original iter_arg is carried.
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<64x64xf16>)
// Filter operand simplified in place: base op pinned to %c0_i32 + scalar affine tail.
//       CHECK:     %[[FPTRS:.*]], %[[FMASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c1_i32]
//       CHECK:     %[[FD:.*]] = arith.subi %[[IV]], %{{.*}} : i32
//       CHECK:     %[[FOFF:.*]] = arith.muli %[[FD]], %{{.*}} : i32
//       CHECK:     %[[FOFFT:.*]] = tt.splat %[[FOFF]] : i32 -> tensor<64x64xi32>
//       CHECK:     %[[FPTR:.*]] = arith.addi %[[FPTRS]], %[[FOFFT]] : tensor<64x64xi32>
//       CHECK:     rock.blockwise_load_ptr %[[FPTR]][%[[FMASK]]]
// Input operand NOT simplified: its transforms_to_ptr stays in the loop, still
// indexed by the induction variable %[[IV]]:
//       CHECK:     %[[IPTRS:.*]], %[[IMASK:.*]] = rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32, %c0_i32, %c1_i32]
//       CHECK:     rock.blockwise_load_ptr %[[IPTRS]][%[[IMASK]]]
//       CHECK:     scf.yield
func.func @affine_one_of_two(%filter: tensor<32768xf16>, %input: tensor<32512xf16>, %init: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32

  // Filter root view (reducible: no pad, offset linear in k_loop).
  %f0 = rock.transform %filter by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : tensor<32768xf16> to tensor<1x256x128xf16>
  // Input root view (K is only 254 here, padded to 256 inside the loop).
  %i0 = rock.transform %input by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{254, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 254, 128] -> [32512]> : tensor<32512xf16> to tensor<1x254x128xf16>

  %res = scf.for %k = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<64x64xf16>) : i32 {
    // Filter tiling view + load.
    %f1 = rock.transform %f0 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]> : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %fptr, %fmask = rock.transforms_to_ptr %f1[%k, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %fload = rock.blockwise_load_ptr %fptr[%fmask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

    // Input Pad (K 254 -> 256) makes the mask coordinate-dependent: not simplified.
    %i1 = rock.transform %i0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Pad{0, 2} ["kpad"] at [1] -> ["k"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>] bounds = [1, 256, 128] -> [1, 254, 128]> : tensor<1x254x128xf16> to tensor<1x256x128xf16>
    %i2 = rock.transform %i1 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]> : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %iptr, %imask = rock.transforms_to_ptr %i2[%k, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %iload = rock.blockwise_load_ptr %iptr[%imask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

    %sum = arith.addf %fload, %iload : tensor<64x64xf16>
    scf.yield %sum : tensor<64x64xf16>
  }

  return %res : tensor<64x64xf16>
}

// -----

// Validity-impacting Pad on a dimension the iv does NOT flow through.
// This mirrors a 3x3-conv filter operand: gemmM is padded (Pad{0, 192}) while
// the induction variable (k_loop) lives in gemmK. The mask therefore does not
// vary with the iv, so the op IS simplified even though the chain contains a Pad.
//
// CHECK-LABEL: func @affine_pad_on_non_iv_dim
//  CHECK-SAME: (%[[FILTER:.*]]: tensor<36864xi8>, %[[INIT:.*]]: tensor<256x32xi8>)
// Affine path: only the original iter_arg is carried:
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<256x32xi8>)
// Base op pinned to %c0_i32 (Pad on a non-iv dim keeps the mask iv-invariant):
//       CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
//       CHECK:     %[[D:.*]] = arith.subi %[[IV]], %{{.*}} : i32
//       CHECK:     %[[OFF:.*]] = arith.muli %[[D]], %{{.*}} : i32
//       CHECK:     %[[OFFT:.*]] = tt.splat %[[OFF]] : i32 -> tensor<256x32xi32>
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[OFFT]] : tensor<256x32xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//       CHECK:     scf.yield
func.func @affine_pad_on_non_iv_dim(%filter: tensor<36864xi8>, %init: tensor<256x32xi8>) -> tensor<256x32xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c18_i32 = arith.constant 18 : i32

  // Filter root view: gemmG=1, gemmM=64, gemmK=576.
  %f0 = rock.transform %filter by <affine_map<(d0, d1, d2) -> (d1 * 576 + d2)> by [<Unmerge{64, 576} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 576] -> [36864]> : tensor<36864xi8> to tensor<1x64x576xi8>

  %res = scf.for %k = %c0_i32 to %c18_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<256x32xi8>) : i32 {
    // Pad gemmM 64 -> 256: validity-impacting, but on a non-iv dimension.
    %f1 = rock.transform %f0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Pad{0, 192} ["gemmMPad"] at [1] -> ["m"] at [1]>, <PassThrough ["k"] at [2] -> ["k"] at [2]>] bounds = [1, 256, 576] -> [1, 64, 576]> : tensor<1x64x576xi8> to tensor<1x256x576xi8>
    // Tiling: k_loop (extra index 0, the iv) feeds gemmK; m feeds gemmM.
    %f2 = rock.transform %f1 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 256 + d4, d0 * 32 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{18, 32} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 256} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{3136} ["n_block"] at [3] -> [] at []>] bounds = [18, 1, 1, 3136, 256, 32] -> [1, 256, 576]> : tensor<1x256x576xi8> to tensor<18x1x1x3136x256x32xi8>
    %fptr, %fmask = rock.transforms_to_ptr %f2[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<18x1x1x3136x256x32xi8> -> tensor<256x32xi32>, tensor<256x32xi1>
    %fload = rock.blockwise_load_ptr %fptr[%fmask] {cacheModifier = #rock<CacheModifier none>} : tensor<256x32xi32>, tensor<256x32xi1> -> tensor<256x32xi8>
    scf.yield %fload : tensor<256x32xi8>
  }

  return %res : tensor<256x32xi8>
}

// -----

// Faithful 3x3-conv filter chain. Here gemmK is
// produced by a Merge{64, 3, 3} that decomposes it into (c, 0, 1) via
// floordiv/mod, and the buffer packing below re-linearizes them with matching
// strides. The net offset is raw = gemmM*576 + gemmK, linear in the iv with
// stride 32. The iv (k_loop, stride 32 in gemmK) straddles the Merge{64,3,3}
// factors (9/3/3), so a flattened affine map keeps opaque floordiv/mod; the
// transform-aware index-diff stride computation handles it (constantFold of the
// merge gives component diffs 3/1/2, recombined through the contiguous pack
// strides 9/3/1 = 9*3 + 3*1 + 1*2 = 32) and the op IS simplified.
//
// CHECK-LABEL: func @affine_conv_filter_merge
//  CHECK-SAME: (%[[FILTER:.*]]: tensor<36864xi8>, %[[INIT:.*]]: tensor<256x32xi8>)
// Affine path: only the original iter_arg is carried:
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<256x32xi8>)
// Base op pinned to %c0_i32; the merge is constant-folded away at the lb seed:
//       CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
//       CHECK:     %[[D:.*]] = arith.subi %[[IV]], %{{.*}} : i32
//       CHECK:     %[[OFF:.*]] = arith.muli %[[D]], %{{.*}} : i32
//       CHECK:     %[[OFFT:.*]] = tt.splat %[[OFF]] : i32 -> tensor<256x32xi32>
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[OFFT]] : tensor<256x32xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//       CHECK:     scf.yield
func.func @affine_conv_filter_merge(%filter: tensor<36864xi8>, %init: tensor<256x32xi8>) -> tensor<256x32xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c18_i32 = arith.constant 18 : i32

  // raw buffer (gkc01): k_out=64, c=64, 0=3, 1=3 -> 36864.
  %0 = rock.transform %filter by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 64 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{64, 64, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64, 3, 3] -> [36864]> : tensor<36864xi8> to tensor<1x64x64x3x3xi8>
  // Merge (c, 0, 1) into gemmK (floordiv/mod), gemmM = k_out.
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 floordiv 3) mod 3, d1 mod 3)> by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{64, 3, 3} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 576, 64] -> [1, 64, 64, 3, 3]> : tensor<1x64x64x3x3xi8> to tensor<1x576x64xi8>
  // Transpose gemmM <-> gemmK.
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM", "gemmK"] at [1, 2] -> ["gemmM", "gemmK"] at [2, 1]>] bounds = [1, 64, 576] -> [1, 576, 64]> : tensor<1x576x64xi8> to tensor<1x64x576xi8>

  %res = scf.for %k = %c0_i32 to %c18_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<256x32xi8>) : i32 {
    // Pad gemmM 64 -> 256: validity-impacting on a non-iv dimension.
    %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 192} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmK"] at [2] -> ["gemmK"] at [2]>] bounds = [1, 256, 576] -> [1, 64, 576]> : tensor<1x64x576xi8> to tensor<1x256x576xi8>
    // Tiling: k_loop (extra index 0, the iv) feeds gemmK; m feeds gemmM.
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 256 + d4, d0 * 32 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{18, 32} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [2]>, <Unmerge{1, 256} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [1]>, <AddDim{3136} ["n_block"] at [3] -> [] at []>] bounds = [18, 1, 1, 3136, 256, 32] -> [1, 256, 576]> : tensor<1x256x576xi8> to tensor<18x1x1x3136x256x32xi8>
    %fptr, %fmask = rock.transforms_to_ptr %4[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<18x1x1x3136x256x32xi8> -> tensor<256x32xi32>, tensor<256x32xi1>
    %fload = rock.blockwise_load_ptr %fptr[%fmask] {cacheModifier = #rock<CacheModifier none>} : tensor<256x32xi32>, tensor<256x32xi1> -> tensor<256x32xi8>
    scf.yield %fload : tensor<256x32xi8>
  }

  return %res : tensor<256x32xi8>
}

// -----

// Two simplifiable loads in one loop that point to the same root buffer
// (%arg0) but view it through differently-shaped tiles (64x64 and 64x128).
//
// CHECK-LABEL: func @affine_two_sharing_base
//  CHECK-SAME: (%[[BUF:.*]]: tensor<32768xf16>, %[[INITA:.*]]: tensor<64x64xf16>, %[[INITB:.*]]: tensor<64x128xf16>)
// Affine path: no offset accumulators, only the two original iter_args:
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INITA]], %{{.*}} = %[[INITB]]) -> (tensor<64x64xf16>, tensor<64x128xf16>)
//       CHECK:     %[[PTRS0:.*]], %[[MASK0:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c1_i32] {{.*}} -> tensor<64x64xi32>
//       CHECK:     %[[P0:.*]] = arith.addi %[[PTRS0]], %{{.*}} : tensor<64x64xi32>
//       CHECK:     rock.blockwise_load_ptr %[[P0]][%[[MASK0]]]
//       CHECK:     %[[PTRS1:.*]], %[[MASK1:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32] {{.*}} -> tensor<64x128xi32>
//       CHECK:     %[[P1:.*]] = arith.addi %[[PTRS1]], %{{.*}} : tensor<64x128xi32>
//       CHECK:     rock.blockwise_load_ptr %[[P1]][%[[MASK1]]]
//       CHECK:     scf.yield
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 128 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 128} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 1, 64, 128] -> [1, 256, 128]>
func.func @affine_two_sharing_base(%arg0: tensor<32768xf16>, %initA: tensor<64x64xf16>, %initB: tensor<64x128xf16>) -> (tensor<64x64xf16>, tensor<64x128xf16>) attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = rock.transform %arg0 by #transform_map : tensor<32768xf16> to tensor<1x256x128xf16>
  %r:2 = scf.for %k = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%accA = %initA, %accB = %initB) -> (tensor<64x64xf16>, tensor<64x128xf16>) : i32 {
    // View A: 64x64 tile of the buffer.
    %a = rock.transform %0 by #transform_map1 : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %pa, %ma = rock.transforms_to_ptr %a[%k, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %la = rock.blockwise_load_ptr %pa[%ma] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    // View B: 64x128 tile of the *same* buffer (differently shaped recurrence).
    %b = rock.transform %0 by #transform_map2 : tensor<1x256x128xf16> to tensor<4x1x1x1x64x128xf16>
    %pb, %mb = rock.transforms_to_ptr %b[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<4x1x1x1x64x128xf16> -> tensor<64x128xi32>, tensor<64x128xi1>
    %lb = rock.blockwise_load_ptr %pb[%mb] {cacheModifier = #rock<CacheModifier none>} : tensor<64x128xi32>, tensor<64x128xi1> -> tensor<64x128xf16>
    %sa = arith.addf %accA, %la : tensor<64x64xf16>
    %sb = arith.addf %accB, %lb : tensor<64x128xf16>
    scf.yield %sa, %sb : tensor<64x64xf16>, tensor<64x128xf16>
  }
  return %r#0, %r#1 : tensor<64x64xf16>, tensor<64x128xf16>
}

// -----

// Case where the two trailing merge factors are 1.
// gemmK is therefore Merge{64, 1, 1} of only the
// channel dim c (raw stride 1). The iv (k_loop) advances gemmK by 32, which
// stays entirely within c (32 < 64, no carry), so the offset is linear with
// stride 32 and the op IS simplified. The size-1 dims must be skipped by the
// merge carry-neutrality guard: their stride-0 layout would otherwise spuriously
// fail the contiguity check (stride(c) 1 != stride("0") 0) and bail.
//
// CHECK-LABEL: func @affine_conv_1x1_unit_merge
//  CHECK-SAME: (%[[FILTER:.*]]: tensor<4096xi8>, %[[INIT:.*]]: tensor<64x32xi8>)
// Affine path: only the original iter_arg is carried:
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<64x32xi8>)
//       CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
//       CHECK:     %[[D:.*]] = arith.subi %[[IV]], %{{.*}} : i32
//       CHECK:     %[[OFF:.*]] = arith.muli %[[D]], %{{.*}} : i32
//       CHECK:     %[[OFFT:.*]] = tt.splat %[[OFF]] : i32 -> tensor<64x32xi32>
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[OFFT]] : tensor<64x32xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//       CHECK:     scf.yield
func.func @affine_conv_1x1_unit_merge(%filter: tensor<4096xi8>, %init: tensor<64x32xi8>) -> tensor<64x32xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32

  // raw buffer (gkc01, 1x1 filter): k_out=64, c=64, 0=1, 1=1 -> 4096. The unit
  // spatial dims "0"/"1" are AddDim'd (no presence in the raw buffer).
  %0 = rock.transform %filter by <affine_map<(d0, d1, d2, d3, d4) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["k", "c"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{1} ["0"] at [3] -> [] at []>, <AddDim{1} ["1"] at [4] -> [] at []>] bounds = [1, 64, 64, 1, 1] -> [4096]> : tensor<4096xi8> to tensor<1x64x64x1x1xi8>
  // Merge (c, 0, 1) into gemmK; the trailing two factors are size-1.
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2, 0, 0)> by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [1]>, <Merge{64, 1, 1} ["gemmK"] at [2] -> ["c", "0", "1"] at [2, 3, 4]>] bounds = [1, 64, 64] -> [1, 64, 64, 1, 1]> : tensor<1x64x64x1x1xi8> to tensor<1x64x64xi8>

  %res = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<64x32xi8>) : i32 {
    // Tiling: k_loop (extra index 0, the iv) feeds gemmK with stride 32.
    %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2 * 64 + d4, d0 * 32 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{2, 32} ["k_loop", "k_iter"] at [0, 5] -> ["gemmK"] at [2]>, <Unmerge{1, 64} ["m_block", "m_iter"] at [2, 4] -> ["gemmM"] at [1]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [2, 1, 1, 1, 64, 32] -> [1, 64, 64]> : tensor<1x64x64xi8> to tensor<2x1x1x1x64x32xi8>
    %fptr, %fmask = rock.transforms_to_ptr %2[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<2x1x1x1x64x32xi8> -> tensor<64x32xi32>, tensor<64x32xi1>
    %fload = rock.blockwise_load_ptr %fptr[%fmask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x32xi32>, tensor<64x32xi1> -> tensor<64x32xi8>
    scf.yield %fload : tensor<64x32xi8>
  }

  return %res : tensor<64x32xi8>
}

// -----

// Coordinate/carry path: faithful (1D) conv-input load where the affine fast
// path must bail. gemmK is a Merge{2, 3} of (channel, filter-tap) and the iv
// (k_loop, stride 2 in gemmK) straddles the tap factor, so the offset is
// non-linear in the iv as a whole; the tap also flows through an Embed sliding
// window into a padded width, so the validity mask depends on the iv too. Once
// the merge is decomposed each coordinate linearizes through the packing below
// and the validity coordinate (tap) forms a suffix, so the pass takes the
// full-tile pointer recurrence: it pins the base transforms_to_ptr to the loop
// lower bound, carries the tap coordinate and a full-tile offset accumulator as
// iter_args, advances them with a mixed-radix add/compare/select odometer (no
// floordiv/mod -- cf. rocMLIR's IndexDiffUpdate), rebuilds only the (iv-varying)
// mask, and forms the pointer as base + accumulator. The high channel
// coordinate is dropped; its offset contribution is recovered from the carry.
//
// CHECK-LABEL: func @carry_conv_input
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi8>, %[[INIT:.*]]: tensor<2x4xi8>)
// The loop carries the tap coordinate (full tile) and a full-tile offset
// accumulator alongside the result:
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<2x4xi8>, tensor<2x4xi32>, tensor<2x4xi32>)
// Base pointer rebuilt at iv == lb (all extra indices pinned to %c0_i32):
//       CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
// No division in the offset/mask rebuild (the merge split is maintained by the
// carry, not recomputed):
//   CHECK-NOT:     floordiv
//   CHECK-NOT:     arith.divui
//   CHECK-NOT:     arith.divsi
//   CHECK-NOT:     arith.remui
//   CHECK-NOT:     arith.remsi
// Pointer = base + carried offset accumulator:
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %{{.*}} : tensor<2x4xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][
// Mixed-radix carry update (compare + select) advances the coordinate:
//       CHECK:     arith.cmpi uge
//       CHECK:     arith.select
//       CHECK:     scf.yield
func.func @carry_conv_input(%arg0: tensor<8xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32

  // raw input buffer (ngc1): ci=2, 1i=4 -> 8.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3)> by [<Unmerge{2, 4} ["ci", "1i"] at [2, 3] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 2, 4] -> [8]> : tensor<8xi8> to tensor<1x1x2x4xi8>
  // Halo pad on the width: 4 -> 6 (validity-impacting).
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 - 1)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Pad{1, 1} ["1ipad"] at [3] -> ["1i"] at [3]>] bounds = [1, 1, 2, 6] -> [1, 1, 2, 4]> : tensor<1x1x2x4xi8> to tensor<1x1x2x6xi8>
  // Sliding window: 1ipad = tap("1") + out("1o").
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 + d4)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [3]>] bounds = [1, 1, 2, 3, 4] -> [1, 1, 2, 6]> : tensor<1x1x2x6xi8> to tensor<1x1x2x3x4xi8>

  %res = scf.for %k = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<2x4xi8>) : i32 {
    // Merge channel+tap into gemmK, ni+out into gemmN.
    %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 3, d1 mod 3, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{2, 3} ["gemmK"] at [1] -> ["ci", "1"] at [2, 3]>, <Merge{1, 4} ["gemmN"] at [2] -> ["ni", "1o"] at [0, 4]>] bounds = [1, 6, 4] -> [1, 1, 2, 3, 4]> : tensor<1x1x2x3x4xi8> to tensor<1x6x4xi8>
    // Tiling: k_loop (the iv) feeds gemmK with stride 2 (k_iter = 2).
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 2 + d4, d3 * 4 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{3, 2} ["k_loop", "k_iter"] at [0, 4] -> ["gemmK"] at [1]>, <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [3, 1, 1, 1, 2, 4] -> [1, 6, 4]> : tensor<1x6x4xi8> to tensor<3x1x1x1x2x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %4[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<3x1x1x1x2x4xi8> -> tensor<2x4xi32>, tensor<2x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<2x4xi32>, tensor<2x4xi1> -> tensor<2x4xi8>
    scf.yield %load : tensor<2x4xi8>
  }
  return %res : tensor<2x4xi8>
}

// -----

// Carry path, 2D conv input. gemmK is a Merge{2, 3, 3} of (channel, tap0, tap1)
// and the iv (k_loop, stride 2 in gemmK) straddles the tap factors, so the
// affine fast path bails. Both spatial taps flow through a halo Pad + Embed
// sliding window, so the validity mask is iv-dependent. Once the merge is
// decomposed each coordinate linearizes and the two validity taps form a
// suffix, so the pass takes the full-tile pointer recurrence: it pins the base
// transforms_to_ptr to the loop lower bound, carries the two tap coordinates
// and a full-tile offset accumulator as iter_args, advances them with a
// mixed-radix add/compare/select odometer, and rebuilds only the mask. The high
// channel coordinate is dropped; its offset contribution comes from the carry.
//
// CHECK-LABEL: func @carry_conv2d_input_rank_redundant
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<32xi8>, %[[INIT:.*]]: tensor<2x4xi8>)
// The two tap coordinates (full tile) and a full-tile offset accumulator are
// carried alongside the result:
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<2x4xi8>, tensor<2x4xi32>, tensor<2x4xi32>, tensor<2x4xi32>)
// Base pointer rebuilt at iv == lb (all extra indices pinned to %c0_i32):
//       CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
// No division in the offset/mask rebuild (the merge split is maintained by the
// carry):
//   CHECK-NOT:     floordiv
//   CHECK-NOT:     arith.divui
//   CHECK-NOT:     arith.divsi
//   CHECK-NOT:     arith.remui
//   CHECK-NOT:     arith.remsi
// Pointer = base + carried offset accumulator:
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %{{.*}} : tensor<2x4xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][
// Two carry stages (compare + select) for the two lower digits (tap1, tap0):
//       CHECK:     arith.cmpi uge
//       CHECK:     arith.select
//       CHECK:     arith.cmpi uge
//       CHECK:     arith.select
//       CHECK:     scf.yield
func.func @carry_conv2d_input_rank_redundant(%arg0: tensor<32xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c9_i32 = arith.constant 9 : i32

  // raw input buffer (ngc01): ci=2, 0i=4, 1i=4 -> 32.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 4 + d3) * 4 + d4)> by [<Unmerge{2, 4, 4} ["ci", "0i", "1i"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 2, 4, 4] -> [32]> : tensor<32xi8> to tensor<1x1x2x4x4xi8>
  // Halo pad on both spatial dims: 4 -> 6 (validity-impacting).
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)> by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{1, 1, 1, 1} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>] bounds = [1, 1, 2, 6, 6] -> [1, 1, 2, 4, 4]> : tensor<1x1x2x4x4xi8> to tensor<1x1x2x6x6xi8>
  // Sliding windows: 0ipad = tap0 + 0o ; 1ipad = tap1 + 1o.
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>] bounds = [1, 1, 2, 3, 4, 3, 4] -> [1, 1, 2, 6, 6]> : tensor<1x1x2x6x6xi8> to tensor<1x1x2x3x4x3x4xi8>

  %res = scf.for %k = %c0_i32 to %c9_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<2x4xi8>) : i32 {
    // Merge (ci, tap0, tap1) into gemmK (radix 2,3,3); (ni, 0o, 1o) into gemmN.
    %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 9, (d1 floordiv 3) mod 3, d2 floordiv 4, d1 mod 3, d2 mod 4)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{2, 3, 3} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>, <Merge{1, 4, 4} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 4, 6]>] bounds = [1, 18, 16] -> [1, 1, 2, 3, 4, 3, 4]> : tensor<1x1x2x3x4x3x4xi8> to tensor<1x18x16xi8>
    // Tiling: k_loop (the iv) feeds gemmK with stride 2 (k_iter = 2); n_iter = 4.
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 2 + d4, d3 * 4 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{9, 2} ["k_loop", "k_iter"] at [0, 4] -> ["gemmK"] at [1]>, <Unmerge{4, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [9, 1, 1, 4, 2, 4] -> [1, 18, 16]> : tensor<1x18x16xi8> to tensor<9x1x1x4x2x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %4[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<9x1x1x4x2x4xi8> -> tensor<2x4xi32>, tensor<2x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<2x4xi32>, tensor<2x4xi1> -> tensor<2x4xi8>
    scf.yield %load : tensor<2x4xi8>
  }
  return %res : tensor<2x4xi8>
}

// -----

// Carry path, no-prefix case. Single-channel 2D (3x3) conv input: gemmK is a
// Merge{3, 3} of only the two spatial taps (tap0, tap1) -- there is no channel
// coordinate above them -- and *both* taps flow through a halo Pad + Embed
// sliding window, so every carried coordinate is validity-impacting. The
// impacting coordinates therefore form the *entire* suffix (prefixSize == 0,
// hasPrefix == false), unlike @carry_conv_input /
// @carry_conv2d_input_rank_redundant where the non-impacting channel is a
// dropped prefix.
//
// Consequences of the no-prefix path (see emitCarryStep):
//   - the most-significant carried coordinate (tap0) is the global top of the
//     merge, so the loop upper bound keeps it in range and it is NEVER wrapped
//     (no compare/select emitted for it); it only advances via the carry
//     rippling in from below,
//   - there is no dropped-prefix offset term to recover.
// The iv (k_loop) advances gemmK by 1 per step (k_iter = 1), so the lower
// coordinate (tap1) counts 0,1,2 and wraps, carrying into the top coordinate
// (tap0). Only tap1 needs a wrap guard, so exactly one compare/select stage is
// emitted.
//
// CHECK-LABEL: func @carry_conv2d_input_no_prefix
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<16xi8>, %[[INIT:.*]]: tensor<1x4xi8>)
// Both tap coordinates (full tile) and a full-tile offset accumulator are
// carried alongside the result (no coordinate is dropped as a prefix):
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (tensor<1x4xi8>, tensor<1x4xi32>, tensor<1x4xi32>, tensor<1x4xi32>)
// Base pointer rebuilt at iv == lb (all extra indices pinned to %c0_i32):
//       CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
// No division in the offset/mask rebuild (the merge split is maintained by the
// carry):
//   CHECK-NOT:     floordiv
//   CHECK-NOT:     arith.divui
//   CHECK-NOT:     arith.divsi
//   CHECK-NOT:     arith.remui
//   CHECK-NOT:     arith.remsi
// Pointer = base + carried offset accumulator:
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %{{.*}} : tensor<1x4xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][
// Exactly ONE mixed-radix wrap stage: only the lower coordinate (tap1) has a
// compare/select; the top coordinate (tap0) is never wrapped:
//       CHECK:     arith.cmpi uge
//       CHECK:     arith.select
//   CHECK-NOT:     arith.cmpi uge
//       CHECK:     scf.yield
func.func @carry_conv2d_input_no_prefix(%arg0: tensor<16xi8>, %arg1: tensor<1x4xi8>) -> tensor<1x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c9_i32 = arith.constant 9 : i32

  // raw input buffer (single channel, n g 0i 1i): 0i=4, 1i=4 -> 16.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3)> by [<Unmerge{4, 4} ["0i", "1i"] at [2, 3] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 4, 4] -> [16]> : tensor<16xi8> to tensor<1x1x4x4xi8>
  // Halo pad on both spatial dims: 4 -> 6 (validity-impacting).
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - 1, d3 - 1)> by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <Pad{1, 1, 1, 1} ["0ipad", "1ipad"] at [2, 3] -> ["0i", "1i"] at [2, 3]>] bounds = [1, 1, 6, 6] -> [1, 1, 4, 4]> : tensor<1x1x4x4xi8> to tensor<1x1x6x6xi8>
  // Sliding windows: 0ipad = tap0 + 0o ; 1ipad = tap1 + 1o.
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d3, d4 + d5)> by [<PassThrough ["ni", "gi"] at [0, 1] -> ["ni", "gi"] at [0, 1]>, <Embed{1, 1} ["0", "0o"] at [2, 3] -> ["0ipad"] at [2]>, <Embed{1, 1} ["1", "1o"] at [4, 5] -> ["1ipad"] at [3]>] bounds = [1, 1, 3, 4, 3, 4] -> [1, 1, 6, 6]> : tensor<1x1x6x6xi8> to tensor<1x1x3x4x3x4xi8>

  %res = scf.for %k = %c0_i32 to %c9_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<1x4xi8>) : i32 {
    // Merge the two spatial taps into gemmK (radix 3, 3) -- no channel above
    // them; (ni, 0o, 1o) into gemmN.
    %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 3, d2 floordiv 4, d1 mod 3, d2 mod 4)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{3, 3} ["gemmK"] at [1] -> ["0", "1"] at [2, 4]>, <Merge{1, 4, 4} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 3, 5]>] bounds = [1, 9, 16] -> [1, 1, 3, 4, 3, 4]> : tensor<1x1x3x4x3x4xi8> to tensor<1x9x16xi8>
    // Tiling: k_loop (the iv) feeds gemmK with stride 1 (k_iter = 1), so it
    // advances the merged index by one per step; the odometer splits that into
    // tap1 wrapping and carrying into tap0.
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 + d4, d3 * 4 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{9, 1} ["k_loop", "k_iter"] at [0, 4] -> ["gemmK"] at [1]>, <Unmerge{4, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [9, 1, 1, 4, 1, 4] -> [1, 9, 16]> : tensor<1x9x16xi8> to tensor<9x1x1x4x1x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %4[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<9x1x1x4x1x4xi8> -> tensor<1x4xi32>, tensor<1x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<1x4xi32>, tensor<1x4xi1> -> tensor<1x4xi8>
    scf.yield %load : tensor<1x4xi8>
  }
  return %res : tensor<1x4xi8>
}

// -----

// Loop-variant, non-iv extra index on the AFFINE path. The 4th extra index
// (n_block) is a second loop iter_arg (%vb) that advances every iteration, so
// it is loop-variant but is NOT the induction variable. The affine rewrite
// handles this and does NOT bail: it re-expresses the base op in place, pinning
// the iv (slot 0) to the loop lower bound %c0_i32 while referencing the
// loop-variant %vb (slot 3) directly, and folds the iv contribution into the
// usual scalar affine tail. The base op stays loop-variant through %vb (so it
// is not hoistable), but the per-iteration iv work is still reduced to one
// scalar multiply + splat + add rather than being threaded through the tile
// index computation.
//
// CHECK-LABEL: func @affine_variant_non_iv_index
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<32768xf16>, %[[INIT:.*]]: tensor<64x64xf16>)
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[VB:.*]] = %{{.*}}) -> (tensor<64x64xf16>, i32)
// Base op: iv (slot 0) pinned to %c0_i32; the loop-variant %[[VB]] kept in slot 3:
//       CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %[[VB]]]
//       CHECK:     %[[D:.*]] = arith.subi %[[IV]], %{{.*}} : i32
//       CHECK:     %[[OFF:.*]] = arith.muli %[[D]], %{{.*}} : i32
//       CHECK:     %[[OFFT:.*]] = tt.splat %[[OFF]] : i32 -> tensor<64x64xi32>
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[OFFT]] : tensor<64x64xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//       CHECK:     scf.yield
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
func.func @affine_variant_non_iv_index(%arg0: tensor<32768xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = rock.transform %arg0 by #transform_map : tensor<32768xf16> to tensor<1x256x128xf16>
  // %vb is a second loop-carried index (advanced each iteration): loop-variant
  // but not the iv.
  %1:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %arg1, %vb = %c0_i32) -> (tensor<64x64xf16>, i32) : i32 {
    %2 = rock.transform %0 by #transform_map1 : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %2[%arg2, %c0_i32, %c0_i32, %vb] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %3 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    %vbn = arith.addi %vb, %c1_i32 : i32
    scf.yield %3, %vbn : tensor<64x64xf16>, i32
  }
  return %1#0 : tensor<64x64xf16>
}

// -----

// Loop-variant, non-iv extra index on the CARRY path. Same halo-padded conv
// input chain as @carry_conv_input (the iv straddles a non-contiguous
// Merge and the mask is iv-dependent, so the affine fast path cannot apply),
// but g_block (slot 1) is fed by a second loop iter_arg (%vb) that advances
// every iteration. The carry path rebuilds the base coordinate slice in the
// preheader via cloneSliceBeforeLoop, which only knows how to reconstruct the
// iv; a loop-variant non-iv index would trip its assertion. The pass therefore
// bails and leaves the loop untouched.
//
// CHECK-LABEL: func @carry_variant_non_iv_index
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi8>, %[[INIT:.*]]: tensor<2x4xi8>)
// The pass bails: the loop still carries only the original two iter_args (the
// result accumulator + the loop-variant index) with NO carry odometer state
// added -- the return type list is unchanged:
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[VB:.*]] = %{{.*}}) -> (tensor<2x4xi8>, i32)
// The transforms_to_ptr is left verbatim in the loop, still indexed by the iv
// %[[IV]] (slot 0, NOT pinned to %c0_i32) and the loop-variant %[[VB]] (slot 1):
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%[[IV]], %[[VB]], %c0_i32, %c0_i32]
//       CHECK:     rock.blockwise_load_ptr
// No mixed-radix carry machinery was emitted:
//   CHECK-NOT:     arith.cmpi uge
//   CHECK-NOT:     arith.select
//       CHECK:     scf.yield
#map = affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 - 1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 + d4)>
#map3 = affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 3, d1 mod 3, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 2 + d4, d3 * 4 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{2, 4} ["ci", "1i"] at [2, 3] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 2, 4] -> [8]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Pad{1, 1} ["1ipad"] at [3] -> ["1i"] at [3]>] bounds = [1, 1, 2, 6] -> [1, 1, 2, 4]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [3]>] bounds = [1, 1, 2, 3, 4] -> [1, 1, 2, 6]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{2, 3} ["gemmK"] at [1] -> ["ci", "1"] at [2, 3]>, <Merge{1, 4} ["gemmN"] at [2] -> ["ni", "1o"] at [0, 4]>] bounds = [1, 6, 4] -> [1, 1, 2, 3, 4]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{3, 2} ["k_loop", "k_iter"] at [0, 4] -> ["gemmK"] at [1]>, <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [3, 1, 1, 1, 2, 4] -> [1, 6, 4]>
func.func @carry_variant_non_iv_index(%arg0: tensor<8xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %0 = rock.transform %arg0 by #transform_map : tensor<8xi8> to tensor<1x1x2x4xi8>
  %1 = rock.transform %0 by #transform_map1 : tensor<1x1x2x4xi8> to tensor<1x1x2x6xi8>
  %2 = rock.transform %1 by #transform_map2 : tensor<1x1x2x6xi8> to tensor<1x1x2x3x4xi8>
  // %vb is a second loop-carried index (advanced each iteration): loop-variant
  // but not the iv. Fed into g_block (slot 1).
  %res:2 = scf.for %k = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%acc = %arg1, %vb = %c0_i32) -> (tensor<2x4xi8>, i32) : i32 {
    %3 = rock.transform %2 by #transform_map3 : tensor<1x1x2x3x4xi8> to tensor<1x6x4xi8>
    %4 = rock.transform %3 by #transform_map4 : tensor<1x6x4xi8> to tensor<3x1x1x1x2x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %4[%k, %vb, %c0_i32, %c0_i32] : tensor<3x1x1x1x2x4xi8> -> tensor<2x4xi32>, tensor<2x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<2x4xi32>, tensor<2x4xi1> -> tensor<2x4xi8>
    %vbn = arith.addi %vb, %c1_i32 : i32
    scf.yield %load, %vbn : tensor<2x4xi8>, i32
  }
  return %res#0 : tensor<2x4xi8>
}

// -----

// Negative case: the induction variable is plugged into more than one extra
// index of the transforms_to_ptr (slots 0 and 2 here). We dont support this
// case, so the transform must bail.
//
// CHECK-LABEL: func @iv_in_multiple_extra_indices
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi8>, %[[INIT:.*]]: tensor<2x4xi8>)
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<2x4xi8>)
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32, %[[IV]], %c0_i32]
//       CHECK:     rock.blockwise_load_ptr
// No mixed-radix carry machinery was emitted:
//   CHECK-NOT:     arith.cmpi uge
//   CHECK-NOT:     arith.select
//       CHECK:     scf.yield
func.func @iv_in_multiple_extra_indices(%arg0: tensor<8xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3)> by [<Unmerge{2, 4} ["ci", "1i"] at [2, 3] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 2, 4] -> [8]> : tensor<8xi8> to tensor<1x1x2x4xi8>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 - 1)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Pad{1, 1} ["1ipad"] at [3] -> ["1i"] at [3]>] bounds = [1, 1, 2, 6] -> [1, 1, 2, 4]> : tensor<1x1x2x4xi8> to tensor<1x1x2x6xi8>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 + d4)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [3]>] bounds = [1, 1, 2, 3, 4] -> [1, 1, 2, 6]> : tensor<1x1x2x6xi8> to tensor<1x1x2x3x4xi8>
  %res = scf.for %k = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<2x4xi8>) : i32 {
    %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 3, d1 mod 3, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{2, 3} ["gemmK"] at [1] -> ["ci", "1"] at [2, 3]>, <Merge{1, 4} ["gemmN"] at [2] -> ["ni", "1o"] at [0, 4]>] bounds = [1, 6, 4] -> [1, 1, 2, 3, 4]> : tensor<1x1x2x3x4xi8> to tensor<1x6x4xi8>
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 2 + d4, d3 * 4 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{3, 2} ["k_loop", "k_iter"] at [0, 4] -> ["gemmK"] at [1]>, <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [3, 1, 1, 1, 2, 4] -> [1, 6, 4]> : tensor<1x6x4xi8> to tensor<3x1x1x1x2x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %4[%k, %c0_i32, %k, %c0_i32] : tensor<3x1x1x1x2x4xi8> -> tensor<2x4xi32>, tensor<2x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<2x4xi32>, tensor<2x4xi1> -> tensor<2x4xi8>
    scf.yield %load : tensor<2x4xi8>
  }
  return %res : tensor<2x4xi8>
}

// -----

// CHECK-LABEL: func @skip_non_kernel_func
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<64x64xf16>)
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%[[IV]], %{{.*}}, %{{.*}}, %{{.*}}]
//   CHECK-NOT:     arith.muli
//       CHECK:     scf.yield
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
func.func @skip_non_kernel_func(%arg0: tensor<32768xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = rock.transform %arg0 by #transform_map : tensor<32768xf16> to tensor<1x256x128xf16>
  %1 = scf.for %arg2 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<64x64xf16>)  : i32 {
    %2 = rock.transform %0 by #transform_map1 : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %2[%arg2, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %3 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    scf.yield %3 : tensor<64x64xf16>
  }
  return %1 : tensor<64x64xf16>
}

// -----

// The pointer does not depend on the iv at all (every extra index is a
// constant): there is nothing to incrementalize, so the pass bails.
//
// CHECK-LABEL: func @no_simplify_iv_independent
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<64x64xf16>)
//       CHECK:     rock.transforms_to_ptr
//   CHECK-NOT:     arith.muli
//       CHECK:     scf.yield
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
func.func @no_simplify_iv_independent(%arg0: tensor<32768xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %0 = rock.transform %arg0 by #transform_map : tensor<32768xf16> to tensor<1x256x128xf16>
  %1 = scf.for %arg2 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg3 = %arg1) -> (tensor<64x64xf16>)  : i32 {
    %2 = rock.transform %0 by #transform_map1 : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    // All extra indices are constants (no iv): the load is loop-invariant.
    %pointers, %mask = rock.transforms_to_ptr %2[%c0_i32, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %3 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    scf.yield %3 : tensor<64x64xf16>
  }
  return %1 : tensor<64x64xf16>
}

// -----

// Carry path requires a compile-time-constant loop step to compute the
// per-iteration offset. Here the step is a runtime value, so
// analyzeCarryCandidate bails.
//
// CHECK-LABEL: func @carry_non_constant_step
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<2x4xi8>)
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32, %c0_i32, %c0_i32]
//   CHECK-NOT:     arith.cmpi uge
//       CHECK:     scf.yield
func.func @carry_non_constant_step(%arg0: tensor<8xi8>, %arg1: tensor<2x4xi8>, %step: i32) -> tensor<2x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c3_i32 = arith.constant 3 : i32
  // raw input buffer (ngc1): ci=2, 1i=4 -> 8.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3)> by [<Unmerge{2, 4} ["ci", "1i"] at [2, 3] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 2, 4] -> [8]> : tensor<8xi8> to tensor<1x1x2x4xi8>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3 - 1)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Pad{1, 1} ["1ipad"] at [3] -> ["1i"] at [3]>] bounds = [1, 1, 2, 6] -> [1, 1, 2, 4]> : tensor<1x1x2x4xi8> to tensor<1x1x2x6xi8>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 + d4)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [3]>] bounds = [1, 1, 2, 3, 4] -> [1, 1, 2, 6]> : tensor<1x1x2x6xi8> to tensor<1x1x2x3x4xi8>

  // Runtime (non-constant) loop step.
  %res = scf.for %k = %c0_i32 to %c3_i32 step %step iter_args(%acc = %arg1) -> (tensor<2x4xi8>) : i32 {
    %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 3, d1 mod 3, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{2, 3} ["gemmK"] at [1] -> ["ci", "1"] at [2, 3]>, <Merge{1, 4} ["gemmN"] at [2] -> ["ni", "1o"] at [0, 4]>] bounds = [1, 6, 4] -> [1, 1, 2, 3, 4]> : tensor<1x1x2x3x4xi8> to tensor<1x6x4xi8>
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 2 + d4, d3 * 4 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{3, 2} ["k_loop", "k_iter"] at [0, 4] -> ["gemmK"] at [1]>, <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [3, 1, 1, 1, 2, 4] -> [1, 6, 4]> : tensor<1x6x4xi8> to tensor<3x1x1x1x2x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %4[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<3x1x1x1x2x4xi8> -> tensor<2x4xi32>, tensor<2x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<2x4xi32>, tensor<2x4xi1> -> tensor<2x4xi8>
    scf.yield %load : tensor<2x4xi8>
  }
  return %res : tensor<2x4xi8>
}

// -----

// Carry eligibility guard: the validity-impacting coordinates must form a
// contiguous suffix with at most ONE dropped (non-impacting) prefix coordinate.
// Here gemmK is a Merge{2, 2, 3} of (c0, c1, tap): c0 and c1 are plain buffer
// splits (non-impacting) sitting above the single padded tap, so prefixSize ==
// 2 and the full-tile pointer recurrence does not apply. The pass should bail.
//
// CHECK-LABEL: func @carry_multi_prefix_not_simplified
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<2x4xi8>)
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32, %c0_i32, %c0_i32]
//   CHECK-NOT:     arith.cmpi uge
//       CHECK:     scf.yield
func.func @carry_multi_prefix_not_simplified(%arg0: tensor<16xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c6_i32 = arith.constant 6 : i32
  // raw input buffer (n g c0 c1 1i): c0=2, c1=2, 1i=4 -> raw = c0*8 + c1*4 + 1i, size 16.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 2 + d3) * 4 + d4)> by [<Unmerge{2, 2, 4} ["c0", "c1", "1i"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 2, 2, 4] -> [16]> : tensor<16xi8> to tensor<1x1x2x2x4xi8>
  // Halo pad on the width only: 4 -> 6 (validity-impacting).
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4 - 1)> by [<PassThrough ["ni", "gi", "c0", "c1"] at [0, 1, 2, 3] -> ["ni", "gi", "c0", "c1"] at [0, 1, 2, 3]>, <Pad{1, 1} ["1ipad"] at [4] -> ["1i"] at [4]>] bounds = [1, 1, 2, 2, 6] -> [1, 1, 2, 2, 4]> : tensor<1x1x2x2x4xi8> to tensor<1x1x2x2x6xi8>
  // Sliding window: 1ipad = tap + 1o.
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 + d5)> by [<PassThrough ["ni", "gi", "c0", "c1"] at [0, 1, 2, 3] -> ["ni", "gi", "c0", "c1"] at [0, 1, 2, 3]>, <Embed{1, 1} ["1", "1o"] at [4, 5] -> ["1ipad"] at [4]>] bounds = [1, 1, 2, 2, 3, 4] -> [1, 1, 2, 2, 6]> : tensor<1x1x2x2x6xi8> to tensor<1x1x2x2x3x4xi8>

  %res = scf.for %k = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<2x4xi8>) : i32 {
    // gemmK = Merge{2, 2, 3} of (c0, c1, tap); two non-impacting coords sit
    // above the single impacting tap -> prefixSize == 2.
    %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 6, (d1 floordiv 3) mod 2, d1 mod 3, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{2, 2, 3} ["gemmK"] at [1] -> ["c0", "c1", "1"] at [2, 3, 4]>, <Merge{1, 4} ["gemmN"] at [2] -> ["ni", "1o"] at [0, 5]>] bounds = [1, 12, 4] -> [1, 1, 2, 2, 3, 4]> : tensor<1x1x2x2x3x4xi8> to tensor<1x12x4xi8>
    // Tiling: k_loop (the iv) feeds gemmK with stride 2 (k_iter = 2).
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 2 + d4, d3 * 4 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{6, 2} ["k_loop", "k_iter"] at [0, 4] -> ["gemmK"] at [1]>, <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [6, 1, 1, 1, 2, 4] -> [1, 12, 4]> : tensor<1x12x4xi8> to tensor<6x1x1x1x2x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %4[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<6x1x1x1x2x4xi8> -> tensor<2x4xi32>, tensor<2x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<2x4xi32>, tensor<2x4xi1> -> tensor<2x4xi8>
    scf.yield %load : tensor<2x4xi8>
  }
  return %res : tensor<2x4xi8>
}

// -----

// Carry bail: a validity-impacting map sits AT/ABOVE the iv-traversed merge.
// The mask is reconstructed only from the sub-chain below the merge, so a Pad
// on gemmK (above the merge, where the iv flows) cannot be handled: the pass
// bails.
//
// CHECK-LABEL: func @carry_pad_above_merge
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<2x4xi8>)
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32, %c0_i32, %c0_i32]
//   CHECK-NOT:     arith.cmpi uge
//       CHECK:     scf.yield
func.func @carry_pad_above_merge(%arg0: tensor<8xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  // raw input buffer (ngc1): ci=2, 1i=4 -> 8.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d2 * 4 + d3)> by [<Unmerge{2, 4} ["ci", "1i"] at [2, 3] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 2, 4] -> [8]> : tensor<8xi8> to tensor<1x1x2x4xi8>
  // Sliding window on the (unpadded) width: 1i = tap + 1o (1i = 4, tap = 1, 1o = 4).
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 + d4)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1i"] at [3]>] bounds = [1, 1, 2, 1, 4] -> [1, 1, 2, 4]> : tensor<1x1x2x4xi8> to tensor<1x1x2x1x4xi8>

  %res = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<2x4xi8>) : i32 {
    // gemmK = Merge{2, 1} of (ci, tap); gemmN = Merge{1, 4} of (ni, 1o).
    %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, 0, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{2, 1} ["gemmK"] at [1] -> ["ci", "1"] at [2, 3]>, <Merge{1, 4} ["gemmN"] at [2] -> ["ni", "1o"] at [0, 4]>] bounds = [1, 2, 4] -> [1, 1, 2, 1, 4]> : tensor<1x1x2x1x4xi8> to tensor<1x2x4xi8>
    // Pad gemmK 2 -> 4 ABOVE the merge (validity-impacting, on the iv dim).
    %pk = rock.transform %3 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 2} ["gemmKpad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 4, 4] -> [1, 2, 4]> : tensor<1x2x4xi8> to tensor<1x4x4xi8>
    // Tiling: k_loop (the iv) feeds gemmKpad with stride 2 (k_iter = 2).
    %4 = rock.transform %pk by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 2 + d4, d3 * 4 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{2, 2} ["k_loop", "k_iter"] at [0, 4] -> ["gemmKpad"] at [1]>, <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [2, 1, 1, 1, 2, 4] -> [1, 4, 4]> : tensor<1x4x4xi8> to tensor<2x1x1x1x2x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %4[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<2x1x1x1x2x4xi8> -> tensor<2x4xi32>, tensor<2x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<2x4xi32>, tensor<2x4xi1> -> tensor<2x4xi8>
    scf.yield %load : tensor<2x4xi8>
  }
  return %res : tensor<2x4xi8>
}

// -----

// A nested Merge sits below the iv-traversed merge. gemmK is
// Merge{4, 3} of (ci, tap) and the iv advances gemmK by 2, which delinearizes
// to ci += 0 / tap += 2, so the nested Merge{2, 2} that splits ci into
// (c_hi, c_lo). We bail on such a case.
//
// CHECK-LABEL: func @carry_nested_merge_below_iv_merge
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<16xi8>, %[[INIT:.*]]: tensor<2x4xi8>)
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]]) -> (tensor<2x4xi8>)
//       CHECK:     rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32, %c0_i32, %c0_i32]
//       CHECK:     rock.blockwise_load_ptr
//   CHECK-NOT:     arith.cmpi uge
//   CHECK-NOT:     arith.select
//       CHECK:     scf.yield
func.func @carry_nested_merge_below_iv_merge(%arg0: tensor<16xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x4xi8> attributes {rock.kernel, rock.conv_kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c6_i32 = arith.constant 6 : i32

  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 2 + d3) * 4 + d4)> by [<Unmerge{2, 2, 4} ["c_hi", "c_lo", "1i"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 2, 2, 4] -> [16]> : tensor<16xi8> to tensor<1x1x2x2x4xi8>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4 - 1)> by [<PassThrough ["ni", "gi", "c_hi", "c_lo"] at [0, 1, 2, 3] -> ["ni", "gi", "c_hi", "c_lo"] at [0, 1, 2, 3]>, <Pad{1, 1} ["1ipad"] at [4] -> ["1i"] at [4]>] bounds = [1, 1, 2, 2, 6] -> [1, 1, 2, 2, 4]> : tensor<1x1x2x2x4xi8> to tensor<1x1x2x2x6xi8>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 + d5)> by [<PassThrough ["ni", "gi", "c_hi", "c_lo"] at [0, 1, 2, 3] -> ["ni", "gi", "c_hi", "c_lo"] at [0, 1, 2, 3]>, <Embed{1, 1} ["1", "1o"] at [4, 5] -> ["1ipad"] at [4]>] bounds = [1, 1, 2, 2, 3, 4] -> [1, 1, 2, 2, 6]> : tensor<1x1x2x2x6xi8> to tensor<1x1x2x2x3x4xi8>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 floordiv 2, d2 mod 2, d3, d4)> by [<PassThrough ["ni", "gi"] at [0, 1] -> ["ni", "gi"] at [0, 1]>, <Merge{2, 2} ["ci"] at [2] -> ["c_hi", "c_lo"] at [2, 3]>, <PassThrough ["1", "1o"] at [3, 4] -> ["1", "1o"] at [4, 5]>] bounds = [1, 1, 4, 3, 4] -> [1, 1, 2, 2, 3, 4]> : tensor<1x1x2x2x3x4xi8> to tensor<1x1x4x3x4xi8>

  %res = scf.for %k = %c0_i32 to %c6_i32 step %c1_i32 iter_args(%acc = %arg1) -> (tensor<2x4xi8>) : i32 {
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 3, d1 mod 3, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{4, 3} ["gemmK"] at [1] -> ["ci", "1"] at [2, 3]>, <Merge{1, 4} ["gemmN"] at [2] -> ["ni", "1o"] at [0, 4]>] bounds = [1, 12, 4] -> [1, 1, 4, 3, 4]> : tensor<1x1x4x3x4xi8> to tensor<1x12x4xi8>
    %5 = rock.transform %4 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 2 + d4, d3 * 4 + d5)> by [<PassThrough ["g_block"] at [1] -> ["gemmG"] at [0]>, <Unmerge{6, 2} ["k_loop", "k_iter"] at [0, 4] -> ["gemmK"] at [1]>, <Unmerge{1, 4} ["n_block", "n_iter"] at [3, 5] -> ["gemmN"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [6, 1, 1, 1, 2, 4] -> [1, 12, 4]> : tensor<1x12x4xi8> to tensor<6x1x1x1x2x4xi8>
    %ptr, %mask = rock.transforms_to_ptr %5[%k, %c0_i32, %c0_i32, %c0_i32] : tensor<6x1x1x1x2x4xi8> -> tensor<2x4xi32>, tensor<2x4xi1>
    %load = rock.blockwise_load_ptr %ptr[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<2x4xi32>, tensor<2x4xi1> -> tensor<2x4xi8>
    scf.yield %load : tensor<2x4xi8>
  }
  return %res : tensor<2x4xi8>
}
