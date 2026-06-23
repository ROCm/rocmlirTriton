// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -rock-transforms-invariant-code-motion --split-input-file --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @hoist_linear_load
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<32768xf16>, %[[INIT:.*]]: tensor<64x64xf16>)
//       CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c1_i32]
//       CHECK:   %[[STRIDE:.*]] = arith.muli
//       CHECK:   %[[STRIDET:.*]] = tt.splat %[[STRIDE]] : i32 -> tensor<64x64xi32>
//       CHECK:   arith.constant dense<0> : tensor<64x64xi32>
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %{{.*}}) ->
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[ACC]] : tensor<64x64xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//   CHECK-NOT:     rock.transforms_to_ptr
//       CHECK:     %[[INC:.*]] = arith.addi %[[ACC]], %[[STRIDET]] : tensor<64x64xi32>
//       CHECK:     scf.yield %{{.*}}, %[[INC]]
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
func.func @hoist_linear_load(%arg0: tensor<32768xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.arch = "gfx1201"} {
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
// CHECK-LABEL: func @hoist_index_iv
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<32768xf16>, %[[INIT:.*]]: tensor<64x64xf16>)
//       CHECK:   rock.transforms_to_ptr %{{.*}}[%{{.*}}, %c0_i32, %c0_i32, %c1_i32]
//       CHECK:   %[[STEPC:.*]] = arith.index_cast %{{.*}} : index to i32
//       CHECK:   %[[STRIDE:.*]] = arith.muli %[[STEPC]], %{{.*}} : i32
//       CHECK:   %[[STRIDET:.*]] = tt.splat %[[STRIDE]] : i32 -> tensor<64x64xi32>
//       CHECK:   arith.constant dense<0> : tensor<64x64xi32>
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %{{.*}}) ->
//       CHECK:     %[[PTR:.*]] = arith.addi %{{.*}}, %[[ACC]] : tensor<64x64xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%{{.*}}]
//   CHECK-NOT:     rock.transforms_to_ptr
//       CHECK:     %[[INC:.*]] = arith.addi %[[ACC]], %[[STRIDET]] : tensor<64x64xi32>
//       CHECK:     scf.yield %{{.*}}, %[[INC]]
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
func.func @hoist_index_iv(%arg0: tensor<32768xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.arch = "gfx1201"} {
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
// must conservatively leave the loop untouched: nothing is hoisted before the
// loop and the transforms_to_ptr stays in the body.
//
// CHECK-LABEL: func @no_hoist_with_pad
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
func.func @no_hoist_with_pad(%arg0: tensor<32512xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.arch = "gfx1201"} {
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
//     it IS hoisted (base pointer in the preheader, offset accumulator carried),
//   - the input chain has a Pad on K, so it is NOT hoisted and its
//     transforms_to_ptr stays inside the loop, still indexed by the iv.
//
// CHECK-LABEL: func @hoist_one_of_two
//  CHECK-SAME: (%[[FILTER:.*]]: tensor<32768xf16>, %[[INPUT:.*]]: tensor<32512xf16>, %[[INIT:.*]]: tensor<64x64xf16>)
// Filter operand hoisted (iv replaced by the lower bound %c0_i32):
//       CHECK:   %[[FPTRS:.*]], %[[FMASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c1_i32]
//       CHECK:   %[[STRIDE:.*]] = arith.muli
//       CHECK:   %[[STRIDET:.*]] = tt.splat %[[STRIDE]] : i32 -> tensor<64x64xi32>
//       CHECK:   arith.constant dense<0> : tensor<64x64xi32>
//       CHECK:   scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %{{.*}}) ->
// Filter pointer reconstructed from the hoisted base + accumulator:
//       CHECK:     %[[FPTR:.*]] = arith.addi %[[FPTRS]], %[[ACC]] : tensor<64x64xi32>
//       CHECK:     rock.blockwise_load_ptr %[[FPTR]][%[[FMASK]]]
// Input operand NOT hoisted: its transforms_to_ptr stays in the loop, still
// indexed by the induction variable %[[IV]]:
//       CHECK:     %[[IPTRS:.*]], %[[IMASK:.*]] = rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32, %c0_i32, %c1_i32]
//       CHECK:     rock.blockwise_load_ptr %[[IPTRS]][%[[IMASK]]]
// Accumulator advances by the stride and is yielded alongside the result:
//       CHECK:     %[[INC:.*]] = arith.addi %[[ACC]], %[[STRIDET]] : tensor<64x64xi32>
//       CHECK:     scf.yield %{{.*}}, %[[INC]]
func.func @hoist_one_of_two(%filter: tensor<32768xf16>, %input: tensor<32512xf16>, %init: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.arch = "gfx1201"} {
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

    // Input Pad (K 254 -> 256) makes the mask coordinate-dependent: not hoisted.
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
// vary with the iv, so the op IS hoisted even though the chain contains a Pad.
//
// CHECK-LABEL: func @hoist_pad_on_non_iv_dim
//  CHECK-SAME: (%[[FILTER:.*]]: tensor<36864xi8>, %[[INIT:.*]]: tensor<256x32xi8>)
// Hoisted to the preheader (iv -> lower bound %c0_i32):
//       CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
//       CHECK:   %[[STRIDE:.*]] = arith.muli
//       CHECK:   %[[STRIDET:.*]] = tt.splat %[[STRIDE]] : i32 -> tensor<256x32xi32>
//       CHECK:   arith.constant dense<0> : tensor<256x32xi32>
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %{{.*}}) ->
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[ACC]] : tensor<256x32xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//   CHECK-NOT:     rock.transforms_to_ptr
//       CHECK:     %[[INC:.*]] = arith.addi %[[ACC]], %[[STRIDET]] : tensor<256x32xi32>
//       CHECK:     scf.yield %{{.*}}, %[[INC]]
func.func @hoist_pad_on_non_iv_dim(%filter: tensor<36864xi8>, %init: tensor<256x32xi8>) -> tensor<256x32xi8> attributes {rock.kernel, rock.arch = "gfx1201"} {
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
// strides 9/3/1 = 9*3 + 3*1 + 1*2 = 32) and the op IS hoisted.
//
// CHECK-LABEL: func @hoist_conv_filter_merge
//  CHECK-SAME: (%[[FILTER:.*]]: tensor<36864xi8>, %[[INIT:.*]]: tensor<256x32xi8>)
//       CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
//       CHECK:   %[[STRIDE:.*]] = arith.muli
//       CHECK:   %[[STRIDET:.*]] = tt.splat %[[STRIDE]] : i32 -> tensor<256x32xi32>
//       CHECK:   arith.constant dense<0> : tensor<256x32xi32>
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %{{.*}}) ->
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[ACC]] : tensor<256x32xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//   CHECK-NOT:     rock.transforms_to_ptr
//       CHECK:     %[[INC:.*]] = arith.addi %[[ACC]], %[[STRIDET]] : tensor<256x32xi32>
//       CHECK:     scf.yield %{{.*}}, %[[INC]]
func.func @hoist_conv_filter_merge(%filter: tensor<36864xi8>, %init: tensor<256x32xi8>) -> tensor<256x32xi8> attributes {rock.kernel, rock.arch = "gfx1201"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c18_i32 = arith.constant 18 : i32

  // raw buffer (gkc01): k_out=64, c=64, 0=3, 1=3 -> 36864.
  %0 = rock.transform %filter by <affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 64 + d2) * 3 + d3) * 3 + d4)> by [<Unmerge{64, 64, 3, 3} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64, 3, 3] -> [36864]> : tensor<36864xi8> to tensor<1x64x64x3x3xi8>
  // Merge (c, 0, 1) into gemmK (floordiv/mod), gemmM = k_out.
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 mod 9) floordiv 3, d1 mod 3)> by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{64, 3, 3} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 576, 64] -> [1, 64, 64, 3, 3]> : tensor<1x64x64x3x3xi8> to tensor<1x576x64xi8>
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

// Two hoistable loads in one loop that point to the same root buffer
// (%arg0) but view it through differently-shaped tiles (64x64 and 64x128).
//
// CHECK-LABEL: func @hoist_two_sharing_base
//  CHECK-SAME: (%[[BUF:.*]]: tensor<32768xf16>, %[[INITA:.*]]: tensor<64x64xf16>, %[[INITB:.*]]: tensor<64x128xf16>)
//       CHECK:   %[[PTRS0:.*]], %[[MASK0:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c1_i32] {{.*}} -> tensor<64x64xi32>
//       CHECK:   arith.muli
//       CHECK:   %[[STRIDET0:.*]] = tt.splat %{{.*}} : i32 -> tensor<64x64xi32>
//       CHECK:   arith.constant dense<0> : tensor<64x64xi32>
//       CHECK:   %[[PTRS1:.*]], %[[MASK1:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32] {{.*}} -> tensor<64x128xi32>
//       CHECK:   arith.muli
//       CHECK:   %[[STRIDET1:.*]] = tt.splat %{{.*}} : i32 -> tensor<64x128xi32>
//       CHECK:   arith.constant dense<0> : tensor<64x128xi32>
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INITA]], %{{.*}} = %[[INITB]], %[[ACC0:.*]] = %{{.*}}, %[[ACC1:.*]] = %{{.*}}) ->
//       CHECK:     %[[P0:.*]] = arith.addi %[[PTRS0]], %[[ACC0]] : tensor<64x64xi32>
//       CHECK:     %[[P1:.*]] = arith.addi %[[PTRS1]], %[[ACC1]] : tensor<64x128xi32>
//       CHECK:     rock.blockwise_load_ptr %[[P0]][%[[MASK0]]]
//       CHECK:     rock.blockwise_load_ptr %[[P1]][%[[MASK1]]]
//   CHECK-NOT:     rock.transforms_to_ptr
//       CHECK:     %[[INC0:.*]] = arith.addi %[[ACC0]], %[[STRIDET0]] : tensor<64x64xi32>
//       CHECK:     %[[INC1:.*]] = arith.addi %[[ACC1]], %[[STRIDET1]] : tensor<64x128xi32>
//       CHECK:     scf.yield %{{.*}}, %{{.*}}, %[[INC0]], %[[INC1]]
#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 128 + d5)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{1, 128} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 1, 64, 128] -> [1, 256, 128]>
func.func @hoist_two_sharing_base(%arg0: tensor<32768xf16>, %initA: tensor<64x64xf16>, %initB: tensor<64x128xf16>) -> (tensor<64x64xf16>, tensor<64x128xf16>) attributes {rock.kernel, rock.arch = "gfx1201"} {
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
// stride 32 and the op IS hoisted. The size-1 dims must be skipped by the
// merge carry-neutrality guard: their stride-0 layout would otherwise spuriously
// fail the contiguity check (stride(c) 1 != stride("0") 0) and bail.
//
// CHECK-LABEL: func @hoist_conv_1x1_unit_merge
//  CHECK-SAME: (%[[FILTER:.*]]: tensor<4096xi8>, %[[INIT:.*]]: tensor<64x32xi8>)
//       CHECK:   %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
//       CHECK:   %[[STRIDE:.*]] = arith.muli
//       CHECK:   %[[STRIDET:.*]] = tt.splat %[[STRIDE]] : i32 -> tensor<64x32xi32>
//       CHECK:   arith.constant dense<0> : tensor<64x32xi32>
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[ACC:.*]] = %{{.*}}) ->
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %[[ACC]] : tensor<64x32xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]][%[[MASK]]]
//   CHECK-NOT:     rock.transforms_to_ptr
//       CHECK:     %[[INC:.*]] = arith.addi %[[ACC]], %[[STRIDET]] : tensor<64x32xi32>
//       CHECK:     scf.yield %{{.*}}, %[[INC]]
func.func @hoist_conv_1x1_unit_merge(%filter: tensor<4096xi8>, %init: tensor<64x32xi8>) -> tensor<64x32xi8> attributes {rock.kernel, rock.arch = "gfx1201"} {
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
// non-linear in the iv; the tap also flows through an Embed sliding window into
// a padded width, so the validity mask depends on the iv too. Rather than leave
// the op in the loop, the pass carries the decomposed merge coordinates
// (channel, tap) as extra iter_args and rebuilds the offset + mask each
// iteration with add/mul/compare/select and a mixed-radix carry -- no
// floordiv/mod in the loop (cf. rocMLIR's IndexDiffUpdate).
//
// CHECK-LABEL: func @hoist_conv_input_carry
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi8>, %[[INIT:.*]]: tensor<2x4xi8>)
// Hoisted to the preheader with the iv replaced by the loop lower bound:
//       CHECK:   %[[PTRS:.*]], %{{.*}} = rock.transforms_to_ptr %{{.*}}[%c0_i32, %c0_i32, %c0_i32, %c0_i32]
// The loop carries the decomposed merge coordinates (channel + tap) as extra
// iter_args alongside the result:
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %[[INIT]], %[[CI:.*]] = %{{.*}}, %[[TAP:.*]] = %{{.*}}) ->
// The offset/mask reconstruction uses no division and the op is gone from the
// loop body:
//   CHECK-NOT:     floordiv
//   CHECK-NOT:     arith.divsi
//   CHECK-NOT:     arith.remsi
//   CHECK-NOT:     rock.transforms_to_ptr
// Mask rebuilt from the carried tap coordinate, pointer rebuilt from the
// hoisted base + reconstructed offset, then loaded:
//       CHECK:     arith.cmpi ult
//       CHECK:     %[[PTR:.*]] = arith.addi %[[PTRS]], %{{.*}} : tensor<2x4xi32>
//       CHECK:     rock.blockwise_load_ptr %[[PTR]]
// Mixed-radix carry update (compare + select) advances the coordinates:
//       CHECK:     arith.cmpi uge
//       CHECK:     arith.select
//       CHECK:     scf.yield
func.func @hoist_conv_input_carry(%arg0: tensor<8xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x4xi8> attributes {rock.kernel, rock.arch = "gfx1201"} {
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
