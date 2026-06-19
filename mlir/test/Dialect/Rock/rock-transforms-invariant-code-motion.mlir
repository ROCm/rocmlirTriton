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
    %3 = rock.blockwise_load_ptr %pointers[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
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
    %4 = rock.blockwise_load_ptr %pointers[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
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
    %fload = rock.blockwise_load_ptr %fptr[%fmask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

    // Input Pad (K 254 -> 256) makes the mask coordinate-dependent: not hoisted.
    %i1 = rock.transform %i0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Pad{0, 2} ["kpad"] at [1] -> ["k"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>] bounds = [1, 256, 128] -> [1, 254, 128]> : tensor<1x254x128xf16> to tensor<1x256x128xf16>
    %i2 = rock.transform %i1 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]> : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %iptr, %imask = rock.transforms_to_ptr %i2[%k, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %iload = rock.blockwise_load_ptr %iptr[%imask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>

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
    %fload = rock.blockwise_load_ptr %fptr[%fmask] : tensor<256x32xi32>, tensor<256x32xi1> -> tensor<256x32xi8>
    scf.yield %fload : tensor<256x32xi8>
  }

  return %res : tensor<256x32xi8>
}

// -----

// Faithful 3x3-conv filter chain (matches the real kernel). Here gemmK is
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
    %fload = rock.blockwise_load_ptr %fptr[%fmask] : tensor<256x32xi32>, tensor<256x32xi1> -> tensor<256x32xi8>
    scf.yield %fload : tensor<256x32xi8>
  }

  return %res : tensor<256x32xi8>
}
