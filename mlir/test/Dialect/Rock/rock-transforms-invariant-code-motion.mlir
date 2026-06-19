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
func.func @hoist_linear_load(%arg0: tensor<32768xf16>, %init: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32

  // Loop-invariant root view of the buffer.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : tensor<32768xf16> to tensor<1x256x128xf16>

  %res = scf.for %k = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<64x64xf16>) : i32 {
    // Tiling view: k_loop (extra index 0) is the iv; offset is linear in it.
    %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]> : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %1[%k, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %2 = rock.blockwise_load_ptr %pointers[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    scf.yield %2 : tensor<64x64xf16>
  }

  return %res : tensor<64x64xf16>
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
func.func @no_hoist_with_pad(%arg0: tensor<32512xf16>, %init: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32

  // Root view: K dimension is only 254 here.
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{254, 128} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 254, 128] -> [32512]> : tensor<32512xf16> to tensor<1x254x128xf16>

  %res = scf.for %k = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %init) -> (tensor<64x64xf16>) : i32 {
    // Pad K from 254 to 256: this makes the mask coordinate-dependent.
    %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Pad{0, 2} ["kpad"] at [1] -> ["k"] at [1]>, <PassThrough ["n"] at [2] -> ["n"] at [2]>] bounds = [1, 256, 128] -> [1, 254, 128]> : tensor<1x254x128xf16> to tensor<1x256x128xf16>
    %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d4, d3 * 64 + d5)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 4] -> ["k"] at [1]>, <Unmerge{2, 64} ["n_block", "n_iter"] at [3, 5] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 1, 2, 64, 64] -> [1, 256, 128]> : tensor<1x256x128xf16> to tensor<4x1x1x2x64x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %2[%k, %c0_i32, %c0_i32, %c1_i32] : tensor<4x1x1x2x64x64xf16> -> tensor<64x64xi32>, tensor<64x64xi1>
    %3 = rock.blockwise_load_ptr %pointers[%mask] : tensor<64x64xi32>, tensor<64x64xi1> -> tensor<64x64xf16>
    scf.yield %3 : tensor<64x64xf16>
  }

  return %res : tensor<64x64xf16>
}
