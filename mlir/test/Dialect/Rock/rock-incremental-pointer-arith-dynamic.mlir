// RUN: rocmlir-opt -rock-incremental-pointer-arith --split-input-file %s | FileCheck %s

// Transform chains with dynamic sizes are not incrementalized: the pointers
// keep depending on the iv and are left for the regular lowering.

// The offset advances by the runtime row length times 32 per iteration.
// CHECK-LABEL: func.func @dynamic_affine_bails
//      CHECK:   scf.for %[[IV:arg[0-9]+]] =
//      CHECK:     %[[PTRS:.*]], %[[MASK:.*]] = rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32,
//  CHECK-NOT:     arith.muli
//      CHECK:     rock.blockwise_load_ptr %[[PTRS]][%[[MASK]]]
func.func @dynamic_affine_bails(%arg1: tensor<1x64x?xf16>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel, rock.conv_kernel} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf16>
  %1 = tt.get_program_id x : i32
  %2 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<32x64xf16>)  : i32 {
    %3 = rock.transform %arg1 by <affine_map<(d0, d1, d2)[s0] -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK"] at [1] -> ["gemmK"] at [1]>, <Pad{0, -s0 + (s0 ceildiv 64) * 64} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] symbols = [arg(0, 2)] bounds = [1, 64, (s0 ceildiv 64) * 64] -> [1, 64, s0]> : tensor<1x64x?xf16> to tensor<1x64x?xf16>
    %4 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4)[s0] -> (d1, d0 * 32 + d3, d2 * 64 + d4)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{2, 32} ["k_loop", "k_iter"] at [0, 3] -> ["k"] at [1]>, <Unmerge{s0 ceildiv 64, 64} ["n_block", "n_iter"] at [2, 4] -> ["n"] at [2]>] symbols = [arg(0, 2)] bounds = [2, 1, s0 ceildiv 64, 32, 64] -> [1, 64, (s0 ceildiv 64) * 64]> : tensor<1x64x?xf16> to tensor<2x1x?x32x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %4[%arg3, %c0_i32, %1] : tensor<2x1x?x32x64xf16> -> tensor<32x64xi64>, tensor<32x64xi1>
    %5 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x64xi64>, tensor<32x64xi1> -> tensor<32x64xf16>
    %6 = arith.addf %arg4, %5 : tensor<32x64xf16>
    scf.yield %6 : tensor<32x64xf16>
  }
  return
}

// -----

// Padding along the loop dimension makes the mask depend on the iv.
// CHECK-LABEL: func.func @dynamic_carry_bails
//      CHECK:   scf.for %[[IV:arg[0-9]+]] =
//      CHECK:     rock.transforms_to_ptr %{{.*}}[%[[IV]], %c0_i32]
//  CHECK-NOT:     arith.muli
//      CHECK:     rock.blockwise_load_ptr
func.func @dynamic_carry_bails(%arg0: tensor<?x64xf16>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel, rock.conv_kernel} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf16>
  %0 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<32x64xf16>)  : i32 {
    %1 = rock.transform %arg0 by <affine_map<(d0, d1)[s0] -> (d0, d1)> by [<Pad{0, -s0 + (s0 ceildiv 32) * 32} ["kPad"] at [0] -> ["k"] at [0]>, <PassThrough ["n"] at [1] -> ["n"] at [1]>] symbols = [arg(0, 0)] bounds = [(s0 ceildiv 32) * 32, 64] -> [s0, 64]> : tensor<?x64xf16> to tensor<?x64xf16>
    %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 32 + d2, d1 * 64 + d3)> by [<Unmerge{s0 ceildiv 32, 32} ["k_loop", "k_iter"] at [0, 2] -> ["kPad"] at [0]>, <Unmerge{1, 64} ["n_block", "n_iter"] at [1, 3] -> ["n"] at [1]>] symbols = [arg(0, 0)] bounds = [s0 ceildiv 32, 1, 32, 64] -> [(s0 ceildiv 32) * 32, 64]> : tensor<?x64xf16> to tensor<?x1x32x64xf16>
    %pointers, %mask = rock.transforms_to_ptr %2[%arg3, %c0_i32] : tensor<?x1x32x64xf16> -> tensor<32x64xi64>, tensor<32x64xi1>
    %3 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<32x64xi64>, tensor<32x64xi1> -> tensor<32x64xf16>
    %4 = arith.addf %arg4, %3 : tensor<32x64xf16>
    scf.yield %4 : tensor<32x64xf16>
  }
  return
}
