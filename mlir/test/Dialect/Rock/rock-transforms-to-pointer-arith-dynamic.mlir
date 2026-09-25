// RUN: rocmlir-opt -rock-transforms-to-pointer-arith --split-input-file %s | FileCheck %s

// A dynamic 2-D root is flattened row-major with its runtime row length, the
// padded dimensions are masked against the runtime sizes, and the offsets are
// i64.
// CHECK-LABEL: func.func @dynamic_load
// CHECK-SAME: (%[[ARG0:arg[0-9]+]]: tensor<?x?xf16>)
//      CHECK:   %[[BASE:.*]] = rock.extract_ptr %[[ARG0]] : tensor<?x?xf16> -> i64
//      CHECK:   %[[M:.*]] = tensor.dim %[[ARG0]], %c0
//      CHECK:   %[[MI:.*]] = arith.index_cast %[[M]] : index to i32
//      CHECK:   %[[N:.*]] = tensor.dim %[[ARG0]], %c1
//      CHECK:   %[[NI:.*]] = arith.index_cast %[[N]] : index to i32
//      CHECK:   arith.extui %[[MI]] : i32 to i64
//      CHECK:   %[[N64:.*]] = arith.extui %[[NI]] : i32 to i64
//      CHECK:   %[[MB:.*]] = arith.extui %[[MI]] : i32 to i64
//      CHECK:   tt.splat %[[MB]] : i64 -> tensor<64x1xi64>
//      CHECK:   arith.cmpi ult, {{.*}} : tensor<64x1xi64>
//      CHECK:   %[[NB:.*]] = arith.extui %[[NI]] : i32 to i64
//      CHECK:   tt.splat %[[NB]] : i64 -> tensor<1x64xi64>
//      CHECK:   arith.cmpi ult, {{.*}} : tensor<1x64xi64>
//      CHECK:   %[[STRIDE:.*]] = tt.splat %[[N64]] : i64 -> tensor<64x1xi64>
//      CHECK:   arith.muli %{{.*}}, %[[STRIDE]] overflow<nsw> : tensor<64x1xi64>
//      CHECK:   tt.splat %[[BASE]] : i64 -> tensor<64x64xi64>
//      CHECK:   rock.blockwise_load_ptr {{.*}} : tensor<64x64xi64>, tensor<64x64xi1> -> tensor<64x64xf16>
//  CHECK-NOT:   rock.transforms_to_ptr
func.func @dynamic_load(%arg0: tensor<?x?xf16>) -> tensor<64x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1)[s0, s1] -> (d0, d1)> by [<Pad{0, -s0 + (s0 ceildiv 64) * 64} ["mPad"] at [0] -> ["m"] at [0]>, <Pad{0, -s1 + (s1 ceildiv 64) * 64} ["nPad"] at [1] -> ["n"] at [1]>] symbols = [arg(0, 0), arg(0, 1)] bounds = [(s0 ceildiv 64) * 64, (s1 ceildiv 64) * 64] -> [s0, s1]> : tensor<?x?xf16> to tensor<?x?xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3)[s0, s1] -> (d0 * 64 + d2, d1 * 64 + d3)> by [<Unmerge{s0 ceildiv 64, 64} ["m_block", "m_iter"] at [0, 2] -> ["mPad"] at [0]>, <Unmerge{s1 ceildiv 64, 64} ["n_block", "n_iter"] at [1, 3] -> ["nPad"] at [1]>] symbols = [arg(0, 0), arg(0, 1)] bounds = [s0 ceildiv 64, s1 ceildiv 64, 64, 64] -> [(s0 ceildiv 64) * 64, (s1 ceildiv 64) * 64]> : tensor<?x?xf16> to tensor<?x?x64x64xf16>
  %pointers, %mask = rock.transforms_to_ptr %1[%c0_i32, %c1_i32] : tensor<?x?x64x64xf16> -> tensor<64x64xi64>, tensor<64x64xi1>
  %2 = rock.blockwise_load_ptr %pointers[%mask] {cacheModifier = #rock<CacheModifier none>} : tensor<64x64xi64>, tensor<64x64xi1> -> tensor<64x64xf16>
  return %2 : tensor<64x64xf16>
}

// -----

// Dimensions assumed equal are read once: the mask of the store uses the M
// of the first argument, which the assume ties to the M of the output.
// CHECK-LABEL: func.func @dynamic_store_equal_dims
// CHECK-SAME: (%[[A:arg[0-9]+]]: tensor<?x64xf16>, %[[C:arg[0-9]+]]: tensor<?x64xf16>, %{{.*}}: tensor<64x64xf16>)
//      CHECK:   %[[MA:.*]] = tensor.dim %[[A]], %c0
//      CHECK:   %[[MAI:.*]] = arith.index_cast %[[MA]] : index to i32
//      CHECK:   tensor.dim %[[C]], %c0
//      CHECK:   llvm.intr.assume
//  CHECK-NOT:   tensor.dim
//      CHECK:   arith.extui %[[MAI]] : i32 to i64
//      CHECK:   %[[BOUND:.*]] = arith.extui %[[MAI]] : i32 to i64
//      CHECK:   tt.splat %[[BOUND]] : i64 -> tensor<64x1xi64>
//      CHECK:   arith.cmpi ult
//      CHECK:   rock.blockwise_store_ptr
func.func @dynamic_store_equal_dims(%arg0: tensor<?x64xf16>, %arg1: tensor<?x64xf16>, %arg2: tensor<64x64xf16>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %dimA = tensor.dim %arg0, %c0 : tensor<?x64xf16>
  %mA = arith.index_cast %dimA : index to i32
  %dimC = tensor.dim %arg1, %c0 : tensor<?x64xf16>
  %mC = arith.index_cast %dimC : index to i32
  %eq = arith.cmpi eq, %mA, %mC : i32
  llvm.intr.assume %eq : i1
  %0 = rock.transform %arg1 by <affine_map<(d0, d1)[s0] -> (d0, d1)> by [<Pad{0, -s0 + (s0 ceildiv 64) * 64} ["mPad"] at [0] -> ["m"] at [0]>, <PassThrough ["n"] at [1] -> ["n"] at [1]>] symbols = [arg(1, 0)] bounds = [(s0 ceildiv 64) * 64, 64] -> [s0, 64]> : tensor<?x64xf16> to tensor<?x64xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1, d2)> by [<Unmerge{s0 ceildiv 64, 64} ["m_block", "m_iter"] at [0, 1] -> ["mPad"] at [0]>, <PassThrough ["n"] at [2] -> ["n"] at [1]>] symbols = [arg(0, 0)] bounds = [s0 ceildiv 64, 64, 64] -> [(s0 ceildiv 64) * 64, 64]> : tensor<?x64xf16> to tensor<?x64x64xf16>
  %pointers, %mask = rock.transforms_to_ptr %1[%c0_i32] : tensor<?x64x64xf16> -> tensor<64x64xi64>, tensor<64x64xi1>
  rock.blockwise_store_ptr %arg2 -> %pointers(%mask) by set : tensor<64x64xf16> -> tensor<64x64xi64>(tensor<64x64xi1>)
  return
}
