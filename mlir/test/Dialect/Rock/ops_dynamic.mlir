// RUN: rocmlir-opt -mlir-print-local-scope %s | rocmlir-opt -mlir-print-local-scope | FileCheck %s
// RUN: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt -mlir-print-local-scope | FileCheck %s

// Round-trips symbolic transform maps, #rock.arg_dim, #rock.arg_expr, and rock
// ops on tensors with dynamic dimensions.

// CHECK: module attributes
// CHECK-SAME: rock.const_expr = #rock.arg_expr<4, []>
// CHECK-SAME: rock.dim_args.k = [#rock.arg_dim<0, 1>, #rock.arg_dim<1, 2>]
// CHECK-SAME: rock.grid_size.k = #rock.arg_expr<(s0 ceildiv 128) * (s1 ceildiv 64), [arg(0, 1), arg(1, 2)]>
module attributes {
    rock.grid_size.k = #rock.arg_expr<(s0 ceildiv 128) * (s1 ceildiv 64), [arg(0, 1), arg(1, 2)]>,
    rock.dim_args.k = [#rock.arg_dim<0, 1>, #rock.arg_dim<1, 2>],
    rock.const_expr = #rock.arg_expr<4, []>
} {

// CHECK-LABEL: func.func @pad_merge_unmerge
// CHECK: rock.transform %{{.*}} by <affine_map<(d0, d1, d2)[s0] -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Pad{0, -s0 + (s0 ceildiv 128) * 128} ["mPad"] at [1] -> ["m"] at [1]>, <PassThrough ["k"] at [2] -> ["k"] at [2]>] symbols = [arg(0, 1)] bounds = [1, (s0 ceildiv 128) * 128, 64] -> [1, s0, 64]> : tensor<1x?x64xf16> to tensor<1x?x64xf16>
// CHECK: rock.transform %{{.*}} by <affine_map<(d0)[s0, s1] -> (d0 floordiv s1, d0 mod s1)> by [<Merge{s0, s1} ["mn"] at [0] -> ["m", "n"] at [0, 1]>] symbols = [arg(1, 0), arg(1, 1)] bounds = [s0 * s1] -> [s0, s1]> : tensor<?x?xf32> to tensor<?xf32>
// CHECK: rock.transform %{{.*}} by <affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + d1)> by [<Unmerge{s0, s1} ["m", "n"] at [0, 1] -> ["mn"] at [0]>] symbols = [arg(1, 0), arg(1, 1)] bounds = [s0, s1] -> [s0 * s1]> : tensor<?xf32> to tensor<?x?xf32>
func.func @pad_merge_unmerge(%arg0: tensor<1x?x64xf16>, %arg1: tensor<?x?xf32>) {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2)[s0] -> (d0, d1, d2)> by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <Pad{0, -s0 + (s0 ceildiv 128) * 128} ["mPad"] at [1] -> ["m"] at [1]>, <PassThrough ["k"] at [2] -> ["k"] at [2]>] symbols = [arg(0, 1)] bounds = [1, (s0 ceildiv 128) * 128, 64] -> [1, s0, 64]> : tensor<1x?x64xf16> to tensor<1x?x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0)[s0, s1] -> (d0 floordiv s1, d0 mod s1)> by [<Merge{s0, s1} ["mn"] at [0] -> ["m", "n"] at [0, 1]>] symbols = [arg(1, 0), arg(1, 1)] bounds = [s0 * s1] -> [s0, s1]> : tensor<?x?xf32> to tensor<?xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + d1)> by [<Unmerge{s0, s1} ["m", "n"] at [0, 1] -> ["mn"] at [0]>] symbols = [arg(1, 0), arg(1, 1)] bounds = [s0, s1] -> [s0 * s1]> : tensor<?xf32> to tensor<?x?xf32>
  return
}

// CHECK-LABEL: func.func @slice_embed_adddim
// CHECK: rock.transform %{{.*}} by <affine_map<(d0, d1)[s0, s1] -> (d0 + 1, d1)> by [<Slice{1, s0} ["m"] at [0] -> ["m"] at [0]>, <PassThrough ["n"] at [1] -> ["n"] at [1]>] symbols = [arg(0, 0), arg(0, 1)] bounds = [s0 - 1, s1] -> [s0, s1]> : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK: rock.transform %{{.*}} by <affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + d1)> by [<Embed{s1, 1} ["m", "n"] at [0, 1] -> ["flat"] at [0]>] symbols = [arg(0, 0), arg(0, 1)] bounds = [s0, s1] -> [s0 * s1]> : tensor<?xf32> to tensor<?x?xf32>
// CHECK: rock.transform %{{.*}} by <affine_map<(d0, d1)[s0] -> (d1)> by [<AddDim{4} ["g"] at [0] -> [] at []>, <PassThrough ["n"] at [1] -> ["n"] at [0]>] symbols = [arg(1, 0)] bounds = [4, s0] -> [s0]> : tensor<?xf32> to tensor<4x?xf32>
func.func @slice_embed_adddim(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1)[s0, s1] -> (d0 + 1, d1)> by [<Slice{1, s0} ["m"] at [0] -> ["m"] at [0]>, <PassThrough ["n"] at [1] -> ["n"] at [1]>] symbols = [arg(0, 0), arg(0, 1)] bounds = [s0 - 1, s1] -> [s0, s1]> : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + d1)> by [<Embed{s1, 1} ["m", "n"] at [0, 1] -> ["flat"] at [0]>] symbols = [arg(0, 0), arg(0, 1)] bounds = [s0, s1] -> [s0 * s1]> : tensor<?xf32> to tensor<?x?xf32>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1)[s0] -> (d1)> by [<AddDim{4} ["g"] at [0] -> [] at []>, <PassThrough ["n"] at [1] -> ["n"] at [0]>] symbols = [arg(1, 0)] bounds = [4, s0] -> [s0]> : tensor<?xf32> to tensor<4x?xf32>
  return
}

// The symbolic parser also reads static maps; with no symbols they print in
// the static form.
// CHECK-LABEL: func.func @static_map
// CHECK: rock.transform %{{.*}} by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>] bounds = [8, 16] -> [8, 16]> : tensor<8x16xf32> to tensor<8x16xf32>
func.func @static_map(%arg0: tensor<8x16xf32>) {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>] bounds = [8, 16] -> [8, 16]> : tensor<8x16xf32> to tensor<8x16xf32>
  return
}

// CHECK-LABEL: func.func @dynamic_gemm
// CHECK: rock.gemm %{{.*}} * %{{.*}} : tensor<1x?x64xf16> * tensor<1x64x?xf16> -> tensor<1x?x?xf16>
func.func @dynamic_gemm(%a: tensor<1x?x64xf16>, %b: tensor<1x64x?xf16>) -> tensor<1x?x?xf16> {
  %0 = rock.gemm %a * %b : tensor<1x?x64xf16> * tensor<1x64x?xf16> -> tensor<1x?x?xf16>
  return %0 : tensor<1x?x?xf16>
}

}
