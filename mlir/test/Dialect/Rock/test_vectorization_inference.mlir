// RUN: rocmlir-opt -rock-vectorization-inference-test \
// RUN:   -allow-unregistered-dialect --mlir-print-local-scope %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not="Unexpected op"

#row_major = #rock.transform_map<
  affine_map<(m, n) -> (m * 64 + n)>
  by [<Unmerge{64, 64} ["m", "n"] at [0, 1] -> ["raw"] at [0]>]
  bounds = [64, 64] -> [4096]>

#identity_2d = #rock.transform_map<
  affine_map<(m, n) -> (m, n)>
  by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>]
  bounds = [64, 64] -> [64, 64]>

// The first GEMM is stored and also feeds the second GEMM. The second store
// writes to the same destination view while the first store result is threaded
// through resultAlias to keep the pure store chain live.
//
// CHECK-LABEL: func @chained_gemm_reuses_store_result
// CHECK: [[G1:%.*]] = rock.blockwise_gemm
// CHECK: [[DEST1:%.*]] = rock.transform %{{.*}}
// CHECK: [[R1:%.*]] = rock.blockwise_store [[G1]]
// CHECK: [[G1F16:%.*]] = arith.truncf [[G1]]
// CHECK: rock.blockwise_gemm([[G1F16]]
// CHECK: rock.blockwise_store {{.*}} -> [[DEST1]] alias [[R1]]
// CHECK: "get_length"([[DEST1]])
// CHECK-SAME: bufferVectorSize = 1 : index
// CHECK-SAME: in_dim = 0 : i64
// CHECK-SAME: result = 1 : index
// CHECK: "get_length"([[DEST1]])
// CHECK-SAME: bufferVectorSize = 1 : index
// CHECK-SAME: in_dim = 1 : i64
// CHECK-SAME: result = 64 : index
func.func @chained_gemm_reuses_store_result(
    %a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
    %c: tensor<64x64xf32>, %b2: tensor<64x64xf16>,
    %c2: tensor<64x64xf32>, %dest_raw: tensor<4096xf32>) {
  %g1 = rock.blockwise_gemm(%a, %b, %c)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32>
      -> tensor<64x64xf32>
  %dest1 = rock.transform %dest_raw by #row_major
    : tensor<4096xf32> to tensor<64x64xf32>
  %r1 = rock.blockwise_store %g1 -> %dest1 by set
    : tensor<64x64xf32> -> tensor<64x64xf32> -> tensor<64x64xf32>

  %g1f16 = arith.truncf %g1 : tensor<64x64xf32> to tensor<64x64xf16>
  %g2 = rock.blockwise_gemm(%g1f16, %b2, %c2)
    : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32>
      -> tensor<64x64xf32>
  %r2 = rock.blockwise_store %g2 -> %dest1 alias %r1 by set
    : tensor<64x64xf32> -> tensor<64x64xf32> alias tensor<64x64xf32> -> tensor<64x64xf32>

  "get_length"(%dest1) {in_dim = 0 : i64} : (tensor<64x64xf32>) -> ()
  "get_length"(%dest1) {in_dim = 1 : i64} : (tensor<64x64xf32>) -> ()
  return
}

// A held-constant Unmerge dim of non-unit size must not let slower dims extend
// the contiguous vector length. Here flat = m_i*16 + ni*4 + vec_item, so with
// "ni" held constant only iter=0..3 are contiguous (iter=4 jumps by 16).

#held_const_flat = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 * 4 + d1)>
  by [<Unmerge{32, 4} ["i", "vec_item"] at [0, 1] -> ["flat"] at [0]>]
  bounds = [32, 4] -> [128]>

#held_const_unmerge_i = #rock.transform_map<
  affine_map<(d0, d1, d2) -> (d1 * 4 + d0, d2)>
  by [<Unmerge{8, 4} ["m_i", "ni"] at [1, 0] -> ["i"] at [0]>,
      <PassThrough ["vec_item"] at [2] -> ["vec_item"] at [1]>]
  bounds = [4, 8, 4] -> [32, 4]>

#held_const_merge = #rock.transform_map<
  affine_map<(d0, d1) -> (d0, d1 floordiv 4, d1 mod 4)>
  by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>,
      <Merge{8, 4} ["iter"] at [1] -> ["m_i", "vec_item"] at [1, 2]>]
  bounds = [4, 32] -> [4, 8, 4]>

// CHECK-LABEL: func @unmerge_held_constant_non_unit_dim
// CHECK: "get_length"
// CHECK-SAME: result = 4 : index
func.func @unmerge_held_constant_non_unit_dim(%buf: tensor<128xf32>) {
  %0 = rock.transform %buf by #held_const_flat
    : tensor<128xf32> to tensor<32x4xf32>
  %1 = rock.transform %0 by #held_const_unmerge_i
    : tensor<32x4xf32> to tensor<4x8x4xf32>
  %2 = rock.transform %1 by #held_const_merge
    : tensor<4x8x4xf32> to tensor<4x32xf32>
  "get_length"(%2) {in_dim = 1 : i64, in_dim_len = 8 : i64}
    : (tensor<4x32xf32>) -> ()
  return
}
