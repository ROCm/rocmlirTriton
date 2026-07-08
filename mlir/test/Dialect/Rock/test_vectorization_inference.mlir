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
