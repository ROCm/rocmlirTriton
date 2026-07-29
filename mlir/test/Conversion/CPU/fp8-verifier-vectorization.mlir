// Regression test for the fp8 CPU-verifier vectorization ordering.
//
// The host lowering pipeline must run `emulate-fp8-ext-trunc` AFTER the
// `cpu-lower-verifier` OPTIMIZE phase (phase=1), which tiles the verifier
// matmul and vectorizes it via a named `vector.contract`. If the fp8
// `arith.extf` is rewritten into a `memref.load` table lookup *before*
// vectorization, `transform.structured.vectorize` can no longer form a named
// contraction and aborts with "Attempted to vectorize, but failed", which
// breaks fp8 GEMM/conv verification (and therefore tuning) for any problem
// with M, N, K >= 8.
//
// This test pins the correct order: verifier lowering + vectorization first
// (producing `vector.contract`), then fp8 emulation (which rewrites the
// resulting vector-typed extends into table lookups).

// RUN: rocmlir-opt -pass-pipeline='builtin.module(cpu-conv-to-gemm,cpu-lower-verifier{phase=1},emulate-fp8-ext-trunc)' %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  // CHECK-LABEL: func.func @host_naive_gemm
  // Vectorization of the tiled matmul must succeed and produce a contraction.
  // CHECK: vector.contract
  // fp8 emulation still runs afterwards, lowering the (now vector-typed) extf
  // into a table lookup rather than leaving a raw arith.extf on fp8.
  // CHECK: memref.get_global @__rocmlir_extf_tbl_f8E4M3FN
  // CHECK-NOT: arith.extf %{{.*}} : {{.*}}f8E4M3FN
  func.func @host_naive_gemm(%arg0: tensor<4096xf8E4M3FN>, %arg1: tensor<4096xf8E4M3FN>, %arg2: tensor<4096xf32>) -> tensor<4096xf32> attributes {rock.cpu_verifier} {
    %cst = arith.constant 0.000000e+00 : f32
    %expanded = tensor.expand_shape %arg0 [[0, 1]] output_shape [64, 64] : tensor<4096xf8E4M3FN> into tensor<64x64xf8E4M3FN>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1]] output_shape [64, 64] : tensor<4096xf8E4M3FN> into tensor<64x64xf8E4M3FN>
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%expanded, %expanded_0 : tensor<64x64xf8E4M3FN>, tensor<64x64xf8E4M3FN>) outs(%1 : tensor<64x64xf32>) {
    ^bb0(%in: f8E4M3FN, %in_1: f8E4M3FN, %out: f32):
      %3 = arith.extf %in_1 : f8E4M3FN to f32
      %4 = arith.extf %in : f8E4M3FN to f32
      %5 = arith.mulf %4, %3 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<64x64xf32>
    %collapsed = tensor.collapse_shape %2 [[0, 1]] : tensor<64x64xf32> into tensor<4096xf32>
    return %collapsed : tensor<4096xf32>
  }
}
