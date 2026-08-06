// Unit tests for rock-lower-reduce pass
// Verifies that rock.reduce + rock.store is lowered to
// broadcast transform + atomic store + prefill attribute.

// RUN: rocmlir-opt -rock-lower-reduce -mlir-print-local-scope %s | FileCheck %s

#supported = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#split_k = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 16, splitKFactor = 2, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#non_power = #rock.gemm_params<mPerBlock = 16, nPerBlock = 12, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
#to_1x16x16 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d2)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <PassThrough ["m", "n"] at [1, 2] -> ["m", "n"] at [0, 1]>] bounds = [1, 16, 16] -> [16, 16]>
#flatten_1x16x1 = #rock.transform_map<affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 16, 1} ["flat"] at [0] -> ["g", "m", "n"] at [0, 1, 2]>] bounds = [16] -> [1, 16, 1]>
#to_1x16x24 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d2)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <PassThrough ["m", "n"] at [1, 2] -> ["m", "n"] at [0, 1]>] bounds = [1, 16, 24] -> [16, 24]>
#to_1x18x16 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d2)> by [<AddDim{1} ["g"] at [0] -> [] at []>, <PassThrough ["m", "n"] at [1, 2] -> ["m", "n"] at [0, 1]>] bounds = [1, 18, 16] -> [18, 16]>
#flatten_1x18x1 = #rock.transform_map<affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 18, 1} ["flat"] at [0] -> ["g", "m", "n"] at [0, 1, 2]>] bounds = [18] -> [1, 18, 1]>

// CHECK-LABEL: func.func @test_reduce_sum
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x12x12xf32>, %[[OUTPUT:.*]]: tensor<2x12x1xf32> {rock.prefill = 0.000000e+00 : f32})
// CHECK-NOT: rock.reduce
// CHECK: %[[BC:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Broadcast{{.*}} : tensor<2x12x1xf32> to tensor<2x12x12xf32>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_add : tensor<2x12x12xf32> -> tensor<2x12x1xf32> to tensor<2x12x12xf32>
func.func @test_reduce_sum(%arg0: tensor<2x12x12xf32>, %arg1: tensor<2x12x1xf32>) -> tensor<2x12x1xf32> {
  %reduced = rock.reduce sum %arg0 {axis = 2 : index} : tensor<2x12x12xf32> -> tensor<2x12x1xf32>
  %result = rock.store %reduced to %arg1 by set : tensor<2x12x1xf32> -> tensor<2x12x1xf32> to tensor<2x12x1xf32>
  return %result : tensor<2x12x1xf32>
}

// CHECK-LABEL: func.func @test_reduce_max
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x12x12xf32>, %[[OUTPUT:.*]]: tensor<2x12x1xf32> {rock.prefill = 0xFF800000 : f32})
// CHECK-NOT: rock.reduce
// CHECK: %[[BC:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Broadcast{{.*}} : tensor<2x12x1xf32> to tensor<2x12x12xf32>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_max : tensor<2x12x12xf32> -> tensor<2x12x1xf32> to tensor<2x12x12xf32>
func.func @test_reduce_max(%arg0: tensor<2x12x12xf32>, %arg1: tensor<2x12x1xf32>) -> tensor<2x12x1xf32> {
  %reduced = rock.reduce max %arg0 {axis = 2 : index} : tensor<2x12x12xf32> -> tensor<2x12x1xf32>
  %result = rock.store %reduced to %arg1 by set : tensor<2x12x1xf32> -> tensor<2x12x1xf32> to tensor<2x12x1xf32>
  return %result : tensor<2x12x1xf32>
}

// CHECK-LABEL: func.func @test_reduce_sum_f16
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x12x12xf16>, %[[OUTPUT:.*]]: tensor<2x12x1xf16> {rock.prefill = 0.000000e+00 : f16})
// CHECK-NOT: rock.reduce
// CHECK: %[[BC:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Broadcast{{.*}} : tensor<2x12x1xf16> to tensor<2x12x12xf16>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_add : tensor<2x12x12xf16> -> tensor<2x12x1xf16> to tensor<2x12x12xf16>
func.func @test_reduce_sum_f16(%arg0: tensor<2x12x12xf16>, %arg1: tensor<2x12x1xf16>) -> tensor<2x12x1xf16> {
  %reduced = rock.reduce sum %arg0 {axis = 2 : index} : tensor<2x12x12xf16> -> tensor<2x12x1xf16>
  %result = rock.store %reduced to %arg1 by set : tensor<2x12x1xf16> -> tensor<2x12x1xf16> to tensor<2x12x1xf16>
  return %result : tensor<2x12x1xf16>
}

// CHECK-LABEL: func.func @test_reduce_sum_bf16
// CHECK-SAME: (%[[INPUT:.*]]: tensor<2x12x12xbf16>, %[[OUTPUT:.*]]: tensor<2x12x1xbf16> {rock.prefill = 0.000000e+00 : bf16})
// CHECK-NOT: rock.reduce
// CHECK: %[[BC:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Broadcast{{.*}} : tensor<2x12x1xbf16> to tensor<2x12x12xbf16>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_add : tensor<2x12x12xbf16> -> tensor<2x12x1xbf16> to tensor<2x12x12xbf16>
func.func @test_reduce_sum_bf16(%arg0: tensor<2x12x12xbf16>, %arg1: tensor<2x12x1xbf16>) -> tensor<2x12x1xbf16> {
  %reduced = rock.reduce sum %arg0 {axis = 2 : index} : tensor<2x12x12xbf16> -> tensor<2x12x1xbf16>
  %result = rock.store %reduced to %arg1 by set : tensor<2x12x1xbf16> -> tensor<2x12x1xbf16> to tensor<2x12x1xbf16>
  return %result : tensor<2x12x1xbf16>
}

// Verify that an intermediate Merge transform between reduce and store
// is handled: the dest gets an Unmerge (inverse of Merge) followed by Broadcast.

// CHECK-LABEL: func.func @test_reduce_sum_with_intermediate_merge
// CHECK-SAME: (%[[INPUT:.*]]: tensor<1x5x3xf32>, %[[OUTPUT:.*]]: tensor<5xf32> {rock.prefill = 0.000000e+00 : f32})
// CHECK-NOT: rock.reduce
// CHECK: %[[UNMERGE:.*]] = rock.transform %[[OUTPUT]] by {{.*}}Unmerge{{.*}} : tensor<5xf32> to tensor<1x5x1xf32>
// CHECK: %[[BC:.*]] = rock.transform %[[UNMERGE]] by {{.*}}Broadcast{{.*}} : tensor<1x5x1xf32> to tensor<1x5x3xf32>
// CHECK: rock.store %[[INPUT]] to %[[BC]] by atomic_add
func.func @test_reduce_sum_with_intermediate_merge(%arg0: tensor<1x5x3xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %reduced = rock.reduce sum %arg0 {axis = 2 : index} : tensor<1x5x3xf32> -> tensor<1x5x1xf32>
  %flat = rock.transform %reduced by <affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 5, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [5] -> [1, 5, 1]> : tensor<1x5x1xf32> to tensor<5xf32>
  %result = rock.store %flat to %arg1 by set : tensor<5xf32> -> tensor<5xf32> to tensor<5xf32>
  return %result : tensor<5xf32>
}

// CHECK-LABEL: func.func @test_blockwise_candidate
// CHECK: rock.reduce sum {{.*}} {axis = 2 : index, blockwise}
// CHECK: rock.store {{.*}} by set
func.func @test_blockwise_candidate(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
  %gemm = rock.gemm %a * %b {params = #supported} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  %fused = arith.addf %gemm, %gemm : tensor<16x16xf32>
  %view = rock.transform %fused by #to_1x16x16 : tensor<16x16xf32> to tensor<1x16x16xf32>
  %reduced = rock.reduce sum %view {axis = 2 : index} : tensor<1x16x16xf32> -> tensor<1x16x1xf32>
  %flat = rock.transform %reduced by #flatten_1x16x1 : tensor<1x16x1xf32> to tensor<16xf32>
  %result = rock.store %flat to %out by set : tensor<16xf32> -> tensor<16xf32> to tensor<16xf32>
  return %result : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_direct_gemm_fallback
// CHECK-NOT: rock.reduce
// CHECK: rock.store {{.*}} by atomic_add
func.func @test_direct_gemm_fallback(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
  %gemm = rock.gemm %a * %b {params = #supported} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  %view = rock.transform %gemm by #to_1x16x16 : tensor<16x16xf32> to tensor<1x16x16xf32>
  %reduced = rock.reduce sum %view {axis = 2 : index} : tensor<1x16x16xf32> -> tensor<1x16x1xf32>
  %flat = rock.transform %reduced by #flatten_1x16x1 : tensor<1x16x1xf32> to tensor<16xf32>
  %result = rock.store %flat to %out by set : tensor<16xf32> -> tensor<16xf32> to tensor<16xf32>
  return %result : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_transposed_gemm_fallback
// CHECK-NOT: rock.reduce
// CHECK: rock.store {{.*}} by atomic_add
func.func @test_transposed_gemm_fallback(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
  %gemm = rock.gemm %a * %b {oTransposed, params = #supported} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  %fused = arith.addf %gemm, %gemm : tensor<16x16xf32>
  %view = rock.transform %fused by #to_1x16x16 : tensor<16x16xf32> to tensor<1x16x16xf32>
  %reduced = rock.reduce sum %view {axis = 2 : index} : tensor<1x16x16xf32> -> tensor<1x16x1xf32>
  %flat = rock.transform %reduced by #flatten_1x16x1 : tensor<1x16x1xf32> to tensor<16xf32>
  %result = rock.store %flat to %out by set : tensor<16xf32> -> tensor<16xf32> to tensor<16xf32>
  return %result : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_ambiguous_roots_fallback
// CHECK-NOT: rock.reduce
// CHECK: rock.store {{.*}} by atomic_add
func.func @test_ambiguous_roots_fallback(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
  %gemm0 = rock.gemm %a * %b {params = #supported} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  %gemm1 = rock.gemm %b * %a {params = #supported} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  %fused = arith.addf %gemm0, %gemm1 : tensor<16x16xf32>
  %view = rock.transform %fused by #to_1x16x16 : tensor<16x16xf32> to tensor<1x16x16xf32>
  %reduced = rock.reduce sum %view {axis = 2 : index} : tensor<1x16x16xf32> -> tensor<1x16x1xf32>
  %flat = rock.transform %reduced by #flatten_1x16x1 : tensor<1x16x1xf32> to tensor<16xf32>
  %result = rock.store %flat to %out by set : tensor<16xf32> -> tensor<16xf32> to tensor<16xf32>
  return %result : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_split_k_fallback
// CHECK-NOT: rock.reduce
// CHECK: rock.store {{.*}} by atomic_add
func.func @test_split_k_fallback(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
  %gemm = rock.gemm %a * %b {params = #split_k} : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
  %fused = arith.addf %gemm, %gemm : tensor<16x16xf32>
  %view = rock.transform %fused by #to_1x16x16 : tensor<16x16xf32> to tensor<1x16x16xf32>
  %reduced = rock.reduce sum %view {axis = 2 : index} : tensor<1x16x16xf32> -> tensor<1x16x1xf32>
  %flat = rock.transform %reduced by #flatten_1x16x1 : tensor<1x16x1xf32> to tensor<16xf32>
  %result = rock.store %flat to %out by set : tensor<16xf32> -> tensor<16xf32> to tensor<16xf32>
  return %result : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_non_power_of_two_tile_fallback
// CHECK-NOT: rock.reduce
// CHECK: rock.store {{.*}} by atomic_add
func.func @test_non_power_of_two_tile_fallback(%a: tensor<16x16xf32>, %b: tensor<16x24xf32>, %out: tensor<16xf32>) -> tensor<16xf32> {
  %gemm = rock.gemm %a * %b {params = #non_power} : tensor<16x16xf32> * tensor<16x24xf32> -> tensor<16x24xf32>
  %fused = arith.addf %gemm, %gemm : tensor<16x24xf32>
  %view = rock.transform %fused by #to_1x16x24 : tensor<16x24xf32> to tensor<1x16x24xf32>
  %reduced = rock.reduce sum %view {axis = 2 : index} : tensor<1x16x24xf32> -> tensor<1x16x1xf32>
  %flat = rock.transform %reduced by #flatten_1x16x1 : tensor<1x16x1xf32> to tensor<16xf32>
  %result = rock.store %flat to %out by set : tensor<16xf32> -> tensor<16xf32> to tensor<16xf32>
  return %result : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_padding_fallback
// CHECK-NOT: rock.reduce
// CHECK: rock.store {{.*}} by atomic_add
func.func @test_padding_fallback(%a: tensor<18x16xf32>, %b: tensor<16x16xf32>, %out: tensor<18xf32>) -> tensor<18xf32> {
  %gemm = rock.gemm %a * %b {params = #supported} : tensor<18x16xf32> * tensor<16x16xf32> -> tensor<18x16xf32>
  %fused = arith.addf %gemm, %gemm : tensor<18x16xf32>
  %view = rock.transform %fused by #to_1x18x16 : tensor<18x16xf32> to tensor<1x18x16xf32>
  %reduced = rock.reduce sum %view {axis = 2 : index} : tensor<1x18x16xf32> -> tensor<1x18x1xf32>
  %flat = rock.transform %reduced by #flatten_1x18x1 : tensor<1x18x1xf32> to tensor<18xf32>
  %result = rock.store %flat to %out by set : tensor<18xf32> -> tensor<18xf32> to tensor<18xf32>
  return %result : tensor<18xf32>
}
