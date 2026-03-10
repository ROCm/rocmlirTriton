// Ensures that the padding application, group application, etc. in gemm-to-gridwise
// function as expected.

// RUN: rocmlir-opt -rock-gemm-to-gridwise -mlir-print-local-scope %s | FileCheck %s

#general_gemm_params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 8, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#general_gemm_params_splitk = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 8, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#general_gemm_params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 4, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params3 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 2, matrixInstrNonkdim = 0, splitKFactor = 3, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xldops_attn_params_g0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 4, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xldops_attn_params_g1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xldops_attn_params_g1_splitk = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 4, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// CHECK-LABEL: func.func @gemm_easy_case_from_conv
// CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf32>, %[[b:.*]]: tensor<1x72x512xf32>, %[[c:.*]]: tensor<1x128x512xf32>)
// CHECK-SAME: rock.grid_size = 4
func.func @gemm_easy_case_from_conv(%a: tensor<1x72x128xf32>, %b: tensor<1x72x512xf32>, %c: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x72x128xf32> to tensor<1x128x72xf32>
  // CHECK: rock.gridwise_gemm(%[[transA]], %[[b]]) 
  %result = rock.gemm tr %a * %b {
    gridSize = 4 : i32,
    params = #general_gemm_params0
  } : tensor<1x72x128xf32> * tensor<1x72x512xf32> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_splitk
// CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf32>, %[[b:.*]]: tensor<1x72x512xf32>, %[[c:.*]]: tensor<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: rock.grid_size = 8 : i32
func.func @gemm_splitk(%a: tensor<1x72x128xf32>, %b: tensor<1x72x512xf32>, %c: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}})
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm tr %a * %b {
    gridSize = 4 : i32,
    params = #general_gemm_params_splitk
  } : tensor<1x72x128xf32> * tensor<1x72x512xf32> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_easy_case_from_conv_xdlops
// CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf32>, %[[b:.*]]: tensor<1x72x512xf32>, %[[c:.*]]: tensor<1x128x512xf32>)
// CHECK-SAME: rock.grid_size = 16 : i32
func.func @gemm_easy_case_from_conv_xdlops(%a: tensor<1x72x128xf32>, %b: tensor<1x72x512xf32>, %c: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x72x128xf32> to tensor<1x128x72xf32>
  // CHECK: rock.gridwise_gemm(%[[transA]], %[[b]])
  %result = rock.gemm tr %a * %b {
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params0
  } : tensor<1x72x128xf32> * tensor<1x72x512xf32> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// CHECK-LABEL: func.func @gemm_most_general_padding_case
// CHECK-SAME: (%[[a:.*]]: tensor<1x1x1xf32>, %[[b:.*]]: tensor<1x1x1xf32>, %[[c:.*]]: tensor<1x1x1xf32>)
// CHECK-SAME: rock.grid_size = 1
func.func @gemm_most_general_padding_case(%a: tensor<1x1x1xf32>, %b: tensor<1x1x1xf32>, %c: tensor<1x1x1xf32>) -> tensor<1x1x1xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: rock.transform %[[a]] by {{.*}} : tensor<1x1x1xf32> to tensor<1x1x1xf32>
  // CHECK: rock.transform {{.*}} by {{.*}} to tensor<1x64x16xf32>
  // CHECK: rock.transform %[[b]] by {{.*}} : tensor<1x1x1xf32> to tensor<1x16x64xf32>
  // CHECK: rock.transform %[[c]] by {{.*}} : tensor<1x1x1xf32> to tensor<1x64x64xf32>
  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}}) 
  %result = rock.gemm tr %a * %b {
    gridSize = 1 : i32,
    params = #general_gemm_params1
  } : tensor<1x1x1xf32> * tensor<1x1x1xf32> -> tensor<1x1x1xf32>
  %out = rock.store %result to %c by set : tensor<1x1x1xf32> -> tensor<1x1x1xf32> to tensor<1x1x1xf32>
  func.return %out : tensor<1x1x1xf32>
}

// CHECK-LABEL: func.func @gemm_in_standard_form
// CHECK-SAME: (%[[a:.*]]: tensor<128x72xf32>, %[[b:.*]]: tensor<72x512xf32>, %[[c:.*]]: tensor<128x512xf32>)
// CHECK-SAME: rock.grid_size = 4
func.func @gemm_in_standard_form(%a: tensor<128x72xf32>, %b: tensor<72x512xf32>, %c: tensor<128x512xf32>) -> tensor<128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: %[[normalizeA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<128x72xf32> to tensor<1x128x72xf32>
  // CHECK: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : tensor<72x512xf32> to tensor<1x72x512xf32>
  // CHECK: %[[normalizeC:.*]] = rock.transform %[[c]] by {{.*}} : tensor<128x512xf32> to tensor<1x128x512xf32>
  // CHECK: rock.gridwise_gemm(%[[normalizeA]], %[[normalizeB]]) 
  %result = rock.gemm %a * %b {
    gridSize = 4 : i32,
    params = #general_gemm_params0
  } : tensor<128x72xf32> * tensor<72x512xf32> -> tensor<128x512xf32>
  %out = rock.store %result to %c by set : tensor<128x512xf32> -> tensor<128x512xf32> to tensor<128x512xf32>
  func.return %out : tensor<128x512xf32>
}

// CHECK-LABEL: func.func @gemm_transposed_from_gridwise
// CHECK-SAME: (%[[a:.*]]: tensor<1x128x72xf32>, %[[b:.*]]: tensor<1x512x72xf32>, %[[c:.*]]: tensor<1x512x128xf32>)
// CHECK-SAME: grid_size = 4
func.func @gemm_transposed_from_gridwise(%a: tensor<1x128x72xf32>, %b: tensor<1x512x72xf32>, %c: tensor<1x512x128xf32>) -> tensor<1x512x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] {{.*}} : tensor<1x512x72xf32> to tensor<1x72x512xf32{{.*}}>
  // CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] {{.*}} : tensor<1x512x128xf32> to tensor<1x128x512xf32{{.*}}>
  // CHECK: rock.gridwise_gemm(%[[a]], %[[normalizeB]])
  %result = rock.gemm %a * tr %b {
    gridSize = 4 : i32,
    params = #general_gemm_params0,
    oTransposed
  } : tensor<1x128x72xf32> * tensor<1x512x72xf32> -> tensor<1x512x128xf32>
  %out = rock.store %result to %c by set : tensor<1x512x128xf32> -> tensor<1x512x128xf32> to tensor<1x512x128xf32>
  func.return %out : tensor<1x512x128xf32>
}

// CHECK-LABEL: func.func @gemm_pad_for_split_k
// CHECK-SAME: (%[[a:.*]]: tensor<1x128x238xf32>, %[[b:.*]]: tensor<1x238x512xf32>, %[[c:.*]]: tensor<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// CHECK-SAME: rock.grid_size = 48
func.func @gemm_pad_for_split_k(%a: tensor<1x128x238xf32>, %b: tensor<1x238x512xf32>, %c: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.transform %[[a]] by {{.*}} : tensor<1x128x238xf32> to tensor<1x128x240xf32>
  // CHECK: rock.transform %[[b]] by {{.*}} : tensor<1x238x512xf32> to tensor<1x240x512xf32>
  %alloc = tensor.empty() : tensor<1x128x512xf32>
  // CHECK: rock.gridwise_gemm({{.*}}, {{.*}})
  // CHECK: rock.store {{.*}} by atomic_add
  %result = rock.gemm %a * %b {
    derivedBlockSize = 256 : i32,
    gridSize = 4 : i32,
    params = #xdlops_gemm_params3
  } : tensor<1x128x238xf32> * tensor<1x238x512xf32> -> tensor<1x128x512xf32>
  %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
  func.return %out : tensor<1x128x512xf32>
}

// TODO(roctriton): We need to unbufferize rock.reduce
// DISABLED-CHECK-LABEL: func.func @gemm_reduce_and_split_k
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x128x238xf32>, %[[b:.*]]: tensor<1x238x512xf32>, %[[c:.*]]: tensor<1x128x1xf32> {rock.prefill = {{.*}} : f32}, %[[d:.*]]: tensor<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// DISABLED-CHECK-SAME: grid_size = 48
// func.func @gemm_reduce_and_split_k(%a: tensor<1x128x238xf32>, %b: tensor<1x238x512xf32>, %c: tensor<1x128x1xf32>, %d: tensor<1x128x512xf32>) -> (tensor<1x128x1xf32>, tensor<1x128x512xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK-DAG: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x128x238xf32> to tensor<1x238x128xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[transA]] by {{.*}} : tensor<1x238x128xf32> to tensor<1x240x128xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : tensor<1x238x512xf32> to tensor<1x240x512xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[splitA:.*]] = rock.transform %[[normalizeA]] by {{.*}} : tensor<1x240x128xf32> to tensor<1x3x80x128xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[splitB:.*]] = rock.transform %[[normalizeB]] by {{.*}} : tensor<1x240x512xf32> to tensor<1x3x80x512xf32{{.*}}>
//   %alloc = tensor.empty() : tensor<1x128x512xf32>
//   %alloc2 = tensor.empty() : tensor<1x128x1xf32>
//   // DISABLED-CHECK: rock.gridwise_gemm
//   // DISABLED-CHECK-SAME: storeMethod( atomic_add)
//   %result = rock.gemm %a * %b {
//     derivedBlockSize = 256 : i32,
//     gridSize = 4 : i32,
//     params = #xdlops_gemm_params3
//   } : tensor<1x128x238xf32> * tensor<1x238x512xf32> -> tensor<1x128x512xf32>
//   rock.reduce sum %result into %alloc2 {axis = 2 : index, blockSize = 256 : i32, gridSize = 2 : i32} : tensor<1x128x512xf32> into tensor<1x128x1xf32>
//   %out_d = rock.store %result to %d by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
//   %out_c = rock.store %alloc2 to %c by set : tensor<1x128x1xf32> -> tensor<1x128x1xf32> to tensor<1x128x1xf32>
//   func.return %out_c, %out_d : tensor<1x128x1xf32>, tensor<1x128x512xf32>
// }

// TODO(roctriton): We need to unbufferize rock.reduce
// DISABLED-CHECK-LABEL: func.func @gemm_reduce_and_split_k_return_reduce_directly
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x128x238xf32>, %[[b:.*]]: tensor<1x238x512xf32>, %[[c:.*]]: tensor<1x128x1xf32> {rock.prefill = {{.*}} : f32}, %[[d:.*]]: tensor<1x128x512xf32> {rock.prefill = {{.*}} : f32})
// DISABLED-CHECK-SAME: grid_size = 48
// func.func @gemm_reduce_and_split_k_return_reduce_directly(%a: tensor<1x128x238xf32>, %b: tensor<1x238x512xf32>, %c: tensor<1x128x1xf32>, %d: tensor<1x128x512xf32>) -> (tensor<1x128x1xf32>, tensor<1x128x512xf32>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK-DAG: %[[transA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x128x238xf32> to tensor<1x238x128xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[transA]] by {{.*}} : tensor<1x238x128xf32> to tensor<1x240x128xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] by {{.*}} : tensor<1x238x512xf32> to tensor<1x240x512xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[splitA:.*]] = rock.transform %[[normalizeA]] by {{.*}} : tensor<1x240x128xf32> to tensor<1x3x80x128xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[splitB:.*]] = rock.transform %[[normalizeB]] by {{.*}} : tensor<1x240x512xf32> to tensor<1x3x80x512xf32{{.*}}>
//   %alloc = tensor.empty() : tensor<1x128x512xf32>
//   // DISABLED-CHECK: rock.gridwise_gemm
//   // DISABLED-CHECK-SAME: storeMethod( atomic_add)
//   %result = rock.gemm %a * %b {
//     derivedBlockSize = 256 : i32,
//     gridSize = 4 : i32,
//     params = #xdlops_gemm_params3
//   } : tensor<1x128x238xf32> * tensor<1x238x512xf32> -> tensor<1x128x512xf32>
//   rock.reduce sum %result into %c {axis = 2 : index, blockSize = 256 : i32, gridSize = 2 : i32} : tensor<1x128x512xf32> into tensor<1x128x1xf32>
//   %out_d = rock.store %result to %d by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
//   func.return %c, %out_d : tensor<1x128x1xf32>, tensor<1x128x512xf32>
// }

// TODO(roctriton): Fusion tests need rework - rock.gemm result must go directly to rock.store
// DISABLED-CHECK-LABEL: func.func @gemm_fusion_to_f32_split_k
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x5x4xf16>, %[[b:.*]]: tensor<1x4x3xf16>, %[[c:.*]]: tensor<1x5x3xf16>, %[[d:.*]]: tensor<1x5x3xf32> {rock.prefill = 0.000000e+00 : f32})
// DISABLED-CHECK-SAME: rock.grid_size = 3
// func.func @gemm_fusion_to_f32_split_k(%arg0: tensor<1x5x4xf16>, %arg1: tensor<1x4x3xf16>, %arg2: tensor<1x5x3xf16>, %arg3: tensor<1x5x3xf32>) -> tensor<1x5x3xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   %alloc = tensor.empty() : tensor<1x5x3xf16>
//   // DISABLED-CHECK: rock.gridwise_gemm
//   %result = rock.gemm %arg0 * %arg1 {
//     derivedBlockSize = 256 : i32,
//     gridSize = 4 : i32,
//     params = #xdlops_gemm_params3
//   } : tensor<1x5x4xf16> * tensor<1x4x3xf16> -> tensor<1x5x3xf16>
//   %alloc_0 = tensor.empty() : tensor<1x5x3xf32>
//   %generic_result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%result, %arg2 : tensor<1x5x3xf16>, tensor<1x5x3xf16>) outs(%alloc_0 : tensor<1x5x3xf32>) {
//   ^bb0(%in: f16, %in_1: f16, %out: f32):
//     %7 = arith.addf %in, %in_1 : f16
//     %8 = arith.extf %7 : f16 to f32
//     linalg.yield %8 : f32
//   } -> tensor<1x5x3xf32>
//   %out = rock.store %generic_result to %arg3 by set : tensor<1x5x3xf32> -> tensor<1x5x3xf32> to tensor<1x5x3xf32>
//   return %out : tensor<1x5x3xf32>
// }

// TODO(roctriton): Fusion tests need rework - rock.gemm result must go directly to rock.store
// DISABLED-CHECK-LABEL: func.func @gemm_fusion_to_f16_split_k
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x5x4xf32>, %[[b:.*]]: tensor<1x4x3xf32>, %[[c:.*]]: tensor<1x5x3xf32>, %[[d:.*]]: tensor<1x5x3xf16> {rock.prefill = 0.000000e+00 : f16})
// DISABLED-CHECK-SAME: rock.grid_size = 3 : i32
// func.func @gemm_fusion_to_f16_split_k(%arg0: tensor<1x5x4xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x5x3xf32>, %arg3: tensor<1x5x3xf16>) -> tensor<1x5x3xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   %alloc = tensor.empty() : tensor<1x5x3xf32>
//   // DISABLED-CHECK: rock.gridwise_gemm
//   %result = rock.gemm %arg0 * %arg1 {
//     derivedBlockSize = 256 : i32,
//     gridSize = 4 : i32,
//     params = #xdlops_gemm_params3
//   } : tensor<1x5x4xf32> * tensor<1x4x3xf32> -> tensor<1x5x3xf32>
//   %alloc_0 = tensor.empty() : tensor<1x5x3xf16>
//   %generic_result = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%result, %arg2 : tensor<1x5x3xf32>, tensor<1x5x3xf32>) outs(%alloc_0 : tensor<1x5x3xf16>) {
//   ^bb0(%in: f32, %in_1: f32, %out: f16):
//     %7 = arith.addf %in, %in_1 : f32
//     %8 = arith.truncf %7 : f32 to f16
//     linalg.yield %8 : f16
//   } -> tensor<1x5x3xf16>
//   %out = rock.store %generic_result to %arg3 by set : tensor<1x5x3xf16> -> tensor<1x5x3xf16> to tensor<1x5x3xf16>
//   return %out : tensor<1x5x3xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_simple
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[o:.*]]: tensor<1x1024x64xf32>)
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 32 : i32
// func.func @rock_attention_simple(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.gridwise_attention(%[[q]], %[[k]], %[[v]], %[[o]])
//   %result = rock.attention{
//      qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
//      %arg3 = softmax(qk) * %arg2 : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32>
//   } { 
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 1 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32
//   } -> tensor<1x1024x64xf32>
//   %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
//   return %out : tensor<1x1024x64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_tr_padded
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<1x49x7xf32>, %[[k:.*]]: tensor<1x7x49xf32>, %[[v:.*]]: tensor<1x49x7xf32>, %[[o:.*]]: tensor<1x49x7xf32>)
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 2 : i32
// func.func @rock_attention_tr_padded(%arg0: tensor<1x49x7xf32>, %arg1: tensor<1x7x49xf32>, %arg2: tensor<1x49x7xf32>, %arg3: tensor<1x49x7xf32>) -> tensor<1x49x7xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK-DAG: %[[trQ:.*]] = rock.transform %[[q]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x7x49xf32>
//   // DISABLED-CHECK-DAG: %[[paddedTrQ:.*]] = rock.transform %[[trQ]] by {{.*}} : tensor<1x7x49xf32> to tensor<1x8x64xf32>
//   // DISABLED-CHECK-DAG: %[[paddedK:.*]] = rock.transform %[[k]] by {{.*}} : tensor<1x7x49xf32> to tensor<1x8x64xf32>
//   // DISABLED-CHECK-DAG: %[[paddedV:.*]] = rock.transform %[[v]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x64x32xf32>
//   // DISABLED-CHECK-DAG: %[[paddedO:.*]] = rock.transform %[[o]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x64x32xf32>
//   // DISABLED-CHECK: rock.gridwise_attention(%[[paddedTrQ]], %[[paddedK]], %[[paddedV]], %[[paddedO]])
//   // DISABLED-CHECK-NEXT: prePadG0M = 49 : index, prePadG0N = 49 : index
//   %result = rock.attention{
//     qk = %arg0 * %arg1 : tensor<1x49x7xf32>, tensor<1x7x49xf32>
//     %arg3 = softmax(qk) * %arg2 : tensor<1x49x7xf32> -> tensor<1x49x7xf32>
//   } { 
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 1 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32
//   } -> tensor<1x49x7xf32>
//   %out = rock.store %result to %arg3 by set : tensor<1x49x7xf32> -> tensor<1x49x7xf32> to tensor<1x49x7xf32>
//   return %out : tensor<1x49x7xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_kvcache
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[o:.*]]: tensor<1x1024x64xf32>, %[[currentSeqLen:.*]]: tensor<1xi32>)
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 32 : i32
// func.func @rock_attention_kvcache(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>, %arg4: tensor<1xi32>) -> tensor<1x1024x64xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.gridwise_attention(%[[q]], %[[k]], %[[v]], %[[currentSeqLen]], %[[o]])
//   %result = rock.attention{
//      qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
//      currentSeqLen = (%arg4 : tensor<1xi32>)
//      %arg3 = softmax(qk) * %arg2 : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32>
//   } {
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 1 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32
//   } -> tensor<1x1024x64xf32>
//   %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
//   return %out : tensor<1x1024x64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_causal
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[o:.*]]: tensor<1x1024x64xf32>)
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 32 : i32
// func.func @rock_attention_causal(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.gridwise_attention(%[[q]], %[[k]], %[[v]], %[[o]])
//   // DISABLED-CHECK-NEXT: , causal,
//   %result = rock.attention{
//      qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
//      causal
//      %arg3 = softmax(qk) * %arg2 : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32>
//   } {
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 1 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32
//   } -> tensor<1x1024x64xf32>
//   %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
//   return %out : tensor<1x1024x64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_lse
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[lse:.*]]: tensor<1x1024xf32>, %[[o:.*]]: tensor<1x1024x64xf32>)
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 32 : i32
// func.func @rock_attention_lse(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024xf32>, %arg4: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.gridwise_attention(%[[q]], %[[k]], %[[v]], %[[o]], %[[lse]])
//   %result = rock.attention{
//      qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
//      lse = %arg3 : tensor<1x1024xf32>
//      %arg4 = softmax(qk) * %arg2 : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32>
//   } {
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 1 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32
//   } -> tensor<1x1024x64xf32>
//   %out = rock.store %result to %arg4 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
//   return %out : tensor<1x1024x64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_splitkv
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x1024xf32>, %[[v:.*]]: tensor<1x1024x64xf32>, %[[lse:.*]]: tensor<4x1024xf32>, %[[o:.*]]: tensor<4x1024x64xf32>)
// DISABLED-CHECK-SAME: grid_size = 128
// func.func @rock_attention_splitkv(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<4x1024xf32>, %arg4: tensor<4x1024x64xf32>) -> tensor<4x1024x64xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, grid_size = 1024 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.gridwise_attention(%[[q]], %[[k]], %[[v]], %[[o]], %[[lse]])
//   // DISABLED-CHECK-NEXT: splitKV = 4
//   %result = rock.attention{
//      qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
//      lse = %arg3 : tensor<4x1024xf32>
//      %arg4 = softmax(qk) * %arg2 : tensor<1x1024x64xf32> -> tensor<4x1024x64xf32>
//   } {
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 4 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32
//   } -> tensor<4x1024x64xf32>
//   %out = rock.store %result to %arg4 by set : tensor<4x1024x64xf32> -> tensor<4x1024x64xf32> to tensor<4x1024x64xf32>
//   return %out : tensor<4x1024x64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_splitkv_padding
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf32>, %[[k:.*]]: tensor<1x64x384xf32>, %[[v:.*]]: tensor<1x384x64xf32>, %[[lse:.*]]: tensor<8x1024xf32>, %[[o:.*]]: tensor<8x1024x64xf32>)
// DISABLED-CHECK-SAME: grid_size = 256
// func.func @rock_attention_splitkv_padding(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<8x1024xf32>, %arg4: tensor<8x1024x64xf32>) -> tensor<8x1024x64xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, grid_size = 1024 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK-DAG: %[[kPadding:.*]] = rock.transform %[[k]] by {{.*}} : tensor<1x64x384xf32> to tensor<1x64x512xf32>
//   // DISABLED-CHECK-DAG: %[[vPadding:.*]] = rock.transform %[[v]] by {{.*}} : tensor<1x384x64xf32> to tensor<1x512x64xf32>
//   // DISABLED-CHECK: rock.gridwise_attention(%[[q]], %[[kPadding]], %[[vPadding]], %[[o]], %[[lse]])
//   // DISABLED-CHECK-NEXT: splitKV = 8
//   %result = rock.attention{
//      qk = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x384xf32>
//      lse = %arg3 : tensor<8x1024xf32>
//      %arg4 = softmax(qk) * %arg2 : tensor<1x384x64xf32> -> tensor<8x1024x64xf32>
//   } {
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     splitKV = 8 : i32,
//     storeMethod = #rock<StoreMethod set>,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32
//   } -> tensor<8x1024x64xf32>
//   %out = rock.store %result to %arg4 by set : tensor<8x1024x64xf32> -> tensor<8x1024x64xf32> to tensor<8x1024x64xf32>
//   return %out : tensor<8x1024x64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_softmaxtype
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<1x64x1024xf16>, %[[k:.*]]: tensor<1x64x1024xf16>, %[[v:.*]]: tensor<1x1024x64xf16>, %[[lse:.*]]: tensor<1x1024xf16>, %[[o:.*]]: tensor<1x1024x64xf16>)
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 32 : i32
// func.func @rock_attention_softmaxtype(%arg0: tensor<1x64x1024xf16>, %arg1: tensor<1x64x1024xf16>, %arg2: tensor<1x1024x64xf16>, %arg3: tensor<1x1024xf16>, %arg4: tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.gridwise_attention(%[[q]], %[[k]], %[[v]], %[[o]], %[[lse]])
//   // DISABLED-CHECK: softmaxType = f32
//   %result = rock.attention{
//      qk = tr %arg0 * %arg1 : tensor<1x64x1024xf16>, tensor<1x64x1024xf16>
//      lse = %arg3 : tensor<1x1024xf16>
//      %arg4 = softmax(qk) * %arg2 : tensor<1x1024x64xf16> -> tensor<1x1024x64xf16>
//   } {
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     storeMethod = #rock<StoreMethod set>,
//     splitKV = 1 : i32,
//     numHeadsKV = 1 : i32, 
//     numHeadsQ = 1 : i32,
//     softmaxType = f32
//   } -> tensor<1x1024x64xf16>
//   %out = rock.store %result to %arg4 by set : tensor<1x1024x64xf16> -> tensor<1x1024x64xf16> to tensor<1x1024x64xf16>
//   return %out : tensor<1x1024x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_gemmelementwisegemm_simple
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x64x1024xf32>, %[[b:.*]]: tensor<1x64x1024xf32>, %[[c:.*]]: tensor<1x1024x64xf32>, %[[o:.*]]: tensor<1x1024x64xf32>)
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 32 : i32
// func.func @rock_gemmelementwisegemm_simple(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.gridwise_attention(%[[a]], %[[b]], %[[c]], %[[o]])
//   // DISABLED-CHECK-NEXT: enableSoftmax = false
//   %result = rock.gemm_elementwise_gemm{
//      ab = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
//      %arg3 = ab * %arg2 : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32>
//   } { 
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     storeMethod = #rock<StoreMethod set>
//   } -> tensor<1x1024x64xf32>
//   %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
//   return %out : tensor<1x1024x64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_gemmelementwisegemm_tr_padded
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x49x7xf32>, %[[b:.*]]: tensor<1x7x49xf32>, %[[c:.*]]: tensor<1x49x7xf32>, %[[o:.*]]: tensor<1x49x7xf32>)
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 2 : i32
// func.func @rock_gemmelementwisegemm_tr_padded(%arg0: tensor<1x49x7xf32>, %arg1: tensor<1x7x49xf32>, %arg2: tensor<1x49x7xf32>, %arg3: tensor<1x49x7xf32>) -> tensor<1x49x7xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK-DAG: %[[trA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x7x49xf32>
//   // DISABLED-CHECK-DAG: %[[paddedTrA:.*]] = rock.transform %[[trA]] by {{.*}} : tensor<1x7x49xf32> to tensor<1x8x64xf32>
//   // DISABLED-CHECK-DAG: %[[paddedB:.*]] = rock.transform %[[b]] by {{.*}} : tensor<1x7x49xf32> to tensor<1x8x64xf32>
//   // DISABLED-CHECK-DAG: %[[paddedC:.*]] = rock.transform %[[c]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x64x32xf32>
//   // DISABLED-CHECK-DAG: %[[paddedO:.*]] = rock.transform %[[o]] by {{.*}} : tensor<1x49x7xf32> to tensor<1x64x32xf32>
//   // DISABLED-CHECK: rock.gridwise_attention(%[[paddedTrA]], %[[paddedB]], %[[paddedC]], %[[paddedO]])
//   // DISABLED-CHECK-NEXT: enableSoftmax = false
//   // DISABLED-CHECK-SAME: prePadG0M = 49 : index, prePadG0N = 49 : index
//   %result = rock.gemm_elementwise_gemm{
//     ab = %arg0 * %arg1 : tensor<1x49x7xf32>, tensor<1x7x49xf32>
//     %arg3 = ab * %arg2 : tensor<1x49x7xf32> -> tensor<1x49x7xf32>
//   } { 
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1,
//     firstGemmIndices = array<i64: 0>,
//     storeMethod = #rock<StoreMethod set>
//   } -> tensor<1x49x7xf32>
//   %out = rock.store %result to %arg3 by set : tensor<1x49x7xf32> -> tensor<1x49x7xf32> to tensor<1x49x7xf32>
//   return %out : tensor<1x49x7xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_gemmelementwisegemm_splitk
// DISABLED-CHECK-SAME: (%[[aRaw:.*]]: tensor<1x64x1024xf32>, %[[bRaw:.*]]: tensor<1x64x1024xf32>, %[[cRaw:.*]]: tensor<1x1024x64xf32>, %[[oRaw:.*]]: tensor<1x1024x64xf32>
// DISABLED-CHECK-SAME: {rock.prefill = 0.000000e+00 : f32})
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 128 : i32
// func.func @rock_gemmelementwisegemm_splitk(%arg0: tensor<1x64x1024xf32>, %arg1: tensor<1x64x1024xf32>, %arg2: tensor<1x1024x64xf32>, %arg3: tensor<1x1024x64xf32>) -> tensor<1x1024x64xf32> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK-DAG: %[[bSplit:.*]] = rock.transform %[[bRaw]] by <affine_map<(d0, d1, d2, d3) -> (d0, d3, d1 * 256 + d2)> by [<PassThrough ["gemmG", "gemmK"] at [0, 3] -> ["gemmG", "gemmK"] at [0, 1]>, <Unmerge{4, 256} ["gemmNSplit", "gemmN"] at [1, 2] -> ["gemmN"] at [2]>] bounds = [1, 4, 256, 64] -> [1, 64, 1024]> : tensor<1x64x1024xf32> to tensor<1x4x256x64xf32>
//   // DISABLED-CHECK-DAG: %[[b:.*]] = rock.transform %[[bSplit]] by <affine_map<(d0, d1, d2) -> (0, d0, d2, d1)> by [<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 1]>, <PassThrough ["gemmN", "gemmK"] at [2, 1] -> ["gemmN", "gemmK"] at [2, 3]>] bounds = [4, 64, 256] -> [1, 4, 256, 64]> : tensor<1x4x256x64xf32> to tensor<4x64x256xf32>
//   // DISABLED-CHECK-DAG: %[[cSplit:.*]] = rock.transform %[[cRaw]] by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 256 + d2, d3)> by [<PassThrough ["gemmG", "gemmO"] at [0, 3] -> ["gemmG", "gemmO"] at [0, 2]>, <Unmerge{4, 256} ["gemmNSplit", "gemmN"] at [1, 2] -> ["gemmN"] at [1]>] bounds = [1, 4, 256, 64] -> [1, 1024, 64]> : tensor<1x1024x64xf32> to tensor<1x4x256x64xf32>
//   // DISABLED-CHECK-DAG: %[[c:.*]] = rock.transform %[[cSplit]] by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 1]>, <PassThrough ["gemmN", "gemmO"] at [1, 2] -> ["gemmN", "gemmO"] at [2, 3]>] bounds = [4, 256, 64] -> [1, 4, 256, 64]> : tensor<1x4x256x64xf32> to tensor<4x256x64xf32>
//   // DISABLED-CHECK-DAG: %[[aSplit:.*]] = rock.transform %[[aRaw]] by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)> by [<PassThrough ["gemmG", "gemmK", "gemmM"] at [0, 1, 2] -> ["gemmG", "gemmK", "gemmM"] at [0, 1, 2]>, <AddDim{4} ["gemmNSplit"] at [3] -> [] at []>] bounds = [1, 64, 1024, 4] -> [1, 64, 1024]> : tensor<1x64x1024xf32> to tensor<1x64x1024x4xf32>
//   // DISABLED-CHECK-DAG: %[[a:.*]] = rock.transform %[[aSplit]] by <affine_map<(d0, d1, d2) -> (0, d1, d2, d0)> by [<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 3]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [1, 2]>] bounds = [4, 64, 1024] -> [1, 64, 1024, 4]> : tensor<1x64x1024x4xf32> to tensor<4x64x1024xf32>
//   // DISABLED-CHECK-DAG: %[[oSplit:.*]] = rock.transform %[[oRaw]] by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)> by [<AddDim{4} ["gemmNSplit"] at [1] -> [] at []>, <PassThrough ["gemmG", "gemmM", "gemmO"] at [0, 2, 3] -> ["gemmG", "gemmM", "gemmO"] at [0, 1, 2]>] bounds = [1, 4, 1024, 64] -> [1, 1024, 64]> : tensor<1x1024x64xf32> to tensor<1x4x1024x64xf32>
//   // DISABLED-CHECK-DAG: %[[o:.*]] = rock.transform %[[oSplit]] by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["gemmG"] at [0] -> ["gemmG", "gemmNSplit"] at [0, 1]>, <PassThrough ["gemmM", "gemmO"] at [1, 2] -> ["gemmM", "gemmO"] at [2, 3]>] bounds = [4, 1024, 64] -> [1, 4, 1024, 64]> : tensor<1x4x1024x64xf32> to tensor<4x1024x64xf32>
//   // DISABLED-CHECK: rock.gridwise_attention(%[[a]], %[[b]], %[[c]], %[[o]])
//   // DISABLED-CHECK-NEXT: enableSoftmax = false
//   // DISABLED-CHECK-SAME: gridSize = 128 : i32
//   // DISABLED-CHECK-SAME: storeMethod = #rock<StoreMethod atomic_add>
//   %result = rock.gemm_elementwise_gemm{
//      ab = tr %arg0 * %arg1 : tensor<1x64x1024xf32>, tensor<1x64x1024xf32>
//      %arg3 = ab * %arg2 : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32>
//   } { 
//     params0 = #xldops_attn_params_g0,
//     params1 = #xldops_attn_params_g1_splitk,
//     firstGemmIndices = array<i64: 0>,
//     storeMethod = #rock<StoreMethod set>
//   } -> tensor<1x1024x64xf32>
//   %out = rock.store %result to %arg3 by set : tensor<1x1024x64xf32> -> tensor<1x1024x64xf32> to tensor<1x1024x64xf32>
//   return %out : tensor<1x1024x64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_gemmelementwisegemm_splitk_two_outputs
// DISABLED-CHECK-SAME: (%[[aRaw:.*]]: tensor<4096xf32>, %[[bRaw:.*]]: tensor<4096xf32>, %[[cRaw:.*]]: tensor<4096xf32>, %[[oRaw:.*]]: tensor<4096xf32> {rock.prefill = 0.000000e+00 : f32},
// DISABLED-CHECK-SAME: %[[reduceOut:.*]]: tensor<64xf32> {rock.prefill = 0.000000e+00 : f32})
// DISABLED-CHECK-SAME: rock.block_size = 64 : i32, grid_size = 8 : i32
// func.func @rock_gemmelementwisegemm_splitk_two_outputs(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096xf32>, %arg3: tensor<4096xf32>, %arg4: tensor<64xf32>) -> (tensor<4096xf32>, tensor<64xf32>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.block_size = 64 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : tensor<4096xf32> to tensor<1x64x64xf32>
//   %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : tensor<4096xf32> to tensor<1x64x64xf32>
//   %2 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : tensor<4096xf32> to tensor<1x64x64xf32>
//   %alloc = tensor.empty() : tensor<1x64x64xf32>
// 
//   // DISABLED-CHECK-DAG: %[[aReshaped:.*]] = rock.transform %[[aRaw]] {{.*}} tensor<4096xf32> to tensor<1x64x64xf32>
//   // DISABLED-CHECK-DAG: %[[bReshaped:.*]] = rock.transform %[[bRaw]] {{.*}} tensor<4096xf32> to tensor<1x64x64xf32>
//   // DISABLED-CHECK-DAG: %[[cReshaped:.*]] = rock.transform %[[cRaw]] {{.*}} tensor<4096xf32> to tensor<1x64x64xf32>
// 
//   // DISABLED-CHECK-DAG: %[[gemmOut:.*]] = tensor.empty() : tensor<1x64x64xf32>
// 
//   // DISABLED-CHECK-DAG: %[[bSplit:.*]] = rock.transform %[[bReshaped]] {{.*}} tensor<1x64x64xf32> to tensor<1x4x16x64xf32>
//   // DISABLED-CHECK-DAG: %[[b:.*]] = rock.transform %[[bSplit]] {{.*}} tensor<1x4x16x64xf32> to tensor<4x64x16xf32>
//   // DISABLED-CHECK-DAG: %[[bPad:.*]] = rock.transform %[[b]] {{.*}} tensor<4x64x16xf32> to tensor<4x64x64xf32>
// 
//   // DISABLED-CHECK-DAG: %[[cSplit:.*]] = rock.transform %[[cReshaped]] {{.*}} tensor<1x64x64xf32> to tensor<1x4x16x64xf32>
//   // DISABLED-CHECK-DAG: %[[c:.*]] = rock.transform %[[cSplit]] {{.*}} tensor<1x4x16x64xf32> to tensor<4x16x64xf32>
//   // DISABLED-CHECK-DAG: %[[cPad:.*]] = rock.transform %[[c]] {{.*}} tensor<4x16x64xf32> to tensor<4x64x128xf32>
// 
//   // DISABLED-CHECK-DAG: %[[aSplit:.*]] = rock.transform %[[aReshaped]] {{.*}} tensor<1x64x64xf32> to tensor<1x64x64xf32>
//   // DISABLED-CHECK-DAG: %[[a:.*]] = rock.transform %[[aSplit]] {{.*}} tensor<1x64x64xf32> to tensor<1x64x64x4xf32>
//   // DISABLED-CHECK-DAG: %[[aPad:.*]] = rock.transform %[[a]] {{.*}} tensor<1x64x64x4xf32> to tensor<4x64x64xf32>
// 
//   // DISABLED-CHECK-DAG: %[[oSplit:.*]] = rock.transform %[[gemmOut]] {{.*}} tensor<1x64x64xf32> to tensor<1x4x64x64xf32>
//   // DISABLED-CHECK-DAG: %[[o:.*]] = rock.transform %[[oSplit]] {{.*}} tensor<1x4x64x64xf32> to tensor<4x64x64xf32>
//   // DISABLED-CHECK-DAG: %[[oPad:.*]] = rock.transform %[[o]] {{.*}} tensor<4x64x64xf32> to tensor<4x64x128xf32>
// 
//   // DISABLED-CHECK: rock.gridwise_attention(%[[aPad]], %[[bPad]], %[[cPad]], %[[oPad]])
//   // DISABLED-CHECK-NEXT: enableSoftmax = false
//   // DISABLED-CHECK-SAME: gridSize = 8 : i32
//   // DISABLED-CHECK-SAME: storeMethod = #rock<StoreMethod atomic_add>
//   %result = rock.gemm_elementwise_gemm{
//    ab = %2 * %1 : tensor<1x64x64xf32>, tensor<1x64x64xf32>
//    ab = elementwise {
//   ^bb0(%arg5: tensor<1x64x64xf32>, %arg6: tensor<1x64x64xf32>):
//     rock.yield
//   }
//    %alloc = ab * %0 : tensor<1x64x64xf32> -> tensor<1x64x64xf32>
//   } {firstGemmIndices = array<i64: 0>, params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 32, kPerBlock = 64, kpack = 4, numWaves = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 32, kPerBlock = 64, kpack = 4, numWaves = 8, matrixInstrNonkdim = 0, splitKFactor = 4, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, perf_config = "attn:v2:64,128,32,16,32,16,4,4,1,2,1", storeMethod = #rock<StoreMethod set>} -> tensor<1x64x64xf32>
//   %3 = rock.transform %result by <affine_map<(d0) -> (0, d0 floordiv 64, d0 mod 64)> by [<Merge{1, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [4096] -> [1, 64, 64]> : tensor<1x64x64xf32> to tensor<4096xf32>
//   %alloc_0 = tensor.empty() : tensor<1x64x1xf32>
// 
//   // DISABLED-CHECK-DAG: %[[outCopy:.*]] = rock.transform %[[gemmOut]] {{.*}} tensor<1x64x64xf32> to tensor<4096xf32>
//   // DISABLED-CHECK-DAG: %[[allocReduce:.*]] = tensor.empty() : tensor<1x64x1xf32>
//   // DISABLED-CHECK-DAG: rock.reduce  sum %[[gemmOut]] into %[[allocReduce]] {axis = 2 : index, blockSize = 256 : i32, gridSize = 16 : i32} : tensor<1x64x64xf32> into tensor<1x64x1xf32>
// 
//   // DISABLED-CHECK-DAG: %[[reduceCopy:.*]] = rock.transform %[[allocReduce]] by <affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 64, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [1, 64, 1]> : tensor<1x64x1xf32> to tensor<64xf32>
//   // DISABLED-CHECK-DAG: rock.store %[[outCopy]] to %[[oRaw]] by set : tensor<4096xf32> -> tensor<4096xf32> to tensor<4096xf32>
//   // DISABLED-CHECK-DAG: rock.store %[[reduceCopy]] to %[[reduceOut]] by set : tensor<64xf32> -> tensor<64xf32> to tensor<64xf32>
//   
//   %reduce_result = rock.reduce sum %result into %alloc_0 {axis = 2 : index, blockSize = 256 : i32, gridSize = 16 : i32} : tensor<1x64x64xf32> into tensor<1x64x1xf32>
//   %4 = rock.transform %reduce_result by <affine_map<(d0) -> (0, d0, 0)> by [<Merge{1, 64, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [1, 64, 1]> : tensor<1x64x1xf32> to tensor<64xf32>
//   %out1 = rock.store %3 to %arg3 by set : tensor<4096xf32> -> tensor<4096xf32> to tensor<4096xf32>
//   %out2 = rock.store %4 to %arg4 by set : tensor<64xf32> -> tensor<64xf32> to tensor<64xf32>
//   return %out1, %out2 : tensor<4096xf32>, tensor<64xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_gqa
// DISABLED-CHECK-SAME: (%[[q:.*]]: tensor<64x1x128xf16>, %[[k:.*]]: tensor<8x128x8192xf16>, %[[v:.*]]: tensor<8x8192x128xf16>, %[[lse:.*]]: tensor<256x1xf16>, %[[o:.*]]: tensor<256x1x128xf16>)
// DISABLED-CHECK-SAME: grid_size = 32
// func.func @rock_attention_gqa(%arg0: tensor<64x1x128xf16>, %arg1: tensor<8x128x8192xf16>, %arg2: tensor<8x8192x128xf16>, %arg3: tensor<256x1xf16>, %arg4: tensor<256x1x128xf16>) -> tensor<256x1x128xf16> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.block_size = 64 : i32, grid_size = 1024 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK-DAG: %[[qNormalized:.*]] = rock.transform %[[q]] by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by {{.*}} tensor<64x1x128xf16> to tensor<64x128x1xf16>
//   // DISABLED-CHECK-DAG: %[[qExtractNumRepeats:.+]] = rock.transform %[[qNormalized]] by <affine_map<(d0, d1, d2, d3) -> (d0 * 8 + d3, d1, d2)> by {{.*}} tensor<64x128x1xf16> to tensor<8x128x1x8xf16>
//   // DISABLED-CHECK-DAG: %[[qMoveToSeqLen:.*]] = rock.transform %[[qExtractNumRepeats]] by <affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)> by {{.*}} tensor<8x128x1x8xf16> to tensor<8x128x8xf16>
//   // DISABLED-CHECK-DAG: %[[qPad:.+]] = rock.transform %[[qMoveToSeqLen]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by {{.*}} tensor<8x128x8xf16> to tensor<8x128x32xf16>
//   
//   // DISABLED-CHECK-DAG: %[[outUnmerge:.*]] = rock.transform %[[o]] by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 8 + d3) * 4 + d1, d2, d4)> by {{.*}} tensor<256x1x128xf16> to tensor<8x4x1x8x128xf16>
//   // DISABLED-CHECK-DAG: %[[outMerge:.*]] = rock.transform %[[outUnmerge]] by <affine_map<(d0, d1, d2) -> (d0 floordiv 4, d0 mod 4, 0, d1, d2)> by {{.*}} tensor<8x4x1x8x128xf16> to tensor<32x8x128xf16>
//   // DISABLED-CHECK-DAG: %[[outPad:.*]] = rock.transform %[[outMerge]] by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by {{.*}} tensor<32x8x128xf16> to tensor<32x32x128xf16>
//   
//   // DISABLED-CHECK-DAG: %[[lseUnmerge:.*]] = rock.transform %[[lse]] by <affine_map<(d0, d1, d2, d3) -> ((d0 * 8 + d3) * 4 + d1, d2)> by {{.*}} tensor<256x1xf16> to tensor<8x4x1x8xf16>
//   // DISABLED-CHECK-DAG: %[[lseMerge:.*]] = rock.transform %[[lseUnmerge]] by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, 0, d1)> by {{.*}} tensor<8x4x1x8xf16> to tensor<32x8xf16>
//   // DISABLED-CHECK-DAG: %[[lsePad:.*]] = rock.transform %[[lseMerge]] by <affine_map<(d0, d1) -> (d0, d1)> by {{.*}} tensor<32x8xf16> to tensor<32x32xf16>
// 
//   // DISABLED-CHECK: rock.gridwise_attention(%[[qPad]], %[[k]], %[[v]], %[[outPad]], %[[lsePad]])
//   // DISABLED-CHECK-NEXT: splitKV = 4
//   %result = rock.attention{
//      qk = %arg0 * %arg1 : tensor<64x1x128xf16>, tensor<8x128x8192xf16>
//      lse = %arg3 : tensor<256x1xf16>
//      qk = elementwise {
//     ^bb0(%arg5: tensor<64x1x8192xf16>, %arg6: tensor<64x1x8192xf16>):
//       rock.yield
//     }
//      %arg4 = softmax(qk) * %arg2 : tensor<8x8192x128xf16> -> tensor<256x1x128xf16>
//   } {firstGemmIndices = array<i64: 0>, numHeadsKV = 8 : i32, numHeadsQ = 64 : i32, params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>, softmaxType = f32, splitKV = 4 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<256x1x128xf16>
//   %out = rock.store %result to %arg4 by set : tensor<256x1x128xf16> -> tensor<256x1x128xf16> to tensor<256x1x128xf16>
//   return %out : tensor<256x1x128xf16>
// }

// -----

// Tests for scaled GEMM 

// TODO(roctriton): Scaled gemm tests need rework
// DISABLED-CHECK-LABEL: func.func @gemm_scaled_fp4_already_f8e8m0
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf4E2M1FN>, %[[b:.*]]: tensor<1x72x512xf4E2M1FN>, %[[c:.*]]: tensor<1x128x512xf32>, %[[scaleA:.*]]: tensor<1x128x72xf8E8M0FNU>, %[[scaleB:.*]]: tensor<1x72x512xf8E8M0FNU>)
// DISABLED-CHECK-SAME: grid_size = 16 : i32
// func.func @gemm_scaled_fp4_already_f8e8m0(%a: tensor<1x72x128xf4E2M1FN>, %b: tensor<1x72x512xf4E2M1FN>, %c: tensor<1x128x512xf32>, %scaleA: tensor<1x128x72xf8E8M0FNU>, %scaleB: tensor<1x72x512xf8E8M0FNU>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
//   // DISABLED-CHECK: %[[normalizeScaleA:.*]] = rock.transform %[[scaleA]] by {{.*}} : tensor<1x128x72xf8E8M0FNU> to tensor<1x72x128xf8E8M0FNU{{.*}}>
//   // DISABLED-CHECK: rock.gridwise_gemm(%[[a]], %[[b]], %[[c]], %[[normalizeScaleA]], %[[scaleB]])
//   %result = rock.gemm tr %a scaled by %scaleA * %b scaled by %scaleB {
//     derivedBlockSize = 256 : i32,
//     gridSize = 16 : i32,
//     params = #xdlops_gemm_params0
//   } : tensor<1x72x128xf4E2M1FN> scaled by tensor<1x128x72xf8E8M0FNU> * tensor<1x72x512xf4E2M1FN> scaled by tensor<1x72x512xf8E8M0FNU> -> tensor<1x128x512xf32>
//   %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
//   func.return %out : tensor<1x128x512xf32>
// }

// TODO(roctriton): Scaled gemm tests need rework
// DISABLED-CHECK-LABEL: func.func @gemm_scaled_fp4_with_padding
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x1x1xf4E2M1FN>, %[[b:.*]]: tensor<1x1x1xf4E2M1FN>, %[[c:.*]]: tensor<1x1x1xf32>, %[[scaleA:.*]]: tensor<1x1x1xf8E8M0FNU>, %[[scaleB:.*]]: tensor<1x1x1xf8E8M0FNU>)
// DISABLED-CHECK-SAME: grid_size = 1
// func.func @gemm_scaled_fp4_with_padding(%a: tensor<1x1x1xf4E2M1FN>, %b: tensor<1x1x1xf4E2M1FN>, %c: tensor<1x1x1xf32>, %scaleA: tensor<1x1x1xf8E8M0FNU>, %scaleB: tensor<1x1x1xf8E8M0FNU>) -> tensor<1x1x1xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
//   // DISABLED-CHECK-DAG: %[[padA:.*]] = rock.transform %[[a]] by {{.*}} : tensor<1x1x1xf4E2M1FN> to tensor<1x8x64xf4E2M1FN{{.*}}>
//   // DISABLED-CHECK-DAG: %[[padB:.*]] = rock.transform %[[b]] by {{.*}} : tensor<1x1x1xf4E2M1FN> to tensor<1x8x64xf4E2M1FN{{.*}}>
//   // DISABLED-CHECK-DAG: %[[padC:.*]] = rock.transform %[[c]] by {{.*}} : tensor<1x1x1xf32> to tensor<1x64x64xf32{{.*}}>
//   // DISABLED-CHECK-DAG: %[[padScaleA:.*]] = rock.transform %[[scaleA]] by {{.*}} : tensor<1x1x1xf8E8M0FNU> to tensor<1x8x64xf8E8M0FNU{{.*}}>
//   // DISABLED-CHECK-DAG: %[[padScaleB:.*]] = rock.transform %[[scaleB]] by {{.*}} : tensor<1x1x1xf8E8M0FNU> to tensor<1x8x64xf8E8M0FNU{{.*}}>
//   // DISABLED-CHECK: rock.gridwise_gemm(%[[padA]], %[[padB]], %[[padC]], %[[padScaleA]], %[[padScaleB]])
//   %result = rock.gemm tr %a scaled by %scaleA * %b scaled by %scaleB {
//     derivedBlockSize = 256 : i32,
//     gridSize = 1 : i32,
//     params = #xdlops_gemm_params0
//   } : tensor<1x1x1xf4E2M1FN> scaled by tensor<1x1x1xf8E8M0FNU> * tensor<1x1x1xf4E2M1FN> scaled by tensor<1x1x1xf8E8M0FNU> -> tensor<1x1x1xf32>
//   %out = rock.store %result to %c by set : tensor<1x1x1xf32> -> tensor<1x1x1xf32> to tensor<1x1x1xf32>
//   func.return %out : tensor<1x1x1xf32>
// }

// TODO(roctriton): Scaled gemm tests need rework
// DISABLED-CHECK-LABEL: func.func @gemm_scaled_fp4_transposed
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x128x72xf4E2M1FN>, %[[b:.*]]: tensor<1x512x72xf4E2M1FN>, %[[c:.*]]: tensor<1x512x128xf32>, %[[scaleA:.*]]: tensor<1x72x128xf8E8M0FNU>, %[[scaleB:.*]]: tensor<1x72x512xf8E8M0FNU>)
// DISABLED-CHECK-SAME: grid_size = 16 : i32
// func.func @gemm_scaled_fp4_transposed(%a: tensor<1x128x72xf4E2M1FN>, %b: tensor<1x512x72xf4E2M1FN>, %c: tensor<1x512x128xf32>, %scaleA: tensor<1x72x128xf8E8M0FNU>, %scaleB: tensor<1x72x512xf8E8M0FNU>) -> tensor<1x512x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
//   // DISABLED-CHECK-DAG: %[[normalizeA:.*]] = rock.transform %[[a]] {{.*}} : tensor<1x128x72xf4E2M1FN> to tensor<1x72x128xf4E2M1FN{{.*}}>
//   // DISABLED-CHECK-DAG: %[[normalizeB:.*]] = rock.transform %[[b]] {{.*}} : tensor<1x512x72xf4E2M1FN> to tensor<1x72x512xf4E2M1FN{{.*}}>
//   // DISABLED-CHECK-DAG: %[[normalizeC:.*]] = rock.transform %[[c]] {{.*}} : tensor<1x512x128xf32> to tensor<1x128x512xf32{{.*}}>
//   // DISABLED-CHECK: rock.gridwise_gemm(%[[normalizeA]], %[[normalizeB]], %[[normalizeC]], %[[scaleA]], %[[scaleB]])
//   %result = rock.gemm tr %c = %a scaled by tr %scaleA * tr %b scaled by %scaleB {
//     derivedBlockSize = 256 : i32,
//     gridSize = 16 : i32,
//     params = #xdlops_gemm_params0
//   } : tensor<1x128x72xf4E2M1FN> scaled by tensor<1x72x128xf8E8M0FNU> * tensor<1x512x72xf4E2M1FN> scaled by tensor<1x72x512xf8E8M0FNU> -> tensor<1x512x128xf32>
//   %out = rock.store %result to %c by set : tensor<1x512x128xf32> -> tensor<1x512x128xf32> to tensor<1x512x128xf32>
//   func.return %out : tensor<1x512x128xf32>
// }

// -----

// TODO(roctriton): Scaled gemm tests need rework
// DISABLED-CHECK-LABEL: func.func @gemm_scaled_fp4_with_f32_scales
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf4E2M1FN>, %[[b:.*]]: tensor<1x72x512xf4E2M1FN>, %[[c:.*]]: tensor<1x128x512xf32>, %[[scaleA:.*]]: tensor<1x128x72xf32>, %[[scaleB:.*]]: tensor<1x72x512xf32>)
// DISABLED-CHECK-SAME: grid_size = 16 : i32
// func.func @gemm_scaled_fp4_with_f32_scales(%a: tensor<1x72x128xf4E2M1FN>, %b: tensor<1x72x512xf4E2M1FN>, %c: tensor<1x128x512xf32>, %scaleA: tensor<1x128x72xf32>, %scaleB: tensor<1x72x512xf32>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
//   // DISABLED-CHECK: %[[normalizeScaleA:.*]] = rock.transform %[[scaleA]] by {{.*}} : tensor<1x128x72xf32> to tensor<1x72x128xf32{{.*}}>
//   // DISABLED-CHECK: %[[allocScaleA:.*]] = tensor.empty() : tensor<1x72x128xf8E8M0FNU>
//   // DISABLED-CHECK: linalg.generic {{{.*}}} ins(%[[normalizeScaleA]] : tensor<1x72x128xf32{{.*}}>) outs(%[[allocScaleA]] : tensor<1x72x128xf8E8M0FNU>)
//   // DISABLED-CHECK: %[[allocScaleB:.*]] = tensor.empty() : tensor<1x72x512xf8E8M0FNU>
//   // DISABLED-CHECK: linalg.generic {{{.*}}} ins(%[[scaleB]] : tensor<1x72x512xf32>) outs(%[[allocScaleB]] : tensor<1x72x512xf8E8M0FNU>)
//   // DISABLED-CHECK: rock.gridwise_gemm(%[[a]], %[[b]], %[[c]], %[[allocScaleA]], %[[allocScaleB]])
//   %result = rock.gemm tr %a scaled by %scaleA * %b scaled by %scaleB {
//     derivedBlockSize = 256 : i32,
//     gridSize = 16 : i32,
//     params = #xdlops_gemm_params0
//   } : tensor<1x72x128xf4E2M1FN> scaled by tensor<1x128x72xf32> * tensor<1x72x512xf4E2M1FN> scaled by tensor<1x72x512xf32> -> tensor<1x128x512xf32>
//   %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
//   func.return %out : tensor<1x128x512xf32>
// }

// TODO(roctriton): Scaled gemm tests need rework
// DISABLED-CHECK-LABEL: func.func @gemm_scaled_fp4_splitk
// DISABLED-CHECK-SAME: (%[[a:.*]]: tensor<1x72x128xf4E2M1FN>, %[[b:.*]]: tensor<1x72x512xf4E2M1FN>, %[[c:.*]]: tensor<1x128x512xf32> {rock.prefill = 0.000000e+00 : f32}, %[[scaleA:.*]]: tensor<1x128x72xf8E8M0FNU>, %[[scaleB:.*]]: tensor<1x72x512xf8E8M0FNU>)
// DISABLED-CHECK-SAME: grid_size = 32 : i32
// func.func @gemm_scaled_fp4_splitk(%a: tensor<1x72x128xf4E2M1FN>, %b: tensor<1x72x512xf4E2M1FN>, %c: tensor<1x128x512xf32>, %scaleA: tensor<1x128x72xf8E8M0FNU>, %scaleB: tensor<1x72x512xf8E8M0FNU>) -> tensor<1x128x512xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
//   // Transpose scaleA from MxK to KxM
//   // DISABLED-CHECK: rock.transform %[[scaleA]] by {{.*}} : tensor<1x128x72xf8E8M0FNU> to tensor<1x72x128xf8E8M0FNU>
//   
//   // Padding K from 72 to 96
//   // DISABLED-CHECK-DAG: rock.transform %[[a]] by {{.*}} : tensor<1x72x128xf4E2M1FN> to tensor<1x96x128xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform %[[b]] by {{.*}} : tensor<1x72x512xf4E2M1FN> to tensor<1x96x512xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x72x128xf8E8M0FNU> to tensor<1x96x128xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform %[[scaleB]] by {{.*}} : tensor<1x72x512xf8E8M0FNU> to tensor<1x96x512xf8E8M0FNU>
//   
//   // Split K into 2 parts (96/2 = 48 per split)
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x96x128xf4E2M1FN> to tensor<1x2x48x128xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x2x48x128xf4E2M1FN> to tensor<2x48x128xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x96x512xf4E2M1FN> to tensor<1x2x48x512xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x2x48x512xf4E2M1FN> to tensor<2x48x512xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x96x128xf8E8M0FNU> to tensor<1x2x48x128xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x2x48x128xf8E8M0FNU> to tensor<2x48x128xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x96x512xf8E8M0FNU> to tensor<1x2x48x512xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x2x48x512xf8E8M0FNU> to tensor<2x48x512xf8E8M0FNU>
//   
//   // Split and merge C
//   // DISABLED-CHECK-DAG: rock.transform %[[c]] by {{.*}} : tensor<1x128x512xf32> to tensor<1x2x128x512xf32>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<1x2x128x512xf32> to tensor<2x128x512xf32>
//   
//   // DISABLED-CHECK: rock.gridwise_gemm({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) storeMethod( atomic_add) {{.*}} : tensor<2x48x128xf4E2M1FN>, tensor<2x48x512xf4E2M1FN>, tensor<2x128x512xf32>, tensor<2x48x128xf8E8M0FNU>, tensor<2x48x512xf8E8M0FNU>
//   %result = rock.gemm tr %a scaled by %scaleA * %b scaled by %scaleB {
//     derivedBlockSize = 256 : i32,
//     gridSize = 16 : i32,
//     params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   } : tensor<1x72x128xf4E2M1FN> scaled by tensor<1x128x72xf8E8M0FNU> * tensor<1x72x512xf4E2M1FN> scaled by tensor<1x72x512xf8E8M0FNU> -> tensor<1x128x512xf32>
//   %out = rock.store %result to %c by set : tensor<1x128x512xf32> -> tensor<1x128x512xf32> to tensor<1x128x512xf32>
//   func.return %out : tensor<1x128x512xf32>
// }

// TODO(roctriton): Scaled gemm tests need rework
// DISABLED-CHECK-LABEL: func.func @gemm_scaled_fp4_splitk_odd
// DISABLED-CHECK-SAME: (%[[aRaw:.*]]: tensor<589824xf4E2M1FN>, %[[bRaw:.*]]: tensor<589824xf4E2M1FN>, %[[cRaw:.*]]: tensor<196608xf32> {rock.prefill = 0.000000e+00 : f32}, %[[scaleARaw:.*]]: tensor<18432xf8E8M0FNU>, %[[scaleBRaw:.*]]: tensor<18432xf8E8M0FNU>)
// DISABLED-CHECK-SAME: grid_size = 240 : i32
// func.func @gemm_scaled_fp4_splitk_odd(%arg0: tensor<589824xf4E2M1FN>, %arg1: tensor<589824xf4E2M1FN>, %arg2: tensor<196608xf32>, %arg3: tensor<18432xf8E8M0FNU>, %arg4: tensor<18432xf8E8M0FNU>) -> tensor<196608xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
//   // DISABLED-CHECK-DAG: rock.transform %[[aRaw]] by {{.*}} : tensor<589824xf4E2M1FN> to tensor<3x256x768xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform %[[bRaw]] by {{.*}} : tensor<589824xf4E2M1FN> to tensor<3x768x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform %[[cRaw]] by {{.*}} : tensor<196608xf32> to tensor<3x256x256xf32>
//   // DISABLED-CHECK-DAG: rock.transform %[[scaleARaw]] by {{.*}} : tensor<18432xf8E8M0FNU> to tensor<3x256x24xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform %[[scaleBRaw]] by {{.*}} : tensor<18432xf8E8M0FNU> to tensor<3x24x256xf8E8M0FNU>
//   
//   // Scale broadcasting through AddDim, Broadcast, and Merge transformations
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x256x24xf8E8M0FNU> to tensor<3x256x24x1xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x256x24x1xf8E8M0FNU> to tensor<3x256x24x32xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x256x24x32xf8E8M0FNU> to tensor<3x256x768xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x24x256xf8E8M0FNU> to tensor<3x24x1x256xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x24x1x256xf8E8M0FNU> to tensor<3x24x32x256xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x24x32x256xf8E8M0FNU> to tensor<3x768x256xf8E8M0FNU>
//   
//   // Transpose A and scaleA from MxK to KxM
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x256x768xf4E2M1FN> to tensor<3x768x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x256x768xf8E8M0FNU> to tensor<3x768x256xf8E8M0FNU>
//   
//   // Padding K from 768 to 800
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x768x256xf4E2M1FN> to tensor<3x800x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x768x256xf4E2M1FN> to tensor<3x800x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x768x256xf8E8M0FNU> to tensor<3x800x256xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x768x256xf8E8M0FNU> to tensor<3x800x256xf8E8M0FNU>
//   
//   // Split K into 5 parts (800/5 = 160 per split)
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x800x256xf4E2M1FN> to tensor<3x5x160x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x5x160x256xf4E2M1FN> to tensor<15x160x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x800x256xf4E2M1FN> to tensor<3x5x160x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x5x160x256xf4E2M1FN> to tensor<15x160x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x800x256xf8E8M0FNU> to tensor<3x5x160x256xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x5x160x256xf8E8M0FNU> to tensor<15x160x256xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x800x256xf8E8M0FNU> to tensor<3x5x160x256xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x5x160x256xf8E8M0FNU> to tensor<15x160x256xf8E8M0FNU>
//   
//   // Split and merge C
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x256x256xf32> to tensor<3x5x256x256xf32>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<3x5x256x256xf32> to tensor<15x256x256xf32>
//   
//   // Final padding K from 160 to 512
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<15x160x256xf4E2M1FN> to tensor<15x512x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<15x160x256xf4E2M1FN> to tensor<15x512x256xf4E2M1FN>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<15x160x256xf8E8M0FNU> to tensor<15x512x256xf8E8M0FNU>
//   // DISABLED-CHECK-DAG: rock.transform {{.*}} : tensor<15x160x256xf8E8M0FNU> to tensor<15x512x256xf8E8M0FNU>
//   
//   // DISABLED-CHECK: rock.gridwise_gemm({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) storeMethod( atomic_add) {{.*}} : tensor<15x512x256xf4E2M1FN>, tensor<15x512x256xf4E2M1FN>, tensor<15x256x256xf32>, tensor<15x512x256xf8E8M0FNU>, tensor<15x512x256xf8E8M0FNU>
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 768 + d2)> by [<Unmerge{3, 256, 768} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 768] -> [589824]> : tensor<589824xf4E2M1FN> to tensor<3x256x768xf4E2M1FN>
//   %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 256 + d2)> by [<Unmerge{3, 768, 256} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 256] -> [589824]> : tensor<589824xf4E2M1FN> to tensor<3x768x256xf4E2M1FN>
//   %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 256 + d2)> by [<Unmerge{3, 256, 256} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 256] -> [196608]> : tensor<196608xf32> to tensor<3x256x256xf32>
//   %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 24 + d2)> by [<Unmerge{3, 256, 24} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 256, 24] -> [18432]> : tensor<18432xf8E8M0FNU> to tensor<3x256x24xf8E8M0FNU>
//   %4 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> ((d0 * 24 + d1) * 256 + d2)> by [<Unmerge{3, 24, 256} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 24, 256] -> [18432]> : tensor<18432xf8E8M0FNU> to tensor<3x24x256xf8E8M0FNU>
//   %5 = rock.transform %3 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)> by [<AddDim{1} ["block"] at [3] -> [] at []>, <PassThrough ["g", "m", "kScale"] at [0, 1, 2] -> ["g", "m", "kScale"] at [0, 1, 2]>] bounds = [3, 256, 24, 1] -> [3, 256, 24]> : tensor<3x256x24xf8E8M0FNU> to tensor<3x256x24x1xf8E8M0FNU>
//   %6 = rock.transform %5 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)> by [<Broadcast{1} ["block"] at [3] -> ["block"] at [3]>, <PassThrough ["g", "m", "kScale"] at [0, 1, 2] -> ["g", "m", "kScale"] at [0, 1, 2]>] bounds = [3, 256, 24, 32] -> [3, 256, 24, 1]> : tensor<3x256x24x1xf8E8M0FNU> to tensor<3x256x24x32xf8E8M0FNU>
//   %7 = rock.transform %6 by <affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 32, d2 mod 32)> by [<Merge{24, 32} ["k"] at [2] -> ["kScale", "block"] at [2, 3]>, <PassThrough ["g", "m"] at [0, 1] -> ["g", "m"] at [0, 1]>] bounds = [3, 256, 768] -> [3, 256, 24, 32]> : tensor<3x256x24x32xf8E8M0FNU> to tensor<3x256x768xf8E8M0FNU>
//   %8 = rock.transform %4 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)> by [<AddDim{1} ["block"] at [2] -> [] at []>, <PassThrough ["g", "kScale", "n"] at [0, 1, 3] -> ["g", "kScale", "n"] at [0, 1, 2]>] bounds = [3, 24, 1, 256] -> [3, 24, 256]> : tensor<3x24x256xf8E8M0FNU> to tensor<3x24x1x256xf8E8M0FNU>
//   %9 = rock.transform %8 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)> by [<Broadcast{1} ["block"] at [2] -> ["block"] at [2]>, <PassThrough ["g", "kScale", "n"] at [0, 1, 3] -> ["g", "kScale", "n"] at [0, 1, 3]>] bounds = [3, 24, 32, 256] -> [3, 24, 1, 256]> : tensor<3x24x1x256xf8E8M0FNU> to tensor<3x24x32x256xf8E8M0FNU>
//   %10 = rock.transform %9 by <affine_map<(d0, d1, d2) -> (d0, d1 floordiv 32, d1 mod 32, d2)> by [<PassThrough ["g", "n"] at [0, 2] -> ["g", "n"] at [0, 3]>, <Merge{24, 32} ["k"] at [1] -> ["kScale", "block"] at [1, 2]>] bounds = [3, 768, 256] -> [3, 24, 32, 256]> : tensor<3x24x32x256xf8E8M0FNU> to tensor<3x768x256xf8E8M0FNU>
//   %result = rock.gemm %0 scaled by %7 * %1 scaled by %10 {
//     derivedBlockSize = 256 : i32,
//     gridSize = 12 : i32,
//     params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 512, kpack = 32, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 5, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   } : tensor<3x256x768xf4E2M1FN> scaled by tensor<3x256x768xf8E8M0FNU> * tensor<3x768x256xf4E2M1FN> scaled by tensor<3x768x256xf8E8M0FNU> -> tensor<3x256x256xf32>
//   %result_flat = rock.transform %result by <affine_map<(d0) -> (d0 floordiv 65536, (d0 mod 65536) floordiv 256, d0 mod 256)> by [<Merge{3, 256, 256} ["raw"] at [0] -> ["g", "m", "n"] at [0, 1, 2]>] bounds = [196608] -> [3, 256, 256]> : tensor<3x256x256xf32> to tensor<196608xf32>
//   %out = rock.store %result_flat to %arg2 by set : tensor<196608xf32> -> tensor<196608xf32> to tensor<196608xf32>
//   func.return %out : tensor<196608xf32>
// }
