// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s

func.func @rock_conv(%filter : tensor<?x?x?x?x?xf32>, %input : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32> -> tensor<?x?x?x?x?xf32>
  return %result : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: func.func @rock_conv
// CHECK-NEXT: rock.conv

func.func @rock_conv_f16(%filter : tensor<?x?x?x?x?xf16>, %input : tensor<?x?x?x?x?xf16>) -> tensor<?x?x?x?x?xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g" ,"k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : tensor<?x?x?x?x?xf16>, tensor<?x?x?x?x?xf16> -> tensor<?x?x?x?x?xf16>
  return %result : tensor<?x?x?x?x?xf16>
}
// CHECK-LABEL: func.func @rock_conv_f16
// CHECK-NEXT: rock.conv

func.func @rock_conv_fp8_mixed(%filter : tensor<?x?x?x?x?xf8E4M3FNUZ>, %input : tensor<?x?x?x?x?xf8E5M2FNUZ>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : tensor<?x?x?x?x?xf8E4M3FNUZ>, tensor<?x?x?x?x?xf8E5M2FNUZ> -> tensor<?x?x?x?x?xf32>
  return %result : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: func.func @rock_conv_fp8_mixed
// CHECK-NEXT: rock.conv

func.func @rock_conv_fp8_mixed_ocp(%filter : tensor<?x?x?x?x?xf8E4M3FN>, %input : tensor<?x?x?x?x?xf8E5M2>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : tensor<?x?x?x?x?xf8E4M3FN>, tensor<?x?x?x?x?xf8E5M2> -> tensor<?x?x?x?x?xf32>
  return %result : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: func.func @rock_conv_fp8_mixed
// CHECK-NEXT: rock.conv

func.func @rock_conv_bwd_data(%filter : tensor<?x?x?x?x?xf32>, %output : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %result = rock.conv_bwd_data(%filter, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32> -> tensor<?x?x?x?x?xf32>
  return %result : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: func.func @rock_conv_bwd_data
// CHECK-NEXT: rock.conv_bwd_data

func.func @rock_conv_bwd_data_f16(%filter : tensor<?x?x?x?x?xf16>, %output : tensor<?x?x?x?x?xf16>) -> tensor<?x?x?x?x?xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %result = rock.conv_bwd_data(%filter, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : tensor<?x?x?x?x?xf16>, tensor<?x?x?x?x?xf16> -> tensor<?x?x?x?x?xf16>
  return %result : tensor<?x?x?x?x?xf16>
}
// CHECK-LABEL: func.func @rock_conv_bwd_data_f16
// CHECK-NEXT: rock.conv_bwd_data

func.func @rock_conv_bwd_weight(%input : tensor<?x?x?x?x?xf32>, %output : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %result = rock.conv_bwd_weight(%input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    rock.numCU = 64 : i32,
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32> -> tensor<?x?x?x?x?xf32>
  return %result : tensor<?x?x?x?x?xf32>
}
// CHECK-LABEL: func.func @rock_conv_bwd_weight
// CHECK-NEXT: rock.conv_bwd_weight

func.func @rock_conv_bwd_weight_f16(%input : tensor<?x?x?x?x?xf16>, %output : tensor<?x?x?x?x?xf16>) -> tensor<?x?x?x?x?xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %result = rock.conv_bwd_weight(%input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    rock.numCU = 64 : i32,
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : tensor<?x?x?x?x?xf16>, tensor<?x?x?x?x?xf16> -> tensor<?x?x?x?x?xf16>
  return %result : tensor<?x?x?x?x?xf16>
}

// CHECK-LABEL: func.func @rock_conv_bwd_weight_f16
// CHECK-NEXT: rock.conv_bwd_weight

func.func @rock_gemm(%a : tensor<32x64xf16>, %b : tensor<1x32x128xf16>, %out : tensor<64x128xf32>) -> tensor<64x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %gemm_result = rock.gemm tr %a * %b
  : tensor<32x64xf16> * tensor<1x32x128xf16> -> tensor<64x128xf32>
  %result = rock.store %gemm_result to %out by set : tensor<64x128xf32> -> tensor<64x128xf32> to tensor<64x128xf32>
  func.return %result : tensor<64x128xf32>
}
// CHECK-LABEL: func.func @rock_gemm
// CHECK-NEXT: rock.gemm
// CHECK-NEXT: rock.store

// TODO: Scaled gemm tests need rework
// func.func @rock_scaled_gemm(%a : tensor<32x64xf4E2M1FN>, %b : tensor<1x32x128xf4E2M1FN>, %scaleA : tensor<32x64xf8E8M0FNU>, %scaleB : tensor<1x32x128xf8E8M0FNU>) -> tensor<64x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
//   %result = rock.gemm tr %a scaled by tr %scaleA * %b scaled by %scaleB storeMethod = set
//   : tensor<32x64xf4E2M1FN> scaled by tensor<32x64xf8E8M0FNU> * tensor<1x32x128xf4E2M1FN> scaled by tensor<1x32x128xf8E8M0FNU> -> tensor<64x128xf32>
//   func.return %result : tensor<64x128xf32>
// }
// DISABLED-CHECK-LABEL: func.func @rock_scaled_gemm
// DISABLED-CHECK-NEXT: rock.gemm


// Affine maps needed when testing transform
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d2, d3 - 1, d4 - 2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 512,
  (d1 mod 512) floordiv 16, d1 mod 16)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) ->
  (d1, d0, d2, d3 + d4, d5 + d6)>

// test 1-1 dimension mappings.
func.func @rock_transform_1_to_1(%tensor: tensor<1x2x3x4x5xf32>) -> tensor<2x1x3x6x9xf32, #map0> {
  %transformed_tensor = rock.transform %tensor by
    <#map0 by [
      <PassThrough ["g"] at [0] -> ["g"] at [1]>,
      <PassThrough ["n"] at [1] -> ["n"] at [0]>,
      <PassThrough ["c"] at [2] -> ["c"] at [2]>,
      <Pad{1, 1} ["0ipad"] at [3] -> ["0i"] at [3]>,
      <Pad{2, 2} ["1ipad"] at [4] -> ["1i"] at [4]>
    ] bounds = [2, 1, 3, 6, 9] -> [1, 2, 3, 4, 5]>
  : tensor<1x2x3x4x5xf32> to tensor<2x1x3x6x9xf32, #map0>
  return %transformed_tensor : tensor<2x1x3x6x9xf32, #map0>
}
// CHECK-LABEL: func.func @rock_transform_1_to_1
//  CHECK-NEXT: rock.transform

// test multiple source dimensions map to 1 target dimension.
func.func @rock_transform_n_to_1(%tensor : tensor<1x128x64x32x16xf32>) -> tensor<1x32768x128xf32, #map1> {
  %transformed_tensor = rock.transform %tensor by
    <#map1 by [
      #rock.transform<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>,
      #rock.transform<Merge{64, 32, 16} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>,
      #rock.transform<PassThrough ["gemmM"] at [2] -> ["k"] at [1]>
    ] bounds = [1, 32768, 128] -> [1, 128, 64, 32, 16]>
  : tensor<1x128x64x32x16xf32> to tensor<1x32768x128xf32, #map1>
  return %transformed_tensor : tensor<1x32768x128xf32, #map1>
}
// CHECK-LABEL: func.func @rock_transform_n_to_1
//  CHECK-NEXT: rock.transform

// test 1 source dimension map to multiple target dimensions.
func.func @rock_transform_1_to_n(%tensor : tensor<1x128x64x32x16xf32>) -> tensor<128x1x64x32x1x16x1xf32, #map2> {
  %transformed_tensor = rock.transform %tensor by
    <#map2 by [
      #rock.transform<PassThrough ["n", "g", "c"] at [0, 1, 2] ->
        ["n", "g", "c"] at [1, 0, 2]>,
      #rock.transform<Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>,
      #rock.transform<Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>
     ] bounds = [128, 1, 64, 32, 1, 16, 1] -> [1, 128, 64, 32, 16]>
  : tensor<1x128x64x32x16xf32> to tensor<128x1x64x32x1x16x1xf32, #map2>
  return %transformed_tensor : tensor<128x1x64x32x1x16x1xf32, #map2>
}

// CHECK-LABEL: func.func @rock_transform_1_to_n
//  CHECK-NEXT: rock.transform

func.func @rock_gridwise_gemm(%A : tensor<2x1024x1024xf32>, %B : tensor<2x1024x2048xf32>, %out : tensor<2x1024x2048xf32>) -> tensor<2x1024x2048xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.numCU = 64 : i32} {
  %gemm_result = rock.gridwise_gemm(%A, %B) {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #rock.gemm_params<
      mPerBlock = 128,
      nPerBlock = 128,
      kPerBlock = 4,
      kpack = 1,
      numWaves = 4,
      matrixInstrNonkdim = 0,
      splitKFactor = 1,
      numStages = 2,
      wavesPerEU = 0,
      gridGroupSize = 0,
      numCTAs = 1>
  } : tensor<2x1024x1024xf32>, tensor<2x1024x2048xf32> -> tensor<2x1024x2048xf32>
  %result = rock.store %gemm_result to %out by set : tensor<2x1024x2048xf32> -> tensor<2x1024x2048xf32> to tensor<2x1024x2048xf32>
  return %result : tensor<2x1024x2048xf32>
}

// CHECK-LABEL: func.func @rock_gridwise_gemm
// CHECK-NEXT: rock.gridwise_gemm
// CHECK-NEXT: rock.store

// TODO: Scaled gemm tests need rework
// func.func @rock_gridwise_scaled_gemm(%A : tensor<2x1024x1024xf4E2M1FN>, %B : tensor<2x1024x2048xf4E2M1FN>, %scaleA : tensor<2x1024x1024xf8E8M0FNU>, %scaleB : tensor<2x1024x2048xf8E8M0FNU>) -> tensor<2x1024x2048xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.numCU = 256 : i32} {
//   %result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) storeMethod (set) {
//     blockSize = 256 : i32,
//     gridSize = 1 : i32,
//     params = #rock.gemm_params<
//       kPerBlock = 4,
//       kpack = 1,
//       mPerBlock = 128,
//       nPerBlock = 128,
//       numWaves = 4,
//       matrixInstrNonkdim = 0,
//       splitKFactor = 1,
//       numStages = 2,
//       wavesPerEU = 0,
//       gridGroupSize = 0,
//       numCTAs = 1>
//   } : tensor<2x1024x1024xf4E2M1FN>, tensor<2x1024x2048xf4E2M1FN>, tensor<2x1024x1024xf8E8M0FNU>, tensor<2x1024x2048xf8E8M0FNU> -> tensor<2x1024x2048xf32>
//   return %result : tensor<2x1024x2048xf32>
// }
//
// DISABLED-CHECK-LABEL: func.func @rock_gridwise_scaled_gemm
// DISABLED-CHECK-NEXT: rock.gridwise_gemm

// TODO: Scaled gemm tests need rework
// func.func @rock_blockwise_gemm_scaled(%matrixA : tensor<256xvector<2xf4E2M1FN>>, 
//                                                 %matrixB : tensor<256xvector<2xf4E2M1FN>>,
//                                                 %matrixScaleA : tensor<256xvector<2xf8E8M0FNU>>,
//                                                 %matrixScaleB : tensor<256xvector<2xf8E8M0FNU>>,
//                                                 %bufferA : tensor<4xf4E2M1FN>, 
//                                                 %bufferB : tensor<4xf4E2M1FN>,
//                                                 %bufferScaleA : tensor<4xf8E8M0FNU>,
//                                                 %bufferScaleB : tensor<4xf8E8M0FNU>,
//                                                 %matrixC : tensor<4xvector<16xf32>>) -> tensor<4xvector<16xf32>> {
//   %result = rock.blockwise_gemm %matrixC += %bufferA from %matrixA scaled by %bufferScaleA from %matrixScaleA * %bufferB from %matrixB scaled by %bufferScaleB from %matrixScaleB {
//     rock.arch = "amdgcn-amd-amdhsa:gfx950",
//     blockSize = 256 : i32,
//     matrixParamsA = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 64, inDPerThread = 2>, 
//     matrixParamsB = #rock.blockwise_matrix_params<elementType = f4E2M1FN, elementTypeLoad = f4E2M1FN, rotateDWithK = false, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 1, d = 256, inDPerThread = 2>,
//     params = #rock.gemm_params<
//       kPerBlock = 2,
//       kpack = 1,
//       mPerBlock = 128,
//       nPerBlock = 128,
//       numWaves = 4,
//       matrixInstrNonkdim = 0,
//       splitKFactor = 1,
//       numStages = 2,
//       wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   } : tensor<4xvector<16xf32>> += tensor<4xf4E2M1FN> from tensor<256xvector<2xf4E2M1FN>> scaled by tensor<4xf8E8M0FNU> from tensor<256xvector<2xf8E8M0FNU>> * tensor<4xf4E2M1FN> from tensor<256xvector<2xf4E2M1FN>> scaled by tensor<4xf8E8M0FNU> from tensor<256xvector<2xf8E8M0FNU>> -> tensor<4xvector<16xf32>>
//   return %result : tensor<4xvector<16xf32>>
// }
//
// DISABLED-CHECK-LABEL: @rock_blockwise_gemm_scaled
// DISABLED-CHECK-NEXT: rock.blockwise_gemm

// ----

// TODO: Scaled gemm tests need rework
// func.func @rock_threadwise_gemm_accel_scaled(%matrixA : tensor<1x4xvector<4xf4E2M1FN>>,
//                                                 %matrixB : tensor<1x4xvector<4xf4E2M1FN>>,
//                                                 %matrixC : tensor<1x1xvector<32xf32>>, %scaleA : tensor<1x4xvector<4xf8E8M0FNU>>, %scaleB : tensor<1x4xvector<4xf8E8M0FNU>>) -> tensor<1x1xvector<32xf32>> {
//   %c0 = arith.constant 0 : index
//   %result = rock.threadwise_gemm_accel %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] {
//     rock.arch = "amdgcn-amd-amdhsa:gfx950",
//     params = #rock.gemm_params<
//       mPerBlock = 256,
//       nPerBlock = 256,
//       numWaves = 8,
//       kPerBlock = 16,
//       matrixInstrNonkdim = 0,
//       kpack = 1,
//       splitKFactor = 1,
//       numStages = 2,
//       wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   } : tensor<1x1xvector<32xf32>> += tensor<1x4xvector<4xf4E2M1FN>> scaled by tensor<1x4xvector<4xf8E8M0FNU>> * tensor<1x4xvector<4xf4E2M1FN>> scaled by tensor<1x4xvector<4xf8E8M0FNU>> -> tensor<1x1xvector<32xf32>>
//   return %result : tensor<1x1xvector<32xf32>>
// }
// DISABLED-CHECK-LABEL: func.func @rock_threadwise_gemm_accel_scaled
// DISABLED-CHECK: rock.threadwise_gemm_accel

// TODO(roctriton): We need to "unbufferize" attention
// DISABLED-CHECK-LABEL: func.func @gridwise_attn_atomic_add
// DISABLED-CHECK: rock.gridwise_attention
// func.func @gridwise_attn_atomic_add(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x64x384xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {rock.block_size = 64 : i32, grid_size = 24 : i32, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : tensor<1x384x64xf32> to tensor<1x64x384xf32>
//   %result = rock.gridwise_attention(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
//     blockSize = 64 : i32,
//     gridSize = 24 : i32,
//     params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
//     params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
//     storeMethod = #rock<StoreMethod atomic_add>,
//     splitKV = 1 : i32,
//     enableSoftmax = false,
//     operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0>
//   } : tensor<1x64x384xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
//   return %result : tensor<1x384x64xf32>
// }
//
// DISABLED-CHECK-LABEL: func.func @attention
// DISABLED-CHECK: rock.attention
// func.func @attention(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<1x384x64xf16>
//   return %result : tensor<1x384x64xf16>
// }

