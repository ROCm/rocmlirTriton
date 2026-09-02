// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt | FileCheck %s
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt | rocmlir-opt | FileCheck %s

func.func @rock_conv(%filter : tensor<?x?x?x?x?xf32>, %input : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
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

func.func @rock_conv_f16(%filter : tensor<?x?x?x?x?xf16>, %input : tensor<?x?x?x?x?xf16>) -> tensor<?x?x?x?x?xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
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

func.func @rock_conv_fp8_mixed(%filter : tensor<?x?x?x?x?xf8E4M3FNUZ>, %input : tensor<?x?x?x?x?xf8E5M2FNUZ>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
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

func.func @rock_conv_fp8_mixed_ocp(%filter : tensor<?x?x?x?x?xf8E4M3FN>, %input : tensor<?x?x?x?x?xf8E5M2>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
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

func.func @rock_conv_bwd_data(%filter : tensor<?x?x?x?x?xf32>, %output : tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
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

func.func @rock_conv_bwd_data_f16(%filter : tensor<?x?x?x?x?xf16>, %output : tensor<?x?x?x?x?xf16>) -> tensor<?x?x?x?x?xf16> attributes {rock.arch = "##TOKEN_ARCH##"} {
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

func.func @rock_gemm(%a : tensor<32x64xf16>, %b : tensor<1x32x128xf16>, %out : tensor<64x128xf32>) -> tensor<64x128xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %gemm_result = rock.gemm tr %a * %b
  : tensor<32x64xf16> * tensor<1x32x128xf16> -> tensor<64x128xf32>
  %result = rock.store %gemm_result to %out by set : tensor<64x128xf32> -> tensor<64x128xf32> to tensor<64x128xf32>
  func.return %result : tensor<64x128xf32>
}
// CHECK-LABEL: func.func @rock_gemm
// CHECK-NEXT: rock.gemm
// CHECK-NEXT: rock.store

func.func @rock_store_result_view_fanout(%source : tensor<4x4xf32>, %dest : tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) attributes {rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  %view0 = rock.transform %result by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>] bounds = [4, 4] -> [4, 4]> : tensor<4x4xf32> to tensor<4x4xf32>
  %view1 = rock.transform %result by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["m", "n"] at [0, 1] -> ["m", "n"] at [0, 1]>] bounds = [4, 4] -> [4, 4]> : tensor<4x4xf32> to tensor<4x4xf32>
  func.return %view0, %view1 : tensor<4x4xf32>, tensor<4x4xf32>
}
// CHECK-LABEL: func.func @rock_store_result_view_fanout
// CHECK: rock.store
// CHECK: rock.transform
// CHECK: rock.transform

func.func @rock_store_result_store_chain(%source : tensor<4x4xf32>, %dest : tensor<4x4xf32>) -> tensor<4x4xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %result0 = rock.store %source to %dest by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  %result1 = rock.store %source to %dest alias %result0 by set : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32> alias tensor<4x4xf32>
  func.return %result1 : tensor<4x4xf32>
}
// CHECK-LABEL: func.func @rock_store_result_store_chain
// CHECK: rock.store
// CHECK: rock.store

// A is K x M (tr), so aScaleTransposed=true means scaleA layout is K/qbs x M.
// B is G x K x N, so scaleB layout is G x N x K/qbs (per GemmOp::verify).
// With K=32 and quantBlockSize=32, K/qbs = 1.
func.func @rock_scaled_gemm(%a : tensor<32x64xf4E2M1FN>, %b : tensor<1x32x128xf4E2M1FN>, %scaleA : tensor<1x64xf8E8M0FNU>, %scaleB : tensor<1x128x1xf8E8M0FNU>, %out : tensor<64x128xf32>) -> tensor<64x128xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %gemm_result = rock.gemm tr %a scaled by tr %scaleA * %b scaled by %scaleB {quantBlockSize = 32 : i64}
  : tensor<32x64xf4E2M1FN> scaled by tensor<1x64xf8E8M0FNU> * tensor<1x32x128xf4E2M1FN> scaled by tensor<1x128x1xf8E8M0FNU> -> tensor<64x128xf32>
  %result = rock.store %gemm_result to %out by set : tensor<64x128xf32> -> tensor<64x128xf32> to tensor<64x128xf32>
  func.return %result : tensor<64x128xf32>
}
// CHECK-LABEL: func.func @rock_scaled_gemm
// CHECK-NEXT: rock.gemm
// CHECK-NEXT: rock.store


// Affine maps needed when testing transform
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d0, d2, d3 - 1, d4 - 2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 512,
  (d1 floordiv 16) mod 32, d1 mod 16)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) ->
  (d1, d0, d2, d3 + d4, d5 + d6)>

// test 1-1 dimension mappings.
func.func @rock_transform_1_to_1(%tensor: tensor<1x2x3x4x5xf32>) -> tensor<2x1x3x6x9xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %transformed_tensor = rock.transform %tensor by
    <#map0 by [
      <PassThrough ["g"] at [0] -> ["g"] at [1]>,
      <PassThrough ["n"] at [1] -> ["n"] at [0]>,
      <PassThrough ["c"] at [2] -> ["c"] at [2]>,
      <Pad{1, 1} ["0ipad"] at [3] -> ["0i"] at [3]>,
      <Pad{2, 2} ["1ipad"] at [4] -> ["1i"] at [4]>
    ] bounds = [2, 1, 3, 6, 9] -> [1, 2, 3, 4, 5]>
  : tensor<1x2x3x4x5xf32> to tensor<2x1x3x6x9xf32>
  return %transformed_tensor : tensor<2x1x3x6x9xf32>
}
// CHECK-LABEL: func.func @rock_transform_1_to_1
//  CHECK-NEXT: rock.transform

// test multiple source dimensions map to 1 target dimension.
func.func @rock_transform_n_to_1(%tensor : tensor<1x128x64x32x16xf32>) -> tensor<1x32768x128xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %transformed_tensor = rock.transform %tensor by
    <#map1 by [
      #rock.transform<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>,
      #rock.transform<Merge{64, 32, 16} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>,
      #rock.transform<PassThrough ["gemmM"] at [2] -> ["k"] at [1]>
    ] bounds = [1, 32768, 128] -> [1, 128, 64, 32, 16]>
  : tensor<1x128x64x32x16xf32> to tensor<1x32768x128xf32>
  return %transformed_tensor : tensor<1x32768x128xf32>
}
// CHECK-LABEL: func.func @rock_transform_n_to_1
//  CHECK-NEXT: rock.transform

// test 1 source dimension map to multiple target dimensions.
func.func @rock_transform_1_to_n(%tensor : tensor<1x128x64x32x16xf32>) -> tensor<128x1x64x32x1x16x1xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %transformed_tensor = rock.transform %tensor by
    <#map2 by [
      #rock.transform<PassThrough ["n", "g", "c"] at [0, 1, 2] ->
        ["n", "g", "c"] at [1, 0, 2]>,
      #rock.transform<Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>,
      #rock.transform<Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>
     ] bounds = [128, 1, 64, 32, 1, 16, 1] -> [1, 128, 64, 32, 16]>
  : tensor<1x128x64x32x16xf32> to tensor<128x1x64x32x1x16x1xf32>
  return %transformed_tensor : tensor<128x1x64x32x1x16x1xf32>
}

// CHECK-LABEL: func.func @rock_transform_1_to_n
//  CHECK-NEXT: rock.transform

func.func @rock_gridwise_gemm(%A : tensor<2x1024x1024xf32>, %B : tensor<2x1024x2048xf32>, %out : tensor<2x1024x2048xf32>) -> tensor<2x1024x2048xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %gemm_result = rock.gridwise_gemm(%A, %B) {
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

// A is G x M x K, B is G x K x N. Per the op description, the scale tensors are
// [G] x M x ceil(K/qbs) for scaleA and [G] x N x ceil(K/qbs) for scaleB.
// With K=1024 and quantBlockSize=32, K/qbs = 32.
func.func @rock_gridwise_scaled_gemm(%A : tensor<2x1024x1024xf4E2M1FN>, %B : tensor<2x1024x2048xf4E2M1FN>, %scaleA : tensor<2x1024x32xf8E8M0FNU>, %scaleB : tensor<2x2048x32xf8E8M0FNU>, %out : tensor<2x1024x2048xf32>) -> tensor<2x1024x2048xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %gemm_result = rock.gridwise_gemm(%A, %B, %scaleA, %scaleB) {
    quantBlockSize = 32 : i64,
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
  } : tensor<2x1024x1024xf4E2M1FN>, tensor<2x1024x2048xf4E2M1FN>, tensor<2x1024x32xf8E8M0FNU>, tensor<2x2048x32xf8E8M0FNU> -> tensor<2x1024x2048xf32>
  %result = rock.store %gemm_result to %out by set : tensor<2x1024x2048xf32> -> tensor<2x1024x2048xf32> to tensor<2x1024x2048xf32>
  return %result : tensor<2x1024x2048xf32>
}

// CHECK-LABEL: func.func @rock_gridwise_scaled_gemm
// CHECK-NEXT: rock.gridwise_gemm
// CHECK: rock.store
func.func @rock_blockwise_gemm_scaled(
    %a: tensor<64x64xf4E2M1FN>, %b: tensor<64x64xf4E2M1FN>,
    %scaleA: tensor<64x2xf8E8M0FNU>, %scaleB: tensor<64x2xf8E8M0FNU>,
    %c: tensor<64x64xf32>) -> tensor<64x64xf32>
    attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.blockwise_gemm(%a scaled by %scaleA, %b scaled by %scaleB, %c)
    {quantBlockSize = 32 : i64}
    : tensor<64x64xf4E2M1FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf4E2M1FN> scaled by tensor<64x2xf8E8M0FNU>,
      tensor<64x64xf32> -> tensor<64x64xf32>
  return %result : tensor<64x64xf32>
}
// CHECK-LABEL: func.func @rock_blockwise_gemm_scaled
// CHECK: rock.blockwise_gemm

// ----

func.func @rock_gridwise_attention(%q: tensor<1x384x64xf32>, %k: tensor<1x64x384xf32>, %v: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {rock.block_size = 64 : i32, rock.grid_size = 24 : i32, rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x384x384xf32>):
    rock.yield %arg_qk : tensor<1x384x384xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    splitKV = 1 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}
// CHECK-LABEL: func.func @rock_gridwise_attention
// CHECK: rock.gridwise_attention
// CHECK-NOT: slidingWindowLookBack

func.func @rock_gridwise_attention_sliding_window(%q: tensor<1x384x64xf32>, %k: tensor<1x64x384xf32>, %v: tensor<1x384x64xf32>, %lastValidKVIndex: tensor<1xi32>) -> tensor<1x384x64xf32> attributes {rock.block_size = 64 : i32, rock.grid_size = 24 : i32, rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.gridwise_attention(%q, %k, %v, %lastValidKVIndex) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x384x384xf32>):
    rock.yield %arg_qk : tensor<1x384x384xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 1, 0>,
    params0 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    params1 = #rock.gemm_params<kPerBlock = 32, mPerBlock = 32, nPerBlock = 32, numWaves = 1, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>,
    splitKV = 1 : i32,
    slidingWindowLookBack = 383 : i32
  } : tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1x384x64xf32>, tensor<1xi32> -> tensor<1x384x64xf32>
  return %result : tensor<1x384x64xf32>
}
// CHECK-LABEL: func.func @rock_gridwise_attention_sliding_window
// CHECK: rock.gridwise_attention
// CHECK: slidingWindowLookBack = 383

func.func @rock_attention(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.attention{
    qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
    softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32} -> tensor<1x384x64xf16>
  return %result : tensor<1x384x64xf16>
}
// CHECK-LABEL: func.func @rock_attention
// CHECK: rock.attention
// CHECK-NOT: slidingWindowLookBack

func.func @rock_attention_sliding_window(%q: tensor<1x384x64xf16>, %k: tensor<1x384x64xf16>, %v: tensor<1x384x64xf16>, %lastValidKVIndex: tensor<1xi32>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.attention{
    qk = %q * tr %k : tensor<1x384x64xf16>, tensor<1x384x64xf16>
    lastValidKVIndex = (%lastValidKVIndex : tensor<1xi32>)
    softmax(qk) * %v : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, slidingWindowLookBack = 383 : i32} -> tensor<1x384x64xf16>
  return %result : tensor<1x384x64xf16>
}
// CHECK-LABEL: func.func @rock_attention_sliding_window
// CHECK: rock.attention
// CHECK: lastValidKVIndex = (%{{.*}} : tensor<1xi32>)
// CHECK: slidingWindowLookBack = 383

func.func @rock_reduce_sum(%in: tensor<8x32xf32>) -> tensor<8x1xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.reduce sum %in {axis = 1 : index} : tensor<8x32xf32> -> tensor<8x1xf32>
  return %result : tensor<8x1xf32>
}
// CHECK-LABEL: func.func @rock_reduce_sum
// CHECK: rock.reduce sum

func.func @rock_reduce_max(%in: tensor<8x32xf32>) -> tensor<8x1xf32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.reduce max %in {axis = 1 : index} : tensor<8x32xf32> -> tensor<8x1xf32>
  return %result : tensor<8x1xf32>
}
// CHECK-LABEL: func.func @rock_reduce_max
// CHECK: rock.reduce max

// Integer reductions are legal: they lower to an integer atomic add, which the
// backend expands to a compare-and-swap loop where no native instruction
// exists.
func.func @rock_reduce_sum_i32(%in: tensor<8x32xi32>) -> tensor<8x1xi32> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.reduce sum %in {axis = 1 : index} : tensor<8x32xi32> -> tensor<8x1xi32>
  return %result : tensor<8x1xi32>
}
// CHECK-LABEL: func.func @rock_reduce_sum_i32
// CHECK: rock.reduce sum

func.func @rock_reduce_sum_i8(%in: tensor<8x32xi8>) -> tensor<8x1xi8> attributes {rock.arch = "##TOKEN_ARCH##"} {
  %result = rock.reduce sum %in {axis = 1 : index} : tensor<8x32xi8> -> tensor<8x1xi8>
  return %result : tensor<8x1xi8>
}
// CHECK-LABEL: func.func @rock_reduce_sum_i8
// CHECK: rock.reduce sum

