// Verify that ops marked with the `Pure` trait are dead-code eliminated by
// `-canonicalize` when their results are unused. Each function instantiates
// one Pure op, ignores its result, and returns an unrelated value.

// RUN: rocmlir-opt --canonicalize -split-input-file %s | FileCheck %s

#xform = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)>
  by [<Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 1] -> ["k"] at [0]>,
      <PassThrough ["n"] at [2] -> ["n"] at [1]>]
  bounds = [4, 64, 128] -> [256, 128]>

// CHECK-LABEL: func.func @dce_transform
// CHECK-NOT:     rock.transform
// CHECK:         return %arg1
func.func @dce_transform(%src: tensor<256x128xf16>, %sink: tensor<4x64x128xf16>)
    -> tensor<4x64x128xf16> {
  %unused = rock.transform %src by #xform
      : tensor<256x128xf16> to tensor<4x64x128xf16>
  return %sink : tensor<4x64x128xf16>
}

// -----

// CHECK-LABEL: func.func @dce_untile
// CHECK-NOT:     rock.untile
// CHECK:         return %arg1
func.func @dce_untile(%tile: tensor<64x64xf32>, %sink: tensor<1x256x128xf32>)
    -> tensor<1x256x128xf32> {
  %unused = rock.untile %tile : tensor<64x64xf32> -> tensor<1x256x128xf32>
  return %sink : tensor<1x256x128xf32>
}

// -----

#load_marker_tmap = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)>
  by [<Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 1] -> ["k"] at [0]>,
      <PassThrough ["n"] at [2] -> ["n"] at [1]>]
  bounds = [4, 64, 128] -> [256, 128]>

// CHECK-LABEL: func.func @dce_load_marker
// CHECK-NOT:     rock.load_marker
// CHECK:         return %arg2
func.func @dce_load_marker(%src: tensor<256x128xf16>, %idx: i32,
                           %sink: tensor<64x128xf16>) -> tensor<64x128xf16> {
  %unused = rock.load_marker %src views [#load_marker_tmap] [%idx]
      : tensor<256x128xf16> -> tensor<64x128xf16>
  return %sink : tensor<64x128xf16>
}

// -----

#store_marker_tmap = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) ->
    (d0, d1 * 64 + d3, d2 * 64 + d4)>
  by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>,
      <Unmerge{1, 64} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>,
      <Unmerge{2, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>]
  bounds = [1, 1, 2, 64, 64] -> [1, 64, 128]>

// CHECK-LABEL: func.func @dce_store_marker
// CHECK-NOT:     rock.store_marker
// CHECK:         return %arg4
func.func @dce_store_marker(%tile: tensor<64x64xf32>,
                            %g: i32, %m: i32, %n: i32,
                            %sink: tensor<1x64x128xf32>) -> tensor<1x64x128xf32> {
  %unused = rock.store_marker %tile views [#store_marker_tmap] [%g, %m, %n]
      : tensor<64x64xf32> -> tensor<1x64x128xf32>
  return %sink : tensor<1x64x128xf32>
}

// -----

// CHECK-LABEL: func.func @dce_store
// CHECK-NOT:     rock.store
// CHECK:         return %arg2
func.func @dce_store(%src: tensor<4x4xf32>, %dest: tensor<4x4xf32>,
                     %sink: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %unused = rock.store %src to %dest by set
      : tensor<4x4xf32> -> tensor<4x4xf32> to tensor<4x4xf32>
  return %sink : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @dce_gemm
// CHECK-NOT:     rock.gemm
// CHECK:         return %arg2
func.func @dce_gemm(%a: tensor<32x64xf16>, %b: tensor<1x32x128xf16>,
                    %sink: tensor<64x128xf32>) -> tensor<64x128xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %unused = rock.gemm tr %a * %b
      : tensor<32x64xf16> * tensor<1x32x128xf16> -> tensor<64x128xf32>
  return %sink : tensor<64x128xf32>
}

// -----

// CHECK-LABEL: func.func @dce_conv
// CHECK-NOT:     rock.conv
// CHECK:         return %arg2
func.func @dce_conv(%filter: tensor<?x?x?x?x?xf32>, %input: tensor<?x?x?x?x?xf32>,
                    %sink: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %unused = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32> -> tensor<?x?x?x?x?xf32>
  return %sink : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @dce_conv_bwd_data
// CHECK-NOT:     rock.conv_bwd_data
// CHECK:         return %arg2
func.func @dce_conv_bwd_data(%filter: tensor<?x?x?x?x?xf32>,
                             %output: tensor<?x?x?x?x?xf32>,
                             %sink: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %unused = rock.conv_bwd_data(%filter, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32> -> tensor<?x?x?x?x?xf32>
  return %sink : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @dce_conv_bwd_weight
// CHECK-NOT:     rock.conv_bwd_weight
// CHECK:         return %arg2
func.func @dce_conv_bwd_weight(%input: tensor<?x?x?x?x?xf32>,
                               %output: tensor<?x?x?x?x?xf32>,
                               %sink: tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %unused = rock.conv_bwd_weight(%input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    rock.numCU = 64 : i32,
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32> -> tensor<?x?x?x?x?xf32>
  return %sink : tensor<?x?x?x?x?xf32>
}

// -----

// CHECK-LABEL: func.func @dce_reduce
// CHECK-NOT:     rock.reduce
// CHECK:         return %arg1
func.func @dce_reduce(%input: tensor<2x32x4096xf32>, %sink: tensor<2x32x1xf32>) -> tensor<2x32x1xf32> {
  %unused = rock.reduce sum %input {axis = 2 : index}
      : tensor<2x32x4096xf32> -> tensor<2x32x1xf32>
  return %sink : tensor<2x32x1xf32>
}

// -----

// CHECK-LABEL: func.func @dce_blockwise_reduce
// CHECK-NOT:     rock.blockwise_reduce
// CHECK:         return %arg1
func.func @dce_blockwise_reduce(%input: tensor<64x64xf32>, %sink: tensor<64xf32>) -> tensor<64xf32> {
  %unused = rock.blockwise_reduce sum %input {axis = 1 : index}
      : tensor<64x64xf32> -> tensor<64xf32>
  return %sink : tensor<64xf32>
}

// -----

// CHECK-LABEL: func.func @dce_blockwise_store
// CHECK-NOT:     rock.blockwise_store
// CHECK:         return %arg3
func.func @dce_blockwise_store(%src: tensor<64x64xf32>,
                               %dest: tensor<1x64x64xf32>, %idx: i32,
                               %sink: tensor<4096xf32>) -> tensor<4096xf32> {
  %unused = rock.blockwise_store %src -> %dest[%idx] by set
      : tensor<64x64xf32> -> tensor<1x64x64xf32> -> tensor<4096xf32>
  return %sink : tensor<4096xf32>
}

// -----

// CHECK-LABEL: func.func @dce_gridwise_gemm
// CHECK-NOT:     rock.gridwise_gemm
// CHECK:         return %arg2
func.func @dce_gridwise_gemm(%a: tensor<2x1024x1024xf32>,
                             %b: tensor<2x1024x2048xf32>,
                             %sink: tensor<2x1024x2048xf32>) -> tensor<2x1024x2048xf32>
    attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.numCU = 64 : i32} {
  %unused = rock.gridwise_gemm(%a, %b) {
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
  return %sink : tensor<2x1024x2048xf32>
}

// -----

// CHECK-LABEL: func.func @dce_blockwise_gemm
// CHECK-NOT:     rock.blockwise_gemm
// CHECK:         return %arg3
func.func @dce_blockwise_gemm(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                              %c: tensor<64x64xf32>, %sink: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %unused = rock.blockwise_gemm(%a, %b, %c)
      : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %sink : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @dce_blockwise_load
// CHECK-NOT:     rock.blockwise_load
// CHECK:         return %arg1
func.func @dce_blockwise_load(%src: tensor<64x64xf32>, %sink: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %unused = rock.blockwise_load %src : tensor<64x64xf32> -> tensor<64x64xf32>
  return %sink : tensor<64x64xf32>
}

// -----

// Has two results (pointers, mask) — both must be DCE'd along with the op.
// CHECK-LABEL: func.func @dce_transforms_to_ptr
// CHECK-NOT:     rock.transforms_to_ptr
// CHECK:         return %arg1
func.func @dce_transforms_to_ptr(%src: tensor<64x64xf32>, %sink: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  %unused_ptrs, %unused_mask = rock.transforms_to_ptr %src
      : tensor<64x64xf32> -> tensor<64x64xi32>, tensor<64x64xi1>
  return %sink : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @dce_cast_to_ptr
// CHECK-NOT:     rock.cast_to_ptr
// CHECK:         return %arg1
func.func @dce_cast_to_ptr(%src: tensor<64x64xi32>,
                           %sink: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %unused = rock.cast_to_ptr %src
      : tensor<64x64xi32> -> tensor<64x64x!tt.ptr<f16>>
  return %sink : tensor<64x64xf32>
}

// -----

// CHECK-LABEL: func.func @dce_extract_ptr
// CHECK-NOT:     rock.extract_ptr
// CHECK:         return %arg1
func.func @dce_extract_ptr(%src: tensor<64x64xf32>, %sink: i32) -> i32 {
  %unused = rock.extract_ptr %src : tensor<64x64xf32> -> i32
  return %sink : i32
}

// -----

// Region-carrying op with a pure body: RecursiveMemoryEffects /
// RecursivelySpeculatable should let -canonicalize DCE it.
// CHECK-LABEL: func.func @dce_attention
// CHECK-NOT:     rock.attention
// CHECK:         return %arg3
func.func @dce_attention(%q: tensor<1x8x4xf16>, %k: tensor<1x4x8xf16>,
                         %v: tensor<1x8x4xf16>, %sink: tensor<1x8x4xf16>)
    -> tensor<1x8x4xf16> {
  %unused = rock.attention {
    qk = %q * %k : tensor<1x8x4xf16>, tensor<1x4x8xf16>
    softmax(qk) * %v : tensor<1x8x4xf16>
  } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32,
     numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x8x4xf16>
  return %sink : tensor<1x8x4xf16>
}

// -----

// CHECK-LABEL: func.func @dce_gemm_elementwise_gemm
// CHECK-NOT:     rock.gemm_elementwise_gemm
// CHECK:         return %arg3
func.func @dce_gemm_elementwise_gemm(%a: tensor<1x32x16xf16>,
                                     %b: tensor<1x16x64xf16>,
                                     %c: tensor<1x64x8xf16>,
                                     %sink: tensor<1x32x8xf16>)
    -> tensor<1x32x8xf16> {
  %unused = rock.gemm_elementwise_gemm {
    ab = %a * %b : tensor<1x32x16xf16>, tensor<1x16x64xf16>
    out = ab * %c : tensor<1x64x8xf16>
  } {firstGemmIndices = array<i64: 0>} -> tensor<1x32x8xf16>
  return %sink : tensor<1x32x8xf16>
}

// -----

// CHECK-LABEL: func.func @dce_conv_elementwise_gemm
// CHECK-NOT:     rock.conv_elementwise_gemm
// CHECK:         return %arg3
func.func @dce_conv_elementwise_gemm(%filter: tensor<1x256x3x3x64xf32>,
                                     %input: tensor<64x14x14x1x64xf32>,
                                     %c: tensor<1x256x256xf32>,
                                     %sink: tensor<1x9216x256xf32>)
    -> tensor<1x9216x256xf32> {
  %unused = rock.conv_elementwise_gemm {
    ab = conv(%filter, %input)
        : tensor<1x256x3x3x64xf32>, tensor<64x14x14x1x64xf32>
    out = ab * %c : tensor<1x256x256xf32>
  } {dilations = [1 : index, 1 : index],
     filter_layout = ["g", "k", "0", "1", "c"],
     firstGemmIndices = array<i64: 0>,
     input_layout = ["ni", "0i", "1i", "gi", "ci"],
     padding = [0 : index, 0 : index, 0 : index, 0 : index],
     strides = [1 : index, 1 : index]} -> tensor<1x9216x256xf32>
  return %sink : tensor<1x9216x256xf32>
}

// -----

// CHECK-LABEL: func.func @dce_gridwise_attention
// CHECK-NOT:     rock.gridwise_attention
// CHECK:         return %arg3
func.func @dce_gridwise_attention(%q: tensor<1x64x32xf32>,
                                  %k: tensor<1x32x64xf32>,
                                  %v: tensor<1x64x32xf32>,
                                  %sink: tensor<1x64x32xf32>)
    -> tensor<1x64x32xf32> {
  %unused = rock.gridwise_attention(%q, %k, %v) preSoftmaxOps = {
  ^bb0(%arg_qk: tensor<1x64x64xf32>):
    rock.yield %arg_qk : tensor<1x64x64xf32>
  } {
    operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0>,
    params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32,
                                kPerBlock = 32, kpack = 1, numCTAs = 1,
                                numWaves = 4, matrixInstrNonkdim = 0,
                                splitKFactor = 1, numStages = 1,
                                wavesPerEU = 0, gridGroupSize = 0>,
    params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32,
                                kPerBlock = 32, kpack = 1, numCTAs = 1,
                                numWaves = 4, matrixInstrNonkdim = 0,
                                splitKFactor = 1, numStages = 1,
                                wavesPerEU = 0, gridGroupSize = 0>,
    splitKV = 1 : i32
  } : tensor<1x64x32xf32>, tensor<1x32x64xf32>, tensor<1x64x32xf32>
      -> tensor<1x64x32xf32>
  return %sink : tensor<1x64x32xf32>
}

// -----

#xform_used = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)>
  by [<Unmerge{4, 64} ["k_loop", "k_iter"] at [0, 1] -> ["k"] at [0]>,
      <PassThrough ["n"] at [2] -> ["n"] at [1]>]
  bounds = [4, 64, 128] -> [256, 128]>

// CHECK-LABEL: func.func @keep_used_transform
// CHECK:         rock.transform
// CHECK:         return
func.func @keep_used_transform(%src: tensor<256x128xf16>) -> tensor<4x64x128xf16> {
  %used = rock.transform %src by #xform_used
      : tensor<256x128xf16> to tensor<4x64x128xf16>
  return %used : tensor<4x64x128xf16>
}
