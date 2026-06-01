// This tests checks the following aspects of the lowering:
// * convolution tuning parameters are set as expected
// If versions of these tests appear in lowering_top_level, then changes to the tuning
// parameters made here should be reflected in that file

// RUN: rocmlir-driver -mlir-print-local-scope -rock-affix-params  %s | FileCheck %s --check-prefix=CHECK
// RUN: rocmlir-driver -mlir-print-local-scope -rock-affix-params -rock-lower-reduce -rock-regularize-output -rock-regularize-inter-gemm-fusion -rock-conv-to-gemm -rock-fusion-splitk-regularization -rock-gemm-to-gridwise -rock-attn-to-gridwise %s | FileCheck %s --check-prefix=GRID

// CHECK-LABEL: @rock_conv
// GRID-LABEL: rock_conv
func.func @rock_conv(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x128x30x30xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32> -> tensor<128x1x128x30x30xf32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32> to tensor<128x1x128x30x30xf32>
  return %out : tensor<128x1x128x30x30xf32>
}

// CHECK-LABEL: @rock_conv_numstages2
// GRID-LABEL: rock_conv_numstages2
func.func @rock_conv_numstages2(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x128x30x30xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32> -> tensor<128x1x128x30x30xf32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32> to tensor<128x1x128x30x30xf32>
  return %out : tensor<128x1x128x30x30xf32>
}

// CHECK-LABEL: func.func @rock_conv_f16
// GRID-LABEL: rock_conv_f16
func.func @rock_conv_f16(%filter : tensor<1x128x8x3x3xf16>, %input : tensor<128x1x8x32x32xf16>, %output : tensor<128x1x128x30x30xf16>) -> tensor<128x1x128x30x30xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<1x128x8x3x3xf16>, tensor<128x1x8x32x32xf16> -> tensor<128x1x128x30x30xf16>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf16> -> tensor<128x1x128x30x30xf16> to tensor<128x1x128x30x30xf16>
  return %out : tensor<128x1x128x30x30xf16>
}

// CHECK-LABEL: func.func @rock_conv_i8
// GRID-LABEL: rock_conv_i8
func.func @rock_conv_i8(%filter : tensor<1x128x8x3x3xi8>, %input : tensor<128x1x8x32x32xi8>, %output : tensor<128x1x128x30x30xi32>) -> tensor<128x1x128x30x30xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 2, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:64,64,64,2,1,4,32,1,2,0,0"
  } : tensor<1x128x8x3x3xi8>, tensor<128x1x8x32x32xi8> -> tensor<128x1x128x30x30xi32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xi32> -> tensor<128x1x128x30x30xi32> to tensor<128x1x128x30x30xi32>
  return %out : tensor<128x1x128x30x30xi32>
}

// CHECK-LABEL: func.func @rock_conv_bwd_data
// GRID-LABEL: func.func @rock_conv_bwd_data
func.func @rock_conv_bwd_data(%filter: tensor<1x1024x1024x1x1xf32>, %input: tensor<128x1x1024x14x14xf32>, %output: tensor<128x1x1024x14x14xf32>) -> tensor<128x1x1024x14x14xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID-SAME: rock.grid_size = 12544
  // GRID: rock.gridwise_gemm
  %result = rock.conv_bwd_data(%filter, %output) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index]
  } : tensor<1x1024x1024x1x1xf32>, tensor<128x1x1024x14x14xf32> -> tensor<128x1x1024x14x14xf32>
  %out = rock.store %result to %input by set : tensor<128x1x1024x14x14xf32> -> tensor<128x1x1024x14x14xf32> to tensor<128x1x1024x14x14xf32>
  return %out : tensor<128x1x1024x14x14xf32>
}

// CHECK-LABEL: @rock_conv_bwd_data_f16
// GRID-LABEL: @rock_conv_bwd_data_f16
func.func @rock_conv_bwd_data_f16(%filter: tensor<1x1024x1024x1x1xf16>, %input: tensor<128x1x1024x14x14xf16>, %output: tensor<128x1x1024x14x14xf16>) -> tensor<128x1x1024x14x14xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID-SAME: rock.grid_size = 12544
  // GRID: rock.gridwise_gemm
  %result = rock.conv_bwd_data(%filter, %output) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index]
  } : tensor<1x1024x1024x1x1xf16>, tensor<128x1x1024x14x14xf16> -> tensor<128x1x1024x14x14xf16>
  %out = rock.store %result to %output by set : tensor<128x1x1024x14x14xf16> -> tensor<128x1x1024x14x14xf16> to tensor<128x1x1024x14x14xf16>
  return %out : tensor<128x1x1024x14x14xf16>
}

// CHECK-LABEL: func.func @rock_conv_bwd_data_padMN
// GRID-LABEL: func.func @rock_conv_bwd_data_padMN
func.func @rock_conv_bwd_data_padMN(%filter : tensor<1x64x3x1x1xf32>, %input : tensor<11x1x3x15x15xf32>, %output : tensor<11x1x64x15x15xf32>) -> tensor<11x1x3x15x15xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID-SAME: rock.grid_size = 78
  // GRID: rock.gridwise_gemm
  %result = rock.conv_bwd_data(%filter, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x64x3x1x1xf32>, tensor<11x1x64x15x15xf32> -> tensor<11x1x3x15x15xf32>
  %out = rock.store %result to %input by set : tensor<11x1x3x15x15xf32> -> tensor<11x1x3x15x15xf32> to tensor<11x1x3x15x15xf32>
  return %out : tensor<11x1x3x15x15xf32>
}

// CHECK-LABEL: @rock_conv_bwd_data_padMK
// GRID-LABEL: @rock_conv_bwd_data_padMK
func.func @rock_conv_bwd_data_padMK(%filter : tensor<1x11x3x1x1xf32>, %input : tensor<128x1x3x15x15xf32>, %output : tensor<128x1x11x15x15xf32>) -> tensor<128x1x3x15x15xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID-SAME: rock.grid_size = 900
  // GRID: rock.gridwise_gemm
  %result = rock.conv_bwd_data(%filter, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x11x3x1x1xf32>, tensor<128x1x11x15x15xf32> -> tensor<128x1x3x15x15xf32>
  %out = rock.store %result to %input by set : tensor<128x1x3x15x15xf32> -> tensor<128x1x3x15x15xf32> to tensor<128x1x3x15x15xf32>
  return %out : tensor<128x1x3x15x15xf32>
}

// CHECK-LABEL: @rock_conv_bwd_weight
// GRID-LABEL: rock_conv_bwd_weight
func.func @rock_conv_bwd_weight(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<1x128x8x3x3xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 4
  // GRID: rock.gridwise_gemm
  %result = rock.conv_bwd_weight(%input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<1x128x8x3x3xf32>
  %out = rock.store %result to %filter by set : tensor<1x128x8x3x3xf32> -> tensor<1x128x8x3x3xf32> to tensor<1x128x8x3x3xf32>
  return %out : tensor<1x128x8x3x3xf32>
}

// CHECK-LABEL: @rock_conv_bwd_weight_f16
// GRID-LABEL: rock_conv_bwd_weight_f16
func.func @rock_conv_bwd_weight_f16(%filter : tensor<1x128x8x3x3xf16>, %input : tensor<128x1x8x32x32xf16>, %output : tensor<128x1x128x30x30xf16>) -> tensor<1x128x8x3x3xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 4
  %result = rock.conv_bwd_weight(%input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<128x1x8x32x32xf16>, tensor<128x1x128x30x30xf16> -> tensor<1x128x8x3x3xf16>
  %out = rock.store %result to %filter by set : tensor<1x128x8x3x3xf16> -> tensor<1x128x8x3x3xf16> to tensor<1x128x8x3x3xf16>
  return %out : tensor<1x128x8x3x3xf16>
}

// CHECK-LABEL: func.func @rock_conv_bwd_weight_padALL
// GRID-LABEL: rock_conv_bwd_weight_padALL
func.func @rock_conv_bwd_weight_padALL(%filter : tensor<1x20x8x3x3xf32>, %input : tensor<7x1x8x32x32xf32>, %output : tensor<7x1x20x30x30xf32>) -> tensor<1x20x8x3x3xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 2
  // GRID: rock.gridwise_gemm
  %result = rock.conv_bwd_weight(%input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<7x1x8x32x32xf32>, tensor<7x1x20x30x30xf32> -> tensor<1x20x8x3x3xf32>
  %out = rock.store %result to %filter by set : tensor<1x20x8x3x3xf32> -> tensor<1x20x8x3x3xf32> to tensor<1x20x8x3x3xf32>
  return %out : tensor<1x20x8x3x3xf32>
}

// CHECK-LABEL: @rock_conv_bwd_weight_padALL_f16
// GRID-LABEL: rock_conv_bwd_weight_padALL_f16
func.func @rock_conv_bwd_weight_padALL_f16(%filter : tensor<1x20x8x3x3xf16>, %input : tensor<7x1x8x32x32xf16>, %output : tensor<7x1x20x30x30xf16>) -> tensor<1x20x8x3x3xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 2
  // GRID: rock.gridwise_gemm
  %result = rock.conv_bwd_weight(%input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<7x1x8x32x32xf16>, tensor<7x1x20x30x30xf16> -> tensor<1x20x8x3x3xf16>
  %out = rock.store %result to %filter by set : tensor<1x20x8x3x3xf16> -> tensor<1x20x8x3x3xf16> to tensor<1x20x8x3x3xf16>
  return %out : tensor<1x20x8x3x3xf16>
}

// CHECK-LABEL: @rock_conv_7x7_tuning
// GRID-LABEL: @rock_conv_7x7_tuning
func.func @rock_conv_7x7_tuning(%arg0: tensor<1x64x3x7x7xf32>, %arg1: tensor<256x1x3x230x230xf32>, %arg2: tensor<256x1x64x112x112xf32>) -> tensor<256x1x64x112x112xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>,
  // GRID-SAME: rock.grid_size = 50176
  // GRID: rock.gridwise_gemm
  %result = rock.conv(%arg0, %arg1) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    // Restore this once the kPack + padding support works
    // perf_config = "v3:64,256,8,64,64,4,1,1,2,1,1",
    // rocMLIR perf_config:
    // perf_config = "v3:64,256,8,64,64,1,1,1,2,1,2",
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0",
    strides = [2 : index, 2 : index]
  } : tensor<1x64x3x7x7xf32>, tensor<256x1x3x230x230xf32> -> tensor<256x1x64x112x112xf32>
  %out = rock.store %result to %arg2 by set : tensor<256x1x64x112x112xf32> -> tensor<256x1x64x112x112xf32> to tensor<256x1x64x112x112xf32>
  return %out : tensor<256x1x64x112x112xf32>
}

// CHECK-LABEL: @rock_conv_7x7
// GRID-LABEL: rock_conv_7x7
func.func @rock_conv_7x7(%arg0: tensor<1x64x3x7x7xf32>, %arg1: tensor<256x1x3x230x230xf32>, %arg2: tensor<256x1x64x112x112xf32>) -> tensor<256x1x64x112x112xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 50176
  // GRID: rock.gridwise_gemm
  %result = rock.conv(%arg0, %arg1) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [2 : index, 2 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<1x64x3x7x7xf32>, tensor<256x1x3x230x230xf32> -> tensor<256x1x64x112x112xf32>
  %out = rock.store %result to %arg2 by set : tensor<256x1x64x112x112xf32> -> tensor<256x1x64x112x112xf32> to tensor<256x1x64x112x112xf32>
  return %out : tensor<256x1x64x112x112xf32>
}

// CHECK-LABEL: @rock_conv_bwd_weight_7x7
// GRID-LABEL: rock_conv_bwd_weight_7x7
func.func @rock_conv_bwd_weight_7x7(%arg0: tensor<1x64x3x7x7xf32>, %arg1: tensor<256x1x3x230x230xf32>, %arg2: tensor<256x1x64x112x112xf32>) -> tensor<1x64x3x7x7xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3
  // GRID: rock.gridwise_gemm
  %result = rock.conv_bwd_weight(%arg1, %arg2) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [2 : index, 2 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"
  } : tensor<256x1x3x230x230xf32>, tensor<256x1x64x112x112xf32> -> tensor<1x64x3x7x7xf32>
  %out = rock.store %result to %arg0 by set : tensor<1x64x3x7x7xf32> -> tensor<1x64x3x7x7xf32> to tensor<1x64x3x7x7xf32>
  return %out : tensor<1x64x3x7x7xf32>
}

// CHECK-LABEL: @rock_gemm_from_conv
// GRID-LABEL: rock_gemm_from_conv
func.func @rock_gemm_from_conv(%a : tensor<1x72x128xf32>, %b : tensor<1x72x115200xf32>, %c : tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.gemm tr %a * %b {perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"}
  : tensor<1x72x128xf32> * tensor<1x72x115200xf32> -> tensor<1x128x115200xf32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out : tensor<1x128x115200xf32>
}

// CHECK-LABEL: func.func @rock_gemm_from_i8_conv
// GRID-LABEL: rock_gemm_from_i8_conv
func.func @rock_gemm_from_i8_conv(%a : tensor<1x72x128xi8>, %b : tensor<1x72x115200xi8>, %c : tensor<1x128x115200xi32>) -> tensor<1x128x115200xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 2, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.gemm tr %a * %b {perf_config = "gemm:v1:64,64,64,2,1,4,32,1,2,0,0"}
  : tensor<1x72x128xi8> * tensor<1x72x115200xi8> -> tensor<1x128x115200xi32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xi32> -> tensor<1x128x115200xi32> to tensor<1x128x115200xi32>
  return %out : tensor<1x128x115200xi32>
}

// CHECK-LABEL: func.func @rock_gemm_from_i8_conv_numstages2
// GRID-LABEL: rock_gemm_from_i8_conv_numstages2
func.func @rock_gemm_from_i8_conv_numstages2(%a : tensor<1x72x128xi8>, %b : tensor<1x72x115200xi8>, %c : tensor<1x128x115200xi32>) -> tensor<1x128x115200xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 2, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.gemm tr %a * %b {perf_config = "gemm:v1:64,64,64,2,1,4,32,1,2,0,0"}
  : tensor<1x72x128xi8> * tensor<1x72x115200xi8> -> tensor<1x128x115200xi32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xi32> -> tensor<1x128x115200xi32> to tensor<1x128x115200xi32>
  return %out : tensor<1x128x115200xi32>
}

// The available xdlops for int8 change on gfx942, verify that different tuning
// parameters are picked.

// CHECK-LABEL: func.func @rock_gemm_from_i8_conv_gfx942
// GRID-LABEL: rock_gemm_from_i8_conv_gfx942
func.func @rock_gemm_from_i8_conv_gfx942(%a : tensor<1x72x128xi8>, %b : tensor<1x72x115200xi8>, %c : tensor<1x128x115200xi32>) -> tensor<1x128x115200xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 2, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.gemm tr %a * %b {perf_config = "gemm:v1:64,64,64,2,1,4,32,1,2,0,0"}
  : tensor<1x72x128xi8> * tensor<1x72x115200xi8> -> tensor<1x128x115200xi32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xi32> -> tensor<1x128x115200xi32> to tensor<1x128x115200xi32>
  return %out : tensor<1x128x115200xi32>
}

// And verify that 8-bit floats have the same tuning behavior as i8.
// CHECK-LABEL: func.func @rock_gemm_xdlops_fp8_bf8
// GRID-LABEL: rock_gemm_xdlops_fp8_bf8
func.func @rock_gemm_xdlops_fp8_bf8(%a : tensor<1x72x128xf8E4M3FNUZ>, %b : tensor<1x72x115200xf8E5M2FNUZ>, %c : tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", rock.num_cu = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 2, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.gemm tr %a * %b {perf_config = "gemm:v1:64,64,64,2,1,4,32,1,2,0,0"}
  : tensor<1x72x128xf8E4M3FNUZ> * tensor<1x72x115200xf8E5M2FNUZ> -> tensor<1x128x115200xf32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out : tensor<1x128x115200xf32>
}

// And verify that 8-bit floats have the same tuning behavior as i8.
// CHECK-LABEL: func.func @rock_gemm_xdlops_fp8_bf8_ocp
// GRID-LABEL: rock_gemm_xdlops_fp8_bf8_ocp
func.func @rock_gemm_xdlops_fp8_bf8_ocp(%a : tensor<1x72x128xf8E4M3FN>, %b : tensor<1x72x115200xf8E5M2>, %c : tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.num_cu = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.grid_size = 3600
  // GRID: rock.gridwise_gemm
  %result = rock.gemm tr %a * %b {perf_config = "gemm:v1:64,64,64,1,1,4,32,1,2,0,0"}
  : tensor<1x72x128xf8E4M3FN> * tensor<1x72x115200xf8E5M2> -> tensor<1x128x115200xf32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out : tensor<1x128x115200xf32>
}

// CHECK-LABEL: func.func @rock_attention_default
// CHECK-SAME: rock.block_size = 128
// GRID-LABEL: func.func @rock_attention_default
// GRID-SAME: rock.grid_size = 12
func.func @rock_attention_default(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.attention
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.attention{
   qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %arg2 : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32} -> tensor<1x384x64xf16>
  %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
  return %out : tensor<1x384x64xf16>
}

// CHECK-LABEL: func.func @rock_attention_large
// CHECK-SAME: rock.block_size = 256
// GRID-LABEL: func.func @rock_attention_large
// GRID-SAME: rock.grid_size = 128
func.func @rock_attention_large(%arg0: tensor<1x16384x512xf32>, %arg1: tensor<1x512x16384xf32>, %arg2: tensor<1x16384x512xf32>, %arg3: tensor<1x16384x512xf32>) -> tensor<1x16384x512xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-"} {
  // CHECK: rock.attention
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 512, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.attention{
    qk = %arg0 * %arg1 : tensor<1x16384x512xf32>, tensor<1x512x16384xf32>
    softmax(qk) * %arg2 : tensor<1x16384x512xf32>
  } {perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0", numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x16384x512xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32> to tensor<1x16384x512xf32>
  return %out : tensor<1x16384x512xf32>
}

// CHECK-LABEL: func.func @rock_attention_mperblockg1_wmma
// CHECK-SAME: rock.block_size = 128
// GRID-LABEL: func.func @rock_attention_mperblockg1_wmma
// GRID-SAME: rock.grid_size = 3
func.func @rock_attention_mperblockg1_wmma(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.attention
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 64, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.attention{
   qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %arg2 : tensor<1x384x64xf16>
  } {perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0", numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x384x64xf16>
  %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
  return %out : tensor<1x384x64xf16>
}

// CHECK-LABEL: func.func @rock_attention_mperblockg1_mfma
// CHECK-SAME: rock.block_size = 256
// GRID-LABEL: func.func @rock_attention_mperblockg1_mfma
// GRID-SAME: rock.grid_size = 3
func.func @rock_attention_mperblockg1_mfma(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "gfx942:sramecc+:xnack-"} {
  // CHECK: rock.attention
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 64, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.attention{
   qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %arg2 : tensor<1x384x64xf16>
  } {perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0", numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32} -> tensor<1x384x64xf16>
  %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
  return %out : tensor<1x384x64xf16>
}

// CHECK-LABEL: func.func @rock_gemm_gemm_default
// CHECK-SAME: rock.block_size = 128
// GRID-LABEL: func.func @rock_gemm_gemm_default
// GRID-SAME: rock.grid_size = 12
func.func @rock_gemm_gemm_default(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.gemm_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.gemm_elementwise_gemm{
   ab = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   out = ab * %arg2 : tensor<1x384x64xf16>
  } -> tensor<1x384x64xf16>
  %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
  return %out : tensor<1x384x64xf16>
}

// CHECK-LABEL: func.func @rock_gemm_gemm_v1
// CHECK-SAME: rock.block_size = 256
// GRID-LABEL: func.func @rock_gemm_gemm_v1
// GRID-SAME: rock.grid_size = 128
func.func @rock_gemm_gemm_v1(%arg0: tensor<1x16384x512xf32>, %arg1: tensor<1x512x16384xf32>, %arg2: tensor<1x16384x512xf32>, %arg3: tensor<1x16384x512xf32>) -> tensor<1x16384x512xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-"} {
  // CHECK: rock.gemm_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 2, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 512, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.gemm_elementwise_gemm{
    ab = %arg0 * %arg1 : tensor<1x16384x512xf32>, tensor<1x512x16384xf32>
    out = ab * %arg2 : tensor<1x16384x512xf32>
  } {perf_config = "attn:v1:128,128,2,1,1,4,0,1,2,0,0"} -> tensor<1x16384x512xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32> to tensor<1x16384x512xf32>
  return %out : tensor<1x16384x512xf32>
}

// CHECK-LABEL: func.func @rock_gemm_gemm_large
// CHECK-SAME: rock.block_size = 256
// GRID-LABEL: func.func @rock_gemm_gemm_large
// GRID-SAME: rock.grid_size = 128
func.func @rock_gemm_gemm_large(%arg0: tensor<1x16384x512xf32>, %arg1: tensor<1x512x16384xf32>, %arg2: tensor<1x16384x512xf32>, %arg3: tensor<1x16384x512xf32>) -> tensor<1x16384x512xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-"} {
  // CHECK: rock.gemm_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 512, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.gemm_elementwise_gemm{
    ab = %arg0 * %arg1 : tensor<1x16384x512xf32>, tensor<1x512x16384xf32>
    out = ab * %arg2 : tensor<1x16384x512xf32>
  } {perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0"} -> tensor<1x16384x512xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32> to tensor<1x16384x512xf32>
  return %out : tensor<1x16384x512xf32>
}

// CHECK-LABEL: func.func @rock_gemm_gemm_mperblockg1_wmma
// CHECK-SAME: rock.block_size = 128
// GRID-LABEL: func.func @rock_gemm_gemm_mperblockg1_wmma
// GRID-SAME: rock.grid_size = 3
func.func @rock_gemm_gemm_mperblockg1_wmma(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.gemm_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 64, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.gemm_elementwise_gemm{
   ab = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   out = ab * %arg2 : tensor<1x384x64xf16>
  } {perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0"} -> tensor<1x384x64xf16>
  %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
  return %out : tensor<1x384x64xf16>
}

// CHECK-LABEL: func.func @rock_gemm_gemm_mperblockg1_mfma
// CHECK-SAME: rock.block_size = 256
// GRID-LABEL: func.func @rock_gemm_gemm_mperblockg1_mfma
// GRID-SAME: rock.grid_size = 3
func.func @rock_gemm_gemm_mperblockg1_mfma(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {rock.kernel, rock.arch = "gfx942:sramecc+:xnack-"} {
  // CHECK: rock.gemm_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 64, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.gemm_elementwise_gemm{
   ab = %arg0 * tr %arg1 : tensor<1x384x64xf32>, tensor<1x384x64xf32>
   out = ab * %arg2 : tensor<1x384x64xf32>
  } {perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0"} -> tensor<1x384x64xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x384x64xf32> -> tensor<1x384x64xf32> to tensor<1x384x64xf32>
  return %out : tensor<1x384x64xf32>
}

// CHECK-LABEL: func.func @rock_conv_gemm_default
// CHECK-SAME: rock.block_size = 128
// GRID-LABEL: func.func @rock_conv_gemm_default
// GRID-SAME: rock.grid_size = 64
func.func @rock_conv_gemm_default(%arg0: tensor<1x128x256x1x1xf16>, %arg1: tensor<2x1x256x32x32xf16>, %arg2: tensor<1x128x64xf16>, %arg3: tensor<1x2048x64xf16>) -> tensor<1x2048x64xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.conv_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.conv_elementwise_gemm{
   ab = conv(%arg0, %arg1) : tensor<1x128x256x1x1xf16>, tensor<2x1x256x32x32xf16>
   out = ab * %arg2 : tensor<1x128x64xf16>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} -> tensor<1x2048x64xf16>
  %out = rock.store %result to %arg3 by set : tensor<1x2048x64xf16> -> tensor<1x2048x64xf16> to tensor<1x2048x64xf16>
  return %out : tensor<1x2048x64xf16>
}

// CHECK-LABEL: func.func @rock_conv_gemm_splitk
// CHECK-SAME: rock.block_size = 256
// GRID-LABEL: func.func @rock_conv_gemm_splitk
// GRID-SAME: rock.grid_size = 2048
func.func @rock_conv_gemm_splitk(%arg0: tensor<1x128x256x3x3xf32>, %arg1: tensor<2x1x256x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32768x128xf32>) -> tensor<1x32768x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  // CHECK: rock.conv_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 8, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.conv_elementwise_gemm{
   ab = conv(%arg0, %arg1) : tensor<1x128x256x3x3xf32>, tensor<2x1x256x128x128xf32>
   out = ab * %arg2 : tensor<1x128x128xf32>
  } {dilations = [1 : index, 1 : index], perf_config = "attn:v1:128,128,16,1,1,4,0,8,1,0,0", filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} -> tensor<1x32768x128xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x32768x128xf32> -> tensor<1x32768x128xf32> to tensor<1x32768x128xf32>
  return %out : tensor<1x32768x128xf32>
}

// CHECK-LABEL: func.func @rock_conv_gemm_large
// CHECK-SAME: rock.block_size = 256
// GRID-LABEL: func.func @rock_conv_gemm_large
// GRID-SAME: rock.grid_size = 256
func.func @rock_conv_gemm_large(%arg0: tensor<1x128x256x3x3xf32>, %arg1: tensor<2x1x256x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32768x128xf32>) -> tensor<1x32768x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  // CHECK: rock.conv_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.conv_elementwise_gemm{
   ab = conv(%arg0, %arg1) : tensor<1x128x256x3x3xf32>, tensor<2x1x256x128x128xf32>
   out = ab * %arg2 : tensor<1x128x128xf32>
  } {dilations = [1 : index, 1 : index], perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0", filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} -> tensor<1x32768x128xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x32768x128xf32> -> tensor<1x32768x128xf32> to tensor<1x32768x128xf32>
  return %out : tensor<1x32768x128xf32>
}

// CHECK-LABEL: func.func @rock_conv_gemm_mperblockg1_wmma
// CHECK-SAME: rock.block_size = 128
// GRID-LABEL: func.func @rock_conv_gemm_mperblockg1_wmma
// GRID-SAME: rock.grid_size = 256
func.func @rock_conv_gemm_mperblockg1_wmma(%arg0: tensor<1x128x256x1x1xf16>, %arg1: tensor<2x1x256x128x128xf16>, %arg2: tensor<1x128x128xf16>, %arg3: tensor<1x32768x128xf16>) -> tensor<1x32768x128xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.conv_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.conv_elementwise_gemm{
   ab = conv(%arg0, %arg1) : tensor<1x128x256x1x1xf16>, tensor<2x1x256x128x128xf16>
   out = ab * %arg2 : tensor<1x128x128xf16>
  } {dilations = [1 : index, 1 : index], perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0", filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} -> tensor<1x32768x128xf16>
  %out = rock.store %result to %arg3 by set : tensor<1x32768x128xf16> -> tensor<1x32768x128xf16> to tensor<1x32768x128xf16>
  return %out : tensor<1x32768x128xf16>
}

// CHECK-LABEL: func.func @rock_conv_gemm_mperblockg1_mfma
// CHECK-SAME: rock.block_size = 256
// GRID-LABEL: func.func @rock_conv_gemm_mperblockg1_mfma
// GRID-SAME: rock.grid_size = 256
func.func @rock_conv_gemm_mperblockg1_mfma(%arg0: tensor<1x128x256x1x1xf32>, %arg1: tensor<2x1x256x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32768x128xf32>) -> tensor<1x32768x128xf32> attributes {rock.kernel, rock.arch = "gfx942:sramecc+:xnack-"} {
  // CHECK: rock.conv_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 128, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 1, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.conv_elementwise_gemm{
   ab = conv(%arg0, %arg1) : tensor<1x128x256x1x1xf32>, tensor<2x1x256x128x128xf32>
   out = ab * %arg2 : tensor<1x128x128xf32>
  } {dilations = [1 : index, 1 : index], perf_config = "attn:v1:128,128,16,1,1,4,0,1,1,0,0", filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} -> tensor<1x32768x128xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x32768x128xf32> -> tensor<1x32768x128xf32> to tensor<1x32768x128xf32>
  return %out : tensor<1x32768x128xf32>
}

// CHECK-LABEL: func.func @rock_conv_tuning
// GRID-LABEL: func.func @rock_conv_tuning
func.func @rock_conv_tuning(%arg0: tensor<1x1x1x3x3xf32>, %arg1: tensor<64x1x1x14x14xf32>, %arg2: tensor<64x1x1x14x14xf32>) -> tensor<64x1x1x14x14xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  %result = rock.conv(%arg0, %arg1) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    rock.num_cu = 110 : i32,
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [1 : index, 1 : index, 1 : index, 1 : index],
    perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0",
    strides = [1 : index, 1 : index]} : tensor<1x1x1x3x3xf32>, tensor<64x1x1x14x14xf32> -> tensor<64x1x1x14x14xf32>
  %out = rock.store %result to %arg2 by set : tensor<64x1x1x14x14xf32> -> tensor<64x1x1x14x14xf32> to tensor<64x1x1x14x14xf32>
  return %out : tensor<64x1x1x14x14xf32>
}

// CHECK-LABEL: @rock_attn_perfconfig_numstages2
func.func @rock_attn_perfconfig_numstages2(%arg0: tensor<32768xf16>, %arg1: tensor<32768xf16>, %arg2: tensor<32768xf16>, %arg3: tensor<32768xf16>) -> tensor<32768xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 32 + d2)> by [<Unmerge{1024, 32} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 32] -> [32768]> : tensor<32768xf16> to tensor<1x1024x32xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 1024 + d2)> by [<Unmerge{32, 1024} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 1024] -> [32768]> : tensor<32768xf16> to tensor<1x32x1024xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 32 + d2)> by [<Unmerge{1024, 32} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 32] -> [32768]> : tensor<32768xf16> to tensor<1x1024x32xf16>
  // CHECK: rock.attention
  // CHECK: numStages = 2
  %result = rock.attention{
    qk = %0 * %1 : tensor<1x1024x32xf16>, tensor<1x32x1024xf16>
    qk = elementwise {
  ^bb0(%arg4: tensor<1x1024x1024xf16>):
    rock.yield %arg4 : tensor<1x1024x1024xf16>
  }
    softmax(qk) * %2 : tensor<1x1024x32xf16>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, perf_config = "attn:v1:32,32,32,1,1,1,0,1,2,0,0", softmaxType = f32, splitKV = 1 : i32} -> tensor<1x1024x32xf16>
  %3 = rock.transform %result by <affine_map<(d0) -> (0, d0 floordiv 32, d0 mod 32)> by [<Merge{1024, 32} ["raw"] at [0] -> ["seq_q", "head_v"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [32768] -> [1, 1024, 32]> : tensor<1x1024x32xf16> to tensor<32768xf16>
  %4 = rock.store %3 to %arg3 by  set : tensor<32768xf16> -> tensor<32768xf16> to tensor<32768xf16>
  return %4 : tensor<32768xf16>
}

// CHECK-LABEL: @rock_attn_perfconfig_numstages3
func.func @rock_attn_perfconfig_numstages3(%arg0: tensor<32768xf16>, %arg1: tensor<32768xf16>, %arg2: tensor<32768xf16>, %arg3: tensor<32768xf16>) -> tensor<32768xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 32 + d2)> by [<Unmerge{1024, 32} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 32] -> [32768]> : tensor<32768xf16> to tensor<1x1024x32xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 1024 + d2)> by [<Unmerge{32, 1024} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 1024] -> [32768]> : tensor<32768xf16> to tensor<1x32x1024xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 32 + d2)> by [<Unmerge{1024, 32} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 32] -> [32768]> : tensor<32768xf16> to tensor<1x1024x32xf16>
  // CHECK: rock.attention
  // CHECK: numStages = 3
  %result = rock.attention{
    qk = %0 * %1 : tensor<1x1024x32xf16>, tensor<1x32x1024xf16>
    qk = elementwise {
  ^bb0(%arg4: tensor<1x1024x1024xf16>):
    rock.yield %arg4 : tensor<1x1024x1024xf16>
  }
    softmax(qk) * %2 : tensor<1x1024x32xf16>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, perf_config = "attn:v1:32,32,32,1,1,1,0,1,3,0,0", softmaxType = f32, splitKV = 1 : i32} -> tensor<1x1024x32xf16>
  %3 = rock.transform %result by <affine_map<(d0) -> (0, d0 floordiv 32, d0 mod 32)> by [<Merge{1024, 32} ["raw"] at [0] -> ["seq_q", "head_v"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [32768] -> [1, 1024, 32]> : tensor<1x1024x32xf16> to tensor<32768xf16>
  %4 = rock.store %3 to %arg3 by  set : tensor<32768xf16> -> tensor<32768xf16> to tensor<32768xf16>
  return %4 : tensor<32768xf16>
}

// CHECK-LABEL: @rock_attn_perfconfig_numstages4
func.func @rock_attn_perfconfig_numstages4(%arg0: tensor<32768xf16>, %arg1: tensor<32768xf16>, %arg2: tensor<32768xf16>, %arg3: tensor<32768xf16>) -> tensor<32768xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", rock.kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 32 + d2)> by [<Unmerge{1024, 32} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 32] -> [32768]> : tensor<32768xf16> to tensor<1x1024x32xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 1024 + d2)> by [<Unmerge{32, 1024} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 32, 1024] -> [32768]> : tensor<32768xf16> to tensor<1x32x1024xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 32 + d2)> by [<Unmerge{1024, 32} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 32] -> [32768]> : tensor<32768xf16> to tensor<1x1024x32xf16>
  // CHECK: rock.attention
  // CHECK: numStages = 4
  %result = rock.attention{
    qk = %0 * %1 : tensor<1x1024x32xf16>, tensor<1x32x1024xf16>
    qk = elementwise {
  ^bb0(%arg4: tensor<1x1024x1024xf16>):
    rock.yield %arg4 : tensor<1x1024x1024xf16>
  }
    softmax(qk) * %2 : tensor<1x1024x32xf16>
  } {numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, perf_config = "attn:v1:32,32,32,1,1,1,0,1,4,0,0", softmaxType = f32, splitKV = 1 : i32} -> tensor<1x1024x32xf16>
  %3 = rock.transform %result by <affine_map<(d0) -> (0, d0 floordiv 32, d0 mod 32)> by [<Merge{1024, 32} ["raw"] at [0] -> ["seq_q", "head_v"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [32768] -> [1, 1024, 32]> : tensor<1x1024x32xf16> to tensor<32768xf16>
  %4 = rock.store %3 to %arg3 by  set : tensor<32768xf16> -> tensor<32768xf16> to tensor<32768xf16>
  return %4 : tensor<32768xf16>
}

// CHECK-LABEL: @rock_attn_schedule_default
// CHECK-SAME: rock.block_size = 128
// GRID-LABEL: @rock_attn_schedule_default
// GRID-SAME: rock.grid_size = 12
func.func @rock_attn_schedule_default(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.attention
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 64, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.attention{
   qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
   softmax(qk) * %arg2 : tensor<1x384x64xf16>
  } {splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32} -> tensor<1x384x64xf16>
  %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
  return %out : tensor<1x384x64xf16>
}

// CHECK-LABEL: @rock_gemm_gemm_splitk
// CHECK-SAME: rock.block_size = 64
// GRID-LABEL: @rock_gemm_gemm_splitk
// GRID-SAME: rock.grid_size = 256
func.func @rock_gemm_gemm_splitk(%arg0: tensor<1474560xf16>, %arg1: tensor<1474560xf16>, %arg2: tensor<1474560xf16>, %arg3: tensor<1474560xf16>) -> tensor<1474560xf16> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : tensor<1474560xf16> to tensor<1x4096x360xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 4096 + d2)> by [<Unmerge{360, 4096} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 360, 4096] -> [1474560]> : tensor<1474560xf16> to tensor<1x360x4096xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : tensor<1474560xf16> to tensor<1x4096x360xf16>
  // CHECK: rock.gemm_elementwise_gemm
  // CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 512, kPerBlock = 32, kpack = 1, numCTAs = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID: rock.gridwise_attention
  %result = rock.gemm_elementwise_gemm{
    ab = %0 * %1 : tensor<1x4096x360xf16>, tensor<1x360x4096xf16>
    ab = elementwise {
    ^bb0(%arg4: tensor<1x4096x4096xf16>):
      rock.yield %arg4 : tensor<1x4096x4096xf16>
    }
    out = ab * %2 : tensor<1x4096x360xf16>
  } {perf_config = "attn:v1:32,32,32,1,1,1,0,2,2,0,0"} -> tensor<1x4096x360xf16>
  %4 = rock.transform %result by <affine_map<(d0) -> (0, d0 floordiv 360, d0 mod 360)> by [<Merge{4096, 360} ["raw"] at [0] -> ["m", "gemmO"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [1474560] -> [1, 4096, 360]> : tensor<1x4096x360xf16> to tensor<1474560xf16>
  %out = rock.store %4 to %arg3 by set : tensor<1474560xf16> -> tensor<1474560xf16> to tensor<1474560xf16>
  return %out : tensor<1474560xf16>
}

// CHECK-LABEL: @mlir_dot_splitk
// GRID-LABEL: @mlir_dot_splitk
func.func @mlir_dot_splitk(%arg1: tensor<1x2x1280xf32>, %arg2: tensor<1x1280x320xf32>, %arg3: tensor<1x2x320xf32>) -> tensor<1x2x320xf32> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = tensor.empty() : tensor<1x2x320xf32>
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 64, kpack = 1, numCTAs = 1, numWaves = 4, matrixInstrNonkdim = 16, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0>
  // GRID-SAME: rock.grid_size = 5
  // GRID: rock.gridwise_gemm
  %result = rock.gemm %arg1 * %arg2 {rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", perf_config = "gemm:v1:64,64,64,1,1,4,16,1,2,0,0"} : tensor<1x2x1280xf32> * tensor<1x1280x320xf32> -> tensor<1x2x320xf32>
  %out = rock.store %result to %arg3 by set : tensor<1x2x320xf32> -> tensor<1x2x320xf32> to tensor<1x2x320xf32>
  return %out : tensor<1x2x320xf32>
}
