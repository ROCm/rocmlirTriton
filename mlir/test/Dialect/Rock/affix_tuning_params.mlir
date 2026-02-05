// This tests checks the following aspects of the lowering:
// * convolution tuning parameters are set as expected
// If versions of these tests appear in lowering_top_level, then changes to the tuning
// parameters made here should be reflected in that file

// RUN: rocmlir-driver -mlir-print-local-scope -rock-affix-params -verify-passes %s | FileCheck %s --check-prefix=CHECK
// RUN: rocmlir-driver -mlir-print-local-scope -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise %s | FileCheck %s --check-prefix=GRID

// CHECK-LABEL: @rock_conv
// GRID-LABEL: rock_conv
func.func @rock_conv(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x128x30x30xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32> to tensor<128x1x128x30x30xf32>
  return %out : tensor<128x1x128x30x30xf32>
}

// CHECK-LABEL: @rock_conv_schedulev2
// GRID-LABEL: rock_conv_schedulev2
func.func @rock_conv_schedulev2(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x128x30x30xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32> to tensor<128x1x128x30x30xf32>
  return %out : tensor<128x1x128x30x30xf32>
}

// CHECK-LABEL: func.func @rock_conv_f16
// GRID-LABEL: rock_conv_f16
func.func @rock_conv_f16(%filter : tensor<1x128x8x3x3xf16>, %input : tensor<128x1x8x32x32xf16>, %output : tensor<128x1x128x30x30xf16>) -> tensor<128x1x128x30x30xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x128x8x3x3xf16>, tensor<128x1x8x32x32xf16>, tensor<128x1x128x30x30xf16> -> tensor<128x1x128x30x30xf16>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf16> -> tensor<128x1x128x30x30xf16> to tensor<128x1x128x30x30xf16>
  return %out : tensor<128x1x128x30x30xf16>
}

// CHECK-LABEL: func.func @rock_conv_i8
// GRID-LABEL: rock_conv_i8
func.func @rock_conv_i8(%filter : tensor<1x128x8x3x3xi8>, %input : tensor<128x1x8x32x32xi8>, %output : tensor<128x1x128x30x30xi32>) -> tensor<128x1x128x30x30xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv(%filter, %input, %output) features = mfma|dot|atomic_add|atomic_add_f16 {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x128x8x3x3xi8>, tensor<128x1x8x32x32xi8>, tensor<128x1x128x30x30xi32> -> tensor<128x1x128x30x30xi32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xi32> -> tensor<128x1x128x30x30xi32> to tensor<128x1x128x30x30xi32>
  return %out : tensor<128x1x128x30x30xi32>
}

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: func.func @rock_conv_bwd_data
// DISABLED-GRID-LABEL: func.func @rock_conv_bwd_data
// func.func @rock_conv_bwd_data(%filter: tensor<1x1024x1024x1x1xf32>, %input: tensor<128x1x1024x14x14xf32>, %output: tensor<128x1x1024x14x14xf32>) -> tensor<128x1x1024x14x14xf32> attributes {kernel = 0 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.conv_bwd_data
//   // DISABLED-CHECK-SAME: params = #rock.gemm_params<
//   // DISABLED-GRID: rock.gridwise_gemm
//   // DISABLED-GRID-SAME: gridSize = 25088
//   %result = rock.conv_bwd_data(%filter, %input, %output) features = mfma|dot|atomic_add|atomic_add_f16 {
//     dilations = [1 : index, 1 : index],
//     filter_layout = ["g", "k", "c", "0", "1"],
//     kernelId = 0 : index,
//     input_layout = ["ni", "gi", "ci", "0i", "1i"],
//     output_layout = ["no", "go", "ko", "0o", "1o"],
//     padding = [0 : index, 0 : index, 0 : index, 0 : index],
//     strides = [1 : index, 1 : index],
//     usesV4R1 = true
//   } : tensor<1x1024x1024x1x1xf32>, tensor<128x1x1024x14x14xf32>, tensor<128x1x1024x14x14xf32> -> tensor<128x1x1024x14x14xf32>
//   %out = rock.store %result to %output by set : tensor<128x1x1024x14x14xf32> -> tensor<128x1x1024x14x14xf32> to tensor<128x1x1024x14x14xf32>
//   return %out : tensor<128x1x1024x14x14xf32>
// }

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: @rock_conv_bwd_data_f16
// DISABLED-GRID-LABEL: @rock_conv_bwd_data_f16
// func.func @rock_conv_bwd_data_f16(%filter: tensor<1x1024x1024x1x1xf16>, %input: tensor<128x1x1024x14x14xf16>, %output: tensor<128x1x1024x14x14xf16>) -> tensor<128x1x1024x14x14xf16> attributes {kernel = 0 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.conv_bwd_data
//   // DISABLED-CHECK-SAME: params = #rock.gemm_params<
//   // DISABLED-GRID: rock.gridwise_gemm
//   // DISABLED-GRID-SAME: gridSize = 25088
//   %result = rock.conv_bwd_data(%filter, %input, %output) features = mfma|dot|atomic_add|atomic_add_f16 {
//     dilations = [1 : index, 1 : index],
//     filter_layout = ["g", "k", "c", "0", "1"],
//     kernelId = 0 : index,
//     input_layout = ["ni", "gi", "ci", "0i", "1i"],
//     output_layout = ["no", "go", "ko", "0o", "1o"],
//     padding = [0 : index, 0 : index, 0 : index, 0 : index],
//     strides = [1 : index, 1 : index],
//     usesV4R1 = true
//   } : tensor<1x1024x1024x1x1xf16>, tensor<128x1x1024x14x14xf16>, tensor<128x1x1024x14x14xf16> -> tensor<128x1x1024x14x14xf16>
//   %out = rock.store %result to %output by set : tensor<128x1x1024x14x14xf16> -> tensor<128x1x1024x14x14xf16> to tensor<128x1x1024x14x14xf16>
//   return %out : tensor<128x1x1024x14x14xf16>
// }

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: func.func @rock_conv_bwd_data_padMN
// DISABLED-GRID-LABEL: func.func @rock_conv_bwd_data_padMN
// func.func @rock_conv_bwd_data_padMN(%filter : tensor<1x64x3x1x1xf32>, %input : tensor<11x1x3x15x15xf32>, %output : tensor<11x1x64x15x15xf32>) -> tensor<11x1x3x15x15xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
//   // DISABLED-CHECK: rock.conv_bwd_data
//   // DISABLED-CHECK-SAME: params = #rock.gemm_params<
//   // DISABLED-GRID: rock.gridwise_gemm
//   // DISABLED-GRID-SAME: gridSize = 39
//   %result = rock.conv_bwd_data(%filter, %input, %output) features = none {
//     filter_layout = ["g", "k", "c", "0", "1"],
//     input_layout = ["ni", "gi", "ci", "0i", "1i"],
//     output_layout = ["no", "go", "ko", "0o", "1o"],
//     dilations = [1 : index, 1 : index],
//     strides = [1 : index, 1 : index],
//     padding = [0 : index, 0 : index, 0 : index, 0 : index],
//     kernelId = 0 : index,
//     usesV4R1 = true
//   } : tensor<1x64x3x1x1xf32>, tensor<11x1x3x15x15xf32>, tensor<11x1x64x15x15xf32> -> tensor<11x1x3x15x15xf32>
//   %out = rock.store %result to %input by set : tensor<11x1x3x15x15xf32> -> tensor<11x1x3x15x15xf32> to tensor<11x1x3x15x15xf32>
//   return %out : tensor<11x1x3x15x15xf32>
// }

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: @rock_conv_bwd_data_padMK
// DISABLED-GRID-LABEL: @rock_conv_bwd_data_padMK
// func.func @rock_conv_bwd_data_padMK(%filter : tensor<1x11x3x1x1xf32>, %input : tensor<128x1x3x15x15xf32>, %output : tensor<128x1x11x15x15xf32>) -> tensor<128x1x3x15x15xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
//   // DISABLED-CHECK: rock.conv_bwd_data
//   // DISABLED-CHECK-SAME: params = #rock.gemm_params<
//   // DISABLED-GRID: rock.gridwise_gemm
//   // DISABLED-GRID-SAME: gridSize = 225
//   %result = rock.conv_bwd_data(%filter, %input, %output) features = none {
//     filter_layout = ["g", "k", "c", "0", "1"],
//     input_layout = ["ni", "gi", "ci", "0i", "1i"],
//     output_layout = ["no", "go", "ko", "0o", "1o"],
//     dilations = [1 : index, 1 : index],
//     strides = [1 : index, 1 : index],
//     padding = [0 : index, 0 : index, 0 : index, 0 : index],
//     kernelId = 0 : index,
//     usesV4R1 = true
//   } : tensor<1x11x3x1x1xf32>, tensor<128x1x3x15x15xf32>, tensor<128x1x11x15x15xf32> -> tensor<128x1x3x15x15xf32>
//   %out = rock.store %result to %input by set : tensor<128x1x3x15x15xf32> -> tensor<128x1x3x15x15xf32> to tensor<128x1x3x15x15xf32>
//   return %out : tensor<128x1x3x15x15xf32>
// }

// CHECK-LABEL: @rock_conv_bwd_weight
// GRID-LABEL: rock_conv_bwd_weight
func.func @rock_conv_bwd_weight(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<1x128x8x3x3xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 64 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv_bwd_weight(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<1x128x8x3x3xf32>
  %out = rock.store %result to %filter by set : tensor<1x128x8x3x3xf32> -> tensor<1x128x8x3x3xf32> to tensor<1x128x8x3x3xf32>
  return %out : tensor<1x128x8x3x3xf32>
}

// CHECK-LABEL: @rock_conv_bwd_weight_f16
// GRID-LABEL: rock_conv_bwd_weight_f16
func.func @rock_conv_bwd_weight_f16(%filter : tensor<1x128x8x3x3xf16>, %input : tensor<128x1x8x32x32xf16>, %output : tensor<128x1x128x30x30xf16>) -> tensor<1x128x8x3x3xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 64 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv_bwd_weight(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x128x8x3x3xf16>, tensor<128x1x8x32x32xf16>, tensor<128x1x128x30x30xf16> -> tensor<1x128x8x3x3xf16>
  %out = rock.store %result to %filter by set : tensor<1x128x8x3x3xf16> -> tensor<1x128x8x3x3xf16> to tensor<1x128x8x3x3xf16>
  return %out : tensor<1x128x8x3x3xf16>
}

// CHECK-LABEL: func.func @rock_conv_bwd_weight_padALL
// GRID-LABEL: rock_conv_bwd_weight_padALL
func.func @rock_conv_bwd_weight_padALL(%filter : tensor<1x20x8x3x3xf32>, %input : tensor<7x1x8x32x32xf32>, %output : tensor<7x1x20x30x30xf32>) -> tensor<1x20x8x3x3xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 64 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv_bwd_weight(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x20x8x3x3xf32>, tensor<7x1x8x32x32xf32>, tensor<7x1x20x30x30xf32> -> tensor<1x20x8x3x3xf32>
  %out = rock.store %result to %filter by set : tensor<1x20x8x3x3xf32> -> tensor<1x20x8x3x3xf32> to tensor<1x20x8x3x3xf32>
  return %out : tensor<1x20x8x3x3xf32>
}

// CHECK-LABEL: @rock_conv_bwd_weight_padALL_f16
// GRID-LABEL: rock_conv_bwd_weight_padALL_f16
func.func @rock_conv_bwd_weight_padALL_f16(%filter : tensor<1x20x8x3x3xf16>, %input : tensor<7x1x8x32x32xf16>, %output : tensor<7x1x20x30x30xf16>) -> tensor<1x20x8x3x3xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 64 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv_bwd_weight(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : tensor<1x20x8x3x3xf16>, tensor<7x1x8x32x32xf16>, tensor<7x1x20x30x30xf16> -> tensor<1x20x8x3x3xf16>
  %out = rock.store %result to %filter by set : tensor<1x20x8x3x3xf16> -> tensor<1x20x8x3x3xf16> to tensor<1x20x8x3x3xf16>
  return %out : tensor<1x20x8x3x3xf16>
}

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: @rock_conv_7x7_tuning
// DISABLED-GRID-LABEL: @rock_conv_7x7_tuning
// func.func @rock_conv_7x7_tuning(%arg0: tensor<1x64x3x7x7xf32>, %arg1: tensor<256x1x3x230x230xf32>, %arg2: tensor<256x1x64x112x112xf32>) -> tensor<256x1x64x112x112xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
//   // DISABLED-CHECK: rock.conv
//   // DISABLED-CHECK-SAME: params = #rock.gemm_params<mPerBlock = 64, nPerBlock = 256, numWaves = 1, kPerBlock = 8, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_gemm
//   // DISABLED-GRID-SAME: gridSize = 12544
//   %result = rock.conv(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add|atomic_add_f16 {
//     dilations = [1 : index, 1 : index],
//     filter_layout = ["g", "k", "c", "0", "1"],
//     input_layout = ["ni", "gi", "ci", "0i", "1i"],
//     output_layout = ["no", "go", "ko", "0o", "1o"],
//     padding = [0 : index, 0 : index, 0 : index, 0 : index],
//     // Restore this once the kPack + padding support works
//     // perf_config = "v3:64,256,8,64,64,4,1,1,2,1,1",
//     perf_config = "v3:64,256,8,64,64,1,1,1,2,1,2",
//     strides = [2 : index, 2 : index]
//   } : tensor<1x64x3x7x7xf32>, tensor<256x1x3x230x230xf32>, tensor<256x1x64x112x112xf32> -> tensor<256x1x64x112x112xf32>
//   %out = rock.store %result to %arg2 by set : tensor<256x1x64x112x112xf32> -> tensor<256x1x64x112x112xf32> to tensor<256x1x64x112x112xf32>
//   return %out : tensor<256x1x64x112x112xf32>
// }

// CHECK-LABEL: @rock_conv_7x7
// GRID-LABEL: rock_conv_7x7
func.func @rock_conv_7x7(%arg0: tensor<1x64x3x7x7xf32>, %arg1: tensor<256x1x3x230x230xf32>, %arg2: tensor<256x1x64x112x112xf32>) -> tensor<256x1x64x112x112xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add|atomic_add_f16 {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [2 : index, 2 : index]
  } : tensor<1x64x3x7x7xf32>, tensor<256x1x3x230x230xf32>, tensor<256x1x64x112x112xf32> -> tensor<256x1x64x112x112xf32>
  %out = rock.store %result to %arg2 by set : tensor<256x1x64x112x112xf32> -> tensor<256x1x64x112x112xf32> to tensor<256x1x64x112x112xf32>
  return %out : tensor<256x1x64x112x112xf32>
}

// CHECK-LABEL: @rock_conv_bwd_weight_7x7
// GRID-LABEL: rock_conv_bwd_weight_7x7
func.func @rock_conv_bwd_weight_7x7(%arg0: tensor<1x64x3x7x7xf32>, %arg1: tensor<256x1x3x230x230xf32>, %arg2: tensor<256x1x64x112x112xf32>) -> tensor<1x64x3x7x7xf32> attributes {kernel = 0 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 120 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.conv_bwd_weight(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add|atomic_add_f16 {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [2 : index, 2 : index]
  } : tensor<1x64x3x7x7xf32>, tensor<256x1x3x230x230xf32>, tensor<256x1x64x112x112xf32> -> tensor<1x64x3x7x7xf32>
  %out = rock.store %result to %arg0 by set : tensor<1x64x3x7x7xf32> -> tensor<1x64x3x7x7xf32> to tensor<1x64x3x7x7xf32>
  return %out : tensor<1x64x3x7x7xf32>
}

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: @rock_conv_bwd_data_7x7_tuning
// DISABLED-GRID-LABEL: @rock_conv_bwd_data_7x7_tuning
// func.func @rock_conv_bwd_data_7x7_tuning(%arg0: tensor<1x64x3x7x7xf32>, %arg1: tensor<256x1x3x230x230xf32>, %arg2: tensor<256x1x64x112x112xf32>) -> tensor<256x1x3x230x230xf32> attributes {kernel = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
//   // DISABLED-CHECK: rock.conv_bwd_data
//   // DISABLED-CHECK-SAME: params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 128, numWaves = 4, kPerBlock = 8, kpack = 4, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_gemm
//   // DISABLED-GRID-SAME: gridSize = 26450
//   %result = rock.conv_bwd_data(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add|atomic_add_f16 {
//     dilations = [1 : index, 1 : index],
//     filter_layout = ["g", "k", "c", "0", "1"],
//     kernelId = 1 : index,
//     input_layout = ["ni", "gi", "ci", "0i", "1i"],
//     output_layout = ["no", "go", "ko", "0o", "1o"],
//     padding = [0 : index, 0 : index, 0 : index, 0 : index],
//     perf_config = "v3:16,128,8,16,16,4,1,1,2,1,1",
//     strides = [2 : index, 2 : index],
//     usesV4R1 = true
//   } : tensor<1x64x3x7x7xf32>, tensor<256x1x3x230x230xf32>, tensor<256x1x64x112x112xf32> -> tensor<256x1x3x230x230xf32>
//   %out = rock.store %result to %arg1 by set : tensor<256x1x3x230x230xf32> -> tensor<256x1x3x230x230xf32> to tensor<256x1x3x230x230xf32>
//   return %out : tensor<256x1x3x230x230xf32>
// }

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: @rock_conv_bwd_data_7x7
// DISABLED-GRID-LABEL: @rock_conv_bwd_data_7x7
// func.func @rock_conv_bwd_data_7x7(%arg0: tensor<1x64x3x7x7xf32>, %arg1: tensor<256x1x3x230x230xf32>, %arg2: tensor<256x1x64x112x112xf32>) -> tensor<256x1x3x230x230xf32> attributes {kernel = 1 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
//   // DISABLED-CHECK: rock.conv_bwd_data
//   // DISABLED-CHECK-SAME: params = #rock.gemm_params<
//   // DISABLED-GRID: rock.gridwise_gemm
//   // DISABLED-GRID-SAME: gridSize = 211600
//   %result = rock.conv_bwd_data(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add|atomic_add_f16 {
//     dilations = [1 : index, 1 : index],
//     filter_layout = ["g", "k", "c", "0", "1"],
//     kernelId = 1 : index,
//     input_layout = ["ni", "gi", "ci", "0i", "1i"],
//     output_layout = ["no", "go", "ko", "0o", "1o"],
//     padding = [0 : index, 0 : index, 0 : index, 0 : index],
//     strides = [2 : index, 2 : index],
//     usesV4R1 = true
//   } : tensor<1x64x3x7x7xf32>, tensor<256x1x3x230x230xf32>, tensor<256x1x64x112x112xf32> -> tensor<256x1x3x230x230xf32>
//   %out = rock.store %result to %arg1 by set : tensor<256x1x3x230x230xf32> -> tensor<256x1x3x230x230xf32> to tensor<256x1x3x230x230xf32>
//   return %out : tensor<256x1x3x230x230xf32>
// }

// CHECK-LABEL: @rock_gemm_from_conv
// GRID-LABEL: rock_gemm_from_conv
func.func @rock_gemm_from_conv(%a : tensor<1x72x128xf32>, %b : tensor<1x72x115200xf32>, %c : tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 64 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.gemm tr %a * %b features = none
  : tensor<1x72x128xf32> * tensor<1x72x115200xf32> -> tensor<1x128x115200xf32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out : tensor<1x128x115200xf32>
}

// CHECK-LABEL: func.func @rock_gemm_from_i8_conv
// GRID-LABEL: rock_gemm_from_i8_conv
func.func @rock_gemm_from_i8_conv(%a : tensor<1x72x128xi8>, %b : tensor<1x72x115200xi8>, %c : tensor<1x128x115200xi32>) -> tensor<1x128x115200xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", numCU = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.gemm tr %a * %b features = mfma|dot|atomic_add|atomic_add_f16
  : tensor<1x72x128xi8> * tensor<1x72x115200xi8> -> tensor<1x128x115200xi32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xi32> -> tensor<1x128x115200xi32> to tensor<1x128x115200xi32>
  return %out : tensor<1x128x115200xi32>
}

// CHECK-LABEL: func.func @rock_gemm_from_i8_conv_schedule_v2
// GRID-LABEL: rock_gemm_from_i8_conv_schedule_v2
func.func @rock_gemm_from_i8_conv_schedule_v2(%a : tensor<1x72x128xi8>, %b : tensor<1x72x115200xi8>, %c : tensor<1x128x115200xi32>) -> tensor<1x128x115200xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", numCU = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.gemm tr %a * %b features = mfma|dot|atomic_add|atomic_add_f16
  : tensor<1x72x128xi8> * tensor<1x72x115200xi8> -> tensor<1x128x115200xi32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xi32> -> tensor<1x128x115200xi32> to tensor<1x128x115200xi32>
  return %out : tensor<1x128x115200xi32>
}

// The available xdlops for int8 change on gfx942, verify that different tuning
// parameters are picked.

// CHECK-LABEL: func.func @rock_gemm_from_i8_conv_gfx942
// GRID-LABEL: rock_gemm_from_i8_conv_gfx942
func.func @rock_gemm_from_i8_conv_gfx942(%a : tensor<1x72x128xi8>, %b : tensor<1x72x115200xi8>, %c : tensor<1x128x115200xi32>) -> tensor<1x128x115200xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", numCU = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.gemm tr %a * %b features = mfma|dot|atomic_add|atomic_add_f16
  : tensor<1x72x128xi8> * tensor<1x72x115200xi8> -> tensor<1x128x115200xi32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xi32> -> tensor<1x128x115200xi32> to tensor<1x128x115200xi32>
  return %out : tensor<1x128x115200xi32>
}

// And verify that 8-bit floats have the same tuning behavior as i8.
// CHECK-LABEL: func.func @rock_gemm_xdlops_fp8_bf8
// GRID-LABEL: rock_gemm_xdlops_fp8_bf8
func.func @rock_gemm_xdlops_fp8_bf8(%a : tensor<1x72x128xf8E4M3FNUZ>, %b : tensor<1x72x115200xf8E5M2FNUZ>, %c : tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942", numCU = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.gemm tr %a * %b features = mfma|dot|atomic_add|atomic_add_f16
  : tensor<1x72x128xf8E4M3FNUZ> * tensor<1x72x115200xf8E5M2FNUZ> -> tensor<1x128x115200xf32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out : tensor<1x128x115200xf32>
}

// And verify that 8-bit floats have the same tuning behavior as i8.
// CHECK-LABEL: func.func @rock_gemm_xdlops_fp8_bf8_ocp
// GRID-LABEL: rock_gemm_xdlops_fp8_bf8_ocp
func.func @rock_gemm_xdlops_fp8_bf8_ocp(%a : tensor<1x72x128xf8E4M3FN>, %b : tensor<1x72x115200xf8E5M2>, %c : tensor<1x128x115200xf32>) -> tensor<1x128x115200xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950", numCU = 120 : i32} {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.gemm_params<
  // GRID: rock.gridwise_gemm
  // GRID: rock.grid_size =
  %result = rock.gemm tr %a * %b features = mfma|dot|atomic_add|atomic_add_f16|atomic_add_bf16
  : tensor<1x72x128xf8E4M3FN> * tensor<1x72x115200xf8E5M2> -> tensor<1x128x115200xf32>
  %out = rock.store %result to %c by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out : tensor<1x128x115200xf32>
}

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_default
// DISABLED-CHECK-SAME: block_size = 32
// DISABLED-GRID-LABEL: func.func @rock_attention_default
// DISABLED-GRID-SAME: grid_size = 12
// func.func @rock_attention_default(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_large
// DISABLED-CHECK-SAME: block_size = 256
// DISABLED-GRID-LABEL: func.func @rock_attention_large
// DISABLED-GRID-SAME: grid_size = 128
// func.func @rock_attention_large(%arg0: tensor<1x16384x512xf32>, %arg1: tensor<1x512x16384xf32>, %arg2: tensor<1x16384x512xf32>, %arg3: tensor<1x16384x512xf32>) -> tensor<1x16384x512xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-"} {
//   %alloc = tensor.empty() : tensor<1x16384x512xf32>
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.attention{
//     qk = %arg0 * %arg1 : tensor<1x16384x512xf32>, tensor<1x512x16384xf32>
//     %arg3 = softmax(qk) * %arg2 : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v2:128,128,128,2,64,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32> to tensor<1x16384x512xf32>
//   return %out : tensor<1x16384x512xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_mperblockg1_wmma
// DISABLED-CHECK-SAME: block_size = 128
// DISABLED-GRID-LABEL: func.func @rock_attention_mperblockg1_wmma
// DISABLED-GRID-SAME: grid_size = 3
// func.func @rock_attention_mperblockg1_wmma(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 256, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, perf_config = "attn:v2:128,256,128,2,64,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: func.func @rock_attention_mperblockg1_mfma
// DISABLED-CHECK-SAME: block_size = 256
// DISABLED-GRID-LABEL: func.func @rock_attention_mperblockg1_mfma
// DISABLED-GRID-SAME: grid_size = 3
// func.func @rock_attention_mperblockg1_mfma(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {kernel, rock.arch = "gfx942:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 256, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v2:128,256,128,2,64,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_gemm_gemm_default
// DISABLED-CHECK-SAME: block_size = 32
// DISABLED-GRID-LABEL: func.func @rock_gemm_gemm_default
// DISABLED-GRID-SAME: grid_size = 12
// func.func @rock_gemm_gemm_default(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.gemm_elementwise_gemm
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.gemm_elementwise_gemm{
//    ab = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = ab * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_gemm_gemm_v1
// DISABLED-CHECK-SAME: block_size = 256
// DISABLED-GRID-LABEL: func.func @rock_gemm_gemm_v1
// DISABLED-GRID-SAME: grid_size = 128
// func.func @rock_gemm_gemm_v1(%arg0: tensor<1x16384x512xf32>, %arg1: tensor<1x512x16384xf32>, %arg2: tensor<1x16384x512xf32>, %arg3: tensor<1x16384x512xf32>) -> tensor<1x16384x512xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-"} {
//   %alloc = tensor.empty() : tensor<1x16384x512xf32>
//   // DISABLED-CHECK: rock.gemm_elementwise_gemm
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.gemm_elementwise_gemm{
//     ab = %arg0 * %arg1 : tensor<1x16384x512xf32>, tensor<1x512x16384xf32>
//     %arg3 = ab * %arg2 : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v1:128,128,128,2,64,64,8,1", firstGemmIndices = array<i64: 0>}
//   %out = rock.store %result to %arg3 by set : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32> to tensor<1x16384x512xf32>
//   return %out : tensor<1x16384x512xf32>
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_gemm_gemm_large
// DISABLED-CHECK-SAME: block_size = 256
// DISABLED-GRID-LABEL: func.func @rock_gemm_gemm_large
// DISABLED-GRID-SAME: grid_size = 128
// func.func @rock_gemm_gemm_large(%arg0: tensor<1x16384x512xf32>, %arg1: tensor<1x512x16384xf32>, %arg2: tensor<1x16384x512xf32>, %arg3: tensor<1x16384x512xf32>) -> tensor<1x16384x512xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-"} {
//   %alloc = tensor.empty() : tensor<1x16384x512xf32>
//   // DISABLED-CHECK: rock.gemm_elementwise_gemm
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.gemm_elementwise_gemm{
//     ab = %arg0 * %arg1 : tensor<1x16384x512xf32>, tensor<1x512x16384xf32>
//     %arg3 = ab * %arg2 : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v2:128,128,128,2,64,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x16384x512xf32> -> tensor<1x16384x512xf32> to tensor<1x16384x512xf32>
//   return %out : tensor<1x16384x512xf32>
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_gemm_gemm_mperblockg1_wmma
// DISABLED-CHECK-SAME: block_size = 128
// DISABLED-GRID-LABEL: func.func @rock_gemm_gemm_mperblockg1_wmma
// DISABLED-GRID-SAME: grid_size = 3
// func.func @rock_gemm_gemm_mperblockg1_wmma(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.gemm_elementwise_gemm
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 256, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.gemm_elementwise_gemm{
//    ab = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = ab * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, perf_config = "attn:v2:128,256,128,2,64,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_gemm_gemm_mperblockg1_mfma
// DISABLED-CHECK-SAME: block_size = 256
// DISABLED-GRID-LABEL: func.func @rock_gemm_gemm_mperblockg1_mfma
// DISABLED-GRID-SAME: grid_size = 3
// func.func @rock_gemm_gemm_mperblockg1_mfma(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {kernel, rock.arch = "gfx942:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.gemm_elementwise_gemm
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 256, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.gemm_elementwise_gemm{
//    ab = %arg0 * tr %arg1 : tensor<1x384x64xf32>, tensor<1x384x64xf32>
//    %arg3 = ab * %arg2 : tensor<1x384x64xf32> -> tensor<1x384x64xf32>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v2:128,256,128,2,64,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf32> -> tensor<1x384x64xf32> to tensor<1x384x64xf32>
//   return %out : tensor<1x384x64xf32>
// }

// TODO(roctriton): conv_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_conv_gemm_default
// DISABLED-CHECK-SAME: block_size = 32
// DISABLED-GRID-LABEL: func.func @rock_conv_gemm_default
// DISABLED-GRID-SAME: grid_size = 64
// func.func @rock_conv_gemm_default(%arg0: tensor<1x128x256x1x1xf16>, %arg1: tensor<2x1x256x32x32xf16>, %arg2: tensor<1x128x64xf16>, %arg3: tensor<1x2048x64xf16>) -> tensor<1x2048x64xf16> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.conv_elementwise_gemm
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.conv_elementwise_gemm{
//    ab = conv(%arg0, %arg1) : tensor<1x128x256x1x1xf16>, tensor<2x1x256x32x32xf16>
//    %arg3 = ab * %arg2 : tensor<1x128x64xf16> -> tensor<1x2048x64xf16>
//   } {dilations = [1 : index, 1 : index], features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_fmax_f32>, filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]}
//   %out = rock.store %result to %arg3 by set : tensor<1x2048x64xf16> -> tensor<1x2048x64xf16> to tensor<1x2048x64xf16>
//   return %out : tensor<1x2048x64xf16>
// }

// TODO(roctriton): conv_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_conv_gemm_splitk
// DISABLED-CHECK-SAME: block_size = 256
// DISABLED-GRID-LABEL: func.func @rock_conv_gemm_splitk
// DISABLED-GRID-SAME: grid_size = 2048
// func.func @rock_conv_gemm_splitk(%arg0: tensor<1x128x256x3x3xf32>, %arg1: tensor<2x1x256x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32768x128xf32>) -> tensor<1x32768x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.conv_elementwise_gemm
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 8, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.conv_elementwise_gemm{
//    ab = conv(%arg0, %arg1) : tensor<1x128x256x3x3xf32>, tensor<2x1x256x128x128xf32>
//    %arg3 = ab * %arg2 : tensor<1x128x128xf32> -> tensor<1x32768x128xf32>
//   } {dilations = [1 : index, 1 : index], features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v2:128,128,128,2,64,64,8,8,1,2,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]}
//   %out = rock.store %result to %arg3 by set : tensor<1x32768x128xf32> -> tensor<1x32768x128xf32> to tensor<1x32768x128xf32>
//   return %out : tensor<1x32768x128xf32>
// }

// TODO(roctriton): conv_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_conv_gemm_large
// DISABLED-CHECK-SAME: block_size = 256
// DISABLED-GRID-LABEL: func.func @rock_conv_gemm_large
// DISABLED-GRID-SAME: grid_size = 256
// func.func @rock_conv_gemm_large(%arg0: tensor<1x128x256x3x3xf32>, %arg1: tensor<2x1x256x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32768x128xf32>) -> tensor<1x32768x128xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.conv_elementwise_gemm
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.conv_elementwise_gemm{
//    ab = conv(%arg0, %arg1) : tensor<1x128x256x3x3xf32>, tensor<2x1x256x128x128xf32>
//    %arg3 = ab * %arg2 : tensor<1x128x128xf32> -> tensor<1x32768x128xf32>
//   } {dilations = [1 : index, 1 : index], features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v2:128,128,128,2,64,64,8,1,1,2,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]}
//   %out = rock.store %result to %arg3 by set : tensor<1x32768x128xf32> -> tensor<1x32768x128xf32> to tensor<1x32768x128xf32>
//   return %out : tensor<1x32768x128xf32>
// }

// TODO(roctriton): conv_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_conv_gemm_mperblockg1_wmma
// DISABLED-CHECK-SAME: block_size = 128
// DISABLED-GRID-LABEL: func.func @rock_conv_gemm_mperblockg1_wmma
// DISABLED-GRID-SAME: grid_size = 256
// func.func @rock_conv_gemm_mperblockg1_wmma(%arg0: tensor<1x128x256x1x1xf16>, %arg1: tensor<2x1x256x128x128xf16>, %arg2: tensor<1x128x128xf16>, %arg3: tensor<1x32768x128xf16>) -> tensor<1x32768x128xf16> attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.conv_elementwise_gemm
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 256, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.conv_elementwise_gemm{
//    ab = conv(%arg0, %arg1) : tensor<1x128x256x1x1xf16>, tensor<2x1x256x128x128xf16>
//    %arg3 = ab * %arg2 : tensor<1x128x128xf16> -> tensor<1x32768x128xf16>
//   } {dilations = [1 : index, 1 : index], features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_fmax_f32>, perf_config = "attn:v2:128,256,128,2,64,64,8,1,1,2,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]}
//   %out = rock.store %result to %arg3 by set : tensor<1x32768x128xf16> -> tensor<1x32768x128xf16> to tensor<1x32768x128xf16>
//   return %out : tensor<1x32768x128xf16>
// }

// TODO(roctriton): conv_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: func.func @rock_conv_gemm_mperblockg1_mfma
// DISABLED-CHECK-SAME: block_size = 256
// DISABLED-GRID-LABEL: func.func @rock_conv_gemm_mperblockg1_mfma
// DISABLED-GRID-SAME: grid_size = 256
// func.func @rock_conv_gemm_mperblockg1_mfma(%arg0: tensor<1x128x256x1x1xf32>, %arg1: tensor<2x1x256x128x128xf32>, %arg2: tensor<1x128x128xf32>, %arg3: tensor<1x32768x128xf32>) -> tensor<1x32768x128xf32> attributes {kernel, rock.arch = "gfx942:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.conv_elementwise_gemm
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, numWaves = 4, kPerBlock = 2, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: #rock.gemm_params<mPerBlock = 256, nPerBlock = 128, numWaves = 4, kPerBlock = 16, kpack = 8, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.conv_elementwise_gemm{
//    ab = conv(%arg0, %arg1) : tensor<1x128x256x1x1xf32>, tensor<2x1x256x128x128xf32>
//    %arg3 = ab * %arg2 : tensor<1x128x128xf32> -> tensor<1x32768x128xf32>
//   } {dilations = [1 : index, 1 : index], features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v2:128,256,128,2,64,64,8,1,1,2,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]}
//   %out = rock.store %result to %arg3 by set : tensor<1x32768x128xf32> -> tensor<1x32768x128xf32> to tensor<1x32768x128xf32>
//   return %out : tensor<1x32768x128xf32>
// }

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: func.func @rock_conv_tuning
// DISABLED-GRID-LABEL: func.func @rock_conv_tuning
// func.func @rock_conv_tuning(%arg0: tensor<1x1x1x3x3xf32>, %arg1: tensor<64x1x1x14x14xf32>, %arg2: tensor<64x1x1x14x14xf32>) -> tensor<64x1x1x14x14xf32> attributes {kernel = 0 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
//   %result = rock.conv(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add|atomic_add_f16 {
//     dilations = [1 : index, 1 : index],
//     filter_layout = ["g", "k", "c", "0", "1"],
//     input_layout = ["ni", "gi", "ci", "0i", "1i"],
//     numCU = 110 : i32,
//     output_layout = ["no", "go", "ko", "0o", "1o"],
//     padding = [1 : index, 1 : index, 1 : index, 1 : index],
//     perf_config = "v3:32,128,4,32,32,4,1,1,2,1,1",
//     strides = [1 : index, 1 : index]} : tensor<1x1x1x3x3xf32>, tensor<64x1x1x14x14xf32>, tensor<64x1x1x14x14xf32> -> tensor<64x1x1x14x14xf32>
//   %out = rock.store %result to %arg2 by set : tensor<64x1x1x14x14xf32> -> tensor<64x1x1x14x14xf32> to tensor<64x1x1x14x14xf32>
//   return %out : tensor<64x1x1x14x14xf32>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @rock_attn_schedulev2
// DISABLED-GRID-LABEL: @rock_attn_schedulev2
// func.func @rock_attn_schedulev2(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_attention
//   // DISABLED-GRID: gridSize = 12
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @rock_attn_schedulev3
// DISABLED-GRID-LABEL: @rock_attn_schedulev3
// func.func @rock_attn_schedulev3(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 3, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 3, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_attention
//   // DISABLED-GRID: gridSize = 12
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @rock_attn_schedulev4
// DISABLED-GRID-LABEL: @rock_attn_schedulev4
// func.func @rock_attn_schedulev4(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 4, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 4, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_attention
//   // DISABLED-GRID: gridSize = 12
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @rock_attn_perfconfig_schedulev2
// DISABLED-GRID-LABEL: @rock_attn_perfconfig_schedulev2
// func.func @rock_attn_perfconfig_schedulev2(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_attention
//   // DISABLED-GRID: gridSize = 12
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, perf_config = "attn:v2:32,32,32,32,32,32,1,1,2,2,1"}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @rock_attn_perfconfig_schedulev3
// DISABLED-GRID-LABEL: @rock_attn_perfconfig_schedulev3
// func.func @rock_attn_perfconfig_schedulev3(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_attention
//   // DISABLED-GRID: gridSize = 12
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, perf_config = "attn:v2:32,32,32,32,32,32,1,1,3,2,1"}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @rock_attn_perfconfig_schedulev4
// DISABLED-GRID-LABEL: @rock_attn_perfconfig_schedulev4
// func.func @rock_attn_perfconfig_schedulev4(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_attention
//   // DISABLED-GRID: gridSize = 12
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, perf_config = "attn:v2:32,32,32,32,32,32,1,1,4,2,1"}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): We need to unbufferize attention
// DISABLED-CHECK-LABEL: @rock_attn_schedule_default
// DISABLED-GRID-LABEL: @rock_attn_schedule_default
// func.func @rock_attn_schedule_default(%arg0: tensor<1x384x64xf16>, %arg1: tensor<1x384x64xf16>, %arg2: tensor<1x384x64xf16>, %arg3: tensor<1x384x64xf16>) -> tensor<1x384x64xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // DISABLED-CHECK: rock.attention
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK-SAME: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-GRID: rock.gridwise_attention
//   // DISABLED-GRID: gridSize = 12
//   %result = rock.attention{
//    qk = %arg0 * tr %arg1 : tensor<1x384x64xf16>, tensor<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : tensor<1x384x64xf16> -> tensor<1x384x64xf16>
//   } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32}
//   %out = rock.store %result to %arg3 by set : tensor<1x384x64xf16> -> tensor<1x384x64xf16> to tensor<1x384x64xf16>
//   return %out : tensor<1x384x64xf16>
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// DISABLED-CHECK-LABEL: @rock_gemm_gemm_splitk
// DISABLED-GRID-LABEL: @rock_gemm_gemm_splitk
// DISABLED-GRID: grid_size = 256
// func.func @rock_gemm_gemm_splitk(%arg0: tensor<1474560xf16>, %arg1: tensor<1474560xf16>, %arg2: tensor<1474560xf16>, %arg3: tensor<1474560xf16>) -> tensor<1x4096x360xf16> attributes {enable_splitk_for_tuning, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>} {
//   %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : tensor<1474560xf16> to tensor<1x4096x360xf16>
//   %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 4096 + d2)> by [<Unmerge{360, 4096} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 360, 4096] -> [1474560]> : tensor<1474560xf16> to tensor<1x360x4096xf16>
//   %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : tensor<1474560xf16> to tensor<1x4096x360xf16>
//   %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 1 + d2)> by [<Unmerge{4096, 360} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : tensor<1474560xf16> to tensor<1x4096x360xf16>
//   %alloc = tensor.empty() : tensor<1x4096x360xf16>
//   // DISABLED-CHECK: rock.gemm_elementwise_gemm
//   // DISABLED-CHECK: params0 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   // DISABLED-CHECK: params1 = #rock.gemm_params<mPerBlock = 32, nPerBlock = 32, numWaves = 1, kPerBlock = 32, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 2, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.gemm_elementwise_gemm{
//     ab = %0 * %1 : tensor<1x4096x360xf16>, tensor<1x360x4096xf16>
//     ab = elementwise {
//   ^bb0(%arg4: tensor<1x4096x4096xf16>, %arg5: tensor<1x4096x4096xf16>):
//     linalg.copy ins(%arg4 : tensor<1x4096x4096xf16>) outs(%arg5 : tensor<1x4096x4096xf16>)
//     rock.yield
//   }
//     %alloc = ab * %2 : tensor<1x4096x360xf16> -> tensor<1x4096x360xf16>
//   } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b>, firstGemmIndices = array<i64: 0>, perf_config="attn:v3:32,32,32,32,32,32,16,1,2,1,2,0,1"}
//   %out = rock.store %result to %3 by set : tensor<1x4096x360xf16> -> tensor<1x4096x360xf16> to tensor<1x4096x360xf16>
//   return %out : tensor<1x4096x360xf16>
// }

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// DISABLED-CHECK-LABEL: @mlir_dot_splitk
// DISABLED-GRID-LABEL: @mlir_dot_splitk
// DISABLED-GRID: grid_size = 100
// func.func @mlir_dot_splitk(%arg1: tensor<1x2x1280xf32>, %arg2: tensor<1x1280x320xf32>, %arg3: tensor<1x2x320xf32>) -> tensor<1x2x320xf32> attributes {enable_splitk_for_tuning, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
//   %cst = arith.constant 0.000000e+00 : f32
//   %alloc = tensor.empty() : tensor<1x2x320xf32>
//   // DISABLED-CHECK: rock.gemm
//   // DISABLED-CHECK-SAME: params = #rock.gemm_params<mPerBlock = 16, nPerBlock = 16, numWaves = 1, kPerBlock = 4, kpack = 1, matrixInstrNonkdim = 0, splitKFactor = 5, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
//   %result = rock.gemm %arg1 * %arg2 features =  mfma|dot|atomic_add|atomic_add_f16 {rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", perf_config = "v4:16,16,4,16,16,16,1,5,1,2,0,0,1,1"} : tensor<1x2x1280xf32> * tensor<1x1280x320xf32> -> tensor<1x2x320xf32>
//   %out = rock.store %result to %arg3 by set : tensor<1x2x320xf32> -> tensor<1x2x320xf32> to tensor<1x2x320xf32>
//   return %out : tensor<1x2x320xf32>
// }
