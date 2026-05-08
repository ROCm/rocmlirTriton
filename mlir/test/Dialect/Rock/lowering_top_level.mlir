// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly
// * Can pass arguments in the right sequence
// * Have, in most cases, the correct transformations
// * Have one gridwise_gemm
// * Can support F32 and F16

// RUN: rocmlir-opt --rock-affix-params --rock-lower-reduce --rock-regularize-output --rock-regularize-inter-gemm-fusion --rock-conv-to-gemm --mlir-print-local-scope %s | FileCheck %s

#gemm_params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 8, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#gemm_params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 4, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

// CHECK-LABEL: func.func @rock_conv
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x128x8x3x3xf32>, %[[arg1:.*]]: tensor<128x1x8x32x32xf32>, %[[arg2:.*]]: tensor<128x1x128x30x30xf32>)
// CHECK-NOT:   rock.conv
// CHECK:       %[[FILTER:.*]] = rock.transform %[[arg0]] by {{.*}}Merge{8, 3, 3} ["gemmK"]{{.*}}PassThrough ["gemmM"]
// CHECK:       %[[IN1:.*]] = rock.transform %[[arg1]] by {{.*}}Pad{0, 0, 0, 0}
// CHECK:       %[[IN2:.*]] = rock.transform %[[IN1]] by {{.*}}Embed{1, 1}{{.*}}Embed{1, 1}
// CHECK:       %[[IN3:.*]] = rock.transform %[[IN2]] by {{.*}}Merge{8, 3, 3} ["gemmK"]{{.*}}Merge{128, 30, 30} ["gemmN"]
// CHECK:       rock.gemm tr %[[FILTER]] * %[[IN3]]
func.func @rock_conv(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x128x30x30xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel, rock.num_cu = 120 : i32} {
  %result = rock.conv(%filter, %input) {
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params0,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32> -> tensor<128x1x128x30x30xf32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32> to tensor<128x1x128x30x30xf32>
  return %out : tensor<128x1x128x30x30xf32>
}

// CHECK-LABEL: func.func @rock_conv_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x128x8x3x3xf16>, %[[arg1:.*]]: tensor<128x1x8x32x32xf16>, %[[arg2:.*]]: tensor<128x1x128x30x30xf16>)
// CHECK-NOT:   rock.conv
// CHECK:       %[[FILTER:.*]] = rock.transform %[[arg0]] by {{.*}}Merge{8, 3, 3} ["gemmK"]{{.*}}PassThrough ["gemmM"]
// CHECK:       %[[IN1:.*]] = rock.transform %[[arg1]] by {{.*}}Pad{0, 0, 0, 0}
// CHECK:       %[[IN2:.*]] = rock.transform %[[IN1]] by {{.*}}Embed{1, 1}{{.*}}Embed{1, 1}
// CHECK:       %[[IN3:.*]] = rock.transform %[[IN2]] by {{.*}}Merge{8, 3, 3} ["gemmK"]{{.*}}Merge{128, 30, 30} ["gemmN"]
// CHECK:       rock.gemm tr %[[FILTER]] * %[[IN3]]
func.func @rock_conv_f16(%filter : tensor<1x128x8x3x3xf16>, %input : tensor<128x1x8x32x32xf16>, %output : tensor<128x1x128x30x30xf16>) -> tensor<128x1x128x30x30xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel, rock.num_cu = 120 : i32} {
  %result = rock.conv(%filter, %input) {
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params0,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xf16>, tensor<128x1x8x32x32xf16> -> tensor<128x1x128x30x30xf16>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf16> -> tensor<128x1x128x30x30xf16> to tensor<128x1x128x30x30xf16>
  return %out : tensor<128x1x128x30x30xf16>
}

// CHECK-LABEL: func.func @rock_conv_i8
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x128x8x3x3xi8>, %[[arg1:.*]]: tensor<128x1x8x32x32xi8>, %[[arg2:.*]]: tensor<128x1x128x30x30xi32>)
// CHECK-NOT:   rock.conv
// CHECK:       %[[FILTER:.*]] = rock.transform %[[arg0]] by {{.*}}Merge{8, 3, 3} ["gemmK"]{{.*}}PassThrough ["gemmM"]
// CHECK:       %[[IN1:.*]] = rock.transform %[[arg1]] by {{.*}}Pad{0, 0, 0, 0}
// CHECK:       %[[IN2:.*]] = rock.transform %[[IN1]] by {{.*}}Embed{1, 1}{{.*}}Embed{1, 1}
// CHECK:       %[[IN3:.*]] = rock.transform %[[IN2]] by {{.*}}Merge{8, 3, 3} ["gemmK"]{{.*}}Merge{128, 30, 30} ["gemmN"]
// CHECK:       rock.gemm tr %[[FILTER]] * %[[IN3]]
func.func @rock_conv_i8(%filter : tensor<1x128x8x3x3xi8>, %input : tensor<128x1x8x32x32xi8>, %output : tensor<128x1x128x30x30xi32>) -> tensor<128x1x128x30x30xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel, rock.num_cu = 120 : i32} {
  %result = rock.conv(%filter, %input) {
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params0,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xi8>, tensor<128x1x8x32x32xi8> -> tensor<128x1x128x30x30xi32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xi32> -> tensor<128x1x128x30x30xi32> to tensor<128x1x128x30x30xi32>
  return %out : tensor<128x1x128x30x30xi32>
}

// CHECK-LABEL: func.func @rock_conv_bwd_data
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x1024x1024x1x1xf32>, %[[arg1:.*]]: tensor<128x1x1024x14x14xf32>, %[[arg2:.*]]: tensor<128x1x1024x14x14xf32>)
// CHECK-NOT:   rock.conv_bwd_data
// CHECK:       %[[FIL1:.*]] = rock.transform %[[arg0]] by {{.*}}Embed{1, 1} ["0dot", "0tilda"]{{.*}}Embed{1, 1} ["1dot", "1tilda"]
// CHECK:       %[[FIL2:.*]] = rock.transform %[[FIL1]] by {{.*}}Slice{0, 1, 0, 1} ["0dotslice", "1dotslice"]{{.*}}Slice{0, 1, 0, 1} ["0tildaslice", "1tildaslice"]
// CHECK:       %[[FIL3:.*]] = rock.transform %[[FIL2]] by {{.*}}Merge{1024, 1, 1} ["gemmK"]{{.*}}Merge{1024, 1, 1} ["gemmM"]
// CHECK:       %[[OUT1:.*]] = rock.transform %[[arg2]] by {{.*}}Embed{-1, 1} ["0dot", "0tilda"]{{.*}}Embed{-1, 1} ["1dot", "1tilda"]
// CHECK:       %[[OUT2:.*]] = rock.transform %[[OUT1]] by {{.*}}Slice{0, 1, 0, 1} ["0slice", "1slice"]{{.*}}Slice{0, 14, 0, 14} ["0islice", "1islice"]
// CHECK:       %[[OUT3:.*]] = rock.transform %[[OUT2]] by {{.*}}Merge{1024, 1, 1} ["gemmK"]{{.*}}Merge{128, 14, 14} ["gemmN"]
// CHECK:       rock.gemm tr %[[FIL3]] * %[[OUT3]]
func.func @rock_conv_bwd_data(%filter: tensor<1x1024x1024x1x1xf32>, %input: tensor<128x1x1024x14x14xf32>, %output: tensor<128x1x1024x14x14xf32>) -> tensor<128x1x1024x14x14xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  %result = rock.conv_bwd_data(%filter, %output) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params1,
    strides = [1 : index, 1 : index]
  } : tensor<1x1024x1024x1x1xf32>, tensor<128x1x1024x14x14xf32> -> tensor<128x1x1024x14x14xf32>
  %out = rock.store %result to %input by set : tensor<128x1x1024x14x14xf32> -> tensor<128x1x1024x14x14xf32> to tensor<128x1x1024x14x14xf32>
  return %out : tensor<128x1x1024x14x14xf32>
}

// CHECK-LABEL: func.func @rock_conv_bwd_data_small
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x128x8x3x3xf32>, %[[arg1:.*]]: tensor<128x1x8x32x32xf32>, %[[arg2:.*]]: tensor<128x1x128x30x30xf32>)
// CHECK-NOT:   rock.conv_bwd_data
// CHECK:       %[[FIL1:.*]] = rock.transform %[[arg0]] by {{.*}}Embed{1, 1} ["0dot", "0tilda"]{{.*}}Embed{1, 1} ["1dot", "1tilda"]
// CHECK:       %[[FIL2:.*]] = rock.transform %[[FIL1]] by {{.*}}Slice{0, 3, 0, 3} ["0dotslice", "1dotslice"]{{.*}}Slice{0, 1, 0, 1} ["0tildaslice", "1tildaslice"]
// CHECK:       %[[FIL3:.*]] = rock.transform %[[FIL2]] by {{.*}}Merge{128, 3, 3} ["gemmK"]{{.*}}Merge{8, 1, 1} ["gemmM"]
// CHECK:       %[[OUT1:.*]] = rock.transform %[[arg2]] by {{.*}}Embed{-1, 1} ["0dot", "0tilda"]{{.*}}Embed{-1, 1} ["1dot", "1tilda"]
// CHECK:       %[[OUT2:.*]] = rock.transform %[[OUT1]] by {{.*}}Slice{0, 3, 0, 3} ["0slice", "1slice"]{{.*}}Slice{0, 32, 0, 32} ["0islice", "1islice"]
// CHECK:       %[[OUT3:.*]] = rock.transform %[[OUT2]] by {{.*}}Merge{128, 3, 3} ["gemmK"]{{.*}}Merge{128, 32, 32} ["gemmN"]
// CHECK:       rock.gemm tr %[[FIL3]] * %[[OUT3]]
func.func @rock_conv_bwd_data_small(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x8x32x32xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel, rock.num_cu = 120 : i32} {
  %result = rock.conv_bwd_data(%filter, %output) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params1,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x128x30x30xf32> -> tensor<128x1x8x32x32xf32>
  %out = rock.store %result to %input by set : tensor<128x1x8x32x32xf32> -> tensor<128x1x8x32x32xf32> to tensor<128x1x8x32x32xf32>
  return %out : tensor<128x1x8x32x32xf32>
}

// CHECK-LABEL: func.func @rock_conv_bwd_data_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x1024x1024x1x1xf16>, %[[arg1:.*]]: tensor<128x1x1024x14x14xf16>, %[[arg2:.*]]: tensor<128x1x1024x14x14xf16>)
// CHECK-NOT:   rock.conv_bwd_data
// CHECK:       %[[FIL1:.*]] = rock.transform %[[arg0]] by {{.*}}Embed{1, 1} ["0dot", "0tilda"]{{.*}}Embed{1, 1} ["1dot", "1tilda"]
// CHECK:       %[[FIL2:.*]] = rock.transform %[[FIL1]] by {{.*}}Slice{0, 1, 0, 1} ["0dotslice", "1dotslice"]{{.*}}Slice{0, 1, 0, 1} ["0tildaslice", "1tildaslice"]
// CHECK:       %[[FIL3:.*]] = rock.transform %[[FIL2]] by {{.*}}Merge{1024, 1, 1} ["gemmK"]{{.*}}Merge{1024, 1, 1} ["gemmM"]
// CHECK:       %[[OUT1:.*]] = rock.transform %[[arg2]] by {{.*}}Embed{-1, 1} ["0dot", "0tilda"]{{.*}}Embed{-1, 1} ["1dot", "1tilda"]
// CHECK:       %[[OUT2:.*]] = rock.transform %[[OUT1]] by {{.*}}Slice{0, 1, 0, 1} ["0slice", "1slice"]{{.*}}Slice{0, 14, 0, 14} ["0islice", "1islice"]
// CHECK:       %[[OUT3:.*]] = rock.transform %[[OUT2]] by {{.*}}Merge{1024, 1, 1} ["gemmK"]{{.*}}Merge{128, 14, 14} ["gemmN"]
// CHECK:       rock.gemm tr %[[FIL3]] * %[[OUT3]]
func.func @rock_conv_bwd_data_f16(%filter: tensor<1x1024x1024x1x1xf16>, %input: tensor<128x1x1024x14x14xf16>, %output: tensor<128x1x1024x14x14xf16>) -> tensor<128x1x1024x14x14xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.num_cu = 120 : i32} {
  %result = rock.conv_bwd_data(%filter, %output) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params1,
    strides = [1 : index, 1 : index]
  } : tensor<1x1024x1024x1x1xf16>, tensor<128x1x1024x14x14xf16> -> tensor<128x1x1024x14x14xf16>
  %out = rock.store %result to %input by set : tensor<128x1x1024x14x14xf16> -> tensor<128x1x1024x14x14xf16> to tensor<128x1x1024x14x14xf16>
  return %out : tensor<128x1x1024x14x14xf16>
}

// CHECK-LABEL: func.func @rock_conv_bwd_weight
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x128x8x3x3xf32>, %[[arg1:.*]]: tensor<128x1x8x32x32xf32>, %[[arg2:.*]]: tensor<128x1x128x30x30xf32>)
// CHECK-NOT:   rock.conv_bwd_weight
// CHECK:       %[[FIL1:.*]] = rock.transform %[[arg0]] by {{.*}}PassThrough ["g", "k", "c", "0", "1"]
// CHECK:       %[[FIL2:.*]] = rock.transform %[[FIL1]] by {{.*}}PassThrough ["gemmM"]{{.*}}Merge{8, 3, 3} ["gemmN"]
// CHECK:       %[[IN1:.*]] = rock.transform %[[arg1]] by {{.*}}Pad{0, 0, 0, 0}
// CHECK:       %[[IN2:.*]] = rock.transform %[[IN1]] by {{.*}}Embed{1, 1}{{.*}}Embed{1, 1}
// CHECK:       %[[IN3:.*]] = rock.transform %[[IN2]] by {{.*}}Merge{128, 30, 30} ["gemmK"]{{.*}}Merge{8, 3, 3} ["gemmN"]
// CHECK:       %[[OUT:.*]] = rock.transform %[[arg2]] by {{.*}}Merge{128, 30, 30} ["gemmK"]{{.*}}PassThrough ["gemmM"]
// CHECK:       rock.gemm tr %[[OUT]] * %[[IN3]]
func.func @rock_conv_bwd_weight(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<1x128x8x3x3xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel, rock.num_cu = 120 : i32} {
  %result = rock.conv_bwd_weight(%input, %output) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params1,
    strides = [1 : index,  1 : index]
  } : tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<1x128x8x3x3xf32>
  %out = rock.store %result to %filter by set : tensor<1x128x8x3x3xf32> -> tensor<1x128x8x3x3xf32> to tensor<1x128x8x3x3xf32>
  return %out : tensor<1x128x8x3x3xf32>
}

// CHECK-LABEL: func.func @rock_conv_bwd_weight_f16
// CHECK-SAME: (%[[arg0:.*]]: tensor<1x128x8x3x3xf16>, %[[arg1:.*]]: tensor<128x1x8x32x32xf16>, %[[arg2:.*]]: tensor<128x1x128x30x30xf16>)
// CHECK-NOT:   rock.conv_bwd_weight
// CHECK:       %[[FIL1:.*]] = rock.transform %[[arg0]] by {{.*}}PassThrough ["g", "k", "c", "0", "1"]
// CHECK:       %[[FIL2:.*]] = rock.transform %[[FIL1]] by {{.*}}PassThrough ["gemmM"]{{.*}}Merge{8, 3, 3} ["gemmN"]
// CHECK:       %[[IN1:.*]] = rock.transform %[[arg1]] by {{.*}}Pad{0, 0, 0, 0}
// CHECK:       %[[IN2:.*]] = rock.transform %[[IN1]] by {{.*}}Embed{1, 1}{{.*}}Embed{1, 1}
// CHECK:       %[[IN3:.*]] = rock.transform %[[IN2]] by {{.*}}Merge{128, 30, 30} ["gemmK"]{{.*}}Merge{8, 3, 3} ["gemmN"]
// CHECK:       %[[OUT:.*]] = rock.transform %[[arg2]] by {{.*}}Merge{128, 30, 30} ["gemmK"]{{.*}}PassThrough ["gemmM"]
// CHECK:       rock.gemm tr %[[OUT]] * %[[IN3]]
func.func @rock_conv_bwd_weight_f16(%filter : tensor<1x128x8x3x3xf16>, %input : tensor<128x1x8x32x32xf16>, %output : tensor<128x1x128x30x30xf16>) -> tensor<1x128x8x3x3xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel, rock.num_cu = 120 : i32} {
  %result = rock.conv_bwd_weight(%input, %output) {
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params1,
    strides = [1 : index,  1 : index]
  } : tensor<128x1x8x32x32xf16>, tensor<128x1x128x30x30xf16> -> tensor<1x128x8x3x3xf16>
  %out = rock.store %result to %filter by set : tensor<1x128x8x3x3xf16> -> tensor<1x128x8x3x3xf16> to tensor<1x128x8x3x3xf16>
  return %out : tensor<1x128x8x3x3xf16>
}
