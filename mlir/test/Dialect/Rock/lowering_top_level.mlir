// UNSUPPORTED: true
// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton

// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly
// * Can pass arguments in the right sequence
// * Have, in most cases, the correct transformations
// * Have one gridwise_gemm
// * Can support F32 and F16

// RUN: rocmlir-opt -rock-conv-to-gemm %s | FileCheck %s

// CHECK-DAG: #[[$MAP_FILTER_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [1, 72, 128] -> [1, 128, 8, 3, 3]>
// CHECK-DAG: #[[$MAP_INPUT1_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [128, 1, 8, 32, 32] -> [128, 1, 8, 32, 32]>
// CHECK-DAG: #[[$MAP_INPUT2_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [128, 1, 8, 3, 30, 3, 30] -> [128, 1, 8, 32, 32]>
// CHECK-DAG: #[[$MAP_INPUT3_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [1, 72, 115200] -> [128, 1, 8, 3, 30, 3, 30]>
// CHECK-DAG: #[[$MAP_OUTPUT_FWD:transform_map[0-9]*]] = #rock.transform_map<{{.*}} bounds = [1, 128, 115200] -> [128, 1, 128, 30, 30]>

// CHECK-DAG: #[[$MAP_BWD_DATA_FIL1_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["g", "k", "c"] at [0, 1, 2] -> ["g", "k", "c"] at [0, 1, 2]>, <Embed{1, 1} ["0dot", "0tilda"] at [3, 4] -> ["0"] at [3]>, <Embed{1, 1} ["1dot", "1tilda"] at [5, 6] -> ["1"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_FIL2_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["g", "k", "c"] at [0, 1, 2] -> ["g", "k", "c"] at [0, 1, 2]>, <Slice{0, 1, 0, 1} ["0dotslice", "1dotslice"] at [3, 5] -> ["0dot", "1dot"] at [3, 5]>, <Slice{0, 1, 0, 1} ["0tildaslice", "1tildaslice"] at [4, 6] -> ["0tilda", "1tilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_FIL3_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{1024, 1, 1} ["gemmK"] at [1] -> ["k", "0dotslice", "1dotslice"] at [1, 3, 5]>, <Merge{1024, 1, 1} ["gemmM"] at [2] -> ["c", "0tildaslice", "1tildaslice"] at [2, 4, 6]>]

// CHECK-DAG: #[[$MAP_BWD_DATA_IN1_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Pad{0, 0, 0, 0} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>
// CHECK-DAG: #[[$MAP_BWD_DATA_IN2_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Embed{1, 1} ["0ftilda", "0itilda"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1ftilda", "1itilda"] at [5, 6] -> ["1ipad"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_IN3_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Slice{0, 1, 0, 1} ["0slice", "1slice"] at [3, 5] -> ["0ftilda", "1ftilda"] at [3, 5]>, <Slice{0, 14, 0, 14} ["0islice", "1islice"] at [4, 6] -> ["0itilda", "1itilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_IN4_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{1024, 1, 1} ["gemmM"] at [1] -> ["ci", "0slice", "1slice"] at [2, 3, 5]>, <Merge{128, 14, 14} ["gemmN"] at [2] -> ["ni", "0islice", "1islice"] at [0, 4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT1_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["go", "no", "ko"] at [1, 0, 2] -> ["go", "no", "ko"] at [1, 0, 2]>, <Embed{-1, 1} ["0dot", "0tilda"] at [3, 4] -> ["0o"] at [3]>, <Embed{-1, 1} ["1dot", "1tilda"] at [5, 6] -> ["1o"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT2_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["go", "no", "ko"] at [1, 0, 2] -> ["go", "no", "ko"] at [1, 0, 2]>, <Slice{0, 1, 0, 1} ["0slice", "1slice"] at [3, 5] -> ["0dot", "1dot"] at [3, 5]>, <Slice{0, 14, 0, 14} ["0islice", "1islice"] at [4, 6] -> ["0tilda", "1tilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT3_NO_PAD:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <Merge{1024, 1, 1} ["gemmK"] at [1] -> ["ko", "0slice", "1slice"] at [2, 3, 5]>, <Merge{128, 14, 14} ["gemmN"] at [2] -> ["no", "0islice", "1islice"] at [0, 4, 6]>]

// CHECK-DAG: #[[$MAP_BWD_WEIGHT_FIL1:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [1]>, <Merge{8, 3, 3} ["gemmN"] at [2] -> ["c", "0", "1"] at [2, 3, 4]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_IN3:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{128, 30, 30} ["gemmK"] at [1] -> ["ni", "0o", "1o"] at [0, 4, 6]>, <Merge{8, 3, 3} ["gemmN"] at [2] -> ["ci", "0", "1"] at [2, 3, 5]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_OUT:transform_map[0-9]*]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <Merge{128, 30, 30} ["gemmK"] at [1] -> ["no", "0o", "1o"] at [0, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["ko"] at [2]>]

#gemm_params0 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 8, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#gemm_params1 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 16, kpack = 1, numWaves = 1, matrixInstrNonkdim = 0, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params0 = #rock.gemm_params<mPerBlock = 64, nPerBlock = 64, kPerBlock = 8, kpack = 1, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>
#xdlops_gemm_params1 = #rock.gemm_params<mPerBlock = 128, nPerBlock = 128, kPerBlock = 16, kpack = 4, numWaves = 4, matrixInstrNonkdim = 32, splitKFactor = 1, numStages = 2, wavesPerEU = 0, gridGroupSize = 0, numCTAs = 1>

func.func @rock_conv(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x128x30x30xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %result = rock.conv(%filter, %input, %output) features = none {
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params0,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32> to tensor<128x1x128x30x30xf32>
  return %out : tensor<128x1x128x30x30xf32>
}
// CHECK-LABEL: func.func {{@rock_conv.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv
// CHECK-NEXT:  %[[FILTER:.*]] = rock.transform %arg0 by #[[$MAP_FILTER_FWD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_INPUT3_FWD]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_OUTPUT_FWD]]
// CHECK-NEXT:  {{%.*}} = rock.gemm tr %[[FILTER]] * %[[IN3]]

func.func @rock_conv_f16(%filter : tensor<1x128x8x3x3xf16>, %input : tensor<128x1x8x32x32xf16>, %output : tensor<128x1x128x30x30xf16>) -> tensor<128x1x128x30x30xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906"} {
  %result = rock.conv(%filter, %input, %output) features = none {
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params0,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xf16>, tensor<128x1x8x32x32xf16>, tensor<128x1x128x30x30xf16> -> tensor<128x1x128x30x30xf16>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xf16> -> tensor<128x1x128x30x30xf16> to tensor<128x1x128x30x30xf16>
  return %out : tensor<128x1x128x30x30xf16>
}
// CHECK-LABEL: func.func {{@rock_conv_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv
// CHECK-NEXT:  %[[FILTER:.*]] = rock.transform %arg0 by #[[$MAP_FILTER_FWD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_INPUT3_FWD]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_OUTPUT_FWD]]
// CHECK-NEXT:  {{%.*}} = rock.gemm tr %[[FILTER]] * %[[IN3]]

func.func @rock_conv_i8(%filter : tensor<1x128x8x3x3xi8>, %input : tensor<128x1x8x32x32xi8>, %output : tensor<128x1x128x30x30xi32>) -> tensor<128x1x128x30x30xi32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result = rock.conv(%filter, %input, %output) {
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params0,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xi8>, tensor<128x1x8x32x32xi8>, tensor<128x1x128x30x30xi32> -> tensor<128x1x128x30x30xi32>
  %out = rock.store %result to %output by set : tensor<128x1x128x30x30xi32> -> tensor<128x1x128x30x30xi32> to tensor<128x1x128x30x30xi32>
  return %out : tensor<128x1x128x30x30xi32>
}
// CHECK-LABEL: func.func {{@rock_conv_i8.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv
// CHECK-NEXT:  %[[FILTER:.*]] = rock.transform %arg0 by #[[$MAP_FILTER_FWD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_INPUT3_FWD]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_OUTPUT_FWD]]
// CHECK-NEXT:  {{%.*}} = rock.gemm tr %[[FILTER]] * %[[IN3]]


func.func @rock_conv_bwd_data(%filter: tensor<1x1024x1024x1x1xf32>, %input: tensor<128x1x1024x14x14xf32>, %output: tensor<128x1x1024x14x14xf32>) -> tensor<128x1x1024x14x14xf32> attributes {rock.kernel = 0 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result = rock.conv_bwd_data(%filter, %input, %output) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 0 : index,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params1,
    strides = [1 : index, 1 : index],
    usesV4R1 = true
  } : tensor<1x1024x1024x1x1xf32>, tensor<128x1x1024x14x14xf32>, tensor<128x1x1024x14x14xf32> -> tensor<128x1x1024x14x14xf32>
  %out = rock.store %result to %input by set : tensor<128x1x1024x14x14xf32> -> tensor<128x1x1024x14x14xf32> to tensor<128x1x1024x14x14xf32>
  return %out : tensor<128x1x1024x14x14xf32>
}

// CHECK-LABEL: func.func {{@rock_conv_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv_bwd_data
// CHECK-NEXT:  %[[FIL1:.*]] = rock.transform %arg0 by #[[$MAP_BWD_DATA_FIL1_NO_PAD]]
// CHECK-NEXT:  %[[FIL2:.*]] = rock.transform %[[FIL1]] by #[[$MAP_BWD_DATA_FIL2_NO_PAD]]
// CHECK-NEXT:  %[[FIL3:.*]] = rock.transform %[[FIL2]] by #[[$MAP_BWD_DATA_FIL3_NO_PAD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_BWD_DATA_IN1_NO_PAD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_BWD_DATA_IN2_NO_PAD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_BWD_DATA_IN3_NO_PAD]]
// CHECK-NEXT:  %[[IN4:.*]] = rock.transform %[[IN3]] by #[[$MAP_BWD_DATA_IN4_NO_PAD]]
// CHECK-NEXT:  %[[OUT1:.*]] = rock.transform %arg2 by #[[$MAP_BWD_DATA_OUT1_NO_PAD]]
// CHECK-NEXT:  %[[OUT2:.*]] = rock.transform %[[OUT1]] by #[[$MAP_BWD_DATA_OUT2_NO_PAD]]
// CHECK-NEXT:  %[[OUT3:.*]] = rock.transform %[[OUT2]] by #[[$MAP_BWD_DATA_OUT3_NO_PAD]]
// CHECK-NEXT:  {{%.*}} = rock.gemm tr %[[FIL3]] * %[[OUT3]]{{.*}}

func.func @rock_conv_bwd_data_nov4r1(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x8x32x32xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 64 : i32} {
  %result = rock.conv_bwd_data(%filter, %input, %output) features = none {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params1,
    strides = [1 : index,  1 : index],
    usesV4R1 = false,
    kernelId = 0 : index
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<128x1x8x32x32xf32>
  %out = rock.store %result to %input by set : tensor<128x1x8x32x32xf32> -> tensor<128x1x8x32x32xf32> to tensor<128x1x8x32x32xf32>
  return %out : tensor<128x1x8x32x32xf32>
}

// CHECK-LABEL: func.func {{@rock_conv_bwd_data_nov4r1.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT: rock.conv_bwd_data
// CHECK-NEXT: %[[FIL1:.*]] = rock.transform %arg0
// CHECK-NEXT: %[[FIL2:.*]] = rock.transform %[[FIL1]]
// CHECK-NEXT: %[[FIL3:.*]] = rock.transform %[[FIL2]]
// CHECK-NEXT: %[[IN1:.*]] = rock.transform %arg1
// CHECK-NEXT: %[[IN2:.*]] = rock.transform %[[IN1]]
// CHECK-NEXT: %[[IN3:.*]] = rock.transform %[[IN2]]
// CHECK-NEXT: %[[IN4:.*]] = rock.transform %[[IN3]]
// CHECK-NEXT: %[[OUT1:.*]] = rock.transform %arg2
// CHECK-NEXT: %[[OUT2:.*]] = rock.transform %[[OUT1]]
// CHECK-NEXT: %[[OUT3:.*]] = rock.transform %[[OUT2]]
// CHECK-NEXT: {{%.*}} = rock.gemm tr %[[FIL3]] * %[[OUT3]]

func.func @rock_conv_bwd_data_f16(%filter: tensor<1x1024x1024x1x1xf16>, %input: tensor<128x1x1024x14x14xf16>, %output: tensor<128x1x1024x14x14xf16>) -> tensor<128x1x1024x14x14xf16> attributes {rock.kernel = 0 : i32, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
  %result = rock.conv_bwd_data(%filter, %input, %output) {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 0 : index,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #xdlops_gemm_params1,
    strides = [1 : index, 1 : index],
    usesV4R1 = true
  } : tensor<1x1024x1024x1x1xf16>, tensor<128x1x1024x14x14xf16>, tensor<128x1x1024x14x14xf16> -> tensor<128x1x1024x14x14xf16>
  %out = rock.store %result to %input by set : tensor<128x1x1024x14x14xf16> -> tensor<128x1x1024x14x14xf16> to tensor<128x1x1024x14x14xf16>
  return %out : tensor<128x1x1024x14x14xf16>
}

// CHECK-LABEL: func.func {{@rock_conv_bwd_data_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv_bwd_data
// CHECK-NEXT:  %[[FIL1:.*]] = rock.transform %arg0 by #[[$MAP_BWD_DATA_FIL1_NO_PAD]]
// CHECK-NEXT:  %[[FIL2:.*]] = rock.transform %[[FIL1]] by #[[$MAP_BWD_DATA_FIL2_NO_PAD]]
// CHECK-NEXT:  %[[FIL3:.*]] = rock.transform %[[FIL2]] by #[[$MAP_BWD_DATA_FIL3_NO_PAD]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_BWD_DATA_IN1_NO_PAD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_BWD_DATA_IN2_NO_PAD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_BWD_DATA_IN3_NO_PAD]]
// CHECK-NEXT:  %[[IN4:.*]] = rock.transform %[[IN3]] by #[[$MAP_BWD_DATA_IN4_NO_PAD]]
// CHECK-NEXT:  %[[OUT1:.*]] = rock.transform %arg2 by #[[$MAP_BWD_DATA_OUT1_NO_PAD]]
// CHECK-NEXT:  %[[OUT2:.*]] = rock.transform %[[OUT1]] by #[[$MAP_BWD_DATA_OUT2_NO_PAD]]
// CHECK-NEXT:  %[[OUT3:.*]] = rock.transform %[[OUT2]] by #[[$MAP_BWD_DATA_OUT3_NO_PAD]]
// CHECK-NEXT:  {{%.*}} = rock.gemm tr %[[FIL3]] * %[[OUT3]]{{.*}}

func.func @rock_conv_bwd_weight(%filter : tensor<1x128x8x3x3xf32>, %input : tensor<128x1x8x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<1x128x8x3x3xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 64 : i32} {
  %result = rock.conv_bwd_weight(%filter, %input, %output) features = none {
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params1,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xf32>, tensor<128x1x8x32x32xf32>, tensor<128x1x128x30x30xf32> -> tensor<1x128x8x3x3xf32>
  %out = rock.store %result to %filter by set : tensor<1x128x8x3x3xf32> -> tensor<1x128x8x3x3xf32> to tensor<1x128x8x3x3xf32>
  return %out : tensor<1x128x8x3x3xf32>
}
// CHECK-LABEL: func.func {{@rock_conv_bwd_weight.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv_bwd_weight
// CHECK-NEXT:  %[[FIL1:.*]] = rock.transform %arg0 by #[[$MAP_BWD_WEIGHT_FIL1]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_BWD_WEIGHT_IN3]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_BWD_WEIGHT_OUT]]
// CHECK-NEXT:  {{%.*}} = rock.gemm tr %[[OUT]] * %[[IN3]]{{.*}}

func.func @rock_conv_bwd_weight_f16(%filter : tensor<1x128x8x3x3xf16>, %input : tensor<128x1x8x32x32xf16>, %output : tensor<128x1x128x30x30xf16>) -> tensor<1x128x8x3x3xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx906", numCU = 64 : i32} {
  %result = rock.conv_bwd_weight(%filter, %input, %output) features = none {
    dilations = [1 : index,  1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    params = #gemm_params1,
    strides = [1 : index,  1 : index]
  } : tensor<1x128x8x3x3xf16>, tensor<128x1x8x32x32xf16>, tensor<128x1x128x30x30xf16> -> tensor<1x128x8x3x3xf16>
  %out = rock.store %result to %filter by set : tensor<1x128x8x3x3xf16> -> tensor<1x128x8x3x3xf16> to tensor<1x128x8x3x3xf16>
  return %out : tensor<1x128x8x3x3xf16>
}
// CHECK-LABEL: func.func {{@rock_conv_bwd_weight_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   rock.conv_bwd_weight
// CHECK-NEXT:  %[[FIL1:.*]] = rock.transform %arg0 by #[[$MAP_BWD_WEIGHT_FIL1]]
// CHECK-NEXT:  %[[IN1:.*]] = rock.transform %arg1 by #[[$MAP_INPUT1_FWD]]
// CHECK-NEXT:  %[[IN2:.*]] = rock.transform %[[IN1]] by #[[$MAP_INPUT2_FWD]]
// CHECK-NEXT:  %[[IN3:.*]] = rock.transform %[[IN2]] by #[[$MAP_BWD_WEIGHT_IN3]]
// CHECK-NEXT:  %[[OUT:.*]] = rock.transform %arg2 by #[[$MAP_BWD_WEIGHT_OUT]]
// CHECK-NEXT:  {{%.*}} = rock.gemm tr %[[OUT]] * %[[IN3]]{{.*}}
