// This tests checks the following aspects of lowering component:
// * The correct padding transformations are generated and added to the gemm

// RUN: rocmlir-opt --rock-affix-params --rock-lower-reduce --rock-regularize-output --rock-regularize-inter-gemm-fusion --rock-conv-to-gemm --rock-fusion-splitk-regularization --rock-gemm-to-gridwise --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: func.func @rock_conv_kcyx_nchw_nkhw_padding_kernel
// CHECK-SAME: %[[filter:.*]]: tensor<32x128x2x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = rock.transform %[[filter]] by {{.*}}Merge{2, 3, 3} ["gemmK"]{{.*}}PassThrough ["gemmM"]
// CHECK: %[[swapped:.*]] = rock.transform %[[gemmFilter]] by {{.*}}PassThrough ["gemmM", "gemmK"]{{.*}}-> ["gemmM", "gemmK"]
// CHECK: %[[padK:.*]] = rock.transform %[[swapped]] by {{.*}}Pad{0, 14} ["gemmKPad"]{{.*}}-> ["gemmK"]
// CHECK: %[[inputK:.*]] = rock.transform %{{.*}} by {{.*}}Pad{0, 14} ["gemmKPad"]{{.*}}-> ["gemmK"]
// CHECK: rock.gridwise_gemm(%[[padK]], %[[inputK]])
func.func @rock_conv_kcyx_nchw_nkhw_padding_kernel(%filter : tensor<32x128x2x3x3xf32>, %input : tensor<64x32x2x11x11xf32>, %output : tensor<64x32x128x9x9xf32>) -> tensor<64x32x128x9x9xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel} {
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni","gi", "ci", "0i", "1i"],
    output_layout = ["no", "go",  "ko", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:128,64,16,1,1,4,0,1,2,0,0"
  } : tensor<32x128x2x3x3xf32>, tensor<64x32x2x11x11xf32> -> tensor<64x32x128x9x9xf32>
  %stored = rock.store %result to %output by set : tensor<64x32x128x9x9xf32> -> tensor<64x32x128x9x9xf32> to tensor<64x32x128x9x9xf32>
  return %stored : tensor<64x32x128x9x9xf32>
}

// CHECK-LABEL: func.func @rock_conv_kcyx_nchw_nkhw_no_extra_padding
// CHECK-SAME: %[[filter:.*]]: tensor<1x128x64x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = rock.transform %[[filter]] by {{.*}}Merge{64, 3, 3} ["gemmK"]{{.*}}PassThrough ["gemmM"]
// CHECK: %[[swapped:.*]] = rock.transform %[[gemmFilter]] by {{.*}}PassThrough ["gemmM", "gemmK"]{{.*}}-> ["gemmM", "gemmK"]
// CHECK-NOT: Pad{{.*}}gemmKPad
// CHECK: rock.gridwise_gemm(%[[swapped]], %{{.*}})
func.func @rock_conv_kcyx_nchw_nkhw_no_extra_padding(%filter : tensor<1x128x64x3x3xf32>, %input : tensor<128x1x64x32x32xf32>, %output : tensor<128x1x128x30x30xf32>) -> tensor<128x1x128x30x30xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel} {
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni","gi", "ci", "0i", "1i"],
    output_layout = ["no", "go",  "ko", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:128,64,16,1,1,4,0,1,2,0,0"
  } : tensor<1x128x64x3x3xf32>, tensor<128x1x64x32x32xf32> -> tensor<128x1x128x30x30xf32>
  %stored = rock.store %result to %output by set : tensor<128x1x128x30x30xf32> -> tensor<128x1x128x30x30xf32> to tensor<128x1x128x30x30xf32>
  return %stored : tensor<128x1x128x30x30xf32>
}

// CHECK-LABEL: func.func @rock_conv_kcyx_nchw_nkhw_partial_padding_kernel
// CHECK-SAME: %[[filter:.*]]: tensor<32x128x2x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = rock.transform %[[filter]] by {{.*}}Merge{2, 3, 3} ["gemmK"]{{.*}}PassThrough ["gemmM"]
// CHECK: %[[swapped:.*]] = rock.transform %[[gemmFilter]] by {{.*}}PassThrough ["gemmM", "gemmK"]{{.*}}-> ["gemmM", "gemmK"]
// CHECK: %[[padK:.*]] = rock.transform %[[swapped]] by {{.*}}Pad{0, 14} ["gemmKPad"]{{.*}}-> ["gemmK"]
// CHECK: %[[inputK:.*]] = rock.transform %{{.*}} by {{.*}}Pad{0, 14} ["gemmKPad"]{{.*}}-> ["gemmK"]
// CHECK: rock.gridwise_gemm(%[[padK]], %[[inputK]])
func.func @rock_conv_kcyx_nchw_nkhw_partial_padding_kernel(%filter : tensor<32x128x2x3x3xf32>, %input : tensor<128x32x2x11x11xf32>, %output : tensor<128x32x128x9x9xf32>) -> tensor<128x32x128x9x9xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx908", rock.kernel} {
  %result = rock.conv(%filter, %input) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni","gi", "ci", "0i", "1i"],
    output_layout = ["no", "go",  "ko", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "gemm:v1:128,64,16,1,1,4,0,1,2,0,0"
  } : tensor<32x128x2x3x3xf32>, tensor<128x32x2x11x11xf32> -> tensor<128x32x128x9x9xf32>
  %stored = rock.store %result to %output by set : tensor<128x32x128x9x9xf32> -> tensor<128x32x128x9x9xf32> to tensor<128x32x128x9x9xf32>
  return %stored : tensor<128x32x128x9x9xf32>
}
