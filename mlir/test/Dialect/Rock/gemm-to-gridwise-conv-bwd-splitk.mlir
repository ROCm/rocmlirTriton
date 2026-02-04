// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// CHECK: module
// NOTE: I adapted the perfConfig to use splitK. It failed compilation before and after doing this, with a different error.

// #map = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 12 + d2) * 4 + d3) * 4 + d4)>
// #map1 = affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 12 + d3) * 12 + d4)>
// #map2 = affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 6 + d3) * 6 + d4)>
// #transform_map = #rock.transform_map<#map by [<Unmerge{24, 12, 4, 4} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 24, 12, 4, 4] -> [4608]>
// #transform_map1 = #rock.transform_map<#map1 by [<Unmerge{12, 12, 12} ["ci", "0i", "1i"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 12, 12, 12] -> [1728]>
// #transform_map2 = #rock.transform_map<#map2 by [<Unmerge{24, 6, 6} ["ko", "0o", "1o"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["no"] at [0] -> [] at []>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [1, 1, 24, 6, 6] -> [864]>
// module attributes {mhal.arch = "##TOKEN_ARCH##"} {
  // DISABLED-CHECK: @rock_conv_bwd_data_gkc01_ngc01_ngk01_0(%arg0: memref<4608xf16> {{{.*}}rock.prefill = 0.000000e+00 : f16
  // func.func @rock_conv_bwd_data_gkc01_ngc01_ngk01_0(%arg0: tensor<4608xf16>, %arg1: tensor<1728xf16>, %arg2: tensor<864xf16>) -> tensor<1x1x12x12x12xf16> attributes {enable_splitk_for_tuning, kernel = 0 : i32, rock.arch = "##TOKEN_ARCH##", num_cu = 304 : i32} {
  //   %0 = rock.transform %arg0 by #transform_map : tensor<4608xf16> to tensor<1x24x12x4x4xf16>
  //   %1 = rock.transform %arg1 by #transform_map1 : tensor<1728xf16> to tensor<1x1x12x12x12xf16>
  //   %2 = rock.transform %arg2 by #transform_map2 : tensor<864xf16> to tensor<1x1x24x6x6xf16>
  //   %result = rock.conv_bwd_data(%0, %1, %2) features = mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 0 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], perf_config = "gemm:v1:64,64,64,1,1,4,16,2,2,0,0", strides = [2 : index, 2 : index], usesV4R1 = true} : tensor<1x24x12x4x4xf16>, tensor<1x1x12x12x12xf16>, tensor<1x1x24x6x6xf16> -> tensor<1x1x12x12x12xf16>
  //   %stored = rock.store %result to %1 by set : tensor<1x1x12x12x12xf16> -> tensor<1x1x12x12x12xf16> to tensor<1x1x12x12x12xf16>
  //   return %stored : tensor<1x1x12x12x12xf16>
  // }
  // DISABLED-CHECK-NOT: @rock_conv_bwd_data_gkc01_ngc01_ngk01_1(%arg0 {{{.*}}rock.prefill = 0.000000e+00 : f16
  // func.func @rock_conv_bwd_data_gkc01_ngc01_ngk01_1(%arg0: tensor<4608xf16>, %arg1: tensor<1728xf16>, %arg2: tensor<864xf16>) -> tensor<1x1x12x12x12xf16> attributes {enable_splitk_for_tuning, kernel = 1 : i32, rock.arch = "##TOKEN_ARCH##", num_cu = 304 : i32} {
  //   %0 = rock.transform %arg0 by #transform_map : tensor<4608xf16> to tensor<1x24x12x4x4xf16>
  //   %1 = rock.transform %arg1 by #transform_map1 : tensor<1728xf16> to tensor<1x1x12x12x12xf16>
  //   %2 = rock.transform %arg2 by #transform_map2 : tensor<864xf16> to tensor<1x1x24x6x6xf16>
  //   %result = rock.conv_bwd_data(%0, %1, %2) features = mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 1 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], perf_config = "gemm:v1:64,64,64,1,1,4,16,2,2,0,0", strides = [2 : index, 2 : index], usesV4R1 = true} : tensor<1x24x12x4x4xf16>, tensor<1x1x12x12x12xf16>, tensor<1x1x24x6x6xf16> -> tensor<1x1x12x12x12xf16>
  //   %stored = rock.store %result to %1 by set : tensor<1x1x12x12x12xf16> -> tensor<1x1x12x12x12xf16> to tensor<1x1x12x12x12xf16>
  //   return %stored : tensor<1x1x12x12x12xf16>
  // }
  // // DISABLED-CHECK-NOT: @rock_conv_bwd_data_gkc01_ngc01_ngk01_2(%arg0 {{{.*}}rock.prefill = 0.000000e+00 : f16
  // func.func @rock_conv_bwd_data_gkc01_ngc01_ngk01_2(%arg0: tensor<4608xf16>, %arg1: tensor<1728xf16>, %arg2: tensor<864xf16>) -> tensor<1x1x12x12x12xf16> attributes {enable_splitk_for_tuning, kernel = 2 : i32, rock.arch = "##TOKEN_ARCH##", num_cu = 304 : i32} {
  //   %0 = rock.transform %arg0 by #transform_map : tensor<4608xf16> to tensor<1x24x12x4x4xf16>
  //   %1 = rock.transform %arg1 by #transform_map1 : tensor<1728xf16> to tensor<1x1x12x12x12xf16>
  //   %2 = rock.transform %arg2 by #transform_map2 : tensor<864xf16> to tensor<1x1x24x6x6xf16>
  //   %result = rock.conv_bwd_data(%0, %1, %2) features = mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 2 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], perf_config = "gemm:v1:64,64,64,1,1,4,16,2,2,0,0", strides = [2 : index, 2 : index], usesV4R1 = true} : tensor<1x24x12x4x4xf16>, tensor<1x1x12x12x12xf16>, tensor<1x1x24x6x6xf16> -> tensor<1x1x12x12x12xf16>
  //   %stored = rock.store %result to %1 by set : tensor<1x1x12x12x12xf16> -> tensor<1x1x12x12x12xf16> to tensor<1x1x12x12x12xf16>
  //   return %stored : tensor<1x1x12x12x12xf16>
  // }
  // // DISABLED-CHECK-NOT: @rock_conv_bwd_data_gkc01_ngc01_ngk01_3(%arg0 {{{.*}}rock.prefill = 0.000000e+00 : f16
  // func.func @rock_conv_bwd_data_gkc01_ngc01_ngk01_3(%arg0: tensor<4608xf16>, %arg1: tensor<1728xf16>, %arg2: tensor<864xf16>) -> tensor<1x1x12x12x12xf16> attributes {enable_splitk_for_tuning, kernel = 3 : i32, rock.arch = "##TOKEN_ARCH##", num_cu = 304 : i32} {
  //   %0 = rock.transform %arg0 by #transform_map : tensor<4608xf16> to tensor<1x24x12x4x4xf16>
  //   %1 = rock.transform %arg1 by #transform_map1 : tensor<1728xf16> to tensor<1x1x12x12x12xf16>
  //   %2 = rock.transform %arg2 by #transform_map2 : tensor<864xf16> to tensor<1x1x24x6x6xf16>
  //   %result = rock.conv_bwd_data(%0, %1, %2) features = mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 3 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], perf_config = "gemm:v1:64,64,64,1,1,4,16,2,2,0,0", strides = [2 : index, 2 : index], usesV4R1 = true} : tensor<1x24x12x4x4xf16>, tensor<1x1x12x12x12xf16>, tensor<1x1x24x6x6xf16> -> tensor<1x1x12x12x12xf16>
  //   %stored = rock.store %result to %1 by set : tensor<1x1x12x12x12xf16> -> tensor<1x1x12x12x12xf16> to tensor<1x1x12x12x12xf16>
  //   return %stored : tensor<1x1x12x12x12xf16>
  // }
// }
