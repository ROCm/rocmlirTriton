// RUN: rocmlir-driver -c --mlir-print-ir-after=rock-gemm-to-gridwise %s -o /dev/null 2>&1 | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 12 + d2) * 4 + d3) * 4 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 12 + d3) * 12 + d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 6 + d3) * 6 + d4)>
#transform_map = #rock.transform_map<#map by [<Unmerge{24, 12, 4, 4} ["k", "c", "0", "1"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 24, 12, 4, 4] -> [4608]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{12, 12, 12} ["ci", "0i", "1i"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 12, 12, 12] -> [1728]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{24, 6, 6} ["ko", "0o", "1o"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["no"] at [0] -> [] at []>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [1, 1, 24, 6, 6] -> [864]>
module attributes {rock.arch = "gfx950:sramecc+:xnack-"} {
  // CHECK: @rock_conv_bwd_data_gkc01_ngc01_ngk01_0(%arg0: tensor<4608xf16>, %arg1: tensor<1728xf16> {rock.prefill = 0.000000e+00 : f16
  func.func @rock_conv_bwd_data_gkc01_ngc01_ngk01_0(%arg0: tensor<4608xf16>, %arg1: tensor<1728xf16>, %arg2: tensor<864xf16>) -> tensor<1x1x12x12x12xf16> attributes {rock.enable_splitk_for_tuning, rock.arch = "gfx950:sramecc+:xnack-", rock.kernel} {
    %0 = rock.transform %arg0 by #transform_map : tensor<4608xf16> to tensor<1x24x12x4x4xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<1728xf16> to tensor<1x1x12x12x12xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<864xf16> to tensor<1x1x24x6x6xf16>
    %3 = rock.conv_bwd_data(%0, %1, %2) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], perf_config = "gemm:v1:64,64,64,1,1,4,16,2,2,0,0", strides = [2 : index, 2 : index]} : tensor<1x24x12x4x4xf16>, tensor<1x1x12x12x12xf16>, tensor<1x1x24x6x6xf16> -> tensor<1x1x12x12x12xf16>
    %4 = rock.store %3 to %1 by  set : tensor<1x1x12x12x12xf16> -> tensor<1x1x12x12x12xf16> to tensor<1x1x12x12x12xf16>
    return %4 : tensor<1x1x12x12x12xf16>
  }
}
