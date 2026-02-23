#map = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 3 + d2) * 3 + d3) * 8 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 8 + d2) * 8 + d3) * 8 + d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 4 + d2) * 8 + d3) * 8 + d4)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 3, 3, 8} ["k", "0", "1", "c"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4, 3, 3, 8] -> [288]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{2, 8, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [2, 1, 8, 8, 8] -> [1024]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{2, 4, 8, 8} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [2, 1, 4, 8, 8] -> [512]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  func.func @rock_conv(%arg0: tensor<288xf16>, %arg1: tensor<1024xf16>, %inputfusion: tensor<1024xf16>, %arg2: tensor<512xf16>, %arg3: tensor<512xf16>) -> tensor<512xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 8 : i64, rock.num_cu = 304 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<288xf16> to tensor<1x4x3x3x8xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<1024xf16> to tensor<2x1x8x8x8xf16>
    %input = rock.transform %inputfusion by #transform_map1 : tensor<1024xf16> to tensor<2x1x8x8x8xf16>
    %a = arith.addf %1, %input : tensor<2x1x8x8x8xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<512xf16> to tensor<2x1x4x8x8xf16>
    %3 = rock.transform %arg3 by #transform_map2 : tensor<512xf16> to tensor<2x1x4x8x8xf16>
    %4 = rock.conv(%0, %a, %3) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : tensor<1x4x3x3x8xf16>, tensor<2x1x8x8x8xf16>, tensor<2x1x4x8x8xf16> -> tensor<2x1x4x8x8xf16>
    %fusion = arith.addf %4, %2 : tensor<2x1x4x8x8xf16>
    %5 = rock.store %fusion to %3 by  set : tensor<2x1x4x8x8xf16> -> tensor<512xf16> to tensor<2x1x4x8x8xf16>
    return %5 : tensor<512xf16>
  }
}
