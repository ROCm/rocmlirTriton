// RUN: rocmlir-driver -kernel-pipeline=gpu %s | rocmlir-opt

#map = affine_map<(d0, d1, d2) -> (d1 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 + d2)>
#map2 = affine_map<(d0) -> (0, 0, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{1, 1} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1, 1] -> [1]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{1, 1} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1, 1] -> [1]>
#transform_map2 = #rock.transform_map<#map2 by [<Merge{1, 1} ["raw"] at [0] -> ["m", "n"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [1] -> [1, 1, 1]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<1xf16>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
    %0 = rock.transform %arg0 by #transform_map : tensor<1xf16> to tensor<1x1x1xf16>
    %ext = arith.extf %0 : tensor<1x1x1xf16> to tensor<1x1x1xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<1xf32> to tensor<1x1x1xf32>
    %2 = rock.gemm %ext * %1 : tensor<1x1x1xf32> * tensor<1x1x1xf32> -> tensor<1x1x1xf32>
    %3 = rock.transform %2 by #transform_map2 : tensor<1x1x1xf32> to tensor<1xf32>
    %4 = rock.store %3 to %arg2 by set : tensor<1xf32> -> tensor<1xf32> to tensor<1xf32>
    return %4 : tensor<1xf32>
  }
}
