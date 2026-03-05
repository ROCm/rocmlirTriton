#map = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %inputfusion: tensor<10000xf16>, %arg2: tensor<10000xf32>, %arg3: tensor<10000xf32>) -> tensor<10000xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<10000xf16> to tensor<1x100x100xf16>
    %input = rock.transform %inputfusion by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %a = arith.addf %0, %input : tensor<1x100x100xf16>
    %b = arith.addf %1, %input : tensor<1x100x100xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<10000xf32> to tensor<1x100x100xf32>
    %3 = rock.transform %arg3 by #transform_map2 : tensor<10000xf32> to tensor<1x100x100xf32>
    %4 = rock.gemm %a * %b {perf_config = "gemm:v1:64,64,64,1,1,4,16,3,2,0,0"} : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf32>
    %fusion = arith.addf %4, %2 : tensor<1x100x100xf32>
    %5 = rock.store %fusion to %3 by  set : tensor<1x100x100xf32> -> tensor<10000xf32> to tensor<1x100x100xf32>
    return %5 : tensor<10000xf32>
  }
}
