// RUN: rocmlir-driver -kernel-pipeline=gpu %s | rocmlir-opt | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1 * 769 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 512 + d2)>
#map2 = affine_map<(d0) -> (0, d0 floordiv 512, d0 mod 512)>
#transform_map = #rock.transform_map<#map by [<Unmerge{1024, 769} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 1024, 769] -> [787456]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{769, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 769, 512] -> [393728]>
#transform_map2 = #rock.transform_map<#map2 by [<Merge{1024, 512} ["raw"] at [0] -> ["m", "n"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [524288] -> [1, 1024, 512]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<787456xf16>, %arg1: tensor<393728xf32>, %arg2: tensor<524288xf32>) -> tensor<524288xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
    %0 = rock.transform %arg0 by #transform_map : tensor<787456xf16> to tensor<1x1024x769xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<393728xf32> to tensor<1x769x512xf32>
    %2 = rock.gemm %0 * %1 : tensor<1x1024x769xf16> * tensor<1x769x512xf32> -> tensor<1x1024x512xf32>
    %3 = rock.transform %2 by #transform_map2 : tensor<1x1024x512xf32> to tensor<524288xf32>
    %4 = rock.store %3 to %arg2 by set : tensor<524288xf32> -> tensor<524288xf32> to tensor<524288xf32>
    return %4 : tensor<524288xf32>
  }
}

// CHECK: arith.extf {{.*}} : tensor<{{.*}}xf16> to tensor<{{.*}}xf32>
