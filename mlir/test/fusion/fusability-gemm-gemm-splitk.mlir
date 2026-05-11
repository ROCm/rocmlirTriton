// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,4,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
// Kernel dims are M=K=N=gemmO=64 so the prospective LDS footprint computed by
// testFusionLegalityGemmGemmLDS fits the gfx90a 64 KiB budget for both
// perfConfigs above. The test still exercises the GEG + reduce-sum fusibility
// path (testFusionLegalityReduce plus the splitK branch of
// testFusionLegalitySplitK).
#map = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#map1 = affine_map<(d0) -> (0, d0 floordiv 1, d0 mod 1)>
#transform_map = #rock.transform_map<#map by [<Unmerge{64, 64} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{64, 64} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{64, 64} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]>
#transform_map3 = #rock.transform_map<#map1 by [<Merge{64, 1} ["raw"] at [0] -> ["m", "gemmO"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [64] -> [1, 64, 1]>
module {
  func.func @rock_gemm_gemm(%arg0: tensor<4096xf16>, %arg1: tensor<4096xf16>, %arg2: tensor<4096xf16>, %arg3: tensor<64xf16>) -> tensor<64xf16> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %0 = rock.transform %arg0 by #transform_map : tensor<4096xf16> to tensor<1x64x64xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<4096xf16> to tensor<1x64x64xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<4096xf16> to tensor<1x64x64xf16>
    %3 = rock.gemm_elementwise_gemm{
     ab = %0 * %1 : tensor<1x64x64xf16>, tensor<1x64x64xf16>
     ab = elementwise {
    ^bb0(%arg4: tensor<1x64x64xf16>):
      rock.yield %arg4 : tensor<1x64x64xf16>
    }
     out = ab * %2 : tensor<1x64x64xf16>
    } -> tensor<1x64x64xf16>
    %4 = rock.reduce sum %3 {axis = 2 : index} : tensor<1x64x64xf16> -> tensor<1x64x1xf16>
    %5 = rock.transform %4 by #transform_map3 : tensor<1x64x1xf16> to tensor<64xf16>
    %6 = rock.store %5 to %arg3 by set : tensor<64xf16> -> tensor<64xf16> to tensor<64xf16>
    return %6 : tensor<64xf16>
  }
}
