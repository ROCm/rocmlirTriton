// An additive epilogue after gemm1 is legal under split-k: it survives being
// applied once per split because rock-fusion-splitk-regularization divides the
// bias by the split factor before the atomic_add sums the partials.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,4,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
#map = affine_map<(d0, d1, d2) -> (d1 * 360 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 4096 + d2)>
#map2 = affine_map<(d0) -> (0, d0 floordiv 360, d0 mod 360)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4096, 360} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{360, 4096} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 360, 4096] -> [1474560]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{4096, 360} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]>
#transform_map3 = #rock.transform_map<#map by [<Unmerge{4096, 360} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]>
#transform_map4 = #rock.transform_map<#map2 by [<Merge{4096, 360} ["raw"] at [0] -> ["m", "gemmO"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [1474560] -> [1, 4096, 360]>
module {
  func.func @rock_gemm_gemm(%arg0: tensor<1474560xf16>, %arg1: tensor<1474560xf16>, %arg2: tensor<1474560xf16>, %arg3: tensor<1474560xf16>, %arg4: tensor<1474560xf16>) -> tensor<1474560xf16> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %0 = rock.transform %arg0 by #transform_map : tensor<1474560xf16> to tensor<1x4096x360xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<1474560xf16> to tensor<1x360x4096xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<1474560xf16> to tensor<1x4096x360xf16>
    %3 = rock.gemm_elementwise_gemm{
     ab = %0 * %1 : tensor<1x4096x360xf16>, tensor<1x360x4096xf16>
     ab = elementwise {
    ^bb0(%arg5: tensor<1x4096x4096xf16>):
      rock.yield %arg5 : tensor<1x4096x4096xf16>
    }
     out = ab * %2 : tensor<1x4096x360xf16>
    } -> tensor<1x4096x360xf16>
    %bias = rock.transform %arg3 by #transform_map3 : tensor<1474560xf16> to tensor<1x4096x360xf16>
    %4 = arith.addf %3, %bias : tensor<1x4096x360xf16>
    %5 = rock.transform %4 by #transform_map4 : tensor<1x4096x360xf16> to tensor<1474560xf16>
    %6 = rock.store %5 to %arg4 by set : tensor<1474560xf16> -> tensor<1474560xf16> to tensor<1474560xf16>
    return %6 : tensor<1474560xf16>
  }
}
