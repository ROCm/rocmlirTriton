// An elementwise body between the two GEMMs is legal under split-k: it is
// pointwise in (gemmM, gemmN) and split-k only partitions gemmN, so each split
// reads its own slice of the body's inputs.

// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,4,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v1:64,64,16,1,1,4,16,1,2,0,0 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
#map = affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 32 + d2)>
#map1 = affine_map<(d0) -> (d0 floordiv 1024, (d0 floordiv 32) mod 32, d0 mod 32)>
#transform_map_a = #rock.transform_map<#map by [<Unmerge{4, 32, 32} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 32, 32] -> [4096]>
#transform_map_b = #rock.transform_map<#map by [<Unmerge{4, 32, 32} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 32, 32] -> [4096]>
#transform_map_c = #rock.transform_map<#map by [<Unmerge{4, 32, 32} ["g", "n", "gemmO"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 32, 32] -> [4096]>
#transform_map_ew = #rock.transform_map<#map by [<Unmerge{4, 32, 32} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [4, 32, 32] -> [4096]>
#transform_map_out = #rock.transform_map<#map1 by [<Merge{4, 32, 32} ["raw"] at [0] -> ["g", "m", "gemmO"] at [0, 1, 2]>] bounds = [4096] -> [4, 32, 32]>
module {
  func.func @mlir_gemm_gemm(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<4096xf32>, %arg3: tensor<4096xf32>, %arg4: tensor<4096xf32>) -> tensor<4096xf32> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %a = rock.transform %arg2 by #transform_map_a : tensor<4096xf32> to tensor<4x32x32xf32>
    %b = rock.transform %arg3 by #transform_map_b : tensor<4096xf32> to tensor<4x32x32xf32>
    %c = rock.transform %arg0 by #transform_map_c : tensor<4096xf32> to tensor<4x32x32xf32>
    %ew = rock.transform %arg1 by #transform_map_ew : tensor<4096xf32> to tensor<4x32x32xf32>
    %0 = rock.gemm_elementwise_gemm{
     ab = %a * tr %b : tensor<4x32x32xf32>, tensor<4x32x32xf32>
     ab = elementwise otherIns(%ew : tensor<4x32x32xf32>) {
    ^bb0(%arg5: tensor<4x32x32xf32>, %arg6: tensor<4x32x32xf32>):
      %mul = arith.mulf %arg5, %arg6 : tensor<4x32x32xf32>
      rock.yield %mul : tensor<4x32x32xf32>
    }
     out = ab * %c : tensor<4x32x32xf32>
    } -> tensor<4x32x32xf32>
    %1 = rock.transform %0 by #transform_map_out : tensor<4x32x32xf32> to tensor<4096xf32>
    %2 = rock.store %1 to %arg4 by set : tensor<4096xf32> -> tensor<4096xf32> to tensor<4096xf32>
    return %2 : tensor<4096xf32>
  }
}
