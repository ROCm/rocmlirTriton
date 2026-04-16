// RUN: not rocmlir-driver -c -arch %arch %s 2>&1 | FileCheck %s
// COM: nested reductions like reduce(reduce(x)) are not allowed
// CHECK: could not find rock.store along single-use chain from reduce

#map = affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 64 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 4096 + d2)>
#map2 = affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 64 + d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map4 = affine_map<(d0, d1) -> (d0 floordiv 64, d0 mod 64, d1)>
#map5 = affine_map<(d0, d1) -> (0, d0, d1)>
#map7 = affine_map<(d0) -> (d0, 0, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{64, 64, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 64, 64] -> [262144]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{64, 64, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 64, 4096] -> [16777216]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{1, 4096, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 64] -> [262144]>
#transform_map3 = #rock.transform_map<#map3 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [64, 4096, 64] -> [1, 4096, 64]>
#transform_map4 = #rock.transform_map<#map4 by [<Merge{64, 64} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [4096, 4096] -> [64, 64, 4096]>
#transform_map5 = #rock.transform_map<#map5 by [<ConstDim{0, 64} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 64] -> [64, 4096, 64]>
#transform_map6 = #rock.transform_map<#map4 by [<Merge{64, 64} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [4096, 64] -> [64, 64, 64]>
#map6 = affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)>
#transform_map6_inv = #rock.transform_map<#map6 by [<Unmerge{64, 64} ["g", "d0"] at [0, 1] -> ["gd0"] at [0]>, <PassThrough ["d1"] at [2] -> ["d1"] at [1]>] bounds = [64, 64, 64] -> [4096, 64]>
#transform_map7 = #rock.transform_map<#map7 by [<Merge{64, 1, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [64] -> [64, 1, 1]>
module {
  func.func @matmul_broadcast_op(%arg0: tensor<262144xf32>, %arg1: tensor<16777216xf32>, %arg2: tensor<262144xf32>, %arg3: tensor<64xf32> {rock.prefill = 0.000000e+00 : f32}) -> tensor<64xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-", rock.enable_splitk_for_tuning, rock.kernel = "mixr"} {
    %0 = rock.transform %arg0 by #transform_map : tensor<262144xf32> to tensor<64x64x64xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<16777216xf32> to tensor<64x64x4096xf32>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<262144xf32> to tensor<1x4096x64xf32>
    %3 = rock.transform %2 by #transform_map3 : tensor<1x4096x64xf32> to tensor<64x4096x64xf32>
    %4 = rock.transform %1 by #transform_map4 : tensor<64x64x4096xf32> to tensor<4096x4096xf32>
    %5 = rock.transform %3 by #transform_map5 : tensor<64x4096x64xf32> to tensor<4096x64xf32>
    %6 = rock.transform %0 by #transform_map6 : tensor<64x64x64xf32> to tensor<4096x64xf32>
    %gemm = rock.gemm %4 * %5 : tensor<4096x4096xf32> * tensor<4096x64xf32> -> tensor<4096x64xf32>
    %gemm_3d = rock.transform %gemm by #transform_map6_inv : tensor<4096x64xf32> to tensor<64x64x64xf32>
    %fused_add = arith.addf %gemm_3d, %0 : tensor<64x64x64xf32>
    %reduced_0 = rock.reduce sum %fused_add {axis = 2 : index} : tensor<64x64x64xf32> -> tensor<64x64x1xf32>
    %reduced_1 = rock.reduce sum %reduced_0 {axis = 1 : index} : tensor<64x64x1xf32> -> tensor<64x1x1xf32>
    %7 = rock.transform %reduced_1 by #transform_map7 : tensor<64x1x1xf32> to tensor<64xf32>
    %out = rock.store %7 to %arg3 by set : tensor<64xf32> -> tensor<64xf32> to tensor<64xf32>
    return %out : tensor<64xf32>
  }
}
