// RUN: rocmlir-driver -kernel-pipeline=gpu %s | rocmlir-opt
#map = affine_map<(d0, d1, d2) -> (d1 * 3 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#map2 = affine_map<(d0) -> (0, d0 floordiv 4, d0 mod 4)>
#transform_map_a = #rock.transform_map<#map by [<Unmerge{2, 3} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 2, 3] -> [6]>
#transform_map_b = #rock.transform_map<#map1 by [<Unmerge{3, 4} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 3, 4] -> [12]>
#transform_map_out = #rock.transform_map<#map2 by [<Merge{2, 4} ["raw"] at [0] -> ["m", "n"] at [1, 2]>, <ConstDim{0, 1} [] at [] -> ["g"] at [0]>] bounds = [8] -> [1, 2, 4]>
func.func @mlir_dot_mul(%arg0: tensor<6xf32>, %arg1: tensor<12xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) attributes {rock.arch = "gfx908:sramecc+:xnack-", rock.kernel = "mixr"} {
  %cst = arith.constant dense<2.500000e-01> : tensor<1x2x4xf32>
  %0 = rock.transform %arg0 by #transform_map_a : tensor<6xf32> to tensor<1x2x3xf32>
  %1 = rock.transform %arg1 by #transform_map_b : tensor<12xf32> to tensor<1x3x4xf32>
  %2 = rock.gemm %0 * %1 : tensor<1x2x3xf32> * tensor<1x3x4xf32> -> tensor<1x2x4xf32>
  %3 = arith.mulf %2, %cst : tensor<1x2x4xf32>
  %4 = rock.transform %2 by #transform_map_out : tensor<1x2x4xf32> to tensor<8xf32>
  %5 = rock.store %4 to %arg2 by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  %6 = rock.transform %3 by #transform_map_out : tensor<1x2x4xf32> to tensor<8xf32>
  %7 = rock.store %6 to %arg3 by set : tensor<8xf32> -> tensor<8xf32> to tensor<8xf32>
  return %5, %7 : tensor<8xf32>, tensor<8xf32>
}
