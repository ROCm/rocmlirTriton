// Fused GEMM test: C = (A + (t1 + t2)) * B + (t1 + t2)
// Tests multi-operand fusion chains with the same two tensors used in both
// input and output fusion.
#map = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %t1: tensor<10000xf16>, %t2: tensor<10000xf16>, %arg2: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<10000xf16> to tensor<1x100x100xf16>
    %t1_in = rock.transform %t1 by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %t2_in = rock.transform %t2 by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %t1_out = rock.transform %t1 by #transform_map2 : tensor<10000xf16> to tensor<1x100x100xf16>
    %t2_out = rock.transform %t2 by #transform_map2 : tensor<10000xf16> to tensor<1x100x100xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<10000xf16> to tensor<1x100x100xf16>
    %sum_in = arith.addf %t1_in, %t2_in : tensor<1x100x100xf16>
    %a = arith.addf %0, %sum_in : tensor<1x100x100xf16>
    %3 = rock.gemm %a * %1 : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %sum_out = arith.addf %t1_out, %t2_out : tensor<1x100x100xf16>
    %fusion = arith.addf %3, %sum_out : tensor<1x100x100xf16>
    %4 = rock.store %fusion to %2 by set : tensor<1x100x100xf16> -> tensor<10000xf16> to tensor<1x100x100xf16>
    return %4 : tensor<10000xf16>
  }
}
