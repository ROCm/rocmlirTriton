// Fused GEMM test: C = (A + shared) * B + shared
// Tests using the same tensor for both input and output fusion.
// The same flat tensor %shared is transformed differently for each use:
// - Input fusion: #transform_map  (m,k layout)
// - Output fusion: #transform_map2 (m,n layout)
// Each gets its own rock.transform -> rock.load chain, so InsertLoads
// creates separate loads. This should work if the pipeline handles the
// two independent uses correctly.
#map = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %shared: tensor<10000xf16>, %arg2: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<10000xf16> to tensor<1x100x100xf16>
    %s_in = rock.transform %shared by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %s_out = rock.transform %shared by #transform_map2 : tensor<10000xf16> to tensor<1x100x100xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<10000xf16> to tensor<1x100x100xf16>
    %a = arith.addf %0, %s_in : tensor<1x100x100xf16>
    %3 = rock.gemm %a * %1 : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %fusion = arith.addf %3, %s_out : tensor<1x100x100xf16>
    %4 = rock.store %fusion to %2 by set : tensor<1x100x100xf16> -> tensor<10000xf16> to tensor<1x100x100xf16>
    return %4 : tensor<10000xf16>
  }
}
