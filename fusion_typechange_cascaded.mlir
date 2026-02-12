// Fused GEMM test: C_f32 = extf(gemm(A_f16, B_f16) + bias_f16)
// Tests cascaded type-changing output fusion: addf in f16 then extf to f32.
// This triggers a bug in LowerStores: convertToTile passes tileType=f32 down
// through extf into addf, causing addf's result to be set to f32 tile while
// its operands are f16 tiles.
#map = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<10000xf16>, %arg1: tensor<10000xf16>, %bias: tensor<10000xf16>, %arg2: tensor<10000xf32>) -> tensor<10000xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<10000xf16> to tensor<1x100x100xf16>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %bias by #transform_map2 : tensor<10000xf16> to tensor<1x100x100xf16>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<10000xf32> to tensor<1x100x100xf32>
    %3 = rock.gemm %0 * %1 : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %fused = arith.addf %3, %b : tensor<1x100x100xf16>
    %fused_f32 = arith.extf %fused : tensor<1x100x100xf16> to tensor<1x100x100xf32>
    %4 = rock.store %fused_f32 to %2 by set : tensor<1x100x100xf32> -> tensor<10000xf32> to tensor<1x100x100xf32>
    return %4 : tensor<10000xf32>
  }
}
