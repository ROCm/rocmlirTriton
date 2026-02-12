// Fused GEMM test: C_f32 = extf(gemm(truncf(A_f32), truncf(B_f32)))
// Tests type-changing fusions: arith.truncf on inputs (f32 -> f16) and
// arith.extf on output (f16 -> f32).
// The issue is that LowerLoads uses a single tileType for the entire
// fusion chain, but arith.truncf changes the element type. The operand
// of truncf is f32, but the tile type is f16 (matching the GEMM), so
// creating a BlockwiseLoadOp for the f32 source with f16 tile type is
// a type mismatch.
#map = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map1 = #rock.transform_map<#map by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map2 = #rock.transform_map<#map by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<10000xf32>, %arg1: tensor<10000xf32>, %arg2: tensor<10000xf32>) -> tensor<10000xf32> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %0 = rock.transform %arg0 by #transform_map : tensor<10000xf32> to tensor<1x100x100xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<10000xf32> to tensor<1x100x100xf32>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<10000xf32> to tensor<1x100x100xf32>
    %a = arith.truncf %0 : tensor<1x100x100xf32> to tensor<1x100x100xf16>
    %b = arith.truncf %1 : tensor<1x100x100xf32> to tensor<1x100x100xf16>
    %3 = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_f32 = arith.extf %3 : tensor<1x100x100xf16> to tensor<1x100x100xf32>
    %4 = rock.store %gemm_f32 to %2 by set : tensor<1x100x100xf32> -> tensor<10000xf32> to tensor<1x100x100xf32>
    return %4 : tensor<10000xf32>
  }
}
