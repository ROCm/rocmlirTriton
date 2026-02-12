// Interleaved transform/fusion output test:
//   gemm [1,100,100]
//   T1 = Unmerge{10,10} on n → [1,100,10,10]
//   addf(T1_result, extra1) in 4D space
//   T2 = Merge{10,10} on (n0,n1) → [1,100,100]
//   addf(T2_result, extra2) in 3D space
//   store to flat 10000 output
//
// After RockRegularize, both fusions should operate in gemm space [1,100,100]:
//   extra1 gets inv(T1) = Merge applied
//   extra2 gets inv(T1∘T2) = identity (Merge undoes Unmerge)
//   store dest gets inv(T1∘T2) = identity
#map_flat = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#map_gemm_to_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 10 + d3)>
#map_4d_to_gemm = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 10, d2 mod 10)>
#map_flat_to_4d = affine_map<(d0, d1, d2, d3) -> (d1 * 100 + d2 * 10 + d3)>
#transform_map_a = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map_b = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_gemm_to_4d = #rock.transform_map<#map_gemm_to_4d by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Unmerge{10, 10} ["n0", "n1"] at [2, 3] -> ["n"] at [2]>] bounds = [1, 100, 10, 10] -> [1, 100, 100]>
#transform_4d_to_gemm = #rock.transform_map<#map_4d_to_gemm by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Merge{10, 10} ["n"] at [2] -> ["n0", "n1"] at [2, 3]>] bounds = [1, 100, 100] -> [1, 100, 10, 10]>
#transform_flat_to_4d = #rock.transform_map<#map_flat_to_4d by [<Unmerge{100, 10, 10} ["m", "n0", "n1"] at [1, 2, 3] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 10, 10] -> [10000]>
#transform_flat_to_3d = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg_a: tensor<10000xf16>, %arg_b: tensor<10000xf16>, %extra1: tensor<10000xf16>, %extra2: tensor<10000xf16>, %arg_c: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %a = rock.transform %arg_a by #transform_map_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg_b by #transform_map_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    // T1: Unmerge n → (n0, n1) to get 4D
    %gemm_4d = rock.transform %gemm by #transform_gemm_to_4d : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    // First fusion in 4D space
    %ext1 = rock.transform %extra1 by #transform_flat_to_4d : tensor<10000xf16> to tensor<1x100x10x10xf16>
    %fused1 = arith.addf %gemm_4d, %ext1 : tensor<1x100x10x10xf16>
    // T2: Merge (n0, n1) back to n to get 3D
    %fused1_3d = rock.transform %fused1 by #transform_4d_to_gemm : tensor<1x100x10x10xf16> to tensor<1x100x100xf16>
    // Second fusion in 3D space
    %ext2 = rock.transform %extra2 by #transform_flat_to_3d : tensor<10000xf16> to tensor<1x100x100xf16>
    %fused2 = arith.addf %fused1_3d, %ext2 : tensor<1x100x100xf16>
    // Store
    %c = rock.transform %arg_c by #transform_flat_to_3d : tensor<10000xf16> to tensor<1x100x100xf16>
    %result = rock.store %fused2 to %c by set : tensor<1x100x100xf16> -> tensor<10000xf16> to tensor<1x100x100xf16>
    return %result : tensor<10000xf16>
  }
}
