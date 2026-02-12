// Complex output fusion DAG where multiple fusion paths trace back to root:
//   gemm [1,100,100]
//   T1 = Unmerge{10,10} on n → [1,100,10,10]
//   exp(T1_result) in 4D
//   mulf(T1_result, exp_result) in 4D — both operands trace to gemm
//   T2 = Merge{10,10} on (n0,n1) → [1,100,100]
//   addf(T2_result, gemm) in 3D — both operands trace to gemm
//   store to flat 10000 output
//
// Flat semantics: C[i] = gemm(A,B)[i] * exp(gemm(A,B)[i]) + gemm(A,B)[i]
//
// RockRegularize must handle:
//   1. mulf with two root-tracing operands (direct + indirect via exp)
//   2. addf with two root-tracing operands (via T1→mulf→T2 and direct gemm)
//   3. Cancelling transforms (T1∘T2 = identity)
#map_flat = affine_map<(d0, d1, d2) -> (d1 * 100 + d2)>
#map_gemm_to_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 10 + d3)>
#map_4d_to_gemm = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 10, d2 mod 10)>
#transform_map_a = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_map_b = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
#transform_gemm_to_4d = #rock.transform_map<#map_gemm_to_4d by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Unmerge{10, 10} ["n0", "n1"] at [2, 3] -> ["n"] at [2]>] bounds = [1, 100, 10, 10] -> [1, 100, 100]>
#transform_4d_to_gemm = #rock.transform_map<#map_4d_to_gemm by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <PassThrough ["m"] at [1] -> ["m"] at [1]>, <Merge{10, 10} ["n"] at [2] -> ["n0", "n1"] at [2, 3]>] bounds = [1, 100, 100] -> [1, 100, 10, 10]>
#transform_flat_to_3d = #rock.transform_map<#map_flat by [<Unmerge{100, 100} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 100, 100] -> [10000]>
module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg_a: tensor<10000xf16>, %arg_b: tensor<10000xf16>, %arg_c: tensor<10000xf16>) -> tensor<10000xf16> attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100", rock.enable_splitk_for_tuning, rock.kernel, rock.num_chiplets = 1 : i64, rock.num_cu = 96 : i64} {
    %a = rock.transform %arg_a by #transform_map_a : tensor<10000xf16> to tensor<1x100x100xf16>
    %b = rock.transform %arg_b by #transform_map_b : tensor<10000xf16> to tensor<1x100x100xf16>
    %gemm = rock.gemm %a * %b : tensor<1x100x100xf16> * tensor<1x100x100xf16> -> tensor<1x100x100xf16>
    %gemm_4d = rock.transform %gemm by #transform_gemm_to_4d : tensor<1x100x100xf16> to tensor<1x100x10x10xf16>
    %exp = math.exp %gemm_4d : tensor<1x100x10x10xf16>
    %fused = arith.mulf %gemm_4d, %exp : tensor<1x100x10x10xf16>
    %fused_3d = rock.transform %fused by #transform_4d_to_gemm : tensor<1x100x10x10xf16> to tensor<1x100x100xf16>
    %added = arith.addf %fused_3d, %gemm : tensor<1x100x100xf16>
    %c = rock.transform %arg_c by #transform_flat_to_3d : tensor<10000xf16> to tensor<1x100x100xf16>
    %result = rock.store %added to %c by set : tensor<1x100x100xf16> -> tensor<10000xf16> to tensor<1x100x100xf16>
    return %result : tensor<10000xf16>
  }
}
