// RUN: rocmlir-driver -kernel-pipeline=gpu %s | rocmlir-opt

#map = affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 64 + d2)>
#map_swap = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map_out = affine_map<(d0) -> (d0 floordiv 4096, (d0 floordiv 64) mod 64, d0 mod 64)>

#transform_map_f16 = #rock.transform_map<#map by [<Unmerge{64, 64, 64} ["d0", "d1", "d2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 64, 64] -> [262144]>
#transform_map = #rock.transform_map<#map by [<Unmerge{64, 64, 64} ["d0", "d1", "d2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [64, 64, 64] -> [262144]>
#transform_swap = #rock.transform_map<#map_swap by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [64, 64, 64] -> [64, 64, 64]>
#transform_map_out = #rock.transform_map<#map_out by [<Merge{64, 64, 64} ["dim0"] at [0] -> ["d0", "d1", "d2"] at [0, 1, 2]>] bounds = [262144] -> [64, 64, 64]>

module attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  func.func @rock_gemm(%arg0: tensor<262144xf16>, %arg1: tensor<262144xf32>, %arg2: tensor<262144xf32>, %arg3: tensor<262144xf32>) -> tensor<262144xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
    %0 = rock.transform %arg0 by #transform_map_f16 : tensor<262144xf16> to tensor<64x64x64xf16>
    %1 = rock.transform %arg1 by #transform_map : tensor<262144xf32> to tensor<64x64x64xf32>
    %3 = rock.transform %arg3 by #transform_map : tensor<262144xf32> to tensor<64x64x64xf32>
    %ext = arith.extf %0 : tensor<64x64x64xf16> to tensor<64x64x64xf32>
    %fused = arith.addf %ext, %3 : tensor<64x64x64xf32>
    %swapped = rock.transform %fused by #transform_swap : tensor<64x64x64xf32> to tensor<64x64x64xf32>
    %gemm = rock.gemm %swapped * %1 : tensor<64x64x64xf32> * tensor<64x64x64xf32> -> tensor<64x64x64xf32>
    %flat = rock.transform %gemm by #transform_map_out : tensor<64x64x64xf32> to tensor<262144xf32>
    %result = rock.store %flat to %arg2 by set : tensor<262144xf32> -> tensor<262144xf32> to tensor<262144xf32>
    return %result : tensor<262144xf32>
  }
}
