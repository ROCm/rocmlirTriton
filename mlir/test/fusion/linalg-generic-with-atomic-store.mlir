// RUN: rocmlir-opt -rock-affix-params %s -verify-diagnostics

#map_a = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 1024 + d2)>
#map_b = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 512 + d2)>
#map_out = affine_map<(d0) -> (0, d0 floordiv 512, d0 mod 512)>

#transform_map_a = #rock.transform_map<#map_a by [<Unmerge{1, 1024, 1024} ["d0", "d1", "d2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 1024, 1024] -> [1048576]>
#transform_map_b = #rock.transform_map<#map_b by [<Unmerge{1, 1024, 512} ["d0", "d1", "d2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 1024, 512] -> [524288]>
#transform_map_out = #rock.transform_map<#map_out by [<Merge{1, 1024, 512} ["dim0"] at [0] -> ["d0", "d1", "d2"] at [0, 1, 2]>] bounds = [524288] -> [1, 1024, 512]>

module {
  func.func @rock_gemm(%arg0: tensor<1048576xf32>, %arg1: tensor<524288xf32>, %arg2: tensor<524288xi16>) -> tensor<524288xi16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a"} {
    %0 = rock.transform %arg0 by #transform_map_a : tensor<1048576xf32> to tensor<1x1024x1024xf32>
    %1 = rock.transform %arg1 by #transform_map_b : tensor<524288xf32> to tensor<1x1024x512xf32>
    // expected-error @+1 {{Fusion with SplitK perfConfig is not legal}}
    %gemm = rock.gemm %0 * %1 {perf_config = "gemm:v1:16,32,4,1,1,4,16,2,1,0,0"} : tensor<1x1024x1024xf32> * tensor<1x1024x512xf32> -> tensor<1x1024x512xf32>
    %add_cst = arith.constant dense<2.000000e+00> : tensor<1x1024x512xf32>
    %added = arith.addf %gemm, %add_cst : tensor<1x1024x512xf32>
    %converted = arith.fptosi %added : tensor<1x1024x512xf32> to tensor<1x1024x512xi16>
    %flat = rock.transform %converted by #transform_map_out : tensor<1x1024x512xi16> to tensor<524288xi16>
    %result = rock.store %flat to %arg2 by set : tensor<524288xi16> -> tensor<524288xi16> to tensor<524288xi16>
    return %result : tensor<524288xi16>
  }
}
