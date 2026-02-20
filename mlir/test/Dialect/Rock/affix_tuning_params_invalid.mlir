// This tests the error handling in the rock-affix-params pass

// RUN: rocmlir-opt -rock-affix-params %s -verify-diagnostics

// TODO(roctriton): We need to unbufferize attention
// func.func @rock_attention_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @+1 {{The provided perf config is not valid}}
//   rock.attention{
//     qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//     %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {perf_config = "attn:v2:128,128,16,8,32,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// func.func @rock_gemm_gemm_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @+1 {{The provided perf config is not valid}}
//   rock.gemm_elementwise_gemm{
//     ab = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//     %arg3 = ab * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {perf_config = "attn:v2:128,128,16,8,32,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// TODO(roctriton): conv_elementwise_gemm are broken
// func.func @rock_conv_gemm_invalid_perf_config(%arg0: memref<1x128x256x1x1xf16>, %arg1: memref<2x1x256x32x32xf16>, %arg2: memref<1x128x128xf16>, %arg3: memref<1x2048x128xf16>) attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @+1 {{The provided perf config is not valid}}
//   rock.conv_elementwise_gemm{
//     ab = conv(%arg0, %arg1) : memref<1x128x256x1x1xf16>, memref<2x1x256x32x32xf16>
//     %arg3 = ab * %arg2 : memref<1x128x128xf16> -> memref<1x2048x128xf16>
//   } {dilations = [1 : index, 1 : index], perf_config = "attn:v2:128,128,16,8,32,64,8,1,1,2,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], storeMethod = #rock<StoreMethod set>, strides = [1 : index, 1 : index]}
//   return
// }

// expected-error @below {{Multiple Fusion Roots detected in a single function. This is not supported.}}
func.func @two_gemms(
    %a0: tensor<1x72x128xf8E4M3FN>, %b0: tensor<1x72x115200xf8E5M2>, %c0: tensor<1x128x115200xf32>,
    %a1: tensor<1x72x128xf8E4M3FN>, %b1: tensor<1x72x115200xf8E5M2>, %c1: tensor<1x128x115200xf32>)
    -> (tensor<1x128x115200xf32>, tensor<1x128x115200xf32>)
    attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // First GEMM
  %result0 = rock.gemm tr %a0 * %b0 
    : tensor<1x72x128xf8E4M3FN> * tensor<1x72x115200xf8E5M2> -> tensor<1x128x115200xf32>
  %out0 = rock.store %result0 to %c0 by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  %result1 = rock.gemm tr %a1 * %b1 
    : tensor<1x72x128xf8E4M3FN> * tensor<1x72x115200xf8E5M2> -> tensor<1x128x115200xf32>
  %out1 = rock.store %result1 to %c1 by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out0, %out1 : tensor<1x128x115200xf32>, tensor<1x128x115200xf32>
}

// TODO(roctriton): We need to unbufferize attention
// func.func @rock_attn_schedulev2(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {schedule_version =  #rock.schedule_version<2>, rock.arch = "amdgcn-amd-amdhsa:gfx942"} {
//   // expected-disabled-error @+1 {{kernel has both perf_config and schedule_version attribute set. Please modify schedule version directly inside perf_config and remove schedule_version}}
//   rock.attention{
//     qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//     %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {perf_config = "attn:v2:128,128,16,8,32,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// TODO(roctriton): We need to unbufferize attention
// func.func @rock_attn_perfconfig_schedulev3_navi(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1200"} {
//   // expected-disabled-error @+1 {{schedule version not supported}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>, perf_config = "attn:v2:32,32,32,32,32,32,1,1,3,2,1"}
//   return
// }

// TODO(roctriton): We need to unbufferize attention
// func.func @rock_attn_perfconfig_schedulev4_navi(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {rock.arch = "amdgcn-amd-amdhsa:gfx1200"} {
//   // expected-disabled-error @+1 {{schedule version not supported}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>, perf_config = "attn:v2:32,32,32,32,32,32,1,1,4,2,1"}
//   return
// }

// TODO(roctriton): We need to unbufferize attention
// func.func @rock_attn_schedulev3_navi(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {schedule_version =  #rock.schedule_version<3>, rock.arch = "amdgcn-amd-amdhsa:gfx1200"} {
//   // expected-disabled-error @+1 {{schedule version not supported}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// TODO(roctriton): We need to unbufferize attention
// func.func @rock_attn_schedulev4_navi(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {schedule_version =  #rock.schedule_version<4>, rock.arch = "amdgcn-amd-amdhsa:gfx1200"} {
//   // expected-disabled-error @+1 {{schedule version not supported}}
//   rock.attention{
//    qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//    %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// func.func @rock_gemm_gemm_splitk(%arg0: memref<1474560xf16>, %arg1: memref<1474560xf16>, %arg2: memref<1474560xf16>, %arg3: memref<1474560xf16>) attributes {enable_splitk_for_tuning, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
//     %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : memref<1474560xf16> to memref<1x4096x360xf16>
//     %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 4096 + d2)> by [<Unmerge{360, 4096} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 360, 4096] -> [1474560]> : memref<1474560xf16> to memref<1x360x4096xf16>
//     %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : memref<1474560xf16> to memref<1x4096x360xf16>
//     %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 1 + d2)> by [<Unmerge{4096, 360} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : memref<1474560xf16> to memref<1x4096x360xf16>
//     %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4096x360xf16>
//     // expected-disabled-error @+1 {{Fusion with SplitK perfConfig is not legal}}
//     rock.gemm_elementwise_gemm{
//      ab = %0 * %1 : memref<1x4096x360xf16>, memref<1x360x4096xf16>
//      ab = elementwise {
//     ^bb0(%arg4: memref<1x4096x4096xf16>, %arg5: memref<1x4096x4096xf16>):
//       memref.copy %arg4, %arg5 : memref<1x4096x4096xf16> to memref<1x4096x4096xf16>
//       rock.yield
//     }
//      %alloc = ab * %2 : memref<1x4096x360xf16> -> memref<1x4096x360xf16>
//     } {firstGemmIndices = array<i64: 0>, storeMethod = #rock<StoreMethod set>, perf_config="attn:v3:32,32,32,32,32,32,16,1,2,1,2,0,1"}
//     %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x4096x360xf16>
//
//     linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<1x4096x360xf16>) outs(%alloc_1 : memref<1x4096x360xf16>) {
//     ^bb0(%in: f16, %out: f16):
//       %5 = arith.fptoui %in : f16 to i8
//       %6 = arith.sitofp %5 : i8 to f16
//       linalg.yield %6 : f16
//     }
//     memref.copy %alloc_1, %3 : memref<1x4096x360xf16> to memref<1x4096x360xf16>
//     return
//   }

// TODO(rocmlirTriton): This fails due to a bug in rocmlirTriton
// func.func @mlir_dot_max_splitk(%arg1: tensor<1x2x1280xf32>, %arg2: tensor<1x1280x320xf32>, %arg3: tensor<1x2x320xf32>) -> tensor<1x2x320xf32> attributes {enable_splitk_for_tuning, kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
//     %cst = arith.constant 0.000000e+00 : f32
//     %empty = tensor.empty() : tensor<1x2x320xf32>
//     // expected-disabled-error @+1 {{Fusion with SplitK perfConfig is not legal}}
//     %gemm_result = rock.gemm %arg1 * %arg2 {rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", perf_config = "v4:16,16,4,16,16,16,1,5,1,2,0,0,1,1"} : tensor<1x2x1280xf32> * tensor<1x1280x320xf32> -> tensor<1x2x320xf32>
//     %alloc = rock.store %gemm_result to %empty by set : tensor<1x2x320xf32> -> tensor<1x2x320xf32> to tensor<1x2x320xf32>
//     %0 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : tensor<1x2x320xf32> to tensor<2x320xf32>
//     %empty_0 = tensor.empty() : tensor<2x320xf32>
//     %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0: tensor<2x320xf32>) outs(%empty_0 : tensor<2x320xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %3 = arith.maximumf %in, %cst : f32
//       linalg.yield %3 : f32
//     } -> tensor<2x320xf32>
//     %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{1, 2} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 320] -> [2, 320]> : tensor<2x320xf32> to tensor<1x2x320xf32>
//     %out = rock.store %2 to %arg3 by set : tensor<1x2x320xf32> -> tensor<1x2x320xf32> to tensor<1x2x320xf32>
//     return %out : tensor<1x2x320xf32>
//   }
