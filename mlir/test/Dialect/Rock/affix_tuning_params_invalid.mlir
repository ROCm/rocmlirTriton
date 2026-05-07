// This tests the error handling in the rock-affix-params pass

// RUN: rocmlir-opt -rock-affix-params %s -verify-diagnostics
// RUN: rocmlir-opt -rock-affix-params %s -verify-diagnostics --mlir-disable-threading \
// RUN:   --mlir-print-ir-after-failure --mlir-print-ir-module-scope 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NA --implicit-check-not=rock.not_applicable

// TODO(roctriton): We need to unbufferize attention
// func.func @rock_attention_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @+1 {{The provided perf config is not valid}}
//   rock.attention{
//     qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//     %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {perf_config = "attn:v1:16,128,64,1,1,1,0,1,1,0,0", splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// TODO(roctriton): gemm_elementwise_gemm are broken
// func.func @rock_gemm_gemm_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @+1 {{The provided perf config is not valid}}
//   rock.gemm_elementwise_gemm{
//     ab = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
//     %arg3 = ab * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
//   } {perf_config = "attn:v1:16,128,64,1,1,1,0,1,1,0,0", splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
//   return
// }

// TODO(roctriton): conv_elementwise_gemm are broken
// func.func @rock_conv_gemm_invalid_perf_config(%arg0: memref<1x128x256x1x1xf16>, %arg1: memref<2x1x256x32x32xf16>, %arg2: memref<1x128x128xf16>, %arg3: memref<1x2048x128xf16>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx1100"} {
//   // expected-disabled-error @+1 {{The provided perf config is not valid}}
//   rock.conv_elementwise_gemm{
//     ab = conv(%arg0, %arg1) : memref<1x128x256x1x1xf16>, memref<2x1x256x32x32xf16>
//     %arg3 = ab * %arg2 : memref<1x128x128xf16> -> memref<1x2048x128xf16>
//   } {dilations = [1 : index, 1 : index], perf_config = "attn:v1:16,128,64,1,1,1,0,1,1,0,0", filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], storeMethod = #rock<StoreMethod set>, strides = [1 : index, 1 : index]}
//   return
// }

// expected-error @below {{Multiple Fusion Roots detected in a single function. This is not supported.}}
func.func @two_gemms(
    %a0: tensor<1x72x128xf8E4M3FN>, %b0: tensor<1x72x115200xf8E5M2>, %c0: tensor<1x128x115200xf32>,
    %a1: tensor<1x72x128xf8E4M3FN>, %b1: tensor<1x72x115200xf8E5M2>, %c1: tensor<1x128x115200xf32>)
    -> (tensor<1x128x115200xf32>, tensor<1x128x115200xf32>)
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // First GEMM
  %result0 = rock.gemm tr %a0 * %b0 
    : tensor<1x72x128xf8E4M3FN> * tensor<1x72x115200xf8E5M2> -> tensor<1x128x115200xf32>
  %out0 = rock.store %result0 to %c0 by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  %result1 = rock.gemm tr %a1 * %b1 
    : tensor<1x72x128xf8E4M3FN> * tensor<1x72x115200xf8E5M2> -> tensor<1x128x115200xf32>
  %out1 = rock.store %result1 to %c1 by set : tensor<1x128x115200xf32> -> tensor<1x128x115200xf32> to tensor<1x128x115200xf32>
  return %out0, %out1 : tensor<1x128x115200xf32>, tensor<1x128x115200xf32>
}

// expected-error @below {{unknown attribute 'kernel' on function 'bare_kernel_attr'}}
func.func @bare_kernel_attr(%arg0: tensor<1x128x128xf32>, %arg1: tensor<1x128x128xf32>, %arg2: tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    attributes {kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %result = rock.gemm %arg0 * %arg1
    : tensor<1x128x128xf32> * tensor<1x128x128xf32> -> tensor<1x128x128xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x128xf32> -> tensor<1x128x128xf32> to tensor<1x128x128xf32>
  return %out : tensor<1x128x128xf32>
}

// expected-error @below {{unknown attribute 'rock.prefil' on argument 0 of function 'unknown_arg_attr'}}
func.func @unknown_arg_attr(%arg0: tensor<1x128x128xf32> {rock.prefil}, %arg1: tensor<1x128x128xf32>, %arg2: tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
    attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %result = rock.gemm %arg0 * %arg1
    : tensor<1x128x128xf32> * tensor<1x128x128xf32> -> tensor<1x128x128xf32>
  %out = rock.store %result to %arg2 by set : tensor<1x128x128xf32> -> tensor<1x128x128xf32> to tensor<1x128x128xf32>
  return %out : tensor<1x128x128xf32>
}

// Verifies that a SplitK perfConfig that fails fusion legality marks the
// module with `rock.not_applicable` so the tuning driver classifies it as a
// non-applicable config rather than a compilation bug.
// NA-LABEL: 'func.func' operation: @rock_gemm_gemm_splitk
// NA: module attributes {rock.not_applicable
func.func @rock_gemm_gemm_splitk(%arg0: tensor<1474560xf16>, %arg1: tensor<1474560xf16>, %arg2: tensor<1474560xf16>, %arg3: tensor<1474560xf16>)  -> tensor<1474560xf16> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : tensor<1474560xf16> to tensor<1x4096x360xf16>
    %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 4096 + d2)> by [<Unmerge{360, 4096} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 360, 4096] -> [1474560]> : tensor<1474560xf16> to tensor<1x360x4096xf16>
    %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : tensor<1474560xf16> to tensor<1x4096x360xf16>
    %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : tensor<1474560xf16> to tensor<1x4096x360xf16>
    // expected-error @+1 {{Fusion with SplitK perfConfig is not legal}}
    %out = rock.gemm_elementwise_gemm{
     ab = %0 * %1 : tensor<1x4096x360xf16>, tensor<1x360x4096xf16>
     ab = elementwise {
    ^bb0(%arg4: tensor<1x4096x4096xf16>, %arg5: tensor<1x4096x4096xf16>):
      rock.yield %arg4 : tensor<1x4096x4096xf16>
    }
     out = ab * %2 : tensor<1x4096x360xf16>
    } {firstGemmIndices = array<i64: 0>, perf_config="attn:v1:32,32,32,1,1,4,0,3,1,0,0"}  -> tensor<1x4096x360xf16>

    %5 = arith.fptoui %out : tensor<1x4096x360xf16> to tensor<1x4096x360xi8>
    %6 = arith.sitofp %5 : tensor<1x4096x360xi8> to tensor<1x4096x360xf16>
    %7 = rock.store %6 to %3 by  set : tensor<1x4096x360xf16> -> tensor<1474560xf16> to tensor<1x4096x360xf16>
    return %7 : tensor<1474560xf16>
  }

// NA-LABEL: 'func.func' operation: @mlir_dot_max_splitk
// NA: module attributes {rock.not_applicable
func.func @mlir_dot_max_splitk(%arg1: tensor<1x2x1280xf32>, %arg2: tensor<1x1280x320xf32>, %arg3: tensor<1x2x320xf32>) -> tensor<1x2x320xf32> attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x2x320xf32>
    // expected-error @+1 {{Fusion with SplitK perfConfig is not legal}}
    %gemm_result = rock.gemm %arg1 * %arg2 {perf_config = "gemm:v1:64,64,64,1,1,4,16,3,2,0,0"} : tensor<1x2x1280xf32> * tensor<1x1280x320xf32> -> tensor<1x2x320xf32>
    %res = arith.maximumf %gemm_result, %cst : tensor<1x2x320xf32>
    %out = rock.store %res to %arg3 by set : tensor<1x2x320xf32> -> tensor<1x2x320xf32> to tensor<1x2x320xf32>
    return %out : tensor<1x2x320xf32>
  }
