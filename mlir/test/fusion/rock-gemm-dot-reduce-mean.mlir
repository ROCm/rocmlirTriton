// RUN: rocmlir-opt -rock-affix-params -rock-lower-reduce -mlir-print-local-scope %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 128 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 128 + d1) * 32 + d2)>
#map2 = affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 32 + d2)>
#map3 = affine_map<(d0, d1) -> (0, d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0 * 32 + d1, d2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{1, 32, 128} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 128] -> [4096]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{1, 128, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 128, 32] -> [4096]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{1, 32, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 32] -> [1024]>
#transform_map3 = #rock.transform_map<#map3 by [<Merge{1, 32} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [32, 32] -> [1, 32, 32]>
#transform_map4 = #rock.transform_map<#map5 by [<Unmerge{1, 32} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 32, 32] -> [32, 32]>

// CHECK-LABEL: func.func @mlir_dot_add_reduce_mean
// CHECK-SAME: %arg3: tensor<1x32x1xf32> {rock.prefill = 0.000000e+00 : f32}
func.func @mlir_dot_add_reduce_mean(%arg0: tensor<4096xf32>, %arg1: tensor<4096xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1x32x1xf32>) -> tensor<1x32x1xf32> attributes {rock.arch = "gfx942:sramecc+:xnack-", rock.enable_splitk_for_tuning, rock.kernel = "mixr"} {
    %cst = arith.constant dense<3.125000e-02> : tensor<32x32xf32>
    %0 = rock.transform %arg0 by #transform_map : tensor<4096xf32> to tensor<1x32x128xf32>
    %1 = rock.transform %arg1 by #transform_map1 : tensor<4096xf32> to tensor<1x128x32xf32>
    %2 = rock.transform %arg2 by #transform_map2 : tensor<1024xf32> to tensor<1x32x32xf32>
    %gemm = rock.gemm %0 * %1 : tensor<1x32x128xf32> * tensor<1x128x32xf32> -> tensor<1x32x32xf32>
    %3 = rock.transform %gemm by #transform_map3 : tensor<1x32x32xf32> to tensor<32x32xf32>
    %4 = rock.transform %2 by #transform_map3 : tensor<1x32x32xf32> to tensor<32x32xf32>
    %fused_add = arith.addf %3, %4 : tensor<32x32xf32>
    %fused_mul = arith.mulf %fused_add, %cst : tensor<32x32xf32>
    %5 = rock.transform %fused_mul by #transform_map4 : tensor<32x32xf32> to tensor<1x32x32xf32>
    // CHECK-NOT: rock.reduce
    // CHECK: rock.transform %arg3 by {{.*}}Broadcast{{.*}}
    // CHECK: rock.store %{{.*}} by atomic_add
    %reduced = rock.reduce sum %5 {axis = 2 : index} : tensor<1x32x32xf32> -> tensor<1x32x1xf32>
    %out = rock.store %reduced to %arg3 by set : tensor<1x32x1xf32> -> tensor<1x32x1xf32> to tensor<1x32x1xf32>
    return %out : tensor<1x32x1xf32>
  }
