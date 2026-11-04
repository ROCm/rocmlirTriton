// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: rocmlir-opt %s -rock-regularize-reduce -cse | FileCheck %s

#map_merge_3x4_to_12 = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
#tf_merge_3x4_to_12 = #rock.transform_map<#map_merge_3x4_to_12 by [<Merge{3, 4} ["dim0"] at [0] -> ["row", "col"] at [0, 1]>] bounds = [12] -> [3, 4]>

module {
  // add -> transform(reshape) -> reduce_sum
  // CHECK-LABEL: func.func @transform_reduce
  func.func @transform_reduce(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<1xf32> attributes {rock.kernel, rock.reduce} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %flat = rock.transform %add by #tf_merge_3x4_to_12 : tensor<3x4xf32> to tensor<12xf32>
    %reduced = rock.reduce sum %flat {axis = 0 : index} : tensor<12xf32> -> tensor<1xf32>
    return %reduced : tensor<1xf32>
  }
  // CHECK-DAG: rock.transform %arg0
  // CHECK-DAG: rock.transform %arg1
  // CHECK: arith.addf

  // add -> mul -> transform(reshape) -> reduce_sum
  // CHECK-LABEL: func.func @pre_reduce_two_elementwise_then_transform
  func.func @pre_reduce_two_elementwise_then_transform(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<1xf32> attributes {rock.kernel, rock.reduce} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %mul = arith.mulf %add, %arg2 : tensor<3x4xf32>
    %flat = rock.transform %mul by #tf_merge_3x4_to_12 : tensor<3x4xf32> to tensor<12xf32>
    %reduced = rock.reduce sum %flat {axis = 0 : index} : tensor<12xf32> -> tensor<1xf32>
    return %reduced : tensor<1xf32>
  }
  // CHECK-DAG: rock.transform %arg0
  // CHECK-DAG: rock.transform %arg1
  // CHECK-DAG: arith.addf
  // CHECK-DAG: rock.transform %arg2
  // CHECK: arith.mulf

  // add(f16) -> extf(f32) -> transform(reshape) -> reduce_sum
  // CHECK-LABEL: func.func @transform_reduce_with_extf
  func.func @transform_reduce_with_extf(%arg0: tensor<3x4xf16>, %arg1: tensor<3x4xf16>) -> tensor<1xf32> attributes {rock.kernel, rock.reduce} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf16>
    %ext = arith.extf %add : tensor<3x4xf16> to tensor<3x4xf32>
    %flat = rock.transform %ext by #tf_merge_3x4_to_12 : tensor<3x4xf32> to tensor<12xf32>
    %reduced = rock.reduce sum %flat {axis = 0 : index} : tensor<12xf32> -> tensor<1xf32>
    return %reduced : tensor<1xf32>
  }
  // CHECK-DAG: rock.transform %arg0
  // CHECK-DAG: rock.transform %arg1
  // CHECK: arith.addf
  // CHECK: arith.extf


  // CHECK-LABEL: func.func @elementwise_multi_use
  func.func @elementwise_multi_use(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<1xf32> attributes {rock.kernel, rock.reduce} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %flat = rock.transform %add by #tf_merge_3x4_to_12 : tensor<3x4xf32> to tensor<12xf32>
    %mul = arith.mulf %add, %arg2 : tensor<3x4xf32>
    %flat_mul = rock.transform %mul by #tf_merge_3x4_to_12 : tensor<3x4xf32> to tensor<12xf32>
    %result = arith.addf %flat, %flat_mul : tensor<12xf32>
    %reduced = rock.reduce sum %result {axis = 0 : index} : tensor<12xf32> -> tensor<1xf32>
    return %reduced : tensor<1xf32>
  }
  // CHECK-DAG: rock.transform %arg0
  // CHECK-DAG: rock.transform %arg1
  // CHECK: arith.addf
  // CHECK: rock.transform %arg2
  // CHECK: arith.mulf
  // CHECK: arith.addf
}
