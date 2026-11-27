// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A view and an elementwise op commute, so every `rock.transform` can be
// pushed down to the leaves of a nongemm kernel, leaving the elementwise ops
// stacked on one shape. This holds whether or not the kernel ends in a
// reduction.

// RUN: rocmlir-opt %s -rock-regularize-nongemm -cse -mlir-print-local-scope | FileCheck %s

#map_merge = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
#tf_merge = #rock.transform_map<#map_merge by [<Merge{3, 4} ["dim0"] at [0] -> ["row", "col"] at [0, 1]>] bounds = [12] -> [3, 4]>

module {
  // A view between two elementwise ops: the second one already works in the
  // viewed shape, so the view has to reach the operands of the first.
  // CHECK-LABEL: func.func @push_inter_fusion_transform
  // CHECK: %[[T0:.*]] = rock.transform %arg0 by
  // CHECK: %[[T1:.*]] = rock.transform %arg1 by
  // CHECK: %[[ADD:.*]] = arith.addf %[[T0]], %[[T1]] : tensor<12xf32>
  // CHECK: arith.mulf %[[ADD]], %arg2 : tensor<12xf32>
  func.func @push_inter_fusion_transform(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel, rock.nongemm} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %t = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %mul = arith.mulf %t, %arg2 : tensor<12xf32>
    %r = rock.store %mul to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }

  // Without the marker the kernel belongs to another pipeline, which has its
  // own reasons for where its views sit.
  // CHECK-LABEL: func.func @skip_unmarked_kernel
  // CHECK: %[[ADD:.*]] = arith.addf %arg0, %arg1 : tensor<3x4xf32>
  // CHECK: %[[T:.*]] = rock.transform %[[ADD]] by
  func.func @skip_unmarked_kernel(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %t = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %r = rock.store %t to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }

  // add -> transform(reshape) -> reduce_sum
  // CHECK-LABEL: func.func @transform_reduce
  func.func @transform_reduce(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<1xf32> attributes {rock.kernel, rock.nongemm} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %flat = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %reduced = rock.reduce sum %flat {axis = 0 : index} : tensor<12xf32> -> tensor<1xf32>
    return %reduced : tensor<1xf32>
  }
  // CHECK-DAG: rock.transform %arg0
  // CHECK-DAG: rock.transform %arg1
  // CHECK: arith.addf

  // add -> mul -> transform(reshape) -> reduce_sum: the view walks past both.
  // CHECK-LABEL: func.func @pre_reduce_two_elementwise_then_transform
  func.func @pre_reduce_two_elementwise_then_transform(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<1xf32> attributes {rock.kernel, rock.nongemm} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %mul = arith.mulf %add, %arg2 : tensor<3x4xf32>
    %flat = rock.transform %mul by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %reduced = rock.reduce sum %flat {axis = 0 : index} : tensor<12xf32> -> tensor<1xf32>
    return %reduced : tensor<1xf32>
  }
  // CHECK-DAG: rock.transform %arg0
  // CHECK-DAG: rock.transform %arg1
  // CHECK-DAG: arith.addf
  // CHECK-DAG: rock.transform %arg2
  // CHECK: arith.mulf

  // Only the shape travels through a cast: `arith.extf` reads f16 and writes
  // f32, so each new view keeps its own operand's element type.
  // CHECK-LABEL: func.func @transform_reduce_with_extf
  func.func @transform_reduce_with_extf(%arg0: tensor<3x4xf16>, %arg1: tensor<3x4xf16>) -> tensor<1xf32> attributes {rock.kernel, rock.nongemm} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf16>
    %ext = arith.extf %add : tensor<3x4xf16> to tensor<3x4xf32>
    %flat = rock.transform %ext by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %reduced = rock.reduce sum %flat {axis = 0 : index} : tensor<12xf32> -> tensor<1xf32>
    return %reduced : tensor<1xf32>
  }
  // CHECK-DAG: rock.transform %arg0
  // CHECK-DAG: rock.transform %arg1
  // CHECK: arith.addf
  // CHECK: arith.extf

  // A fusion op with two viewed users is cloned per view rather than moved,
  // so the other user keeps a definition in the shape it reads.
  // CHECK-LABEL: func.func @elementwise_multi_use
  func.func @elementwise_multi_use(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<1xf32> attributes {rock.kernel, rock.nongemm} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %flat = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %mul = arith.mulf %add, %arg2 : tensor<3x4xf32>
    %flat_mul = rock.transform %mul by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
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
