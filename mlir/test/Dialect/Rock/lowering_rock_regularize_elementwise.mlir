// RUN: rocmlir-opt -rock-regularize-elementwise -mlir-print-local-scope %s | FileCheck %s

#map_merge = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
#tf_merge = #rock.transform_map<#map_merge by [<Merge{3, 4} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [12] -> [3, 4]>

module {
  // CHECK-LABEL: func.func @push_inter_fusion_transform
  // CHECK: %[[T0:.*]] = rock.transform %arg0 by
  // CHECK: %[[T1:.*]] = rock.transform %arg1 by
  // CHECK: %[[ADD:.*]] = arith.addf %[[T0]], %[[T1]] : tensor<12xf32>
  // CHECK: arith.mulf %[[ADD]], %arg2 : tensor<12xf32>
  func.func @push_inter_fusion_transform(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<12xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel, rock.elementwise} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %t = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %mul = arith.mulf %t, %arg2 : tensor<12xf32>
    %r = rock.store %mul to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }

  // CHECK-LABEL: func.func @skip_unmarked_kernel
  // CHECK: %[[ADD:.*]] = arith.addf %arg0, %arg1 : tensor<3x4xf32>
  // CHECK: %[[T:.*]] = rock.transform %[[ADD]] by
  func.func @skip_unmarked_kernel(%arg0: tensor<3x4xf32>, %arg1: tensor<3x4xf32>, %dest: tensor<12xf32>) -> tensor<12xf32> attributes {rock.kernel} {
    %add = arith.addf %arg0, %arg1 : tensor<3x4xf32>
    %t = rock.transform %add by #tf_merge : tensor<3x4xf32> to tensor<12xf32>
    %r = rock.store %t to %dest by set : tensor<12xf32> -> tensor<12xf32> to tensor<12xf32>
    return %r : tensor<12xf32>
  }
}
