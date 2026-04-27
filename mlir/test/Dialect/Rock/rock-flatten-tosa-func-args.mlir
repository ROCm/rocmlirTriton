// RUN: rocmlir-opt --rock-flatten-tosa-func-args --split-input-file %s | FileCheck %s

// CHECK-LABEL: @flatten_args_and_result
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<1728xf32>, %[[ARG1:.*]]: tensor<1728xf32>) -> tensor<1728xf32>
// CHECK:       tosa.reshape %[[ARG0]], %{{.*}} : (tensor<1728xf32>, !tosa.shape<4>) -> tensor<1x12x12x12xf32>
// CHECK:       tosa.reshape %[[ARG1]], %{{.*}} : (tensor<1728xf32>, !tosa.shape<4>) -> tensor<1x12x12x12xf32>
// CHECK:       tosa.reshape %{{.*}}, %{{.*}} : (tensor<1x12x12x12xf32>, !tosa.shape<1>) -> tensor<1728xf32>
// CHECK:       return {{.*}} : tensor<1728xf32>
func.func @flatten_args_and_result(
    %arg0: tensor<1x12x12x12xf32>,
    %arg1: tensor<1x12x12x12xf32>)
    -> (tensor<1x12x12x12xf32>) attributes {rock.kernel} {
  %0 = tosa.add %arg0, %arg1 : (tensor<1x12x12x12xf32>, tensor<1x12x12x12xf32>) -> tensor<1x12x12x12xf32>
  return %0 : tensor<1x12x12x12xf32>
}

// -----

// Already 1-D: nothing should change.
// CHECK-LABEL: @already_flat
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<128xf32>, %[[ARG1:.*]]: tensor<128xf32>) -> tensor<128xf32>
// CHECK-NOT:   tosa.reshape
// CHECK:       return
func.func @already_flat(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> attributes {rock.kernel} {
  %0 = tosa.add %arg0, %arg1 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

// -----

// Mixed: some args are N-D, some are already 1-D or scalar-like.
// Same for returns: one N-D, one already 1-D.
// CHECK-LABEL: @mixed_ranks
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<384xf32>, %[[ARG1:.*]]: tensor<12xf32>, %[[ARG2:.*]]: tensor<1xf32>) -> (tensor<384xf32>, tensor<12xf32>)
// CHECK:       tosa.reshape %[[ARG0]], %{{.*}} : (tensor<384xf32>, !tosa.shape<2>) -> tensor<12x32xf32>
// CHECK-NOT:   tosa.reshape %[[ARG1]]
// CHECK-NOT:   tosa.reshape %[[ARG2]]
// CHECK:       tosa.reshape {{.*}} : (tensor<12x32xf32>, !tosa.shape<1>) -> tensor<384xf32>
// CHECK:       return {{.*}}, %[[ARG1]] : tensor<384xf32>, tensor<12xf32>
func.func @mixed_ranks(
    %arg0: tensor<12x32xf32>,
    %arg1: tensor<12xf32>,
    %arg2: tensor<1xf32>)
    -> (tensor<12x32xf32>, tensor<12xf32>) attributes {rock.kernel} {
  %0 = tosa.add %arg0, %arg0 : (tensor<12x32xf32>, tensor<12x32xf32>) -> tensor<12x32xf32>
  return %0, %arg1 : tensor<12x32xf32>, tensor<12xf32>
}

// -----

// Non-kernel function is also flattened (host functions need 1-D boundaries too).
// CHECK-LABEL: @non_kernel
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<32xf32>, %[[ARG1:.*]]: tensor<32xf32>) -> tensor<32xf32>
// CHECK:       tosa.reshape %[[ARG0]], %{{.*}} : (tensor<32xf32>, !tosa.shape<2>) -> tensor<4x8xf32>
// CHECK:       tosa.reshape %[[ARG1]], %{{.*}} : (tensor<32xf32>, !tosa.shape<2>) -> tensor<4x8xf32>
// CHECK:       tosa.reshape {{.*}} : (tensor<4x8xf32>, !tosa.shape<1>) -> tensor<32xf32>
func.func @non_kernel(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

// Multiple return values, both N-D.
// CHECK-LABEL: @multi_return
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<384xf32>) -> (tensor<384xf32>, tensor<384xf32>)
// CHECK:       tosa.reshape %[[ARG0]], %{{.*}} : (tensor<384xf32>, !tosa.shape<2>) -> tensor<12x32xf32>
// CHECK-DAG:   tosa.reshape {{.*}} : (tensor<12x32xf32>, !tosa.shape<1>) -> tensor<384xf32>
// CHECK-DAG:   tosa.reshape {{.*}} : (tensor<12x32xf32>, !tosa.shape<1>) -> tensor<384xf32>
// CHECK:       return {{.*}}, {{.*}} : tensor<384xf32>, tensor<384xf32>
func.func @multi_return(
    %arg0: tensor<12x32xf32>)
    -> (tensor<12x32xf32>, tensor<12x32xf32>) attributes {rock.kernel} {
  %0 = tosa.abs %arg0 : (tensor<12x32xf32>) -> tensor<12x32xf32>
  return %arg0, %0 : tensor<12x32xf32>, tensor<12x32xf32>
}
