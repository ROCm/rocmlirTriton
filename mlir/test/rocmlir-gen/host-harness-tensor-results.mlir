// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A host-only function that returns tensors gets one harness buffer per
// result, shaped like that result. Without a validator and without
// small-float parameters there are no validation buffers, so the function is
// called on the regular buffers. Non-shaped results are rejected.

// RUN: rocmlir-gen -ph -pr -fut host_two_results %s | rocmlir-opt | FileCheck %s
// RUN: not rocmlir-gen -ph -pr -fut host_scalar_result %s 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-LABEL: func.func @main()
// CHECK: %[[IN:.+]] = memref.alloc() : memref<2x3xf32>
// CHECK: %[[RES0:.+]] = memref.alloc() : memref<6xf32>
// CHECK-NEXT: %[[RES1:.+]] = memref.alloc() : memref<3x2xf32>
// CHECK-NEXT: %[[IN_T:.+]] = bufferization.to_tensor %[[IN]]
// CHECK-NEXT: %[[OUT:.+]]:2 = call @host_two_results(%[[IN_T]]) : (tensor<2x3xf32>) -> (tensor<6xf32>, tensor<3x2xf32>)
// CHECK-NEXT: %[[OUT0:.+]] = bufferization.to_buffer %[[OUT]]#0
// CHECK-NEXT: memref.copy %[[OUT0]], %[[RES0]] : memref<6xf32> to memref<6xf32>
// CHECK-NEXT: %[[OUT1:.+]] = bufferization.to_buffer %[[OUT]]#1
// CHECK-NEXT: memref.copy %[[OUT1]], %[[RES1]] : memref<3x2xf32> to memref<3x2xf32>
// CHECK-NOT: memref.alloc()
// CHECK: memref.cast %[[RES0]]
// CHECK: memref.cast %[[RES1]]

// ERR: error: host harness only supports shaped function results

func.func @host_two_results(%arg0: tensor<2x3xf32>) -> (tensor<6xf32>, tensor<3x2xf32>) {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [3, 2] : tensor<6xf32> into tensor<3x2xf32>
  return %0, %1 : tensor<6xf32>, tensor<3x2xf32>
}

func.func @host_scalar_result(%arg0: tensor<2x3xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %arg0[%c0, %c0] : tensor<2x3xf32>
  return %0 : f32
}
