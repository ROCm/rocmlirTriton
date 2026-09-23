// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A host-only function that returns tensors gets one harness buffer per
// result, shaped like that result. Without a validator and without
// small-float parameters there are no validation buffers, so the function is
// called on the regular buffers. A function without arguments is still called
// through the tensor interface, with no result buffers passed as arguments.
// Results that are not ranked tensors, such as scalars or memrefs, are
// rejected.

// RUN: rocmlir-gen -ph -pr -fut host_two_results %s | rocmlir-opt | FileCheck %s
// RUN: rocmlir-gen -ph -pr -fut host_no_args %s | rocmlir-opt | FileCheck %s --check-prefix=NOARGS
// RUN: not rocmlir-gen -ph -pr -fut host_scalar_result %s 2>&1 | FileCheck %s --check-prefix=ERR
// RUN: not rocmlir-gen -ph -pr -fut host_memref_result %s 2>&1 | FileCheck %s --check-prefix=MEMREF

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

// NOARGS-LABEL: func.func @main()
// NOARGS-NEXT: %[[RES:.+]] = memref.alloc() : memref<4xf32>
// NOARGS-NEXT: %[[OUT:.+]] = call @host_no_args() : () -> tensor<4xf32>
// NOARGS-NEXT: %[[OUT_M:.+]] = bufferization.to_buffer %[[OUT]]
// NOARGS-NEXT: memref.copy %[[OUT_M]], %[[RES]] : memref<4xf32> to memref<4xf32>
// NOARGS-NEXT: memref.cast %[[RES]]

// ERR: error: host harness only supports ranked tensor function results
// MEMREF: error: host harness only supports ranked tensor function results

func.func @host_two_results(%arg0: tensor<2x3xf32>) -> (tensor<6xf32>, tensor<3x2xf32>) {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [3, 2] : tensor<6xf32> into tensor<3x2xf32>
  return %0, %1 : tensor<6xf32>, tensor<3x2xf32>
}

func.func @host_no_args() -> tensor<4xf32> {
  %0 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @host_memref_result(%arg0: tensor<2x3xf32>) -> memref<2x3xf32> {
  %0 = bufferization.to_buffer %arg0 : tensor<2x3xf32> to memref<2x3xf32>
  return %0 : memref<2x3xf32>
}

func.func @host_scalar_result(%arg0: tensor<2x3xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = tensor.extract %arg0[%c0, %c0] : tensor<2x3xf32>
  return %0 : f32
}
