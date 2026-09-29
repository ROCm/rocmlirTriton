// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// A host-only function that returns a tensor shaped differently from its
// argument. The harness converts the result back to a memref of the result's
// own shape, and calls the function through the tensor interface since it is
// not a GPU kernel.

// RUN: rocmlir-gen -ph -pr -fut host_collapse %s | rocmlir-opt | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK: %[[IN:.+]] = memref.alloc() : memref<2x3xf32>
// CHECK: %[[IN_T:.+]] = bufferization.to_tensor %[[IN]]
// CHECK-NEXT: %[[OUT:.+]] = call @host_collapse(%[[IN_T]]) : (tensor<2x3xf32>) -> tensor<6xf32>
// CHECK-NEXT: %[[OUT_M:.+]] = bufferization.to_buffer %[[OUT]] : tensor<6xf32> to memref<6xf32>
// CHECK: memref.cast %[[OUT_M]] : memref<6xf32> to memref<*xf32>
// CHECK: call @printMemrefF32

func.func @host_collapse(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf32> into tensor<6xf32>
  return %0 : tensor<6xf32>
}
