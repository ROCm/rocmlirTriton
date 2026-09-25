// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Without a validator, a host-only function with a small-float parameter is
// called on the harness's validation buffers. Every parameter needs one, even
// the f32 and i32 ones, and each must keep its parameter's element type, so
// that the result slots appended afterwards line up with the regular buffers.

// RUN: rocmlir-gen -ph -pr -fut host_mixed %s | rocmlir-opt | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK: memref.alloc() : memref<2x3xf16>
// CHECK: %[[VAL_A:.+]] = memref.alloc() : memref<2x3xf16>
// CHECK: memref.alloc() : memref<3xf32>
// CHECK: %[[VAL_B:.+]] = memref.alloc() : memref<3xf32>
// CHECK: memref.alloc() : memref<4xi32>
// CHECK: %[[VAL_C:.+]] = memref.alloc() : memref<4xi32>
// CHECK: memref.alloc() : memref<6xf16>
// CHECK-NEXT: memref.alloc() : memref<6xf16>
// CHECK-NEXT: %[[A_T:.+]] = bufferization.to_tensor %[[VAL_A]]
// CHECK-NEXT: %[[B_T:.+]] = bufferization.to_tensor %[[VAL_B]]
// CHECK-NEXT: %[[C_T:.+]] = bufferization.to_tensor %[[VAL_C]]
// CHECK-NEXT: %[[OUT:.+]] = call @host_mixed(%[[A_T]], %[[B_T]], %[[C_T]]) : (tensor<2x3xf16>, tensor<3xf32>, tensor<4xi32>) -> tensor<6xf16>
// CHECK-NEXT: %[[OUT_M:.+]] = bufferization.to_buffer %[[OUT]] : tensor<6xf16> to memref<6xf16>
// CHECK: call @_memcpy_f16_f32_6(%[[OUT_M]], %{{.+}})

func.func @host_mixed(%arg0: tensor<2x3xf16>, %arg1: tensor<3xf32>, %arg2: tensor<4xi32>) -> tensor<6xf16> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x3xf16> into tensor<6xf16>
  return %0 : tensor<6xf16>
}
