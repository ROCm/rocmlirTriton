// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// RUN: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

// CHECK-LABEL: func.func @hip_transpose
// CHECK: hip.transpose
func.func @hip_transpose(%ctx: !hip.context, %arg0: tensor<4x16x256x4xf32>) -> tensor<4x4x16x256xf32> {
  %0 = tensor.empty() : tensor<4x4x16x256xf32>
  %1 = hip.transpose(%ctx) ins(%arg0 : tensor<4x16x256x4xf32>) outs(%0 : tensor<4x4x16x256xf32>) {perm = [0, 3, 1, 2]} : tensor<4x4x16x256xf32>
  return %1 : tensor<4x4x16x256xf32>
}

// CHECK-LABEL: func.func @hip_matmul
// CHECK: hip.matmul
func.func @hip_matmul(%ctx: !hip.context, %arg0: tensor<4x4x256x16xf32>, %arg1: tensor<4x4x16x256xf32>) -> tensor<4x4x256x256xf32> {
  %0 = tensor.empty() : tensor<4x4x256x256xf32>
  %1 = hip.matmul(%ctx) ins(%arg0, %arg1 : tensor<4x4x256x16xf32>, tensor<4x4x16x256xf32>) outs(%0 : tensor<4x4x256x256xf32>) : tensor<4x4x256x256xf32>
  return %1 : tensor<4x4x256x256xf32>
}
