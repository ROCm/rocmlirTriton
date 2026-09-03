// RUN: rocmlir-opt --hip-to-tosa='arch=gfx942' %s | FileCheck %s

// The pass also marks each function as a Rock kernel so TosaToRock accepts it,
// drops the !hip.context argument once nothing uses it any more, and strips the
// onnx.* provenance metadata that Rock's attribute allowlist would reject.
// CHECK-LABEL: func.func @hip_transpose
// CHECK-SAME: (%arg0: tensor<4x16x256x4xf32>)
// CHECK-SAME: attributes {rock.arch = "gfx942", rock.kernel}
// CHECK-NOT: onnx.
// CHECK: tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>}
// CHECK-NOT: hip.transpose
func.func @hip_transpose(%ctx: !hip.context, %arg0: tensor<4x16x256x4xf32> {onnx.name = "transpose:0"}) -> tensor<4x4x16x256xf32> attributes {onnx.graph.name = "Extracted from {tf2onnx}"} {
  %0 = tensor.empty() : tensor<4x4x16x256xf32>
  %1 = hip.transpose(%ctx) ins(%arg0 : tensor<4x16x256x4xf32>) outs(%0 : tensor<4x4x16x256xf32>) {perm = [0, 3, 1, 2]} : tensor<4x4x16x256xf32>
  return %1 : tensor<4x4x16x256xf32>
}

// A 3-D matmul needs no reshaping around tosa.matmul.
// CHECK-LABEL: func.func @hip_matmul_3d
// CHECK: tosa.matmul %arg0, %arg1, %{{.*}}, %{{.*}} {acc_type = f32}
// CHECK-NOT: hip.matmul
func.func @hip_matmul_3d(%ctx: !hip.context, %arg0: tensor<16x256x16xf32>, %arg1: tensor<16x16x256xf32>) -> tensor<16x256x256xf32> {
  %0 = tensor.empty() : tensor<16x256x256xf32>
  %1 = hip.matmul(%ctx) ins(%arg0, %arg1 : tensor<16x256x16xf32>, tensor<16x16x256xf32>) outs(%0 : tensor<16x256x256xf32>) : tensor<16x256x256xf32>
  return %1 : tensor<16x256x256xf32>
}

// A 4-D matmul collapses its leading batch dims for tosa.matmul, then expands back.
// CHECK-LABEL: func.func @hip_matmul_4d
// CHECK: tosa.reshape %arg0, %{{.*}} : (tensor<4x4x256x16xf32>, !tosa.shape<3>) -> tensor<16x256x16xf32>
// CHECK: tosa.reshape %arg1, %{{.*}} : (tensor<4x4x16x256xf32>, !tosa.shape<3>) -> tensor<16x16x256xf32>
// CHECK: %[[MM:.*]] = tosa.matmul
// CHECK: tosa.reshape %[[MM]], %{{.*}} : (tensor<16x256x256xf32>, !tosa.shape<4>) -> tensor<4x4x256x256xf32>
func.func @hip_matmul_4d(%ctx: !hip.context, %arg0: tensor<4x4x256x16xf32>, %arg1: tensor<4x4x16x256xf32>) -> tensor<4x4x256x256xf32> {
  %0 = tensor.empty() : tensor<4x4x256x256xf32>
  %1 = hip.matmul(%ctx) ins(%arg0, %arg1 : tensor<4x4x256x16xf32>, tensor<4x4x16x256xf32>) outs(%0 : tensor<4x4x256x256xf32>) : tensor<4x4x256x256xf32>
  return %1 : tensor<4x4x256x256xf32>
}

// transA/transB are not handled yet, so the op must survive untouched -- and
// with it the !hip.context argument it still uses.
// CHECK-LABEL: func.func @hip_matmul_transposed
// CHECK-SAME: !hip.context
// CHECK: hip.matmul
func.func @hip_matmul_transposed(%ctx: !hip.context, %arg0: tensor<16x16x256xf32>, %arg1: tensor<16x16x256xf32>) -> tensor<16x256x256xf32> {
  %0 = tensor.empty() : tensor<16x256x256xf32>
  %1 = hip.matmul(%ctx) ins(%arg0, %arg1 : tensor<16x16x256xf32>, tensor<16x16x256xf32>) outs(%0 : tensor<16x256x256xf32>) {transA = 1 : i64} : tensor<16x256x256xf32>
  return %1 : tensor<16x256x256xf32>
}
