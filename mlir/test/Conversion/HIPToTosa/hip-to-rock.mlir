// RUN: rocmlir-driver -kernel-pipeline hipep,highlevel -arch gfx942 %s | FileCheck %s

// The hipep phase stamps rock.kernel / rock.arch itself, so unlike the MIGraphX
// path the input carries no Rock attributes.
// CHECK-LABEL: @main_graph
// CHECK-SAME: rock.arch = "amdgcn-amd-amdhsa:gfx942"
// CHECK-SAME: rock.kernel
// CHECK: rock.gemm
// CHECK-NOT: hip.
// CHECK-NOT: tosa.
func.func @main_graph(%arg0: !hip.context, %arg1: tensor<4x16x256x4xf32>, %arg2: tensor<4x256x64xf32>) -> tensor<4x256x256x4xf32> {
  %0 = tensor.empty() : tensor<4x4x16x256xf32>
  %1 = hip.transpose(%arg0) ins(%arg1 : tensor<4x16x256x4xf32>) outs(%0 : tensor<4x4x16x256xf32>) {perm = [0, 3, 1, 2]} : tensor<4x4x16x256xf32>
  %expanded = tensor.expand_shape %arg2 [[0], [1], [2, 3]] output_shape [4, 256, 4, 16] : tensor<4x256x64xf32> into tensor<4x256x4x16xf32>
  %2 = tensor.empty() : tensor<4x4x256x16xf32>
  %3 = hip.transpose(%arg0) ins(%expanded : tensor<4x256x4x16xf32>) outs(%2 : tensor<4x4x256x16xf32>) {perm = [0, 2, 1, 3]} : tensor<4x4x256x16xf32>
  %4 = tensor.empty() : tensor<4x4x256x256xf32>
  %5 = hip.matmul(%arg0) ins(%3, %1 : tensor<4x4x256x16xf32>, tensor<4x4x16x256xf32>) outs(%4 : tensor<4x4x256x256xf32>) : tensor<4x4x256x256xf32>
  %6 = tensor.empty() : tensor<4x256x256x4xf32>
  %7 = hip.transpose(%arg0) ins(%5 : tensor<4x4x256x256xf32>) outs(%6 : tensor<4x256x256x4xf32>) {perm = [0, 2, 3, 1]} : tensor<4x256x256x4xf32>
  return %7 : tensor<4x256x256x4xf32>
}
