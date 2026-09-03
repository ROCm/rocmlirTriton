// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o - | FileCheck %s

// A grouped convolution can't be fused with the GEMM that follows it: the group
// becomes the batch dimension of both GEMMs, while the GEMM after a grouped
// convolution contracts over every group's output channels at once. Report the
// fusion as unsupported rather than build an op that computes the wrong thing.

// CHECK-LABEL: func.func @conv_gemm_grouped
// CHECK-NOT: rock.conv_elementwise_gemm
func.func @conv_gemm_grouped(%arg0: tensor<131072xf32>, %arg1: tensor<18432xf32>, %arg2: tensor<1024xf32>) -> tensor<32768xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xf32> into tensor<1x16x64xf32>
  %0 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1>} : (tensor<1x16x64xf32>) -> tensor<1x64x16xf32>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [64, 3, 3, 32] : tensor<18432xf32> into tensor<64x3x3x32xf32>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 32, 32, 64] : tensor<131072xf32> into tensor<2x32x32x64xf32>
  %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %3 = tosa.conv2d %expanded_1, %expanded_0, %2, %1, %1 {acc_type = f32, dilation = array<i64: 1, 1>, group = 2 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<2x32x32x64xf32>, tensor<64x3x3x32xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x64xf32>
  %collapsed = tensor.collapse_shape %3 [[0, 1, 2], [3]] : tensor<2x32x32x64xf32> into tensor<2048x64xf32>
  %expanded_2 = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [1, 2048, 64] : tensor<2048x64xf32> into tensor<1x2048x64xf32>
  // expected-error @+1 {{'tosa.matmul' op fusing a grouped convolution into conv+gemm is not supported}}
  %4 = tosa.matmul %expanded_2, %0, %1, %1 {acc_type = f32} : (tensor<1x2048x64xf32>, tensor<1x64x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x16xf32>
  %collapsed_3 = tensor.collapse_shape %4 [[0, 1, 2]] : tensor<1x2048x16xf32> into tensor<32768xf32>
  return %collapsed_3 : tensor<32768xf32>
}

// The same graph with group = 1 still fuses, so the CHECK-NOT above can't pass
// vacuously because of an unrelated match failure.

// CHECK-LABEL: func.func @conv_gemm_ungrouped
// CHECK: rock.conv_elementwise_gemm
func.func @conv_gemm_ungrouped(%arg0: tensor<131072xf32>, %arg1: tensor<36864xf32>, %arg2: tensor<1024xf32>) -> tensor<32768xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 16, 64] : tensor<1024xf32> into tensor<1x16x64xf32>
  %0 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1>} : (tensor<1x16x64xf32>) -> tensor<1x64x16xf32>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [64, 3, 3, 64] : tensor<36864xf32> into tensor<64x3x3x64xf32>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [2, 32, 32, 64] : tensor<131072xf32> into tensor<2x32x32x64xf32>
  %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
  %3 = tosa.conv2d %expanded_1, %expanded_0, %2, %1, %1 {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<2x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x64xf32>
  %collapsed = tensor.collapse_shape %3 [[0, 1, 2], [3]] : tensor<2x32x32x64xf32> into tensor<2048x64xf32>
  %expanded_2 = tensor.expand_shape %collapsed [[0, 1], [2]] output_shape [1, 2048, 64] : tensor<2048x64xf32> into tensor<1x2048x64xf32>
  %4 = tosa.matmul %expanded_2, %0, %1, %1 {acc_type = f32} : (tensor<1x2048x64xf32>, tensor<1x64x16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x16xf32>
  %collapsed_3 = tensor.collapse_shape %4 [[0, 1, 2]] : tensor<1x2048x16xf32> into tensor<32768xf32>
  return %collapsed_3 : tensor<32768xf32>
}
