// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt -split-input-file --tosa-to-rock -verify-diagnostics -o -| FileCheck %s

// Q is a plain kernel argument, so its matmul operand is an expand_shape and
// not a collapse_shape. The per-batch lastValidKVIndex must still be broadcast
// across the 4 attention groups instead of being passed through unchanged.
// CHECK-LABEL: func @mlir_attention_plain_q
// CHECK: %[[HEAD_BROADCAST:.*]] = rock.transform %arg3 {{.*}} : tensor<1x1xi32> to tensor<1x4xi32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[HEAD_BROADCAST]]
// CHECK: rock.attention
// CHECK: lastValidKVIndex = (%[[FLAT]] : tensor<4xi32>)
func.func @mlir_attention_plain_q(%arg0: tensor<16xf16>, %arg1: tensor<128xf16>, %arg2: tensor<128xf16>, %arg3: tensor<1x1xi32>) -> (tensor<16xf16>) attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %q = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [4, 1, 4] : tensor<16xf16> into tensor<4x1x4xf16>
  %k = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [4, 4, 8] : tensor<128xf16> into tensor<4x4x8xf16>
  %v = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [4, 8, 4] : tensor<128xf16> into tensor<4x8x4xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = tosa.matmul %q, %k, %a_zp, %b_zp {acc_type = f32} : (tensor<4x1x4xf16>, tensor<4x4x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<4x1x8xf16>
  %expanded = tensor.expand_shape %0 [[0, 1], [2], [3]] output_shape [1, 4, 1, 8] : tensor<4x1x8xf16> into tensor<1x4x1x8xf16>
  %ninf = "tosa.const"() <{values = dense<0xFC00> : tensor<1x4x1x8xf16>}> : () -> tensor<1x4x1x8xf16>
  %scale = "tosa.const"() <{values = dense<1.250000e-01> : tensor<1x4x1x8xf16>}> : () -> tensor<1x4x1x8xf16>
  %range = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %ones = "tosa.const"() <{values = dense<1> : tensor<1x4x1x8xi32>}> : () -> tensor<1x4x1x8xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %range, %ones, %shift : (tensor<1x1x1x8xi32>, tensor<1x4x1x8xi32>, tensor<1xi8>) -> tensor<1x4x1x8xi32>
  %idx = tensor.expand_shape %arg3 [[0], [1, 2, 3]] output_shape [1, 1, 1, 1] : tensor<1x1xi32> into tensor<1x1x1x1xi32>
  %2 = tosa.mul %idx, %ones, %shift : (tensor<1x1x1x1xi32>, tensor<1x4x1x8xi32>, tensor<1xi8>) -> tensor<1x4x1x8xi32>
  %3 = tosa.greater %1, %2 : (tensor<1x4x1x8xi32>, tensor<1x4x1x8xi32>) -> tensor<1x4x1x8xi1>
  %4 = tosa.cast %3 : (tensor<1x4x1x8xi1>) -> tensor<1x4x1x8xi32>
  %5 = tosa.cast %4 : (tensor<1x4x1x8xi32>) -> tensor<1x4x1x8xi8>
  %6 = tosa.mul %expanded, %scale, %shift : (tensor<1x4x1x8xf16>, tensor<1x4x1x8xf16>, tensor<1xi8>) -> tensor<1x4x1x8xf16>
  %7 = tosa.cast %5 : (tensor<1x4x1x8xi8>) -> tensor<1x4x1x8xi1>
  %8 = tosa.select %7, %ninf, %6 : (tensor<1x4x1x8xi1>, tensor<1x4x1x8xf16>, tensor<1x4x1x8xf16>) -> tensor<1x4x1x8xf16>
  %9 = tosa.reduce_max %8 {axis = 3 : i32} : (tensor<1x4x1x8xf16>) -> tensor<1x4x1x1xf16>
  %10 = tosa.sub %8, %9 : (tensor<1x4x1x8xf16>, tensor<1x4x1x1xf16>) -> tensor<1x4x1x8xf16>
  %11 = tosa.exp %10 : (tensor<1x4x1x8xf16>) -> tensor<1x4x1x8xf16>
  %12 = tosa.reduce_sum %11 {axis = 3 : i32} : (tensor<1x4x1x8xf16>) -> tensor<1x4x1x1xf16>
  %13 = tosa.reciprocal %12 : (tensor<1x4x1x1xf16>) -> tensor<1x4x1x1xf16>
  %14 = tosa.mul %11, %13, %shift : (tensor<1x4x1x8xf16>, tensor<1x4x1x1xf16>, tensor<1xi8>) -> tensor<1x4x1x8xf16>
  %collapsed = tensor.collapse_shape %14 [[0, 1], [2], [3]] : tensor<1x4x1x8xf16> into tensor<4x1x8xf16>
  %15 = tosa.matmul %collapsed, %v, %a_zp, %b_zp {acc_type = f32} : (tensor<4x1x8xf16>, tensor<4x8x4xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<4x1x4xf16>
  %out = tensor.collapse_shape %15 [[0, 1, 2]] : tensor<4x1x4xf16> into tensor<16xf16>
  return %out : tensor<16xf16>
}

// -----

// A per-batch index with two batches is broadcast over the two groups of each
// batch, keeping a distinct value per batch.
// CHECK-LABEL: func @mlir_attention_plain_q_batched
// CHECK: %[[HEAD_BROADCAST:.*]] = rock.transform %arg3 {{.*}} : tensor<2x1xi32> to tensor<2x2xi32>
// CHECK: %[[FLAT:.*]] = tensor.collapse_shape %[[HEAD_BROADCAST]]
// CHECK: rock.attention
// CHECK: lastValidKVIndex = (%[[FLAT]] : tensor<4xi32>)
func.func @mlir_attention_plain_q_batched(%arg0: tensor<16xf16>, %arg1: tensor<128xf16>, %arg2: tensor<128xf16>, %arg3: tensor<2x1xi32>) -> (tensor<16xf16>) attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %q = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [4, 1, 4] : tensor<16xf16> into tensor<4x1x4xf16>
  %k = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [4, 4, 8] : tensor<128xf16> into tensor<4x4x8xf16>
  %v = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [4, 8, 4] : tensor<128xf16> into tensor<4x8x4xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = tosa.matmul %q, %k, %a_zp, %b_zp {acc_type = f32} : (tensor<4x1x4xf16>, tensor<4x4x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<4x1x8xf16>
  %expanded = tensor.expand_shape %0 [[0, 1], [2], [3]] output_shape [2, 2, 1, 8] : tensor<4x1x8xf16> into tensor<2x2x1x8xf16>
  %ninf = "tosa.const"() <{values = dense<0xFC00> : tensor<2x2x1x8xf16>}> : () -> tensor<2x2x1x8xf16>
  %scale = "tosa.const"() <{values = dense<1.250000e-01> : tensor<2x2x1x8xf16>}> : () -> tensor<2x2x1x8xf16>
  %range = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %ones = "tosa.const"() <{values = dense<1> : tensor<2x2x1x8xi32>}> : () -> tensor<2x2x1x8xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %range, %ones, %shift : (tensor<1x1x1x8xi32>, tensor<2x2x1x8xi32>, tensor<1xi8>) -> tensor<2x2x1x8xi32>
  %idx = tensor.expand_shape %arg3 [[0], [1, 2, 3]] output_shape [2, 1, 1, 1] : tensor<2x1xi32> into tensor<2x1x1x1xi32>
  %2 = tosa.mul %idx, %ones, %shift : (tensor<2x1x1x1xi32>, tensor<2x2x1x8xi32>, tensor<1xi8>) -> tensor<2x2x1x8xi32>
  %3 = tosa.greater %1, %2 : (tensor<2x2x1x8xi32>, tensor<2x2x1x8xi32>) -> tensor<2x2x1x8xi1>
  %4 = tosa.cast %3 : (tensor<2x2x1x8xi1>) -> tensor<2x2x1x8xi32>
  %5 = tosa.cast %4 : (tensor<2x2x1x8xi32>) -> tensor<2x2x1x8xi8>
  %6 = tosa.mul %expanded, %scale, %shift : (tensor<2x2x1x8xf16>, tensor<2x2x1x8xf16>, tensor<1xi8>) -> tensor<2x2x1x8xf16>
  %7 = tosa.cast %5 : (tensor<2x2x1x8xi8>) -> tensor<2x2x1x8xi1>
  %8 = tosa.select %7, %ninf, %6 : (tensor<2x2x1x8xi1>, tensor<2x2x1x8xf16>, tensor<2x2x1x8xf16>) -> tensor<2x2x1x8xf16>
  %9 = tosa.reduce_max %8 {axis = 3 : i32} : (tensor<2x2x1x8xf16>) -> tensor<2x2x1x1xf16>
  %10 = tosa.sub %8, %9 : (tensor<2x2x1x8xf16>, tensor<2x2x1x1xf16>) -> tensor<2x2x1x8xf16>
  %11 = tosa.exp %10 : (tensor<2x2x1x8xf16>) -> tensor<2x2x1x8xf16>
  %12 = tosa.reduce_sum %11 {axis = 3 : i32} : (tensor<2x2x1x8xf16>) -> tensor<2x2x1x1xf16>
  %13 = tosa.reciprocal %12 : (tensor<2x2x1x1xf16>) -> tensor<2x2x1x1xf16>
  %14 = tosa.mul %11, %13, %shift : (tensor<2x2x1x8xf16>, tensor<2x2x1x1xf16>, tensor<1xi8>) -> tensor<2x2x1x8xf16>
  %collapsed = tensor.collapse_shape %14 [[0, 1], [2], [3]] : tensor<2x2x1x8xf16> into tensor<4x1x8xf16>
  %15 = tosa.matmul %collapsed, %v, %a_zp, %b_zp {acc_type = f32} : (tensor<4x1x8xf16>, tensor<4x8x4xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<4x1x4xf16>
  %out = tensor.collapse_shape %15 [[0, 1, 2]] : tensor<4x1x4xf16> into tensor<16xf16>
  return %out : tensor<16xf16>
}
