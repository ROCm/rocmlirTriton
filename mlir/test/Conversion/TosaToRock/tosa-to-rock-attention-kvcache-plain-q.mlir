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

// -----

// An index whose leading dimension (3) does not divide the 4 attention groups
// cannot be broadcast per group, so match() must decline the fusion. The
// leftover softmax then fails to legalize, which is what pins the decline: if
// the attention had fused, no diagnostic would be emitted and
// -verify-diagnostics would fail this case.
func.func @mlir_attention_plain_q_indivisible(%arg0: tensor<16xf16>, %arg1: tensor<128xf16>, %arg2: tensor<384xf16>, %arg3: tensor<3x1xi32>) -> (tensor<48xf16>) attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %q = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [4, 1, 4] : tensor<16xf16> into tensor<4x1x4xf16>
  %k = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [4, 4, 8] : tensor<128xf16> into tensor<4x4x8xf16>
  %v = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [12, 8, 4] : tensor<384xf16> into tensor<12x8x4xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = tosa.matmul %q, %k, %a_zp, %b_zp {acc_type = f32} : (tensor<4x1x4xf16>, tensor<4x4x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<4x1x8xf16>
  %expanded = tensor.expand_shape %0 [[0, 1], [2], [3]] output_shape [1, 4, 1, 8] : tensor<4x1x8xf16> into tensor<1x4x1x8xf16>
  %ninf = "tosa.const"() <{values = dense<0xFC00> : tensor<3x4x1x8xf16>}> : () -> tensor<3x4x1x8xf16>
  %scale = "tosa.const"() <{values = dense<1.250000e-01> : tensor<1x4x1x8xf16>}> : () -> tensor<1x4x1x8xf16>
  %range = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %ones = "tosa.const"() <{values = dense<1> : tensor<3x4x1x8xi32>}> : () -> tensor<3x4x1x8xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %range, %ones, %shift : (tensor<1x1x1x8xi32>, tensor<3x4x1x8xi32>, tensor<1xi8>) -> tensor<3x4x1x8xi32>
  %idx = tensor.expand_shape %arg3 [[0], [1, 2, 3]] output_shape [3, 1, 1, 1] : tensor<3x1xi32> into tensor<3x1x1x1xi32>
  %2 = tosa.mul %idx, %ones, %shift : (tensor<3x1x1x1xi32>, tensor<3x4x1x8xi32>, tensor<1xi8>) -> tensor<3x4x1x8xi32>
  %3 = tosa.greater %1, %2 : (tensor<3x4x1x8xi32>, tensor<3x4x1x8xi32>) -> tensor<3x4x1x8xi1>
  %4 = tosa.mul %expanded, %scale, %shift : (tensor<1x4x1x8xf16>, tensor<1x4x1x8xf16>, tensor<1xi8>) -> tensor<1x4x1x8xf16>
  %5 = tosa.select %3, %ninf, %4 : (tensor<3x4x1x8xi1>, tensor<3x4x1x8xf16>, tensor<1x4x1x8xf16>) -> tensor<3x4x1x8xf16>
  // expected-error@below {{failed to legalize operation 'tosa.reduce_max'}}
  %6 = tosa.reduce_max %5 {axis = 3 : i32} : (tensor<3x4x1x8xf16>) -> tensor<3x4x1x1xf16>
  %7 = tosa.sub %5, %6 : (tensor<3x4x1x8xf16>, tensor<3x4x1x1xf16>) -> tensor<3x4x1x8xf16>
  %8 = tosa.exp %7 : (tensor<3x4x1x8xf16>) -> tensor<3x4x1x8xf16>
  %9 = tosa.reduce_sum %8 {axis = 3 : i32} : (tensor<3x4x1x8xf16>) -> tensor<3x4x1x1xf16>
  %10 = tosa.reciprocal %9 : (tensor<3x4x1x1xf16>) -> tensor<3x4x1x1xf16>
  %11 = tosa.mul %8, %10, %shift : (tensor<3x4x1x8xf16>, tensor<3x4x1x1xf16>, tensor<1xi8>) -> tensor<3x4x1x8xf16>
  %collapsed = tensor.collapse_shape %11 [[0, 1], [2], [3]] : tensor<3x4x1x8xf16> into tensor<12x1x8xf16>
  %12 = tosa.matmul %collapsed, %v, %a_zp, %b_zp {acc_type = f32} : (tensor<12x1x8xf16>, tensor<12x8x4xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x1x4xf16>
  %out = tensor.collapse_shape %12 [[0, 1, 2]] : tensor<12x1x4xf16> into tensor<48xf16>
  return %out : tensor<48xf16>
}

// -----

// A per-batch prefix offset that reaches the matcher through a tosa.transpose
// is validated against the underlying block argument, but the matcher returns
// the transpose result itself, which the rewrite cannot broadcast per group.
// match() must decline the fusion instead of emitting a 2-element prefixOffset
// against the 4 attention groups.
func.func @mlir_attention_prefix_transposed_offset(%arg0: tensor<64xf16>, %arg1: tensor<128xf16>, %arg2: tensor<128xf16>, %arg3: tensor<1x2xi32>) -> (tensor<64xf16>) attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %q = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [4, 4, 4] : tensor<64xf16> into tensor<4x4x4xf16>
  %k = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [4, 4, 8] : tensor<128xf16> into tensor<4x4x8xf16>
  %v = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [4, 8, 4] : tensor<128xf16> into tensor<4x8x4xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = tosa.matmul %q, %k, %a_zp, %b_zp {acc_type = f32} : (tensor<4x4x4xf16>, tensor<4x4x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<4x4x8xf16>
  %expanded = tensor.expand_shape %0 [[0, 1], [2], [3]] output_shape [2, 2, 4, 8] : tensor<4x4x8xf16> into tensor<2x2x4x8xf16>
  %ninf = "tosa.const"() <{values = dense<0xFC00> : tensor<2x2x4x8xf16>}> : () -> tensor<2x2x4x8xf16>
  %scale = "tosa.const"() <{values = dense<1.250000e-01> : tensor<2x2x4x8xf16>}> : () -> tensor<2x2x4x8xf16>
  %rows = arith.constant dense<[[[[0], [1], [2], [3]]]]> : tensor<1x1x4x1xi32>
  %cols = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %ones = "tosa.const"() <{values = dense<1> : tensor<2x2x4x8xi32>}> : () -> tensor<2x2x4x8xi32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %t = tosa.transpose %arg3 {perms = array<i32: 1, 0>} : (tensor<1x2xi32>) -> tensor<2x1xi32>
  %off = tensor.expand_shape %t [[0], [1, 2, 3]] output_shape [2, 1, 1, 1] : tensor<2x1xi32> into tensor<2x1x1x1xi32>
  %sum = tosa.add %rows, %off : (tensor<1x1x4x1xi32>, tensor<2x1x1x1xi32>) -> tensor<2x1x4x1xi32>
  %1 = tosa.mul %sum, %ones, %shift : (tensor<2x1x4x1xi32>, tensor<2x2x4x8xi32>, tensor<1xi8>) -> tensor<2x2x4x8xi32>
  %2 = tosa.mul %cols, %ones, %shift : (tensor<1x1x1x8xi32>, tensor<2x2x4x8xi32>, tensor<1xi8>) -> tensor<2x2x4x8xi32>
  %3 = tosa.greater %2, %1 : (tensor<2x2x4x8xi32>, tensor<2x2x4x8xi32>) -> tensor<2x2x4x8xi1>
  %4 = tosa.mul %expanded, %scale, %shift : (tensor<2x2x4x8xf16>, tensor<2x2x4x8xf16>, tensor<1xi8>) -> tensor<2x2x4x8xf16>
  %5 = tosa.select %3, %ninf, %4 : (tensor<2x2x4x8xi1>, tensor<2x2x4x8xf16>, tensor<2x2x4x8xf16>) -> tensor<2x2x4x8xf16>
  // expected-error@below {{failed to legalize operation 'tosa.reduce_max'}}
  %6 = tosa.reduce_max %5 {axis = 3 : i32} : (tensor<2x2x4x8xf16>) -> tensor<2x2x4x1xf16>
  %7 = tosa.sub %5, %6 : (tensor<2x2x4x8xf16>, tensor<2x2x4x1xf16>) -> tensor<2x2x4x8xf16>
  %8 = tosa.exp %7 : (tensor<2x2x4x8xf16>) -> tensor<2x2x4x8xf16>
  %9 = tosa.reduce_sum %8 {axis = 3 : i32} : (tensor<2x2x4x8xf16>) -> tensor<2x2x4x1xf16>
  %10 = tosa.reciprocal %9 : (tensor<2x2x4x1xf16>) -> tensor<2x2x4x1xf16>
  %11 = tosa.mul %8, %10, %shift : (tensor<2x2x4x8xf16>, tensor<2x2x4x1xf16>, tensor<1xi8>) -> tensor<2x2x4x8xf16>
  %collapsed = tensor.collapse_shape %11 [[0, 1], [2], [3]] : tensor<2x2x4x8xf16> into tensor<4x4x8xf16>
  %12 = tosa.matmul %collapsed, %v, %a_zp, %b_zp {acc_type = f32} : (tensor<4x4x8xf16>, tensor<4x8x4xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<4x4x4xf16>
  %out = tensor.collapse_shape %12 [[0, 1, 2]] : tensor<4x4x4xf16> into tensor<64xf16>
  return %out : tensor<64xf16>
}
