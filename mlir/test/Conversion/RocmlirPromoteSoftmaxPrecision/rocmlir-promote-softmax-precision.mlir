// RUN: rocmlir-opt --rocmlir-promote-softmax-precision --split-input-file %s | FileCheck %s

// Basic f16 softmax normalization pattern: exp -> reduce_sum -> reciprocal -> mul
// All three operations (reduce_sum, reciprocal, mul) should be promoted to f32.

// CHECK-LABEL: @softmax_norm_f16
// CHECK-DAG: %[[SHIFT:.+]] = "tosa.const"()
// CHECK: %[[EXP:.+]] = tosa.exp
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x384xf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[RECIP:.+]] = tosa.reciprocal %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[MUL:.+]] = tosa.mul %[[CAST_EXP]], %[[RECIP]], %[[SHIFT]] : (tensor<1x384xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x384xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %[[MUL]] : (tensor<1x384xf32>) -> tensor<1x384xf16>
// CHECK: return %[[CAST_BACK]]
func.func @softmax_norm_f16(%arg0: tensor<1x384xf16>) -> tensor<1x384xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  %2 = tosa.reciprocal %1 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.mul %0, %2, %shift : (tensor<1x384xf16>, tensor<1x1xf16>, tensor<1xi8>) -> tensor<1x384xf16>
  func.return %3 : tensor<1x384xf16>
}

// -----

// bf16 variant should also be promoted.

// CHECK-LABEL: @softmax_norm_bf16
// CHECK-DAG: %[[SHIFT:.+]] = "tosa.const"()
// CHECK: %[[EXP:.+]] = tosa.exp
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x384xbf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[RECIP:.+]] = tosa.reciprocal %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[MUL:.+]] = tosa.mul %[[CAST_EXP]], %[[RECIP]], %[[SHIFT]] : (tensor<1x384xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x384xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %[[MUL]] : (tensor<1x384xf32>) -> tensor<1x384xbf16>
// CHECK: return %[[CAST_BACK]]
func.func @softmax_norm_bf16(%arg0: tensor<1x384xbf16>) -> tensor<1x384xbf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xbf16>) -> tensor<1x384xbf16>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xbf16>) -> tensor<1x1xbf16>
  %2 = tosa.reciprocal %1 : (tensor<1x1xbf16>) -> tensor<1x1xbf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.mul %0, %2, %shift : (tensor<1x384xbf16>, tensor<1x1xbf16>, tensor<1xi8>) -> tensor<1x384xbf16>
  func.return %3 : tensor<1x384xbf16>
}

// -----

// f32 softmax should not be promoted (already f32).

// CHECK-LABEL: @softmax_norm_f32_noop
// CHECK-DAG: %[[SHIFT:.+]] = "tosa.const"()
// CHECK: %[[EXP:.+]] = tosa.exp
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[RECIP:.+]] = tosa.reciprocal %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[MUL:.+]] = tosa.mul %[[EXP]], %[[RECIP]], %[[SHIFT]] : (tensor<1x384xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x384xf32>
// CHECK-NOT: tosa.cast
// CHECK: return %[[MUL]]
func.func @softmax_norm_f32_noop(%arg0: tensor<1x384xf32>) -> tensor<1x384xf32> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf32>) -> tensor<1x384xf32>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
  %2 = tosa.reciprocal %1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.mul %0, %2, %shift : (tensor<1x384xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x384xf32>
  func.return %3 : tensor<1x384xf32>
}

// -----

// Operand order: reciprocal on input1, exp on input2 (reversed from typical).

// CHECK-LABEL: @softmax_norm_reversed_operands
// CHECK-DAG: %[[SHIFT:.+]] = "tosa.const"()
// CHECK: %[[EXP:.+]] = tosa.exp
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x384xf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[RECIP:.+]] = tosa.reciprocal %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[MUL:.+]] = tosa.mul %[[CAST_EXP]], %[[RECIP]], %[[SHIFT]] : (tensor<1x384xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x384xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %[[MUL]] : (tensor<1x384xf32>) -> tensor<1x384xf16>
// CHECK: return %[[CAST_BACK]]
func.func @softmax_norm_reversed_operands(%arg0: tensor<1x384xf16>) -> tensor<1x384xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  %2 = tosa.reciprocal %1 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.mul %2, %0, %shift : (tensor<1x1xf16>, tensor<1x384xf16>, tensor<1xi8>) -> tensor<1x384xf16>
  func.return %3 : tensor<1x384xf16>
}

// -----

// Pattern should not fire when reduce_sum input differs from exp feeding the mul.

// CHECK-LABEL: @softmax_norm_mismatched_exp
// CHECK-NOT: -> tensor<{{.*}}xf32>
// CHECK: tosa.mul
// CHECK-SAME: -> tensor<1x384xf16>
func.func @softmax_norm_mismatched_exp(%arg0: tensor<1x384xf16>, %arg1: tensor<1x384xf16>) -> tensor<1x384xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.exp %arg1 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %2 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  %3 = tosa.reciprocal %2 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %4 = tosa.mul %0, %3, %shift : (tensor<1x384xf16>, tensor<1x1xf16>, tensor<1xi8>) -> tensor<1x384xf16>
  func.return %4 : tensor<1x384xf16>
}

// -----

// Pattern should not fire when reciprocal input is not reduce_sum.

// CHECK-LABEL: @softmax_norm_no_reduce_sum
// CHECK-NOT: -> tensor<{{.*}}xf32>
// CHECK: tosa.mul
// CHECK-SAME: -> tensor<1x384xf16>
func.func @softmax_norm_no_reduce_sum(%arg0: tensor<1x384xf16>, %arg1: tensor<1x1xf16>) -> tensor<1x384xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.reciprocal %arg1 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 = tosa.mul %0, %1, %shift : (tensor<1x384xf16>, tensor<1x1xf16>, tensor<1xi8>) -> tensor<1x384xf16>
  func.return %2 : tensor<1x384xf16>
}

// -----

// Pattern should not fire when mul inputs are not exp and reciprocal.

// CHECK-LABEL: @plain_mul_no_softmax
// CHECK-NOT: tosa.cast
// CHECK: tosa.mul
// CHECK-SAME: -> tensor<1x384xf16>
func.func @plain_mul_no_softmax(%arg0: tensor<1x384xf16>, %arg1: tensor<1x384xf16>) -> tensor<1x384xf16> {
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<1x384xf16>, tensor<1x384xf16>, tensor<1xi8>) -> tensor<1x384xf16>
  func.return %0 : tensor<1x384xf16>
}

// -----

// 3D tensor with axis=2 reduction (attention-like shape).

// CHECK-LABEL: @softmax_norm_3d_attention
// CHECK-DAG: %[[SHIFT:.+]] = "tosa.const"()
// CHECK: %[[EXP:.+]] = tosa.exp
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x36x384xf16>) -> tensor<1x36x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 2 : i32} : (tensor<1x36x384xf32>) -> tensor<1x36x1xf32>
// CHECK: %[[RECIP:.+]] = tosa.reciprocal %[[SUM]] : (tensor<1x36x1xf32>) -> tensor<1x36x1xf32>
// CHECK: %[[MUL:.+]] = tosa.mul %[[CAST_EXP]], %[[RECIP]], %[[SHIFT]] : (tensor<1x36x384xf32>, tensor<1x36x1xf32>, tensor<1xi8>) -> tensor<1x36x384xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %[[MUL]] : (tensor<1x36x384xf32>) -> tensor<1x36x384xf16>
// CHECK: return %[[CAST_BACK]]
func.func @softmax_norm_3d_attention(%arg0: tensor<1x36x384xf16>) -> tensor<1x36x384xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x36x384xf16>) -> tensor<1x36x384xf16>
  %1 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<1x36x384xf16>) -> tensor<1x36x1xf16>
  %2 = tosa.reciprocal %1 : (tensor<1x36x1xf16>) -> tensor<1x36x1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.mul %0, %2, %shift : (tensor<1x36x384xf16>, tensor<1x36x1xf16>, tensor<1xi8>) -> tensor<1x36x384xf16>
  func.return %3 : tensor<1x36x384xf16>
}
