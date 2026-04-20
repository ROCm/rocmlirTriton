// RUN: rocmlir-opt --rocmlir-promote-softmax-precision --split-input-file %s | FileCheck %s

// The pass is anchored on `tosa.reduce_sum` whose input is `tosa.exp` of an
// f16/bf16 tensor (the TOSA softmax/LSE numerator-sum). The reduce_sum is
// always rewritten in f32. Recognized consumer chains (softmax tail and LSE)
// are also rewritten in f32 with a single cast back to the original element
// type at the boundary; unrecognized consumers fall back to a single
// `cast(reduce_sum_f32 -> orig)` so semantics are preserved.

// -----

// Basic f16 softmax normalization pattern: exp -> reduce_sum -> reciprocal -> mul.
// All three operations (reduce_sum, reciprocal, mul) should be promoted to f32
// and the original f16 reduce_sum / reciprocal / mul should be eliminated.

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
// `isSoftmaxNormalizeMul` matches either operand order, so the softmax tail
// is still recognized and promoted end-to-end.

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

// reduce_sum-of-exp is anchored and promoted, but the downstream `mul` uses a
// *different* `exp` than the one feeding the reduce_sum, so it is not part of
// a softmax tail. The reduce_sum still gets promoted (with a cast back to
// f16 inserted as the fallback semantics-preserving path) but the
// reciprocal/mul stay in f16.

// CHECK-LABEL: @reduce_sum_no_softmax_match
// CHECK-DAG: %[[SHIFT:.+]] = "tosa.const"()
// CHECK-DAG: %[[EXP0:.+]] = tosa.exp %arg0
// CHECK-DAG: %[[EXP1:.+]] = tosa.exp %arg1
// CHECK: %[[CAST_EXP1:.+]] = tosa.cast %[[EXP1]] : (tensor<1x384xf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM_F32:.+]] = tosa.reduce_sum %[[CAST_EXP1]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %[[SUM_F32]] : (tensor<1x1xf32>) -> tensor<1x1xf16>
// CHECK: %[[RECIP:.+]] = tosa.reciprocal %[[CAST_BACK]] : (tensor<1x1xf16>) -> tensor<1x1xf16>
// CHECK: %[[MUL:.+]] = tosa.mul %[[EXP0]], %[[RECIP]], %[[SHIFT]] : (tensor<1x384xf16>, tensor<1x1xf16>, tensor<1xi8>) -> tensor<1x384xf16>
// CHECK: return %[[MUL]]
func.func @reduce_sum_no_softmax_match(%arg0: tensor<1x384xf16>, %arg1: tensor<1x384xf16>) -> tensor<1x384xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.exp %arg1 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %2 = tosa.reduce_sum %1 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  %3 = tosa.reciprocal %2 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %4 = tosa.mul %0, %3, %shift : (tensor<1x384xf16>, tensor<1x1xf16>, tensor<1xi8>) -> tensor<1x384xf16>
  func.return %4 : tensor<1x384xf16>
}

// -----

// Pattern should not fire when the reciprocal input is not a reduce_sum at
// all (no anchor to match).

// CHECK-LABEL: @softmax_norm_no_reduce_sum
// CHECK-NOT: tosa.cast
// CHECK-NOT: tosa.reduce_sum
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

// Pattern should not fire for an unrelated mul that isn't part of a softmax
// pattern at all.

// CHECK-LABEL: @plain_mul_no_softmax
// CHECK-NOT: tosa.cast
// CHECK-NOT: tosa.reduce_sum
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

// -----

// LSE chain `log(reduce_sum(exp(...))) + max` (the canonical CPU LSE
// pattern). The reduce_sum, log, and the follow-up add are all promoted to
// f32 with a single cast back to f16 at the end of the chain. The other
// operand of the add (`reduce_max`-like value, here `%arg1`) is cast up to
// f32 for the in-f32 add.

// CHECK-LABEL: @lse_log_plus_max_f16
// CHECK: %[[EXP:.+]] = tosa.exp %arg0
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x384xf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK-DAG: %[[LOG:.+]] = tosa.log %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-DAG: %[[CAST_MAX:.+]] = tosa.cast %arg1 : (tensor<1x1xf16>) -> tensor<1x1xf32>
// CHECK: %[[ADD:.+]] = tosa.add %[[CAST_MAX]], %[[LOG]] : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %[[ADD]] : (tensor<1x1xf32>) -> tensor<1x1xf16>
// CHECK: return %[[CAST_BACK]]
func.func @lse_log_plus_max_f16(%arg0: tensor<1x384xf16>, %arg1: tensor<1x1xf16>) -> tensor<1x1xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  %2 = tosa.log %1 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %3 = tosa.add %arg1, %2 : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
  func.return %3 : tensor<1x1xf16>
}

// -----

// LSE chain in bf16 should also be promoted.

// CHECK-LABEL: @lse_log_plus_max_bf16
// CHECK: %[[EXP:.+]] = tosa.exp %arg0
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x384xbf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[LOG:.+]] = tosa.log %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: tosa.add
// CHECK-SAME: tensor<1x1xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %{{.+}} : (tensor<1x1xf32>) -> tensor<1x1xbf16>
// CHECK: return %[[CAST_BACK]]
func.func @lse_log_plus_max_bf16(%arg0: tensor<1x384xbf16>, %arg1: tensor<1x1xbf16>) -> tensor<1x1xbf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xbf16>) -> tensor<1x384xbf16>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xbf16>) -> tensor<1x1xbf16>
  %2 = tosa.log %1 : (tensor<1x1xbf16>) -> tensor<1x1xbf16>
  %3 = tosa.add %arg1, %2 : (tensor<1x1xbf16>, tensor<1x1xbf16>) -> tensor<1x1xbf16>
  func.return %3 : tensor<1x1xbf16>
}

// -----

// Bare LSE: `log(reduce_sum(exp(...)))` without a follow-up add. The log
// is promoted to f32 and a cast back to the original type is inserted on
// its result so downstream consumers still see f16.

// CHECK-LABEL: @lse_log_only_f16
// CHECK: %[[EXP:.+]] = tosa.exp
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x384xf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[LOG:.+]] = tosa.log %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %[[LOG]] : (tensor<1x1xf32>) -> tensor<1x1xf16>
// CHECK: return %[[CAST_BACK]]
func.func @lse_log_only_f16(%arg0: tensor<1x384xf16>) -> tensor<1x1xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  %2 = tosa.log %1 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  func.return %2 : tensor<1x1xf16>
}

// -----

// f32 LSE should not be promoted (already f32).

// CHECK-LABEL: @lse_log_plus_max_f32_noop
// CHECK: %[[EXP:.+]] = tosa.exp
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[LOG:.+]] = tosa.log %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK: %[[ADD:.+]] = tosa.add %arg1, %[[LOG]]
// CHECK-NOT: tosa.cast
// CHECK: return %[[ADD]]
func.func @lse_log_plus_max_f32_noop(%arg0: tensor<1x384xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf32>) -> tensor<1x384xf32>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
  %2 = tosa.log %1 : (tensor<1x1xf32>) -> tensor<1x1xf32>
  %3 = tosa.add %arg1, %2 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  func.return %3 : tensor<1x1xf32>
}

// -----

// Combined softmax + LSE sharing one `reduce_sum(exp(...))` (the
// kvcache-attention-with-LSE pattern). Both consumer chains must be
// rewritten in f32 against the *same* new f32 reduce_sum so the precision
// win is shared and there's no redundant f16 reduce_sum left over.

// CHECK-LABEL: @softmax_and_lse_shared_reduce_sum
// CHECK-DAG: %[[SHIFT:.+]] = "tosa.const"()
// CHECK-DAG: %[[EXP:.+]] = tosa.exp %arg0
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x384xf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// We expect exactly one f32 reduce_sum -- no leftover f16 reduce_sum.
// CHECK-NOT: tosa.reduce_sum {{.*}} -> tensor<1x1xf16>
// Softmax tail in f32:
// CHECK-DAG: %[[RECIP:.+]] = tosa.reciprocal %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-DAG: %[[MUL:.+]] = tosa.mul %[[CAST_EXP]], %[[RECIP]], %[[SHIFT]] : (tensor<1x384xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<1x384xf32>
// CHECK-DAG: %[[CAST_SOFTMAX:.+]] = tosa.cast %[[MUL]] : (tensor<1x384xf32>) -> tensor<1x384xf16>
// LSE chain in f32:
// CHECK-DAG: %[[LOG:.+]] = tosa.log %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-DAG: %[[CAST_MAX:.+]] = tosa.cast %arg1 : (tensor<1x1xf16>) -> tensor<1x1xf32>
// CHECK-DAG: %[[ADD:.+]] = tosa.add %[[CAST_MAX]], %[[LOG]] : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK-DAG: %[[CAST_LSE:.+]] = tosa.cast %[[ADD]] : (tensor<1x1xf32>) -> tensor<1x1xf16>
// CHECK: return %[[CAST_SOFTMAX]], %[[CAST_LSE]]
func.func @softmax_and_lse_shared_reduce_sum(%arg0: tensor<1x384xf16>, %arg1: tensor<1x1xf16>) -> (tensor<1x384xf16>, tensor<1x1xf16>) {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  // Softmax tail.
  %2 = tosa.reciprocal %1 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.mul %0, %2, %shift : (tensor<1x384xf16>, tensor<1x1xf16>, tensor<1xi8>) -> tensor<1x384xf16>
  // LSE.
  %4 = tosa.log %1 : (tensor<1x1xf16>) -> tensor<1x1xf16>
  %5 = tosa.add %arg1, %4 : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
  func.return %3, %5 : tensor<1x384xf16>, tensor<1x1xf16>
}

// -----

// The reduce_sum input is not `tosa.exp` so the anchor doesn't match and
// nothing is promoted -- semantics-preserving no-op.

// CHECK-LABEL: @reduce_sum_not_from_exp
// CHECK-NOT: -> tensor<{{.*}}xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
// CHECK: return %[[SUM]]
func.func @reduce_sum_not_from_exp(%arg0: tensor<1x384xf16>) -> tensor<1x1xf16> {
  %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  func.return %0 : tensor<1x1xf16>
}

// -----

// reduce_sum-of-exp whose only consumer is neither a softmax tail nor an LSE
// log. The reduce_sum is still promoted to f32 and the unrecognized consumer
// sees a `cast(reduce_sum_f32 -> orig)` -- semantically identical to the
// original f16 sum but with the precision-critical reduction done in f32.

// CHECK-LABEL: @reduce_sum_unrecognized_consumer
// CHECK: %[[EXP:.+]] = tosa.exp %arg0
// CHECK: %[[CAST_EXP:.+]] = tosa.cast %[[EXP]] : (tensor<1x384xf16>) -> tensor<1x384xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %[[CAST_EXP]] {axis = 1 : i32} : (tensor<1x384xf32>) -> tensor<1x1xf32>
// CHECK: %[[CAST_BACK:.+]] = tosa.cast %[[SUM]] : (tensor<1x1xf32>) -> tensor<1x1xf16>
// CHECK: %[[ADD:.+]] = tosa.add %arg1, %[[CAST_BACK]] : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
// CHECK: return %[[ADD]]
func.func @reduce_sum_unrecognized_consumer(%arg0: tensor<1x384xf16>, %arg1: tensor<1x1xf16>) -> tensor<1x1xf16> {
  %0 = tosa.exp %arg0 : (tensor<1x384xf16>) -> tensor<1x384xf16>
  %1 = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<1x384xf16>) -> tensor<1x1xf16>
  %2 = tosa.add %arg1, %1 : (tensor<1x1xf16>, tensor<1x1xf16>) -> tensor<1x1xf16>
  func.return %2 : tensor<1x1xf16>
}
