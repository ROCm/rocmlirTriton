// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=32 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=128 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 -pv | rocmlir-opt | FileCheck %s --check-prefix=DEGENERATE
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=-1 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 -pv | rocmlir-opt | FileCheck %s --check-prefix=DEGENERATE
// RUN: not rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -causal_look_back=32 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 2>&1 | FileCheck %s --check-prefix=NEEDS-CAUSAL
// RUN: not rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=0 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 2>&1 | FileCheck %s --check-prefix=NOT-POSITIVE
// RUN: not rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=32 -prefix_offset=4 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 2>&1 | FileCheck %s --check-prefix=NO-PREFIX

// Degeneracy is set by the query length: the band's lower edge is max(0, q - L)
// and clamps to zero for every row only once L covers the longest query row.
// At seq_len_q 256 and L 128 it still masks -- query 200 keeps keys 72..127,
// not 0..127 -- and the attribute cannot represent a band wider than the key
// sequence, so fail instead of quietly widening the mask.
// RUN: not rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=128 -seq_len_q 256 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 2>&1 | FileCheck %s --check-prefix=TOO-WIDE
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=255 -seq_len_q 256 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 -pv 2>&1 | FileCheck %s --check-prefix=WIDE-DEGENERATE
// A square band at exactly seq_len_q - 1 reaches key 0 from the last query row,
// so it is degenerate even though it is still within the verifier's bound.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=127 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 -pv 2>&1 | FileCheck %s --check-prefix=SQUARE-DEGENERATE

// The band changes how many key blocks the n-loop visits, so it has to be part
// of the tuning identity; a plain causal problem must keep the key it had
// before, or every shipped per-problem quick-tuning map stops matching.
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -causal_look_back=32 -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 --emit-tuning-key | FileCheck %s --check-prefix=TUNING-KEY
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention --causal -seq_len_q 128 -seq_len_k 128 -head_dim_qk 64 -head_dim_v 64 -t f32 --emit-tuning-key | FileCheck %s --check-prefix=TUNING-KEY-NO-BAND

// TUNING-KEY: -split_kv 1 -causal_look_back 32 -num_heads_q 1
// TUNING-KEY-NO-BAND: -split_kv 1 -num_heads_q 1
// TUNING-KEY-NO-BAND-NOT: causal_look_back

// CHECK: rock.attention
// CHECK-NEXT: qk = %{{.*}} * %{{.*}}
// CHECK-NEXT: causalLookBack = 32
// CHECK-NEXT: causal
// CHECK: softmax(qk) * %{{.*}}

// CHECK-LABEL: func.func @host_naive_attention

// Causal's upper edge: mask keys strictly after the query.
// CHECK: %[[COLS:.*]] = tosa.mul %{{.*}}, %{{.*}} : (tensor<1x1x128xi32>, tensor<1x128x128xi32>, tensor<1xi8>) -> tensor<1x128x128xi32>
// CHECK: %[[FUTURE:.*]] = tosa.greater %[[COLS]], %{{.*}} : (tensor<1x128x128xi32>, tensor<1x128x128xi32>) -> tensor<1x128x128xi1>
// CHECK: tosa.select %[[FUTURE]], %{{.*}}, %{{.*}} : (tensor<1x128x128xi1>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>

// The band's lower edge, max(0, row - 32). Unlike the sliding window, the bound
// comes from the broadcast row index, so it is a full tensor rather than a
// 1x1x1 scalar, and it moves with the query.
// CHECK: %[[ROWS:.*]] = tosa.mul %{{.*}}, %{{.*}} : (tensor<1x128x1xi32>, tensor<1x128x128xi32>, tensor<1xi8>) -> tensor<1x128x128xi32>
// CHECK: %[[L:.*]] = "tosa.const"() <{values = dense<32> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
// CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
// CHECK: %[[RAW_LB:.*]] = tosa.sub %[[ROWS]], %[[L]] : (tensor<1x128x128xi32>, tensor<1x1x1xi32>) -> tensor<1x128x128xi32>
// CHECK: %[[LB:.*]] = tosa.maximum %[[RAW_LB]], %[[ZERO]] : (tensor<1x128x128xi32>, tensor<1x1x1xi32>) -> tensor<1x128x128xi32>
// CHECK: %[[TOO_OLD:.*]] = tosa.greater %[[LB]], %{{.*}} : (tensor<1x128x128xi32>, tensor<1x128x128xi32>) -> tensor<1x128x128xi1>
// CHECK: tosa.select %[[TOO_OLD]], %{{.*}}, %{{.*}} : (tensor<1x128x128xi1>, tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>

// A narrow band can leave a row with every key masked on some tile, so the
// reference normalizes with a finite max and a denominator clamped at 1.
// CHECK: %[[MAX:.*]] = tosa.reduce_max
// CHECK: %[[LOWEST:.*]] = "tosa.const"() <{values = dense<-3.40282347E+38> : tensor<1x128x1xf32>}> : () -> tensor<1x128x1xf32>
// CHECK: %[[SAFE_MAX:.*]] = tosa.maximum %[[MAX]], %[[LOWEST]]
// CHECK: tosa.sub %{{.*}}, %[[SAFE_MAX]]
// CHECK: tosa.exp
// CHECK: %[[SUM:.*]] = tosa.reduce_sum
// CHECK: %[[ONE:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x128x1xf32>}> : () -> tensor<1x128x1xf32>
// CHECK: %[[SAFE_SUM:.*]] = tosa.maximum %[[SUM]], %[[ONE]]
// CHECK: tosa.reciprocal %[[SAFE_SUM]]
// CHECK: tosa.matmul

// A band at least as wide as the key sequence keeps the whole causal triangle,
// so it degrades to plain causal masking instead of paying for the extra edge.
// DEGENERATE: rock.attention
// DEGENERATE-NOT: causalLookBack
// DEGENERATE: causal
// DEGENERATE-NOT: tosa.maximum %{{.*}} : (tensor<1x128x128xi32>, tensor<1x1x1xi32>)

// NEEDS-CAUSAL: causal_look_back requires -causal
// NOT-POSITIVE: causal_look_back must be -1 or a positive integer
// NO-PREFIX: causal_look_back is not supported with prefix_offset

// TOO-WIDE: causal_look_back (128) must be less than seq_len_k (128) unless it also covers seq_len_q (256)

// Once the band reaches the last query row, every lower bound clamps to zero and
// dropping it really is plain causal.
// WIDE-DEGENERATE: warning: causal_look_back (255) covers the whole query sequence (seq_len_q 256)
// WIDE-DEGENERATE-NOT: causalLookBack

// SQUARE-DEGENERATE: warning: causal_look_back (127) covers the whole query sequence (seq_len_q 128)
// SQUARE-DEGENERATE-NOT: causalLookBack
