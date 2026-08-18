// Verify the host split-KV combine keeps all-masked reductions defined.
//
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -current_seq_len=33 -return_lse -split_kv 8 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 1024 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK-LABEL: func.func @rock_attention_gpu

// LSE re-normalization across the splitKV axis (axis = 1).
// CHECK: %[[MAX:.+]] = tosa.reduce_max %{{.+}} {axis = 1 : i32} : (tensor<4x8x1x1xf32>) -> tensor<4x1x1x1xf32>
// CHECK: %[[LOWEST:.+]] = "tosa.const"() <{values = dense<-3.40282347E+38> : tensor<4x1x1x1xf32>}> : () -> tensor<4x1x1x1xf32>
// CHECK: %[[SAFE_MAX:.+]] = tosa.maximum %[[MAX]], %[[LOWEST]] : (tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>) -> tensor<4x1x1x1xf32>
// CHECK: tosa.sub %{{.+}}, %[[SAFE_MAX]] : (tensor<4x8x1x1xf32>, tensor<4x1x1x1xf32>) -> tensor<4x8x1x1xf32>
// CHECK: tosa.exp %{{.+}} : (tensor<4x8x1x1xf32>) -> tensor<4x8x1x1xf32>
// CHECK: %[[SUM:.+]] = tosa.reduce_sum %{{.+}} {axis = 1 : i32} : (tensor<4x8x1x1xf32>) -> tensor<4x1x1x1xf32>
// CHECK: %[[ONE:.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4x1x1x1xf32>}> : () -> tensor<4x1x1x1xf32>
// CHECK: %[[SAFE_SUM:.+]] = tosa.maximum %[[SUM]], %[[ONE]] : (tensor<4x1x1x1xf32>, tensor<4x1x1x1xf32>) -> tensor<4x1x1x1xf32>
// CHECK: tosa.reciprocal %[[SAFE_SUM]] : (tensor<4x1x1x1xf32>) -> tensor<4x1x1x1xf32>
