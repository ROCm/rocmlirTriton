// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -last_valid_kv_index=33 -sliding_window_look_back=16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -g 2 -sliding_window_look_back=16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv | rocmlir-opt | FileCheck %s --check-prefix=DEFAULT-LAST-VALID-KV-INDEX
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -last_valid_kv_index=63 -sliding_window_look_back=63 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 | rocmlir-opt | FileCheck %s --check-prefix=MAX-LOOK-BACK
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -last_valid_kv_index=0 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 | rocmlir-opt | FileCheck %s --check-prefix=NON-SLIDING

// CHECK: module attributes {rock.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: tensor<32xf32>,
// CHECK-SAME: %[[keysRaw:.*1]]: tensor<2048xf32>,
// CHECK-SAME: %[[valuesRaw:.*2]]: tensor<2048xf32>,
// CHECK-SAME: %[[lastValidKVIndexRaw:.*3]]: tensor<1xi32>,
// CHECK-SAME: %[[outputRaw:.*4]]: tensor<32xf32>)
// CHECK-SAME: attributes {rock.arch = "[[$ARCH]]", rock.kernel{{.*}}}

// CHECK: rock.attention
// CHECK-NEXT: qk = %{{.*}} * %{{.*}}
// CHECK-NEXT: lastValidKVIndex = (%{{.*}} : tensor<1xi32>)
// CHECK-NEXT: slidingWindowLookBack = 16
// CHECK: softmax(qk) * %{{.*}}
// CHECK: return

// MAX-LOOK-BACK: rock.attention
// MAX-LOOK-BACK: lastValidKVIndex = (%{{.*}} : tensor<1xi32>)
// MAX-LOOK-BACK: slidingWindowLookBack = 63

// NON-SLIDING: rock.attention
// NON-SLIDING: lastValidKVIndex = (%{{.*}} : tensor<1xi32>)
// NON-SLIDING-NOT: slidingWindowLookBack
// NON-SLIDING: softmax(qk)

// CHECK-LABEL: func.func @host_naive_attention
// Verify KV-cache masking is applied.
// CHECK: tosa.matmul
// CHECK: tosa.greater
// CHECK: tosa.select

// Verify sliding window masking is applied in the CPU verifier:
// For inclusive index P and look-back L, lowerBound = max(0, P - L).
// Positions where col < lowerBound are then masked with -inf.
// CHECK: tosa.sub %{{.*}}, %{{.*}} : (tensor<1x1x1x64xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x64xi32>
// CHECK: tosa.maximum %{{.*}}, %{{.*}} : (tensor<1x1x1x64xi32>, tensor<1x1x1x1xi32>) -> tensor<1x1x1x64xi32>
// CHECK: tosa.greater %{{.*}}, %{{.*}} : (tensor<1x1x1x64xi32>, tensor<1x1x1x64xi32>) -> tensor<1x1x1x64xi1>
// CHECK: tosa.select %{{.*}}, %{{.*}}, %{{.*}} : (tensor<1x1x1x64xi1>, tensor<1x1x1x64xf32>, tensor<1x1x1x64xf32>) -> tensor<1x1x1x64xf32>

// Verify softmax follows.
// CHECK-DAG: tosa.reduce_max
// CHECK-DAG: tosa.exp
// CHECK-DAG: tosa.reduce_sum
// CHECK-DAG: tosa.reciprocal
// CHECK: tosa.matmul
// CHECK: return

// When last_valid_kv_index is omitted, use the last valid key position for every
// group so tuning-problem keys can be reconstructed by tuningRunner.
// DEFAULT-LAST-VALID-KV-INDEX-LABEL: func.func @rock_attention(
// DEFAULT-LAST-VALID-KV-INDEX-SAME: tensor<2xi32>
// DEFAULT-LAST-VALID-KV-INDEX: lastValidKVIndex = (%{{.*}} : tensor<2xi32>)
// DEFAULT-LAST-VALID-KV-INDEX: slidingWindowLookBack = 16
// DEFAULT-LAST-VALID-KV-INDEX-COUNT-2: arith.constant 63 : i32
