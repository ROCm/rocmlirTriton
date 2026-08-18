// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -last_valid_kv_index=33 -sliding_window_look_back=16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -g 2 -sliding_window_look_back=16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv | rocmlir-opt | FileCheck %s --check-prefix=DEFAULT-CURRENT-SEQ-LEN

// CHECK: module attributes {rock.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: tensor<32xf32>,
// CHECK-SAME: %[[keysRaw:.*1]]: tensor<2048xf32>,
// CHECK-SAME: %[[valuesRaw:.*2]]: tensor<2048xf32>,
// CHECK-SAME: %[[lastValidKVIndexRaw:.*3]]: tensor<1xi32>,
// CHECK-SAME: %[[outputRaw:.*4]]: tensor<32xf32>)
// CHECK-SAME: attributes {rock.arch = "[[$ARCH]]", rock.kernel}

// CHECK: rock.attention
// CHECK-NEXT: qk = %{{.*}} * %{{.*}}
// CHECK-NEXT: lastValidKVIndex = (%{{.*}} : tensor<1xi32>)
// CHECK-NEXT: slidingWindowLookBack = 16
// CHECK: softmax(qk) * %{{.*}}
// CHECK: return

// CHECK-LABEL: func.func @host_naive_attention
// Verify KV-cache masking is applied.
// CHECK: tosa.matmul
// CHECK: tosa.greater
// CHECK: tosa.select

// Verify sliding window masking is applied in the CPU verifier:
// The sliding window masking computes lowerBound = max(0, lastValidKVIndex - windowSize),
// then masks positions where col < lowerBound with -inf.
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
// DEFAULT-CURRENT-SEQ-LEN-LABEL: func.func @rock_attention(
// DEFAULT-CURRENT-SEQ-LEN-SAME: tensor<2xi32>
// DEFAULT-CURRENT-SEQ-LEN: lastValidKVIndex = (%{{.*}} : tensor<2xi32>)
// DEFAULT-CURRENT-SEQ-LEN: slidingWindowLookBack = 16
// DEFAULT-CURRENT-SEQ-LEN-COUNT-2: arith.constant 63 : i32
