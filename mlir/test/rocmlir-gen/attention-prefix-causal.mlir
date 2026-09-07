// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation attention -prefix_offset=5 --causal -num_heads_q 4 -num_heads_kv 4 -seq_len_q 8 -seq_len_k 16 -head_dim_qk 32 -head_dim_v 32 -t f32 -pv | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {rock.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queriesRaw:.*0]]: tensor<1024xf32>,
// CHECK-SAME: %[[keysRaw:.*1]]: tensor<2048xf32>,
// CHECK-SAME: %[[valuesRaw:.*2]]: tensor<2048xf32>,
// CHECK-SAME: %[[prefixOffsetRaw:.*3]]: tensor<1xi32>,
// CHECK-SAME: %[[outputRaw:.*4]]: tensor<1024xf32>)
// CHECK-SAME: attributes {rock.arch = "[[$ARCH]]", rock.kernel}
// CHECK: %[[queries:.*]] = rock.transform %[[queriesRaw]] {{.*}} : tensor<1024xf32> to tensor<4x8x32xf32>
// CHECK: %[[keys:.*]] = rock.transform %[[keysRaw]] {{.*}} : tensor<2048xf32> to tensor<4x32x16xf32>
// CHECK: %[[values:.*]] = rock.transform %[[valuesRaw]] {{.*}} : tensor<2048xf32> to tensor<4x16x32xf32>
// CHECK: %[[prefixOffset:.*]] = rock.transform %[[prefixOffsetRaw]] {{.*}} : tensor<1xi32> to tensor<1xi32>

// Verify rock.attention has causal attribute and prefixOffset
// CHECK: rock.attention
// CHECK: qk = %[[queries]] * %[[keys]]
// CHECK: prefixOffset = (%{{.*}} : tensor<4xi32>)
// CHECK: causal
// CHECK: softmax(qk) * %[[values]]
// CHECK: %[[flatOutput:.*]] = rock.transform %{{.*}} {{.*}}
// CHECK-NEXT: rock.store %[[flatOutput]] to %[[outputRaw]] by {{.*}}set
// CHECK: return

// Verify the host verification function uses prefix causal masking
// CHECK-LABEL: func.func @host_naive_attention
// CHECK: %[[qkTensor:.*]] = tosa.matmul %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : {{.*}} -> tensor<4x8x16xf32>

// Verify reshape to 4D for broadcast
// CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<4x8x16xf32>, !tosa.shape<4>) -> tensor<1x4x8x16xf32>

// Verify row range (dimension 2 = SEQ_LEN_Q = 8)
// CHECK: %[[rowRange:.*]] = "tosa.const"() <{values = {{.*}} : tensor<8xi32>}> : () -> tensor<8xi32>

// Verify col range (dimension 3 = SEQ_LEN_KV = 16)
// CHECK: %[[colRange:.*]] = "tosa.const"() <{values = {{.*}} : tensor<16xi32>}> : () -> tensor<16xi32>

// Verify offset broadcast to 4D
// CHECK: %[[offsetBroadcast:.*]] = tosa.mul %{{.*}}, %{{.*}}, %{{.*}} : (tensor<1x1x1x1xi32>, tensor<1x4x8x16xi32>, tensor<1xi8>) -> tensor<1x4x8x16xi32>

// Verify row + offset computation
// CHECK: %[[rowPlusOffset:.*]] = tosa.add %{{.*}}, %{{.*}} : {{.*}} -> tensor<1x4x8x16xi32>

// Verify prefix causal mask: col > row + offset
// CHECK: %[[mask:.*]] = tosa.greater %{{.*}}, %{{.*}} : {{.*}} -> tensor<1x4x8x16xi1>

// Verify mask application with -inf
// CHECK: %[[negInf:.*]] = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x4x8x16xf32>}> : () -> tensor<1x4x8x16xf32>
// CHECK: tosa.select %{{.*}}, %[[negInf]], %{{.*}} : {{.*}} -> tensor<1x4x8x16xf32>

// Verify reshape back to 3D
// CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<1x4x8x16xf32>, !tosa.shape<3>) -> tensor<4x8x16xf32>

// Verify softmax and final matmul
// CHECK-DAG: tosa.reduce_max %{{.*}} {{.*}}
// CHECK-DAG: tosa.exp %{{.*}}
// CHECK-DAG: tosa.reduce_sum %{{.*}}
// CHECK-DAG: tosa.reciprocal %{{.*}}
// CHECK: tosa.matmul %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : {{.*}} -> tensor<4x8x32xf32>
// CHECK: return

