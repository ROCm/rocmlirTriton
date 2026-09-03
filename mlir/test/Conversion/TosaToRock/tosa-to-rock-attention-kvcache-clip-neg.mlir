// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -split-input-file -verify-diagnostics -o -| FileCheck %s

// This is intentionally a negative test for KV-cache specialization. A
// per-head, non-splat lower bound cannot be folded into the scalar clip bounds
// on the lastValidKVIndex operand, so the pattern must retain %clip_min in the
// elementwise region, where it exercises the dense-constant input path.
// CHECK-LABEL: func @kvcache_nonsplat_clip_bound
// CHECK: %[[CLIP_MIN:.*]] = "tosa.const"() <{values = dense<{{.*0.*1.*}}> : tensor<1x2x1x1xi32>}>
// CHECK: rock.attention
// The retained dense non-splat constant is an elementwise input so downstream
// Rock lowering can tile-load it from compiler-owned GPU storage.
// CHECK: qk = elementwise otherIns(%{{.*}}, %{{.*}}, %[[CLIP_MIN]]
// CHECK-NOT: lastValidKVIndex
// CHECK: tosa.maximum
// CHECK: tosa.minimum
// CHECK: tosa.greater
// CHECK: tosa.select
// CHECK-NOT: lastValidKVIndex
func.func @kvcache_nonsplat_clip_bound(%arg0: tensor<2xi32>, %arg1: tensor<12xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %softmax_ones = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x1x8xf32>}> : () -> tensor<1x2x1x8xf32>
  %broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<1x2x1x8xi32>}> : () -> tensor<1x2x1x8xi32>
  %scale = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %neg_inf = "tosa.const"() <{values = dense<0xFC00> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %zero = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %columns_base = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %clip_min = "tosa.const"() <{values = dense<[[[[0]], [[1]]]]> : tensor<1x2x1x1xi32>}> : () -> tensor<1x2x1x1xi32>
  %clip_max = "tosa.const"() <{values = dense<7> : tensor<1x2x1x1xi32>}> : () -> tensor<1x2x1x1xi32>
  %queries_expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 6, 1, 2] : tensor<12xf16> into tensor<1x6x1x2xf16>
  %queries = tensor.extract_slice %queries_expanded[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %keys_expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 2, 8, 2] : tensor<32xf16> into tensor<1x2x8x2xf16>
  %keys = tosa.transpose %keys_expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %queries_collapsed = tensor.collapse_shape %queries [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %keys_collapsed = tensor.collapse_shape %keys [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %scores = tosa.matmul %queries_collapsed, %keys_collapsed, %zero, %zero {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %scores_expanded = tensor.expand_shape %scores [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %scaled_scores = tosa.mul %scores_expanded, %scale, %shift : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %last_valid_kv_index = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 2, 1, 1] : tensor<2xi32> into tensor<1x2x1x1xi32>
  %clamped = tosa.maximum %last_valid_kv_index, %clip_min : (tensor<1x2x1x1xi32>, tensor<1x2x1x1xi32>) -> tensor<1x2x1x1xi32>
  %clipped = tosa.minimum %clamped, %clip_max : (tensor<1x2x1x1xi32>, tensor<1x2x1x1xi32>) -> tensor<1x2x1x1xi32>
  %columns = tosa.mul %columns_base, %broadcast_ones, %shift : (tensor<1x1x1x8xi32>, tensor<1x2x1x8xi32>, tensor<1xi8>) -> tensor<1x2x1x8xi32>
  %last_valid_kv_index_broadcast = tosa.mul %clipped, %broadcast_ones, %shift : (tensor<1x2x1x1xi32>, tensor<1x2x1x8xi32>, tensor<1xi8>) -> tensor<1x2x1x8xi32>
  %mask = tosa.greater %columns, %last_valid_kv_index_broadcast : (tensor<1x2x1x8xi32>, tensor<1x2x1x8xi32>) -> tensor<1x2x1x8xi1>
  %masked_scores = tosa.select %mask, %neg_inf, %scaled_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %scores_f32 = tosa.cast %masked_scores : (tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf32>
  %max = tosa.reduce_max %scores_f32 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %max_broadcast = tosa.mul %max, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %normalized = tosa.sub %scores_f32, %max_broadcast : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %exp = tosa.exp %normalized : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %sum_broadcast = tosa.mul %sum, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %reciprocal = tosa.reciprocal %sum_broadcast : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %softmax = tosa.mul %exp, %reciprocal, %shift : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %softmax_f16 = tosa.cast %softmax : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf16>
  %softmax_collapsed = tensor.collapse_shape %softmax_f16 [[0, 1], [2], [3]] : tensor<1x2x1x8xf16> into tensor<2x1x8xf16>
  %values = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [2, 8, 2] : tensor<32xf16> into tensor<2x8x2xf16>
  %attention = tosa.matmul %softmax_collapsed, %values, %zero, %zero {acc_type = f32} : (tensor<2x1x8xf16>, tensor<2x8x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x2xf16>
  %result = tensor.collapse_shape %attention [[0, 1, 2]] : tensor<2x1x2xf16> into tensor<4xf16>
  return %result : tensor<4xf16>
}

// -----

// Rock uses unsigned comparisons for lastValidKVIndex masking, so a clip that
// permits a negative effective index must remain in the signed TOSA
// elementwise computation.
// CHECK-LABEL: func @kvcache_negative_clip_bound
// CHECK: rock.attention
// CHECK-NOT: lastValidKVIndex
// CHECK: tosa.maximum
// CHECK: tosa.minimum
// CHECK: tosa.greater
// CHECK: tosa.select
// CHECK-NOT: lastValidKVIndex
func.func @kvcache_negative_clip_bound(%arg0: tensor<2xi32>, %arg1: tensor<12xf16>, %arg2: tensor<32xf16>, %arg3: tensor<32xf16>) -> tensor<4xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %softmax_ones = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x1x8xf32>}> : () -> tensor<1x2x1x8xf32>
  %broadcast_ones = "tosa.const"() <{values = dense<1> : tensor<1x2x1x8xi32>}> : () -> tensor<1x2x1x8xi32>
  %scale = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %neg_inf = "tosa.const"() <{values = dense<0xFC00> : tensor<1x2x1x8xf16>}> : () -> tensor<1x2x1x8xf16>
  %zero = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %columns_base = arith.constant dense<[[[[0, 1, 2, 3, 4, 5, 6, 7]]]]> : tensor<1x1x1x8xi32>
  %clip_min = "tosa.const"() <{values = dense<-1> : tensor<1x2x1x1xi32>}> : () -> tensor<1x2x1x1xi32>
  %clip_max = "tosa.const"() <{values = dense<7> : tensor<1x2x1x1xi32>}> : () -> tensor<1x2x1x1xi32>
  %queries_expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 6, 1, 2] : tensor<12xf16> into tensor<1x6x1x2xf16>
  %queries = tensor.extract_slice %queries_expanded[0, 0, 0, 0] [1, 2, 1, 2] [1, 1, 1, 1] : tensor<1x6x1x2xf16> to tensor<1x2x1x2xf16>
  %keys_expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 2, 8, 2] : tensor<32xf16> into tensor<1x2x8x2xf16>
  %keys = tosa.transpose %keys_expanded {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x8x2xf16>) -> tensor<1x2x2x8xf16>
  %queries_collapsed = tensor.collapse_shape %queries [[0, 1], [2], [3]] : tensor<1x2x1x2xf16> into tensor<2x1x2xf16>
  %keys_collapsed = tensor.collapse_shape %keys [[0, 1], [2], [3]] : tensor<1x2x2x8xf16> into tensor<2x2x8xf16>
  %scores = tosa.matmul %queries_collapsed, %keys_collapsed, %zero, %zero {acc_type = f32} : (tensor<2x1x2xf16>, tensor<2x2x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x8xf16>
  %scores_expanded = tensor.expand_shape %scores [[0, 1], [2], [3]] output_shape [1, 2, 1, 8] : tensor<2x1x8xf16> into tensor<1x2x1x8xf16>
  %scaled_scores = tosa.mul %scores_expanded, %scale, %shift : (tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>, tensor<1xi8>) -> tensor<1x2x1x8xf16>
  %last_valid_kv_index = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 2, 1, 1] : tensor<2xi32> into tensor<1x2x1x1xi32>
  %clamped = tosa.maximum %last_valid_kv_index, %clip_min : (tensor<1x2x1x1xi32>, tensor<1x2x1x1xi32>) -> tensor<1x2x1x1xi32>
  %clipped = tosa.minimum %clamped, %clip_max : (tensor<1x2x1x1xi32>, tensor<1x2x1x1xi32>) -> tensor<1x2x1x1xi32>
  %columns = tosa.mul %columns_base, %broadcast_ones, %shift : (tensor<1x1x1x8xi32>, tensor<1x2x1x8xi32>, tensor<1xi8>) -> tensor<1x2x1x8xi32>
  %last_valid_kv_index_broadcast = tosa.mul %clipped, %broadcast_ones, %shift : (tensor<1x2x1x1xi32>, tensor<1x2x1x8xi32>, tensor<1xi8>) -> tensor<1x2x1x8xi32>
  %mask = tosa.greater %columns, %last_valid_kv_index_broadcast : (tensor<1x2x1x8xi32>, tensor<1x2x1x8xi32>) -> tensor<1x2x1x8xi1>
  %masked_scores = tosa.select %mask, %neg_inf, %scaled_scores : (tensor<1x2x1x8xi1>, tensor<1x2x1x8xf16>, tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf16>
  %scores_f32 = tosa.cast %masked_scores : (tensor<1x2x1x8xf16>) -> tensor<1x2x1x8xf32>
  %max = tosa.reduce_max %scores_f32 {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %max_broadcast = tosa.mul %max, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %normalized = tosa.sub %scores_f32, %max_broadcast : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %exp = tosa.exp %normalized : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x1xf32>
  %sum_broadcast = tosa.mul %sum, %softmax_ones, %shift : (tensor<1x2x1x1xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %reciprocal = tosa.reciprocal %sum_broadcast : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf32>
  %softmax = tosa.mul %exp, %reciprocal, %shift : (tensor<1x2x1x8xf32>, tensor<1x2x1x8xf32>, tensor<1xi8>) -> tensor<1x2x1x8xf32>
  %softmax_f16 = tosa.cast %softmax : (tensor<1x2x1x8xf32>) -> tensor<1x2x1x8xf16>
  %softmax_collapsed = tensor.collapse_shape %softmax_f16 [[0, 1], [2], [3]] : tensor<1x2x1x8xf16> into tensor<2x1x8xf16>
  %values = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [2, 8, 2] : tensor<32xf16> into tensor<2x8x2xf16>
  %attention = tosa.matmul %softmax_collapsed, %values, %zero, %zero {acc_type = f32} : (tensor<2x1x8xf16>, tensor<2x8x2xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x1x2xf16>
  %result = tensor.collapse_shape %attention [[0, 1, 2]] : tensor<2x1x2xf16> into tensor<4xf16>
  return %result : tensor<4xf16>
}
