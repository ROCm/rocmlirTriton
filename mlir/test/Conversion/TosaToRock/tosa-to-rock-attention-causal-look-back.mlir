// Check that an additive mask whose kept region is the band
// max(0, row - L) <= col <= row is recognised as causal masking plus a
// causalLookBack width, instead of being rejected for not being the full
// causal triangle.

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o -| FileCheck %s

// The 8x8 mask below keeps max(0, row - 3) <= col <= row, so L = 3.
// CHECK-LABEL: func @mlir_causal_attention_look_back
// CHECK: rock.attention
// CHECK: causalLookBack = 3
// CHECK: causal
func.func @mlir_causal_attention_look_back(%arg0: tensor<1024xf16>, %arg1: tensor<7168xf16>, %arg2: tensor<7168xf16>) -> tensor<7168xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %2 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x14x8x8xf32>}> : () -> tensor<1x14x8x8xf32>
  %7 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x14x8x8xf16>}> : () -> tensor<1x14x8x8xf16>
  %8 = "tosa.const"() <{values = dense<3.535160e-01> : tensor<1x14x64x8xf16>}> : () -> tensor<1x14x64x8xf16>
  %10 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %11 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x7x64x8xf16>}> : () -> tensor<1x2x7x64x8xf16>
  %13 = "tosa.const"() <{values = dense<[[[[-0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00], [0xFC00, 0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00], [0xFC00, 0xFC00, 0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00], [0xFC00, 0xFC00, 0xFC00, 0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00]]]]> : tensor<1x1x8x8xf16>}> : () -> tensor<1x1x8x8xf16>
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 8, 14, 64] : tensor<7168xf16> into tensor<1x8x14x64xf16>
  %15 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x8x14x64xf16>) -> tensor<1x14x8x64xf16>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 2, 1, 8, 64] : tensor<1024xf16> into tensor<1x2x1x8x64xf16>
  %16 = tosa.transpose %expanded_0 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<1x2x1x8x64xf16>) -> tensor<1x2x1x64x8xf16>
  %17 = tosa.mul %16, %11, %10 : (tensor<1x2x1x64x8xf16>, tensor<1x2x7x64x8xf16>, tensor<1xi8>) -> tensor<1x2x7x64x8xf16>
  %collapsed = tensor.collapse_shape %17 [[0], [1, 2], [3], [4]] : tensor<1x2x7x64x8xf16> into tensor<1x14x64x8xf16>
  %18 = tosa.mul %collapsed, %8, %10 : (tensor<1x14x64x8xf16>, tensor<1x14x64x8xf16>, tensor<1xi8>) -> tensor<1x14x64x8xf16>
  %19 = tosa.mul %13, %7, %10 : (tensor<1x1x8x8xf16>, tensor<1x14x8x8xf16>, tensor<1xi8>) -> tensor<1x14x8x8xf16>
  %collapsed_1 = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<1x14x8x64xf16> into tensor<14x8x64xf16>
  %collapsed_2 = tensor.collapse_shape %18 [[0, 1], [2], [3]] : tensor<1x14x64x8xf16> into tensor<14x64x8xf16>
  %20 = tosa.matmul %collapsed_1, %collapsed_2, %4, %4 {acc_type = f32} : (tensor<14x8x64xf16>, tensor<14x64x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x8x8xf16>
  %expanded_3 = tensor.expand_shape %20 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf16> into tensor<1x14x8x8xf16>
  %21 = tosa.add %expanded_3, %19 : (tensor<1x14x8x8xf16>, tensor<1x14x8x8xf16>) -> tensor<1x14x8x8xf16>
  %22 = tosa.cast %21 : (tensor<1x14x8x8xf16>) -> tensor<1x14x8x8xf32>
  %23 = tosa.reduce_max %22 {axis = 3 : i32} : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x1xf32>
  %24 = tosa.mul %23, %2, %10 : (tensor<1x14x8x1xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %25 = tosa.sub %22, %24 : (tensor<1x14x8x8xf32>, tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %26 = tosa.exp %25 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %27 = tosa.reduce_sum %26 {axis = 3 : i32} : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x1xf32>
  %28 = tosa.mul %27, %2, %10 : (tensor<1x14x8x1xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %29 = tosa.reciprocal %28 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %30 = tosa.mul %26, %29, %10 : (tensor<1x14x8x8xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %31 = tosa.cast %30 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf16>
  %collapsed_4 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x14x8x8xf16> into tensor<14x8x8xf16>
  %expanded_5 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [14, 8, 64] : tensor<7168xf16> into tensor<14x8x64xf16>
  %32 = tosa.matmul %collapsed_4, %expanded_5, %4, %4 {acc_type = f32} : (tensor<14x8x8xf16>, tensor<14x8x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x8x64xf16>
  %collapsed_6 = tensor.collapse_shape %32 [[0, 1, 2]] : tensor<14x8x64xf16> into tensor<7168xf16>
  return %collapsed_6 : tensor<7168xf16>
}

// Same shape, but the band has a hole at (7, 5): not a band and not the full
// triangle, so nothing causal is recognised at all.
// CHECK-LABEL: func @mlir_attention_ragged_mask_not_causal
// CHECK: rock.attention
// CHECK-NOT: causal
func.func @mlir_attention_ragged_mask_not_causal(%arg0: tensor<1024xf16>, %arg1: tensor<7168xf16>, %arg2: tensor<7168xf16>) -> tensor<7168xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %4 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %2 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x14x8x8xf32>}> : () -> tensor<1x14x8x8xf32>
  %7 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x14x8x8xf16>}> : () -> tensor<1x14x8x8xf16>
  %8 = "tosa.const"() <{values = dense<3.535160e-01> : tensor<1x14x64x8xf16>}> : () -> tensor<1x14x64x8xf16>
  %10 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %11 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x7x64x8xf16>}> : () -> tensor<1x2x7x64x8xf16>
  %13 = "tosa.const"() <{values = dense<[[[[-0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00], [0xFC00, 0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00], [0xFC00, 0xFC00, 0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00], [0xFC00, 0xFC00, 0xFC00, 0xFC00, -0.000000e+00, 0xFC00, -0.000000e+00, -0.000000e+00]]]]> : tensor<1x1x8x8xf16>}> : () -> tensor<1x1x8x8xf16>
  %expanded = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 8, 14, 64] : tensor<7168xf16> into tensor<1x8x14x64xf16>
  %15 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x8x14x64xf16>) -> tensor<1x14x8x64xf16>
  %expanded_0 = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [1, 2, 1, 8, 64] : tensor<1024xf16> into tensor<1x2x1x8x64xf16>
  %16 = tosa.transpose %expanded_0 {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<1x2x1x8x64xf16>) -> tensor<1x2x1x64x8xf16>
  %17 = tosa.mul %16, %11, %10 : (tensor<1x2x1x64x8xf16>, tensor<1x2x7x64x8xf16>, tensor<1xi8>) -> tensor<1x2x7x64x8xf16>
  %collapsed = tensor.collapse_shape %17 [[0], [1, 2], [3], [4]] : tensor<1x2x7x64x8xf16> into tensor<1x14x64x8xf16>
  %18 = tosa.mul %collapsed, %8, %10 : (tensor<1x14x64x8xf16>, tensor<1x14x64x8xf16>, tensor<1xi8>) -> tensor<1x14x64x8xf16>
  %19 = tosa.mul %13, %7, %10 : (tensor<1x1x8x8xf16>, tensor<1x14x8x8xf16>, tensor<1xi8>) -> tensor<1x14x8x8xf16>
  %collapsed_1 = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<1x14x8x64xf16> into tensor<14x8x64xf16>
  %collapsed_2 = tensor.collapse_shape %18 [[0, 1], [2], [3]] : tensor<1x14x64x8xf16> into tensor<14x64x8xf16>
  %20 = tosa.matmul %collapsed_1, %collapsed_2, %4, %4 {acc_type = f32} : (tensor<14x8x64xf16>, tensor<14x64x8xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x8x8xf16>
  %expanded_3 = tensor.expand_shape %20 [[0, 1], [2], [3]] output_shape [1, 14, 8, 8] : tensor<14x8x8xf16> into tensor<1x14x8x8xf16>
  %21 = tosa.add %expanded_3, %19 : (tensor<1x14x8x8xf16>, tensor<1x14x8x8xf16>) -> tensor<1x14x8x8xf16>
  %22 = tosa.cast %21 : (tensor<1x14x8x8xf16>) -> tensor<1x14x8x8xf32>
  %23 = tosa.reduce_max %22 {axis = 3 : i32} : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x1xf32>
  %24 = tosa.mul %23, %2, %10 : (tensor<1x14x8x1xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %25 = tosa.sub %22, %24 : (tensor<1x14x8x8xf32>, tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %26 = tosa.exp %25 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %27 = tosa.reduce_sum %26 {axis = 3 : i32} : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x1xf32>
  %28 = tosa.mul %27, %2, %10 : (tensor<1x14x8x1xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %29 = tosa.reciprocal %28 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf32>
  %30 = tosa.mul %26, %29, %10 : (tensor<1x14x8x8xf32>, tensor<1x14x8x8xf32>, tensor<1xi8>) -> tensor<1x14x8x8xf32>
  %31 = tosa.cast %30 : (tensor<1x14x8x8xf32>) -> tensor<1x14x8x8xf16>
  %collapsed_4 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x14x8x8xf16> into tensor<14x8x8xf16>
  %expanded_5 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [14, 8, 64] : tensor<7168xf16> into tensor<14x8x64xf16>
  %32 = tosa.matmul %collapsed_4, %expanded_5, %4, %4 {acc_type = f32} : (tensor<14x8x8xf16>, tensor<14x8x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x8x64xf16>
  %collapsed_6 = tensor.collapse_shape %32 [[0, 1, 2]] : tensor<14x8x64xf16> into tensor<7168xf16>
  return %collapsed_6 : tensor<7168xf16>
}
