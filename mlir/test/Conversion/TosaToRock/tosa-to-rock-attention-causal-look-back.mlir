// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Check that a mask whose kept region is the band
// max(0, row - L) <= col <= row is recognised as causal masking plus a
// causalLookBack width, instead of being rejected for not being the full
// causal triangle.

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o -| FileCheck %s

// The 8x8 mask below keeps max(0, row - 3) <= col <= row, so L = 3.
// `causal` is checked with CHECK-NEXT so it cannot be satisfied by the
// `causal` prefix of the `causalLookBack` line above it.
// CHECK-LABEL: func @mlir_causal_attention_look_back
// CHECK: rock.attention
// CHECK: causalLookBack = 3
// CHECK-NEXT: causal
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

// Same band, but reached through tosa.select rather than the additive tosa.add
// above, so the mask is an i8 constant with 1s inside the band and the
// masking value is -10000.0 on the false branch. The 4x4 mask keeps
// max(0, row - 2) <= col <= row, so L = 2.
// CHECK-LABEL: func @mlir_causal_attention_look_back_select
// CHECK: rock.attention
// CHECK-NOT: lastValidKVIndex =
// CHECK: causalLookBack = 2
// CHECK-NEXT: causal
func.func @mlir_causal_attention_look_back_select(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %neg10k = "tosa.const"() <{values = dense<-1.000000e+04> : tensor<1x2x4x4xf32>}> : () -> tensor<1x2x4x4xf32>
  %scale = "tosa.const"() <{values = dense<1.250000e-01> : tensor<1x2x4x4xf32>}> : () -> tensor<1x2x4x4xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %mask_i8 = "tosa.const"() <{values = dense<"0x01000000010100000101010000010101"> : tensor<1x1x4x4xi8>}> : () -> tensor<1x1x4x4xi8>
  %ones_i8 = "tosa.const"() <{values = dense<1> : tensor<1x2x4x4xi8>}> : () -> tensor<1x2x4x4xi8>
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %q_4d = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 2, 4, 8] : tensor<64xf32> into tensor<1x2x4x8xf32>
  %k_4d = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 2, 4, 8] : tensor<64xf32> into tensor<1x2x4x8xf32>
  %k_t = tosa.transpose %k_4d {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x4x8xf32>) -> tensor<1x2x8x4xf32>
  %q_3d = tensor.collapse_shape %q_4d [[0, 1], [2], [3]] : tensor<1x2x4x8xf32> into tensor<2x4x8xf32>
  %k_3d = tensor.collapse_shape %k_t [[0, 1], [2], [3]] : tensor<1x2x8x4xf32> into tensor<2x8x4xf32>
  %qk = tosa.matmul %q_3d, %k_3d, %zp, %zp {acc_type = f32} : (tensor<2x4x8xf32>, tensor<2x8x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x4xf32>
  %qk_4d = tensor.expand_shape %qk [[0, 1], [2], [3]] output_shape [1, 2, 4, 4] : tensor<2x4x4xf32> into tensor<1x2x4x4xf32>
  %scaled = tosa.mul %qk_4d, %scale, %shift : (tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>, tensor<1xi8>) -> tensor<1x2x4x4xf32>
  %mask_bcast = tosa.mul %mask_i8, %ones_i8, %shift : (tensor<1x1x4x4xi8>, tensor<1x2x4x4xi8>, tensor<1xi8>) -> tensor<1x2x4x4xi8>
  %mask_i1 = tosa.cast %mask_bcast : (tensor<1x2x4x4xi8>) -> tensor<1x2x4x4xi1>
  %masked = tosa.select %mask_i1, %scaled, %neg10k : (tensor<1x2x4x4xi1>, tensor<1x2x4x4xf32>, tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %max = tosa.reduce_max %masked {axis = 3 : i32} : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x1xf32>
  %sub = tosa.sub %masked, %max : (tensor<1x2x4x4xf32>, tensor<1x2x4x1xf32>) -> tensor<1x2x4x4xf32>
  %exp = tosa.exp %sub : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x4xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x2x4x4xf32>) -> tensor<1x2x4x1xf32>
  %recip = tosa.reciprocal %sum : (tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  %softmax = tosa.mul %exp, %recip, %shift : (tensor<1x2x4x4xf32>, tensor<1x2x4x1xf32>, tensor<1xi8>) -> tensor<1x2x4x4xf32>
  %sm_3d = tensor.collapse_shape %softmax [[0, 1], [2], [3]] : tensor<1x2x4x4xf32> into tensor<2x4x4xf32>
  %v_3d = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [2, 4, 8] : tensor<64xf32> into tensor<2x4x8xf32>
  %attn = tosa.matmul %sm_3d, %v_3d, %zp, %zp {acc_type = f32} : (tensor<2x4x4xf32>, tensor<2x4x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4x8xf32>
  %out = tensor.collapse_shape %attn [[0, 1, 2]] : tensor<2x4x8xf32> into tensor<64xf32>
  return %out : tensor<64xf32>
}

// A genuine band that the rock.attention attribute cannot express. There are 8
// queries but only 4 keys, and the mask keeps max(0, row - 4) <= col <= row,
// so L = 4. The lower edge is measured from the query row while the key length
// only clips the upper edge, so the last row keeps only key 3 and the inferred
// width comes out equal to the key length -- and verifyCausalLookBack rejects
// every width >= the key length. Detection has to fail so the mask stays a
// pre-softmax fusion that the kernel applies elementwise, rather than
// producing a rock.attention the verifier would reject.
// CHECK-LABEL: func @mlir_attention_look_back_too_wide_for_keys
// CHECK: rock.attention
// CHECK-NOT: causal
func.func @mlir_attention_look_back_too_wide_for_keys(%arg0: tensor<128xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<128xf32> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  %neg10k = "tosa.const"() <{values = dense<-1.000000e+04> : tensor<1x2x8x4xf32>}> : () -> tensor<1x2x8x4xf32>
  %scale = "tosa.const"() <{values = dense<1.250000e-01> : tensor<1x2x8x4xf32>}> : () -> tensor<1x2x8x4xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %mask_i8 = "tosa.const"() <{values = dense<"0x0100000001010000010101000101010101010101000101010000010100000001"> : tensor<1x1x8x4xi8>}> : () -> tensor<1x1x8x4xi8>
  %ones_i8 = "tosa.const"() <{values = dense<1> : tensor<1x2x8x4xi8>}> : () -> tensor<1x2x8x4xi8>
  %zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %q_4d = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 2, 8, 8] : tensor<128xf32> into tensor<1x2x8x8xf32>
  %k_4d = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 2, 4, 8] : tensor<64xf32> into tensor<1x2x4x8xf32>
  %k_t = tosa.transpose %k_4d {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x2x4x8xf32>) -> tensor<1x2x8x4xf32>
  %q_3d = tensor.collapse_shape %q_4d [[0, 1], [2], [3]] : tensor<1x2x8x8xf32> into tensor<2x8x8xf32>
  %k_3d = tensor.collapse_shape %k_t [[0, 1], [2], [3]] : tensor<1x2x8x4xf32> into tensor<2x8x4xf32>
  %qk = tosa.matmul %q_3d, %k_3d, %zp, %zp {acc_type = f32} : (tensor<2x8x8xf32>, tensor<2x8x4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x8x4xf32>
  %qk_4d = tensor.expand_shape %qk [[0, 1], [2], [3]] output_shape [1, 2, 8, 4] : tensor<2x8x4xf32> into tensor<1x2x8x4xf32>
  %scaled = tosa.mul %qk_4d, %scale, %shift : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>, tensor<1xi8>) -> tensor<1x2x8x4xf32>
  %mask_bcast = tosa.mul %mask_i8, %ones_i8, %shift : (tensor<1x1x8x4xi8>, tensor<1x2x8x4xi8>, tensor<1xi8>) -> tensor<1x2x8x4xi8>
  %mask_i1 = tosa.cast %mask_bcast : (tensor<1x2x8x4xi8>) -> tensor<1x2x8x4xi1>
  %masked = tosa.select %mask_i1, %scaled, %neg10k : (tensor<1x2x8x4xi1>, tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  %max = tosa.reduce_max %masked {axis = 3 : i32} : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x1xf32>
  %sub = tosa.sub %masked, %max : (tensor<1x2x8x4xf32>, tensor<1x2x8x1xf32>) -> tensor<1x2x8x4xf32>
  %exp = tosa.exp %sub : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  %sum = tosa.reduce_sum %exp {axis = 3 : i32} : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x1xf32>
  %recip = tosa.reciprocal %sum : (tensor<1x2x8x1xf32>) -> tensor<1x2x8x1xf32>
  %softmax = tosa.mul %exp, %recip, %shift : (tensor<1x2x8x4xf32>, tensor<1x2x8x1xf32>, tensor<1xi8>) -> tensor<1x2x8x4xf32>
  %sm_3d = tensor.collapse_shape %softmax [[0, 1], [2], [3]] : tensor<1x2x8x4xf32> into tensor<2x8x4xf32>
  %v_3d = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [2, 4, 8] : tensor<64xf32> into tensor<2x4x8xf32>
  %attn = tosa.matmul %sm_3d, %v_3d, %zp, %zp {acc_type = f32} : (tensor<2x8x4xf32>, tensor<2x4x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x8x8xf32>
  %out = tensor.collapse_shape %attn [[0, 1, 2]] : tensor<2x8x8xf32> into tensor<128xf32>
  return %out : tensor<128xf32>
}

// A banded additive mask on top of a prefix-causal graph. Prefix causal moves
// the upper bound off the diagonal, so the band's edges are no longer the ones
// causalLookBack describes and the op verifier rejects the pair outright (see
// attention_causal_look_back_with_prefix_offset in
// test/Dialect/Rock/ops_error.mlir). The conversion must therefore leave the
// band in the pre-softmax elementwise region instead of promoting it: expect
// prefixOffset and causal on the op, and no causalLookBack anywhere.
// The two CHECK-NOTs plus the CHECK-NEXT leave no unchecked gap: the first
// covers the op header up to prefixOffset, the CHECK-NEXT pins causal to the
// very next line, and the second covers everything after it.
// CHECK-LABEL: func @mlir_attention_prefix_causal_with_band
// CHECK: rock.attention
// CHECK-NOT: causalLookBack
// CHECK: prefixOffset = (%{{.*}} : tensor<14xi32>)
// CHECK-NEXT: causal
// CHECK-NOT: causalLookBack
func.func @mlir_attention_prefix_causal_with_band(%arg0: tensor<1xi32>, %arg1: tensor<4608xf16>, %arg2: tensor<2048xf16>, %arg3: tensor<14336xf16>) -> tensor<3584xf16> attributes {rock.kernel, rock.arch = "##TOKEN_ARCH##"} {
  // The 4x16 literal keeps max(0, row - 2) <= col <= row, so L = 2.
  %band = "tosa.const"() <{values = dense<[[[[-0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [-0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00], [0xFC00, -0.000000e+00, -0.000000e+00, -0.000000e+00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00]]]]> : tensor<1x1x4x16xf16>}> : () -> tensor<1x1x4x16xf16>
  %band_ones = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x14x4x16xf16>}> : () -> tensor<1x14x4x16xf16>
  %0 = "tosa.const"() <{values = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]> : tensor<1x16xi32>}> : () -> tensor<1x16xi32>
  %4 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x14x4x16xf32>}> : () -> tensor<1x14x4x16xf32>
  %5 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x14x4x16xf16>}> : () -> tensor<1x14x4x16xf16>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %10 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x2x7x64x16xf16>}> : () -> tensor<1x2x7x64x16xf16>
  %12 = "tosa.const"() <{values = dense<1> : tensor<1x14x4x16xi8>}> : () -> tensor<1x14x4x16xi8>
  %14 = "tosa.const"() <{values = dense<1> : tensor<4x1xi32>}> : () -> tensor<4x1xi32>
  %15 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<1x14x4x16xf16>}> : () -> tensor<1x14x4x16xf16>
  %16 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %17 = "tosa.const"() <{values = dense<1> : tensor<4x16xi32>}> : () -> tensor<4x16xi32>
  %18 = "tosa.const"() <{values = dense<[[0], [1], [2], [3]]> : tensor<4x1xi32>}> : () -> tensor<4x1xi32>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3, 4]] output_shape [1, 2, 1, 16, 64] : tensor<2048xf16> into tensor<1x2x1x16x64xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 4, 18, 64] : tensor<4608xf16> into tensor<1x4x18x64xf16>
  %22 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x4x18x64xf16>) -> tensor<1x18x4x64xf16>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1]] output_shape [1, 1] : tensor<1xi32> into tensor<1x1xi32>
  %23 = tosa.mul %0, %17, %16 : (tensor<1x16xi32>, tensor<4x16xi32>, tensor<1xi8>) -> tensor<4x16xi32>
  %24 = tosa.mul %expanded_1, %14, %16 : (tensor<1x1xi32>, tensor<4x1xi32>, tensor<1xi8>) -> tensor<4x1xi32>
  %25 = tosa.add %24, %18 : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x1xi32>
  %26 = tosa.mul %25, %17, %16 : (tensor<4x1xi32>, tensor<4x16xi32>, tensor<1xi8>) -> tensor<4x16xi32>
  %27 = tosa.greater %23, %26 : (tensor<4x16xi32>, tensor<4x16xi32>) -> tensor<4x16xi1>
  %28 = tosa.cast %27 : (tensor<4x16xi1>) -> tensor<4x16xi32>
  %29 = tosa.cast %28 : (tensor<4x16xi32>) -> tensor<4x16xi8>
  %expanded_2 = tensor.expand_shape %29 [[0, 1, 2], [3]] output_shape [1, 1, 4, 16] : tensor<4x16xi8> into tensor<1x1x4x16xi8>
  %30 = tosa.mul %expanded_2, %12, %16 : (tensor<1x1x4x16xi8>, tensor<1x14x4x16xi8>, tensor<1xi8>) -> tensor<1x14x4x16xi8>
  %extracted_slice = tensor.extract_slice %22[0, 0, 0, 0] [1, 14, 4, 64] [1, 1, 1, 1] : tensor<1x18x4x64xf16> to tensor<1x14x4x64xf16>
  %31 = tosa.transpose %expanded {perms = array<i32: 0, 1, 2, 4, 3>} : (tensor<1x2x1x16x64xf16>) -> tensor<1x2x1x64x16xf16>
  %32 = tosa.mul %31, %10, %16 : (tensor<1x2x1x64x16xf16>, tensor<1x2x7x64x16xf16>, tensor<1xi8>) -> tensor<1x2x7x64x16xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x14x4x64xf16> into tensor<14x4x64xf16>
  %collapsed_3 = tensor.collapse_shape %32 [[0, 1, 2], [3], [4]] : tensor<1x2x7x64x16xf16> into tensor<14x64x16xf16>
  %33 = tosa.matmul %collapsed, %collapsed_3, %7, %7 {acc_type = f32} : (tensor<14x4x64xf16>, tensor<14x64x16xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x4x16xf16>
  %expanded_4 = tensor.expand_shape %33 [[0, 1], [2], [3]] output_shape [1, 14, 4, 16] : tensor<14x4x16xf16> into tensor<1x14x4x16xf16>
  %34 = tosa.mul %expanded_4, %15, %16 : (tensor<1x14x4x16xf16>, tensor<1x14x4x16xf16>, tensor<1xi8>) -> tensor<1x14x4x16xf16>
  %band_bcast = tosa.mul %band, %band_ones, %16 : (tensor<1x1x4x16xf16>, tensor<1x14x4x16xf16>, tensor<1xi8>) -> tensor<1x14x4x16xf16>
  %banded = tosa.add %34, %band_bcast : (tensor<1x14x4x16xf16>, tensor<1x14x4x16xf16>) -> tensor<1x14x4x16xf16>
  %35 = tosa.cast %30 : (tensor<1x14x4x16xi8>) -> tensor<1x14x4x16xi1>
  %36 = tosa.select %35, %5, %banded : (tensor<1x14x4x16xi1>, tensor<1x14x4x16xf16>, tensor<1x14x4x16xf16>) -> tensor<1x14x4x16xf16>
  %37 = tosa.cast %36 : (tensor<1x14x4x16xf16>) -> tensor<1x14x4x16xf32>
  %38 = tosa.reduce_max %37 {axis = 3 : i32} : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x1xf32>
  %39 = tosa.mul %38, %4, %16 : (tensor<1x14x4x1xf32>, tensor<1x14x4x16xf32>, tensor<1xi8>) -> tensor<1x14x4x16xf32>
  %40 = tosa.sub %37, %39 : (tensor<1x14x4x16xf32>, tensor<1x14x4x16xf32>) -> tensor<1x14x4x16xf32>
  %41 = tosa.exp %40 : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x16xf32>
  %42 = tosa.reduce_sum %41 {axis = 3 : i32} : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x1xf32>
  %43 = tosa.mul %42, %4, %16 : (tensor<1x14x4x1xf32>, tensor<1x14x4x16xf32>, tensor<1xi8>) -> tensor<1x14x4x16xf32>
  %44 = tosa.reciprocal %43 : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x16xf32>
  %45 = tosa.mul %41, %44, %16 : (tensor<1x14x4x16xf32>, tensor<1x14x4x16xf32>, tensor<1xi8>) -> tensor<1x14x4x16xf32>
  %46 = tosa.cast %45 : (tensor<1x14x4x16xf32>) -> tensor<1x14x4x16xf16>
  %collapsed_5 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<1x14x4x16xf16> into tensor<14x4x16xf16>
  %expanded_6 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [14, 16, 64] : tensor<14336xf16> into tensor<14x16x64xf16>
  %47 = tosa.matmul %collapsed_5, %expanded_6, %7, %7 {acc_type = f32} : (tensor<14x4x16xf16>, tensor<14x16x64xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<14x4x64xf16>
  %expanded_7 = tensor.expand_shape %47 [[0, 1], [2], [3]] output_shape [1, 14, 4, 64] : tensor<14x4x64xf16> into tensor<1x14x4x64xf16>
  %48 = tosa.transpose %expanded_7 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x14x4x64xf16>) -> tensor<1x4x14x64xf16>
  %collapsed_8 = tensor.collapse_shape %48 [[0, 1, 2, 3]] : tensor<1x4x14x64xf16> into tensor<3584xf16>
  return %collapsed_8 : tensor<3584xf16>
}

// Same shape as the first case, but the band has a hole at (7, 5): not a band
// and not the full triangle, so nothing causal is recognised at all.
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
