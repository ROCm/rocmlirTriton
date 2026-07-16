// RUN: rocmlir-opt --split-input-file --tosa-to-rock %s -o -| FileCheck %s

// Stream-K uses the v5 perf-config format (streamKMultiple = field 12 >= 1,
// splitKFactor = field 8 stays 1). Its remainder wave accumulates partials via
// atomic_add into a zero-prefilled output just like split-K, so the tosa-to-rock
// conversion must attach `rock.prefill = 0` to every kernel result that the
// stream-K gemm/conv feeds.

// CHECK-LABEL: @test_basic
// CHECK-SAME: -> (tensor<2x128x256xf32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_basic(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> tensor<2x128x256xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %c = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32, perf_config="gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x128x256xf32>
  // CHECK: %{{.*}} = rock.gemm %arg0 * %arg1 {perf_config = "gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  return %c : tensor<2x128x256xf32>
}

// -----

// CHECK-LABEL: @test_basic_f16
// CHECK-SAME: -> (tensor<2x128x256xf16> {rock.prefill = 0.000000e+00 : f16})
func.func @test_basic_f16(%a: tensor<2x128x64xf16>, %b: tensor<2x64x256xf16>) -> tensor<2x128x256xf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: %{{.*}} = rock.gemm %arg0 * %arg1 {perf_config = "gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : tensor<2x128x64xf16> * tensor<2x64x256xf16> -> tensor<2x128x256xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %c = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32, perf_config="gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : (tensor<2x128x64xf16>, tensor<2x64x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x128x256xf16>
  return %c : tensor<2x128x256xf16>
}

// -----

// CHECK-LABEL: @test_basic_bf16
// CHECK-SAME: -> (tensor<2x128x256xbf16> {rock.prefill = 0.000000e+00 : bf16})
func.func @test_basic_bf16(%a: tensor<2x128x64xbf16>, %b: tensor<2x64x256xbf16>) -> tensor<2x128x256xbf16> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: %{{.*}} = rock.gemm %arg0 * %arg1 {perf_config = "gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : tensor<2x128x64xbf16> * tensor<2x64x256xbf16> -> tensor<2x128x256xbf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xbf16>}> : () -> tensor<1xbf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xbf16>}> : () -> tensor<1xbf16>
  %c = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32, perf_config="gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : (tensor<2x128x64xbf16>, tensor<2x64x256xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<2x128x256xbf16>
  return %c : tensor<2x128x256xbf16>
}

// -----

// CHECK-LABEL: @test_reduce
// CHECK-SAME: -> (tensor<2x128x1xf32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_reduce(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> tensor<2x128x1xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: %[[gemmOut:.*]] = rock.gemm %arg0 * %arg1 {perf_config = "gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  // CHECK: rock.reduce sum %[[gemmOut]] {axis = 2 : index} : tensor<2x128x256xf32> -> tensor<2x128x1xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %c = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32, perf_config="gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x128x256xf32>
  %1 = "tosa.reduce_sum"(%c) {axis = 2 : i32} : (tensor<2x128x256xf32>) -> tensor<2x128x1xf32>
  return %1 : tensor<2x128x1xf32>
}

// -----

// CHECK-LABEL: @test_reduce_two_outputs
// CHECK-SAME: -> (tensor<2x128x1xf32> {rock.prefill = 0.000000e+00 : f32}
// CHECK-SAME: tensor<2x1x256xf32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_reduce_two_outputs(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> (tensor<2x128x1xf32>, tensor<2x1x256xf32>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: %[[outGemm:.*]] = rock.gemm %arg0 * %arg1 {perf_config = "gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  // CHECK: rock.reduce sum %[[outGemm]] {axis = 2 : index} : tensor<2x128x256xf32> -> tensor<2x128x1xf32>
  // CHECK: rock.reduce sum %[[outGemm]] {axis = 1 : index} : tensor<2x128x256xf32> -> tensor<2x1x256xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %c = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32, perf_config="gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x128x256xf32>
  %1 = "tosa.reduce_sum"(%c) {axis = 2 : i32} : (tensor<2x128x256xf32>) -> tensor<2x128x1xf32>
  %2 = "tosa.reduce_sum"(%c) {axis = 1 : i32} : (tensor<2x128x256xf32>) -> tensor<2x1x256xf32>
  return %1, %2 : tensor<2x128x1xf32>, tensor<2x1x256xf32>
}

// -----

// CHECK-LABEL: @test_reduce_two_outputs2
// CHECK-SAME: -> (tensor<2x128x1xf32> {rock.prefill = 0.000000e+00 : f32}
// CHECK-SAME: tensor<2x128x256xf32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_reduce_two_outputs2(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> (tensor<2x128x1xf32>, tensor<2x128x256xf32>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: %[[outGemm:.*]] = rock.gemm %arg0 * %arg1 {perf_config = "gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  // CHECK: rock.reduce sum %[[outGemm]] {axis = 2 : index} : tensor<2x128x256xf32> -> tensor<2x128x1xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %c = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32, perf_config="gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x128x256xf32>
  %1 = "tosa.reduce_sum"(%c) {axis = 2 : i32} : (tensor<2x128x256xf32>) -> tensor<2x128x1xf32>
  return %1, %c : tensor<2x128x1xf32>, tensor<2x128x256xf32>
}

// -----

// CHECK-LABEL: @test_add_two_outputs
// CHECK-SAME: -> (tensor<2x128x256xf32> {rock.prefill = 0.000000e+00 : f32}
// CHECK-SAME: tensor<2x128x256xf32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_add_two_outputs(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>, %arg3: tensor<2x128x256xf32>) -> (tensor<2x128x256xf32>, tensor<2x128x256xf32>) attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // CHECK: %[[outGemm:.*]] = rock.gemm %arg0 * %arg1 {perf_config = "gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  // CHECK: tosa.add %[[outGemm]], %arg2 : (tensor<2x128x256xf32>, tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %c = "tosa.matmul"(%a, %b, %a_zp, %b_zp) {acc_type = f32, perf_config="gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x128x256xf32>
  %1 = "tosa.add"(%c, %arg3) {} : (tensor<2x128x256xf32>, tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  return %1, %c : tensor<2x128x256xf32>, tensor<2x128x256xf32>
}

// -----

// CHECK-LABEL: @test_backwards_conv
// CHECK-SAME: -> (tensor<524288xf32> {rock.prefill = 0.000000e+00 : f32})
func.func @test_backwards_conv(%arg0: tensor<131072xf32>, %arg1: tensor<4194304xf32>) -> tensor<524288xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx950"} {
  %0 = tosa.const_shape  {values = dense<[512, 512, 4, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.reshape %arg1, %0 : (tensor<4194304xf32>, !tosa.shape<4>) -> tensor<512x512x4x4xf32>
  %2 = tosa.const_shape  {values = dense<[1, 512, 16, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %3 = tosa.reshape %arg0, %2 : (tensor<131072xf32>, !tosa.shape<4>) -> tensor<1x512x16x16xf32>
  %4 = tosa.transpose %3 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x512x16x16xf32>) -> tensor<1x16x16x512xf32>
  %5 = tosa.transpose %1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<512x512x4x4xf32>) -> tensor<512x4x4x512xf32>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %7 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %8 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<512xf32>}> : () -> tensor<512xf32>
  // CHECK: rock.conv_bwd_data
  // CHECK: perf_config = "gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1"
  %9 = tosa.custom %4, %5, %8, %6, %7 {perf_config="gemm:v5:16,32,4,16,16,4,4,1,1,1,1,1,-1,-1,-1,-1,-1,-1", acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x16x16x512xf32>, tensor<512x4x4x512xf32>, tensor<512xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x32x32x512xf32>
  %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x32x32x512xf32>) -> tensor<1x512x32x32xf32>
  %11 = tosa.const_shape  {values = dense<524288> : tensor<1xindex>} : () -> !tosa.shape<1>
  %12 = tosa.reshape %10, %11 : (tensor<1x512x32x32xf32>, !tosa.shape<1>) -> tensor<524288xf32>
  return %12 : tensor<524288xf32>
}
