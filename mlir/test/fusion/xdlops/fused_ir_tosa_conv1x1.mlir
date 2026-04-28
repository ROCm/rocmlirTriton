// RUN: rocmlir-driver -kernel-pipeline highlevel %s | rocmlir-driver -kernel-pipeline gpu -arch %arch | rocmlir-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<64x64x1x1xf32>, %arg2: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> attributes {rock.kernel, rock.arch = "amdgcn-amd-amdhsa:gfx908"} {
    %0 = "tosa.transpose"(%arg0) {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
    %1 = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 3, 1>} : (tensor<64x64x1x1xf32>) -> tensor<64x1x1x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %2 = "tosa.conv2d"(%0, %1, %cst_0, %input_zp, %weight_zp) {xdlopsV2 = true, acc_type = f32, dilation = array<i64: 1, 1>, expected_filter_layout = "kcyx", expected_input_layout = "nchw", expected_output_layout = "nkhw", pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x1x1x64xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x56x56x64xf32>
    %3 = "tosa.transpose"(%2) {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x56x56x64xf32>) -> tensor<1x64x56x56xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %5 = "tosa.clamp"(%4) {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return %5 : tensor<1x64x56x56xf32>
  }
}
// 1. Check gemm lowered to tt.dot
// CHECK: tt.load
// CHECK: tt.load
// CHECK: tt.dot

// 2. Check fused add + relu after gemm
// CHECK: arith.addf
// CHECK: arith.maximumf

// 3. Check result is stored
// CHECK: tt.store
