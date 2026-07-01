// RUN: rocmlir-gen -fut forward__part_17 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel | rocmlir-gen -ph -fut forward__part_17 --verifier clone - | rocmlir-driver -c -arch %arch | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

module {
  func.func private @forward__part_17(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x1x1xf32>, %arg2: tensor<64x1x1xf32>, %arg3: tensor<1x64x1x1xf32>, %arg4: tensor<1x64x1x1xf32>, %arg5: tensor<64x3x3x64xf32>) -> (tensor<1x64x56x56xf32>) {
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
    %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<64xf32>}> : () -> tensor<64xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[1, 64, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %5 = tosa.reshape %arg2, %const_shape : (tensor<64x1x1xf32>, !tosa.shape<4>) -> tensor<1x64x1x1xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %6 = tosa.conv2d %1, %arg5, %3, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x56x56x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x56x56x64xf32>
    %7 = tosa.transpose %6 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x56x56x64xf32>) -> tensor<1x64x56x56xf32>
    %8 = tosa.sub %7, %arg1 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = tosa.mul %8, %5, %shift : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>, tensor<1xi8>) -> tensor<1x64x56x56xf32>
    %10 = tosa.mul %9, %arg3, %shift : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>, tensor<1xi8>) -> tensor<1x64x56x56xf32>
    %11 = tosa.add %10, %arg4 : (tensor<1x64x56x56xf32>, tensor<1x64x1x1xf32>) -> tensor<1x64x56x56xf32>
    %12 = tosa.clamp %11 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return %12 : tensor<1x64x56x56xf32>
  }
}
