// RUN: rocmlir-gen -fut forward__part_7 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel | rocmlir-gen -ph -fut forward__part_7 --verifier clone - | rocmlir-driver -c -arch %arch | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

module {
  func.func private @forward__part_7(%arg0: tensor<1x256x14x14xf32>, %arg1: tensor<256x1x1xf32>, %arg2: tensor<256x3x3x256xf32>) -> (tensor<1x256x14x14xf32>) {
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x256x14x14xf32>) -> tensor<1x14x14x256xf32>
    %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<256xf32>}> : () -> tensor<256xf32>
    %5 = "tosa.const"() <{values = dense<-0.102452755> : tensor<1x256x1x1xf32>}> : () -> tensor<1x256x1x1xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[1, 256, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %6 = tosa.reshape %arg1, %const_shape : (tensor<256x1x1xf32>, !tosa.shape<4>) -> tensor<1x256x1x1xf32>
    %7 = "tosa.const"() <{values = dense<0.248004287> : tensor<1x256x1x1xf32>}> : () -> tensor<1x256x1x1xf32>
    %8 = "tosa.const"() <{values = dense<-0.133214399> : tensor<1x256x1x1xf32>}> : () -> tensor<1x256x1x1xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %9 = tosa.conv2d %1, %arg2, %3, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x14x14x256xf32>, tensor<256x3x3x256xf32>, tensor<256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x14x14x256xf32>
    %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x14x14x256xf32>) -> tensor<1x256x14x14xf32>
    %11 = tosa.sub %10, %5 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %12 = tosa.mul %11, %6, %shift : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>, tensor<1xi8>) -> tensor<1x256x14x14xf32>
    %13 = tosa.mul %12, %7, %shift : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>, tensor<1xi8>) -> tensor<1x256x14x14xf32>
    %14 = tosa.add %13, %8 : (tensor<1x256x14x14xf32>, tensor<1x256x1x1xf32>) -> tensor<1x256x14x14xf32>
    %15 = tosa.clamp %14 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    return %15 : tensor<1x256x14x14xf32>
  }
}
