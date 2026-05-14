// RUN: rocmlir-gen -fut forward__part_13 --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel | rocmlir-gen -ph -fut forward__part_13 --verifier clone - | rocmlir-driver -c -arch %arch | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

module {
  func.func private @forward__part_13(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<128x1x1xf32>, %arg2: tensor<1x128x28x28xf32>, %arg3: tensor<128x1x1x64xf32>) -> (tensor<1x128x28x28xf32>) {
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
    %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<128xf32>}> : () -> tensor<128xf32>
    %5 = "tosa.const"() <{values = dense<-0.211282596> : tensor<1x128x1x1xf32>}> : () -> tensor<1x128x1x1xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[1, 128, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %6 = tosa.reshape %arg1, %const_shape : (tensor<128x1x1xf32>, !tosa.shape<4>) -> tensor<1x128x1x1xf32>
    %7 = "tosa.const"() <{values = dense<0.333436757> : tensor<1x128x1x1xf32>}> : () -> tensor<1x128x1x1xf32>
    %8 = "tosa.const"() <{values = dense<0.0245603509> : tensor<1x128x1x1xf32>}> : () -> tensor<1x128x1x1xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %_slice_start = "tosa.const_shape"() { values = dense<[0, 0, 0, 0]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %_slice_size = "tosa.const_shape"() { values = dense<[1, 55, 55, 64]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %_sliced = tosa.slice %1, %_slice_start, %_slice_size : (tensor<1x56x56x64xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x55x55x64xf32>
    %9 = tosa.conv2d %_sliced, %arg3, %3, %input_zp, %weight_zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x55x55x64xf32>, tensor<128x1x1x64xf32>, tensor<128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x28x28x128xf32>
    %10 = tosa.transpose %9 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x28x28x128xf32>) -> tensor<1x128x28x28xf32>
    %11 = tosa.sub %10, %5 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %12 = tosa.mul %11, %6, %shift : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>, tensor<1xi8>) -> tensor<1x128x28x28xf32>
    %13 = tosa.mul %12, %7, %shift : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>, tensor<1xi8>) -> tensor<1x128x28x28xf32>
    %14 = tosa.add %13, %8 : (tensor<1x128x28x28xf32>, tensor<1x128x1x1xf32>) -> tensor<1x128x28x28xf32>
    %15 = tosa.add %arg2, %14 : (tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %16 = tosa.clamp %15 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    return %16 : tensor<1x128x28x28xf32>
  }
}
