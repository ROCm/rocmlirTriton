// RUN: rocmlir-driver -host-pipeline=highlevel %s |\
// RUN: rocmlir-opt -empty-tensor-to-alloc-tensor \
// RUN:   --one-shot-bufferize="copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
// RUN:   -buffer-results-to-out-params="add-result-attr=false hoist-dynamic-allocs=false hoist-static-allocs=false modify-public-functions=true" |\
// RUN: rocmlir-gen -rand 1 -ph -pr -fut test_fusion - |\
// RUN: rocmlir-driver --host-pipeline=backend --arch %arch |\
// RUN: mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
module {
  // CHECK:  6,     2,     0,     0,     6,     6,     6,     6,     6,     0,     0,     6,     6,     0,     0,     0
  func.func @test_fusion(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x3x3x8xf32>) -> tensor<1x30x30x16xf32> {

    %cst = arith.constant dense<0.0> : tensor<16xf32>
    %input_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %weight_zp = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst, %input_zp, %weight_zp) {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x32x32x8xf32>, tensor<16x3x3x8xf32>, tensor<16xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x30x30x16xf32>
    %1 = "tosa.clamp"(%0) {min_val = 0.0 : f32, max_val = 6.0 : f32} : (tensor<1x30x30x16xf32>) -> tensor<1x30x30x16xf32>
    return %1 : tensor<1x30x30x16xf32>
  }

}

