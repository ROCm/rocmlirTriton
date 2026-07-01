// RUN: rocmlir-driver -host-pipeline=highlevel %s |\
// RUN: rocmlir-opt -empty-tensor-to-alloc-tensor \
// RUN:   --one-shot-bufferize="copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
// RUN:   -buffer-results-to-out-params="add-result-attr=false hoist-dynamic-allocs=false hoist-static-allocs=false modify-public-functions=true" |\
// RUN: rocmlir-gen -rand=none -ph -pr -fut test_fusion - |\
// RUN: rocmlir-driver --host-pipeline=backend --arch %arch |\
// RUN: mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {

  // CHECK: [2,  2,  2,  2,  2,  2,  2,  2,
  func.func @test_fusion(%arg0: tensor<8192xf32>, %arg1: tensor<8192xf32>) -> tensor<8192xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<8192xf32>, tensor<8192xf32>) -> tensor<8192xf32>
    return %0 : tensor<8192xf32>
  }

}

