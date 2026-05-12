// RUN: rocmlir-gen -fut bert_part_7 -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut bert_part_7 --verifier clone - | rocmlir-driver -kernel-pipeline full | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @bert_part_7(%arg0: tensor<1x12x1536xf32>, %arg1: tensor<1536x384xf32>, %arg2: tensor<1x1x384xf32>, %arg3: tensor<1x12x384xf32>) -> (tensor<1x12x384xf32>) {
      %const_shape = "tosa.const_shape"() { values = dense<[1, 1536, 384]> : tensor<3xindex> } : () -> !tosa.shape<3>
      %0 = "tosa.reshape"(%arg1, %const_shape) : (tensor<1536x384xf32>, !tosa.shape<3>) -> tensor<1x1536x384xf32>
      %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
      %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
      %1 = "tosa.matmul"(%arg0, %0, %a_zp, %b_zp) : (tensor<1x12x1536xf32>, tensor<1x1536x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x12x384xf32>
      %2 = "tosa.add"(%1, %arg2) : (tensor<1x12x384xf32>, tensor<1x1x384xf32>) -> tensor<1x12x384xf32>
      %3 = "tosa.add"(%2, %arg3) : (tensor<1x12x384xf32>, tensor<1x12x384xf32>) -> tensor<1x12x384xf32>
      return %3 : tensor<1x12x384xf32>
    }
}
