// RUN: rocmlir-gen -fut bert_part_3 -arch %arch --clone-harness %s | rocmlir-driver -host-pipeline highlevel -kernel-pipeline highlevel -arch %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut bert_part_3 --verifier clone - | rocmlir-driver -kernel-pipeline full | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @bert_part_3(%arg0: tensor<1x12x12x32xf32>, %arg1: tensor<1x12x32x12xf32>, %arg2: tensor<1x1x1x1xf32>, %arg3: tensor<1x1x1x12xf32>) -> (tensor<1x12x12x12xf32>) {
      %const_shape = "tosa.const_shape"() { values = dense<[12, 32, 12]> : tensor<3xindex> } : () -> !tosa.shape<3>
      %0 = "tosa.reshape"(%arg1, %const_shape) : (tensor<1x12x32x12xf32>, !tosa.shape<3>) -> tensor<12x32x12xf32>
      %const_shape2 = "tosa.const_shape"() { values = dense<[12, 12, 32]> : tensor<3xindex> } : () -> !tosa.shape<3>
      %1 = "tosa.reshape"(%arg0, %const_shape2) : (tensor<1x12x12x32xf32>, !tosa.shape<3>) -> tensor<12x12x32xf32>
      %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
      %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
      %2 = "tosa.matmul"(%1, %0, %a_zp, %b_zp) : (tensor<12x12x32xf32>, tensor<12x32x12xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x12x12xf32>
      %const_shape3 = "tosa.const_shape"() { values = dense<[1, 12, 12, 12]> : tensor<4xindex> } : () -> !tosa.shape<4>
      %3 = "tosa.reshape"(%2, %const_shape3) : (tensor<12x12x12xf32>, !tosa.shape<4>) -> tensor<1x12x12x12xf32>
      %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
      %4 = "tosa.mul"(%3, %arg2, %shift) : (tensor<1x12x12x12xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x12x12x12xf32>
      %5 = "tosa.add"(%4, %arg3) : (tensor<1x12x12x12xf32>, tensor<1x1x1x12xf32>) -> tensor<1x12x12x12xf32>
      return %5 : tensor<1x12x12x12xf32>
    }
}
