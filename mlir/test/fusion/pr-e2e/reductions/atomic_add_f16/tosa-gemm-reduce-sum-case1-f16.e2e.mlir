// RUN: rocmlir-gen -fut dot_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline highlevel -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_add --verifier clone - | rocmlir-driver -c -arch %arch | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

func.func private @dot_add(%arg0: tensor<1x128x64xf16>, %arg1: tensor<1x64x256xf16>) -> tensor<1x128x1xf16> attributes {rock.kernel} {
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %0 = "tosa.matmul"(%arg0, %arg1, %a_zp, %b_zp) {acc_type = f32} : (tensor<1x128x64xf16>, tensor<1x64x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x128x256xf16>
  %1 = "tosa.reduce_sum"(%0) {axis = 2 : i32} : (tensor<1x128x256xf16>) -> tensor<1x128x1xf16>
  return %1 : tensor<1x128x1xf16>
}
