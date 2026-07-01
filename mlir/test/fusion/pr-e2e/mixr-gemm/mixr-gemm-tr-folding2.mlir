// RUN: rocmlir-gen -fut mlir_transpose_reshape_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_transpose_reshape_dot --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
func.func @mlir_transpose_reshape_dot(%arg0: !migraphx.shaped<1x2x1x3xf32, 6x3x3x1>, %arg1: !migraphx.shaped<6x6xf32, 6x1>) -> !migraphx.shaped<1x6xf32, 6x1> attributes {rock.kernel} {
  %0 = migraphx.transpose %arg0 {permutation = [0, 2, 1, 3]} : <1x2x1x3xf32, 6x3x3x1> -> <1x1x2x3xf32, 6x6x3x1>
  %1 = migraphx.reshape %0 {dims = [1, 6]} : <1x1x2x3xf32, 6x6x3x1> -> <1x6xf32, 6x1>
  %2 = migraphx.dot %1, %arg1 : <1x6xf32, 6x1>, <6x6xf32, 6x1> -> <1x6xf32, 6x1>
  return %2 : !migraphx.shaped<1x6xf32, 6x1>
}
