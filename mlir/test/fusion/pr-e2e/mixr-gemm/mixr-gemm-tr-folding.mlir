// RUN: rocmlir-gen -fut mlir_transpose_reshape_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_transpose_reshape_dot --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
func.func private @mlir_transpose_reshape_dot(%arg0: !migraphx.shaped<2x8x4x4xf32, 128x16x4x1>, %arg1: !migraphx.shaped<1x8x8xf32, 64x8x1>) -> !migraphx.shaped<2x16x8xf32, 128x8x1> attributes {rock.kernel} {
  %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [2, 8, 8]} : <1x8x8xf32, 64x8x1> -> <2x8x8xf32, 0x8x1>
  %1 = migraphx.transpose %arg0 {permutation = [0, 2, 3, 1]} : <2x8x4x4xf32, 128x16x4x1> -> <2x4x4x8xf32, 128x32x8x1>
  %2 = migraphx.reshape %1 {dims = [2, 16, 8]} : <2x4x4x8xf32, 128x32x8x1> -> <2x16x8xf32, 128x8x1>
  %3 = migraphx.dot %2, %0 : <2x16x8xf32, 128x8x1>, <2x8x8xf32, 0x8x1> -> <2x16x8xf32, 128x8x1>
  return %3 : !migraphx.shaped<2x16x8xf32, 128x8x1>
}
