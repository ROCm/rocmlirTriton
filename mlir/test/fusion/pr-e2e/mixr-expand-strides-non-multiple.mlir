// RUN: rocmlir-gen -fut mlir_dot_sigmoid --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_sigmoid --verifier clone - | rocmlir-driver -c |  mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// Only ~40% of the results will be correct since the non-contiguous strides
// in this example is the result of concatenating 4x5x24 and 4x7x24
//(i.e., 4x12x24). This kernel is specifically dealing with making sure that the
// 4x5x24 elements are all in the correct place in the larger tensor.
// CHECK: relDiff = 0     : {{4[6-9][0-9]}}/1152

module {
  func.func @mlir_dot_sigmoid(%arg0: !migraphx.shaped<4x5x16xf16, 80x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x5x24xf16, 288x24x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <4x5x16xf16, 80x16x1>, <4x16x24xf16, 384x24x1> -> <4x5x24xf16, 120x24x1>
    %1 = migraphx.sigmoid %0 : <4x5x24xf16, 120x24x1> -> <4x5x24xf16, 288x24x1>
    return %1 : !migraphx.shaped<4x5x24xf16, 288x24x1>
  }
}
