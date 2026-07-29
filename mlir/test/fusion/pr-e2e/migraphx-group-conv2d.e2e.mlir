// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_convolution %s | rocmlir-driver --kernel-pipeline=migraphx,highlevel --host-pipeline=migraphx,highlevel | rocmlir-gen -ph -verifier clone -fut mlir_convolution - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]

module {
  func.func @mlir_convolution(%arg0: !migraphx.shaped<1x4x16x16xf32, 1024x256x16x1>, %arg1: !migraphx.shaped<4x1x3x3xf32, 36x9x3x1>) -> !migraphx.shaped<1x4x14x14xf32, 784x196x14x1> attributes {rock.kernel} {
    %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 4 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x4x16x16xf32, 1024x256x16x1>, <4x1x3x3xf32, 36x9x3x1> -> <1x4x14x14xf32, 784x196x14x1>
    return %0 : !migraphx.shaped<1x4x14x14xf32, 784x196x14x1>
  }
}