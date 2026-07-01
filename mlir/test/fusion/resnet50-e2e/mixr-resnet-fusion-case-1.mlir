// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]

module {
  func.func @test(%arg0: !migraphx.shaped<1x128x28x28xf32, 0x1x0x0>, %arg1: !migraphx.shaped<1x128x28x28xf32, 100352x784x28x1>, %arg2: !migraphx.shaped<128x128x3x3xf32, 1152x9x3x1>) -> !migraphx.shaped<1x128x28x28xf32, 100352x784x28x1> attributes {rock.kernel} {
    %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x128x28x28xf32, 100352x784x28x1>, <128x128x3x3xf32, 1152x9x3x1> -> <1x128x28x28xf32, 100352x784x28x1>
    %2 = migraphx.add %1, %arg0 : <1x128x28x28xf32, 100352x784x28x1>, <1x128x28x28xf32, 0x1x0x0> -> <1x128x28x28xf32, 100352x784x28x1>
    %3 = migraphx.relu %2 : <1x128x28x28xf32, 100352x784x28x1> -> <1x128x28x28xf32, 100352x784x28x1>
    return %3 : !migraphx.shaped<1x128x28x28xf32, 100352x784x28x1>
  }
}
