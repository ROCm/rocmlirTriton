// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]

module {
  func.func @test(%scale: !migraphx.shaped<1x1x1x3xf16, 3x3x3x1>, %arg1: !migraphx.shaped<1x128x56x56xf16, 401408x3136x56x1>, %arg2: !migraphx.shaped<128x128x3x3xi8, 1152x9x3x1>) -> !migraphx.shaped<1x128x28x28xf16, 100352x784x28x1> attributes {rock.kernel} {
    %2 = migraphx.dequantizelinear %arg2, %scale : <128x128x3x3xi8, 1152x9x3x1>, <1x1x1x3xf16, 3x3x3x1> -> <128x128x3x3xf16, 1152x9x3x1>
    %1 = migraphx.convolution %arg1, %2 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : <1x128x56x56xf16, 401408x3136x56x1>, <128x128x3x3xf16, 1152x9x3x1> -> <1x128x28x28xf16, 100352x784x28x1>
    return %1 : !migraphx.shaped<1x128x28x28xf16, 100352x784x28x1>
  }
}
