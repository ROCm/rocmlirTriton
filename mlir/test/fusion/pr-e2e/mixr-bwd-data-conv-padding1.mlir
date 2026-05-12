// RUN: rocmlir-gen -fut mlir_bwd_data_conv --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_bwd_data_conv --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
    // CHECK: [1 1 1]
  func.func @mlir_bwd_data_conv(
      %arg0: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>,
      %arg1: !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>
  ) -> !migraphx.shaped<1x1x3x3xf32, 9x9x3x1> attributes {rock.kernel} {
    %0 = migraphx.backwards_data_convolution %arg1, %arg0 {
      dilation = [1, 1],
      group = 1 : i64,
      padding = [1, 1, 1, 1],
      padding_mode = 0 : i64,
      stride = [1, 1]
    } : <1x1x3x3xf32, 9x9x3x1>, <1x1x3x3xf32, 9x9x3x1>
        -> <1x1x3x3xf32, 9x9x3x1>
    return %0 : !migraphx.shaped<1x1x3x3xf32, 9x9x3x1>
  }
}

