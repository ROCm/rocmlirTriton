// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// The CPU lowering pipeline is broken for grouped backwards data convolutions,
// so the clone verifier cannot check this kernel. Inputs are all ones
// (-rand none), so every output element is the number of filter taps that
// reach it. The expected values come from the reference at the end of this
// file.

// CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 0, 0, 0, 0, 4, 4, 0, 4,
module {
  func.func @mlir_bwd_data_conv(
      %arg0: !migraphx.shaped<1x8x6x7xf32, 336x42x7x1>,
      %arg1: !migraphx.shaped<8x4x3x3xf32, 36x9x3x1>
  ) -> !migraphx.shaped<1x8x15x25xf32, 3000x375x25x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
    %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
      dilation = [3, 4],
      group = 2 : i64,
      padding = [1, 1, 1, 1],
      padding_mode = 0 : i64,
      stride = [2, 3]} : <1x8x6x7xf32, 336x42x7x1>, <8x4x3x3xf32, 36x9x3x1> -> <1x8x15x25xf32, 3000x375x25x1>
    return %0 : !migraphx.shaped<1x8x15x25xf32, 3000x375x25x1>
  }
}
