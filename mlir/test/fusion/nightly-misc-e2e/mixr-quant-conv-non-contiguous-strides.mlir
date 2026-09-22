// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// The output strides are non-contiguous: the buffer holds 6144 elements while
// the convolution writes 4096 of them, so the clone verifier cannot be used
// here (it also walks the 2048 trailing elements, which nothing initializes).
// Those elements sit at the end of the buffer, so the values checked below are
// real output. Inputs are all ones (-rand none), which makes every output
// element 4 (input channels) times the number of filter taps that land inside
// the padded image: 4 taps in the corners, 6 along the edges and 9 in the
// interior.

// CHECK: [16, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 16, 24, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 24, 24, 36,
module {
  func.func @mlir_quant_convolution(%arg0: !migraphx.shaped<1x4x16x16xsi8, 1024x256x16x1>, %arg1: !migraphx.shaped<16x4x3x3xsi8, 36x9x3x1>) -> !migraphx.shaped<1x16x16x16xsi32, 6144x256x16x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = migraphx.quant_convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <1x4x16x16xsi8, 1024x256x16x1>, <16x4x3x3xsi8, 36x9x3x1> -> <1x16x16x16xsi32, 6144x256x16x1>
    return %0 : !migraphx.shaped<1x16x16x16xsi32, 6144x256x16x1>
  }
}
