// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// The CPU lowering pipeline is broken for grouped backwards data convolutions
// and rocmlir-gen cannot generate 3D convolutions, so neither the clone
// verifier nor -pv can check this kernel. The inputs are all ones (-rand none),
// which makes every output element the number of filter taps that reach it.
// With one channel per group, that number is the product over (D, H, W) of the
// count of (input index, tap index) pairs satisfying
// in * stride - pad + tap * dilation == out, which is how the expected values
// below were derived. The second pattern spans the boundary between the two
// groups: element 1104 is the last one of channel 0 and element 1105 is the
// first one of channel 1.

// CHECK: [8, 0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 8,
// CHECK-SAME: 0, 0, 8, 0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 12, 0, 0, 0, 8, 8, 0, 0, 0, 12,
func.func @mlir_bwd_data_conv(
    %arg0: !migraphx.shaped<1x2x3x5x5xf32, 150x75x25x5x1>,
    %arg1: !migraphx.shaped<2x1x3x3x3xf32, 27x27x9x3x1>
) -> !migraphx.shaped<1x2x5x13x17xf32, 2210x1105x221x17x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel} {
    %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
        dilation = [2, 3, 4],
        group = 2 : i64,
        padding = [2, 3, 4, 2, 3, 4],
        padding_mode = 0 : i64,
        stride = [2, 3, 4]
    } : <1x2x3x5x5xf32, 150x75x25x5x1>, <2x1x3x3x3xf32, 27x27x9x3x1> -> <1x2x5x13x17xf32, 2210x1105x221x17x1>
    return %0 : !migraphx.shaped<1x2x5x13x17xf32, 2210x1105x221x17x1>
}
