// RUN: rocmlir-driver --host-pipeline=migraphx,highlevel %s | rocmlir-gen -rand=none -ph -pr -fut literal_quantizelinear - | rocmlir-driver -arch %arch -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [4, -128, -128, -128, 2, 0, 127, -128, 26, 27]
func.func @literal_quantizelinear(%dummy : !migraphx.shaped<1xi8, 1>) -> !migraphx.shaped<10xsi8, 1> {
    // IEEE 754: nan = 0x7fc00000, +inf = 0x7f800000, -inf = 0xff800000
    // input: [1.0, -inf, inf, nan, 0.0, -1.0, 127.0, -128.0, 12.1, 12.4]
    %input = migraphx.literal (dense<[1.0, 0xff800000, 0x7f800000, 0x7fc00000, 0.0, -1.0, 127.0, -128.0, 12.1, 12.4]> : tensor<10xf32>) : <10xf32, 1>
    %scale = migraphx.literal (dense<0.5> : tensor<1xf32>) : <1xf32, 1>
    %bias = migraphx.literal (dense<2> : tensor<1xsi8>) : <1xsi8, 1>
    %result = migraphx.quantizelinear %input, %scale, %bias : <10xf32, 1>, <1xf32, 1>, !migraphx.shaped<1xsi8, 1> -> <10xsi8, 1>
    return %result : !migraphx.shaped<10xsi8, 1>
}
