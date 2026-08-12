// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand fixed - | rocmlir-driver -arch %arch -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [2.25,  0,  5.25,  2.25]
func.func @mlir_quantizelinear_f8E4M3FN(
    %input: !migraphx.shaped<2x2xf32, 2x1>,
    %scale: !migraphx.shaped<2x2xf32, 2x1>,
    %bias: !migraphx.shaped<2x2xf8E4M3FN, 2x1>)
    -> !migraphx.shaped<2x2xf32, 2x1>
    attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
  %result = migraphx.quantizelinear %input, %scale, %bias
    : <2x2xf32, 2x1>, <2x2xf32, 2x1>, !migraphx.shaped<2x2xf8E4M3FN, 2x1>
      -> <2x2xf8E4M3FN, 2x1>
  %dotResult = migraphx.quant_dot %result, %result
    : <2x2xf8E4M3FN, 2x1>, <2x2xf8E4M3FN, 2x1>
      -> <2x2xf32, 2x1>
  return %dotResult : !migraphx.shaped<2x2xf32, 2x1>
}
