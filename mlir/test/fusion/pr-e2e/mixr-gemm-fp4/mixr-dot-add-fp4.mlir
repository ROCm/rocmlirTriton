// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK: TODO_FILL_EXPECTED_OUTPUT
  func.func @mlir_dot_add_fp4(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>,
                               %arg1: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>,
                               %arg2: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = migraphx.add %arg0, %arg1 : <1x16x512xf4E2M1FN, 8192x512x1>, <1x16x512xf4E2M1FN, 8192x512x1> -> <1x16x512xf4E2M1FN, 8192x512x1>
    %1 = migraphx.dot %0, %arg2 : <1x16x512xf4E2M1FN, 8192x512x1>, <1x512x16xf4E2M1FN, 8192x16x1> -> <1x16x16xf32, 256x16x1>
    return %1 : !migraphx.shaped<1x16x16xf32, 256x16x1>
  }
}
