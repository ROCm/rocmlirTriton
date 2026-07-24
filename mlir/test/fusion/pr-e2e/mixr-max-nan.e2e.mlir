// RUN: rocmlir-gen -fut dot_max_nan --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -disable-fast-math | rocmlir-gen -ph -print-results -rand none -fut dot_max_nan - | rocmlir-driver -c -disable-fast-math | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Disable unrelated division fast-math while deriving NaN below. The focused
// Rock lit test verifies that Max receives `nsz` without receiving `nnan`.
// CHECK-COUNT-30: nan

module {
  func.func @dot_max_nan(
      %arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>,
      %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>)
      -> (!migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>)
      attributes {rock.kernel, rock.arch = ""} {
    %dot = migraphx.dot %arg0, %arg1
        : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1>
        -> <1x5x3xf32, 15x3x1>
    %zero = migraphx.sub %dot, %dot
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %one_literal = migraphx.literal(
        dense<1.0> : tensor<1x5x3xf32>)
        : <1x5x3xf32, 15x3x1>
    %one = migraphx.add %zero, %one_literal
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %nan = migraphx.div %zero, %zero
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %nan_lhs_result = migraphx.max %nan, %one
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %nan_rhs_result = migraphx.max %one, %nan
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    return %nan_lhs_result, %nan_rhs_result
        : !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>
  }
}
