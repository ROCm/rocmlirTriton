// RUN: rocmlir-gen -fut dot_max_special --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -disable-fast-math | rocmlir-gen -ph -print-results -rand none -fut dot_max_special - | rocmlir-driver -c -disable-fast-math | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Disable unrelated div/reciprocal fast-math while deriving the special values
// below. rock-allow-fast-math-flags.mlir separately verifies that the default
// pipeline leaves Max-derived arith.maximumf operations precise.
//
// Verify NaN propagation from both operand positions, +0 over -0, preservation
// of -0 for max(-0, -0), infinity ordering, and ordinary finite values. Two
// +inf outputs are expected: one from max(-inf, +inf), and one from
// 1 / max(-0, +0). The -inf output comes from 1 / max(-0, -0), distinguishing
// both zero-sign cases without relying on how the runner prints signed zero.
// CHECK-COUNT-30: nan
// CHECK-COUNT-2: [inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf,  inf]
// CHECK: [-inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf,  -inf]
// CHECK: [2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2]

module {
  func.func @dot_max_special(
      %arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>,
      %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>)
      -> (!migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>,
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
    %neg_zero = migraphx.literal(
        dense<-0.0> : tensor<1x5x3xf32>)
        : <1x5x3xf32, 15x3x1>
    %nan_lhs = migraphx.div %zero, %zero
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %nan_lhs_result = migraphx.max %nan_lhs, %one
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %nan_rhs_result = migraphx.max %one, %nan_lhs
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %max_zero = migraphx.max %neg_zero, %zero
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %zero_sign_result = migraphx.div %one, %max_zero
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %max_neg_zero = migraphx.max %neg_zero, %neg_zero
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %same_zero_sign_result = migraphx.div %one, %max_neg_zero
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %pos_inf = migraphx.div %one, %zero
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %neg_one = migraphx.neg %one
        : <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    %neg_inf = migraphx.div %neg_one, %zero
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %inf_result = migraphx.max %neg_inf, %pos_inf
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %two = migraphx.add %one, %one
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %neg_two = migraphx.neg %two
        : <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    %finite_result = migraphx.max %neg_two, %two
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    return %nan_lhs_result, %nan_rhs_result, %zero_sign_result, %inf_result,
           %same_zero_sign_result, %finite_result
        : !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>,
          !migraphx.shaped<1x5x3xf32, 15x3x1>
  }
}
