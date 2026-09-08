// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention -RMS_threshold 0.01 --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

// Exercise rightmost-dimension-contiguous (row-major) addressing of a rank-4,
// multi-head dense constant against the clone harness's CPU reference.
module {
  func.func @mlir_attention(
      %q: !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>,
      %k: !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>,
      %v: !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>)
      -> !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>
      attributes {rock.kernel} {
    %bias = migraphx.literal(
        dense<[[[[0.1, -0.2, 0.3, -0.4],
                  [0.5, -0.6, 0.7, -0.8],
                  [0.9, -1.0, 1.1, -1.2],
                  [1.3, -1.4, 1.5, -1.6]],
                 [[-1.7, 1.8, -1.9, 2.0],
                  [-2.1, 2.2, -2.3, 2.4],
                  [-2.5, 2.6, -2.7, 2.8],
                  [-2.9, 3.0, -3.1, 3.2]]]]>
        : tensor<1x2x4x4xf32>) : <1x2x4x4xf32, 32x16x4x1>
    %kt = migraphx.transpose %k {permutation = [0, 1, 3, 2]}
        : <1x2x4x4xf32, 32x16x4x1>
        -> <1x2x4x4xf32, 32x16x1x4>
    %scores = migraphx.dot %q, %kt
        : <1x2x4x4xf32, 32x16x4x1>,
          <1x2x4x4xf32, 32x16x1x4>
        -> <1x2x4x4xf32, 32x16x4x1>
    %biased = migraphx.add %scores, %bias
        : <1x2x4x4xf32, 32x16x4x1>,
          <1x2x4x4xf32, 32x16x4x1>
        -> <1x2x4x4xf32, 32x16x4x1>
    %probabilities = migraphx.softmax %biased {axis = 3 : i64}
        : <1x2x4x4xf32, 32x16x4x1>
        -> <1x2x4x4xf32, 32x16x4x1>
    %result = migraphx.dot %probabilities, %v
        : <1x2x4x4xf32, 32x16x4x1>,
          <1x2x4x4xf32, 32x16x4x1>
        -> <1x2x4x4xf32, 32x16x4x1>
    return %result : !migraphx.shaped<1x2x4x4xf32, 32x16x4x1>
  }
}
