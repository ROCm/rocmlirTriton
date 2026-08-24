// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | FileCheck %s --check-prefix=IR
// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand_min_int 4 -rand_max_int 5 -rand_type_int_for_inputs=3 -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

// IR-LABEL: func.func @mlir_attention
// IR: %[[MAX:.*]] = arith.maxsi {{.*}} : tensor<2xi32>
// IR: %[[CLIPPED:.*]] = arith.minsi
// IR-SAME: %[[MAX]]{{[, ]}}
// IR-SAME: : tensor<2xi32>
// IR: rock.attention
// IR: lastValidKVIndex = (%[[CLIPPED]] : tensor<2xi32>)

module {
  // E2E: [1 1 1]
  func.func @mlir_attention(
        %arg0: !migraphx.shaped<1x1x12xf16, 12x12x1>,
        %arg1: !migraphx.shaped<1x2x4x2xf16, 16x8x2x1>,
        %arg2: !migraphx.shaped<1x2x4x2xf16, 16x8x2x1>,
        %arg3: !migraphx.shaped<1x1xsi32, 1x1>
    ) -> !migraphx.shaped<1x1x4xf16, 4x4x1> attributes {rock.kernel} {
    %range = migraphx.literal(dense<[0, 1, 2, 3]> : tensor<4xsi32>) : <4xsi32, 1>
    %neg_inf = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %one = migraphx.literal(dense<1.000000e+00> : tensor<1xf16>) : <1xf16, 1>
    %clip_min = migraphx.literal(dense<0> : tensor<1x1xsi32>) : <1x1xsi32, 1x1>
    %clip_max = migraphx.literal(dense<2> : tensor<1x1xsi32>) : <1x1xsi32, 1x1>
    %clipped = migraphx.clip %arg3, %clip_min, %clip_max : <1x1xsi32, 1x1>, <1x1xsi32, 1x1>, <1x1xsi32, 1x1> -> <1x1xsi32, 1x1>
    %q_reshaped = migraphx.reshape %arg0 {dims = [1, 1, 6, 2]} : <1x1x12xf16, 12x12x1> -> <1x1x6x2xf16, 12x12x2x1>
    %q_transposed = migraphx.transpose %q_reshaped {permutation = [0, 2, 1, 3]} : <1x1x6x2xf16, 12x12x2x1> -> <1x6x1x2xf16, 12x2x12x1>
    %seq_len = migraphx.multibroadcast %clipped {out_dyn_dims = [], out_lens = [1, 2]} : <1x1xsi32, 1x1> -> <1x2xsi32, 1x0>
    %q = migraphx.slice %q_transposed {axes = [1], ends = [2], starts = [0]} : <1x6x1x2xf16, 12x2x12x1> -> <1x2x1x2xf16, 12x2x12x1>
    %k = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x2x4x2xf16, 16x8x2x1> -> <1x2x2x4xf16, 16x8x1x2>
    %scores = migraphx.dot %q, %k : <1x2x1x2xf16, 12x2x12x1>, <1x2x2x4xf16, 16x8x1x2> -> <1x2x1x4xf16, 8x4x4x1>
    %columns = migraphx.multibroadcast %range {out_dyn_dims = [], out_lens = [1, 2, 1, 4]} : <4xsi32, 1> -> <1x2x1x4xsi32, 0x0x0x1>
    %neg_inf_broadcast = migraphx.multibroadcast %neg_inf {out_dyn_dims = [], out_lens = [1, 2, 1, 4]} : <1xf16, 1> -> <1x2x1x4xf16, 0x0x0x0>
    %one_broadcast = migraphx.multibroadcast %one {out_dyn_dims = [], out_lens = [1, 2, 1, 4]} : <1xf16, 1> -> <1x2x1x4xf16, 0x0x0x0>
    %scaled_scores = migraphx.mul %scores, %one_broadcast : <1x2x1x4xf16, 8x4x4x1>, <1x2x1x4xf16, 0x0x0x0> -> <1x2x1x4xf16, 8x4x4x1>
    %seq_len_reshaped = migraphx.reshape %seq_len {dims = [1, 2, 1, 1]} : <1x2xsi32, 1x0> -> <1x2x1x1xsi32, 2x1x1x1>
    %seq_len_broadcast = migraphx.multibroadcast %seq_len_reshaped {out_dyn_dims = [], out_lens = [1, 2, 1, 4]} : <1x2x1x1xsi32, 2x1x1x1> -> <1x2x1x4xsi32, 2x1x1x0>
    %mask = migraphx.greater %columns, %seq_len_broadcast : <1x2x1x4xsi32, 0x0x0x1>, <1x2x1x4xsi32, 2x1x1x0> -> <1x2x1x4xsi32, 8x4x4x1>
    %mask_i8 = migraphx.convert %mask {target_type = 0 : i64} : <1x2x1x4xsi32, 8x4x4x1> to <1x2x1x4xsi8, 8x4x4x1>
    %masked_scores = migraphx.where %mask_i8, %neg_inf_broadcast, %scaled_scores : <1x2x1x4xsi8, 8x4x4x1>, <1x2x1x4xf16, 0x0x0x0>, <1x2x1x4xf16, 8x4x4x1> -> <1x2x1x4xf16, 8x4x4x1>
    %softmax = migraphx.softmax %masked_scores {axis = 3 : i64} : <1x2x1x4xf16, 8x4x4x1> -> <1x2x1x4xf16, 8x4x4x1>
    %attn = migraphx.dot %softmax, %arg2 : <1x2x1x4xf16, 8x4x4x1>, <1x2x4x2xf16, 16x8x2x1> -> <1x2x1x2xf16, 4x2x2x1>
    %attn_transposed = migraphx.transpose %attn {permutation = [0, 2, 1, 3]} : <1x2x1x2xf16, 4x2x2x1> -> <1x1x2x2xf16, 4x2x2x1>
    %result = migraphx.reshape %attn_transposed {dims = [1, 1, 4]} : <1x1x2x2xf16, 4x2x2x1> -> <1x1x4xf16, 4x4x1>
    return %result : !migraphx.shaped<1x1x4xf16, 4x4x1>
  }
}
