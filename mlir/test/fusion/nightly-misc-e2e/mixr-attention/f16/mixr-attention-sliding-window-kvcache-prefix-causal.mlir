// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | FileCheck %s --check-prefix=FOLD
// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_attention -rand 1 -rand_type float -rand_min_int 4 -rand_max_int 5 -rand_type_int_for_inputs=2,4 --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

// Flash decoding (splitKV = 2, 8 keys in two chunks of 4) combined with three
// nested masks, all expressed in the [batch, heads, splitKV, seqQ, key] score
// layout with key index j = split * 4 + key:
//   - prefix causal:  j > q + prefixOffset[b]        (prefixOffset = 4)
//   - KV cache:       j > lastValidKVIndex           (clipped to 5)
//   - sliding window: j < lastValidKVIndex - 3
// Each mask removes keys (0-1 by the window, 5 for q = 0 by the prefix, 6-7 by
// the KV cache) but no (split, q) row is fully masked, so the CPU reference
// has no NaNs and the whole output is verified.

// Verify that all three nested selects are folded into one attention op.
// FOLD: rock.attention{
// FOLD-DAG: lastValidKVIndex = (
// FOLD-DAG: prefixOffset = (
// FOLD-DAG: slidingWindowLookBack = 3
// FOLD-DAG: causal
// FOLD: qk = elementwise {
// FOLD-NOT: tosa.select
// FOLD: rock.yield

// E2E: [1 1 1]
// E2E-NEXT: [1 1 1]

module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<2x6x2x2xf16, 24x4x2x1>, %arg1: !migraphx.shaped<2x2x8x2xf16, 32x16x2x1>, %arg2: !migraphx.shaped<2x1xsi32, 1x1>, %arg3: !migraphx.shaped<2x2x8x2xf16, 32x16x2x1>, %arg4: !migraphx.shaped<2x1xsi32, 1x1>) -> (!migraphx.shaped<2x2x2x4xf16, 16x8x4x1>, !migraphx.shaped<2x2x2x2x1xf32, 8x4x2x1x1>) attributes {rock.kernel = "mixr"} {
    %kidx = migraphx.literal(dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xsi32>) : <8xsi32, 1>
    %qidx = migraphx.literal(dense<[[[[[0], [1]]]]]> : tensor<1x1x1x2x1xsi32>) : <1x1x1x2x1xsi32, 2x2x2x1x1>
    %ones = migraphx.literal(dense<1> : tensor<2x1x2x2x4xsi32>) : <2x1x2x2x4xsi32, 16x16x8x4x1>
    %ninf = migraphx.literal(dense<0xFC00> : tensor<1xf16>) : <1xf16, 1>
    %scale = migraphx.literal(dense<5.000000e-01> : tensor<1xf16>) : <1xf16, 1>
    %sliding_offset = migraphx.literal(dense<-3> : tensor<1xsi32>) : <1xsi32, 1>
    %fixed_seq_len = migraphx.literal(dense<5> : tensor<2x1xsi32>) : <2x1xsi32, 1x1>
    %seq_len = migraphx.clip %arg2, %fixed_seq_len, %fixed_seq_len : <2x1xsi32, 1x1>, <2x1xsi32, 1x1>, <2x1xsi32, 1x1> -> <2x1xsi32, 1x1>

    %q5 = migraphx.reshape %arg0 {dims = [2, 6, 1, 2, 2]} : <2x6x2x2xf16, 24x4x2x1> -> <2x6x1x2x2xf16, 24x4x4x2x1>
    %qb = migraphx.multibroadcast %q5 {out_dyn_dims = [], out_lens = [2, 6, 2, 2, 2]} : <2x6x1x2x2xf16, 24x4x4x2x1> -> <2x6x2x2x2xf16, 24x4x0x2x1>
    %q = migraphx.slice %qb {axes = [1], ends = [2], starts = [0]} : <2x6x2x2x2xf16, 24x4x0x2x1> -> <2x2x2x2x2xf16, 24x4x0x2x1>
    %k5 = migraphx.reshape %arg1 {dims = [2, 2, 2, 4, 2]} : <2x2x8x2xf16, 32x16x2x1> -> <2x2x2x4x2xf16, 32x16x8x2x1>
    %kt = migraphx.transpose %k5 {permutation = [0, 1, 2, 4, 3]} : <2x2x2x4x2xf16, 32x16x8x2x1> -> <2x2x2x2x4xf16, 32x16x8x1x2>
    %v5 = migraphx.reshape %arg3 {dims = [2, 2, 2, 4, 2]} : <2x2x8x2xf16, 32x16x2x1> -> <2x2x2x4x2xf16, 32x16x8x2x1>
    %ninf_b = migraphx.multibroadcast %ninf {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 4]} : <1xf16, 1> -> <2x2x2x2x4xf16, 0x0x0x0x0>
    %scale_b = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 4]} : <1xf16, 1> -> <2x2x2x2x4xf16, 0x0x0x0x0>
    %qk = migraphx.dot %q, %kt : <2x2x2x2x2xf16, 24x4x0x2x1>, <2x2x2x2x4xf16, 32x16x8x1x2> -> <2x2x2x2x4xf16, 32x16x8x4x1>
    %scaled = migraphx.mul %qk, %scale_b : <2x2x2x2x4xf16, 32x16x8x4x1>, <2x2x2x2x4xf16, 0x0x0x0x0> -> <2x2x2x2x4xf16, 32x16x8x4x1>

    // Prefix-causal mask.
    %off = migraphx.reshape %arg4 {dims = [2, 1, 1, 1, 1]} : <2x1xsi32, 1x1> -> <2x1x1x1x1xsi32, 1x1x1x1x1>
    %off_b = migraphx.multibroadcast %off {out_dyn_dims = [], out_lens = [2, 1, 1, 2, 1]} : <2x1x1x1x1xsi32, 1x1x1x1x1> -> <2x1x1x2x1xsi32, 1x0x0x0x0>
    %qidx_b = migraphx.multibroadcast %qidx {out_dyn_dims = [], out_lens = [2, 1, 1, 2, 1]} : <1x1x1x2x1xsi32, 2x2x2x1x1> -> <2x1x1x2x1xsi32, 0x0x0x1x0>
    %row = migraphx.add %qidx_b, %off_b : <2x1x1x2x1xsi32, 0x0x0x1x0>, <2x1x1x2x1xsi32, 1x0x0x0x0> -> <2x1x1x2x1xsi32, 2x2x2x1x1>
    %row_b = migraphx.multibroadcast %row {out_dyn_dims = [], out_lens = [2, 1, 2, 2, 4]} : <2x1x1x2x1xsi32, 2x2x2x1x1> -> <2x1x2x2x4xsi32, 2x0x0x1x0>
    %row_m = migraphx.mul %row_b, %ones : <2x1x2x2x4xsi32, 2x0x0x1x0>, <2x1x2x2x4xsi32, 16x16x8x4x1> -> <2x1x2x2x4xsi32, 16x16x8x4x1>
    %col = migraphx.reshape %kidx {dims = [1, 1, 2, 1, 4]} : <8xsi32, 1> -> <1x1x2x1x4xsi32, 8x8x4x4x1>
    %col_b = migraphx.multibroadcast %col {out_dyn_dims = [], out_lens = [2, 1, 2, 2, 4]} : <1x1x2x1x4xsi32, 8x8x4x4x1> -> <2x1x2x2x4xsi32, 0x0x4x0x1>
    %col_m = migraphx.mul %col_b, %ones : <2x1x2x2x4xsi32, 0x0x4x0x1>, <2x1x2x2x4xsi32, 16x16x8x4x1> -> <2x1x2x2x4xsi32, 16x16x8x4x1>
    %prefix_pred = migraphx.greater %col_m, %row_m : <2x1x2x2x4xsi32, 16x16x8x4x1>, <2x1x2x2x4xsi32, 16x16x8x4x1> -> <2x1x2x2x4xsi32, 16x16x8x4x1>
    %prefix_i8 = migraphx.convert %prefix_pred {target_type = 0 : i64} : <2x1x2x2x4xsi32, 16x16x8x4x1> to <2x1x2x2x4xsi8, 16x16x8x4x1>
    %prefix_mask = migraphx.multibroadcast %prefix_i8 {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 4]} : <2x1x2x2x4xsi8, 16x16x8x4x1> -> <2x2x2x2x4xsi8, 16x0x8x4x1>
    %prefix_masked = migraphx.where %prefix_mask, %ninf_b, %scaled : <2x2x2x2x4xsi8, 16x0x8x4x1>, <2x2x2x2x4xf16, 0x0x0x0x0>, <2x2x2x2x4xf16, 32x16x8x4x1> -> <2x2x2x2x4xf16, 32x16x8x4x1>

    // KV-cache mask.
    %cols = migraphx.broadcast %kidx {axis = 1 : i64, out_lens = [2, 8]} : <8xsi32, 1> -> <2x8xsi32, 0x1>
    %seq_len_b = migraphx.multibroadcast %seq_len {out_dyn_dims = [], out_lens = [2, 8]} : <2x1xsi32, 1x1> -> <2x8xsi32, 1x0>
    %kv_pred = migraphx.greater %cols, %seq_len_b : <2x8xsi32, 0x1>, <2x8xsi32, 1x0> -> <2x8xsi32, 8x1>
    %kv_i8 = migraphx.convert %kv_pred {target_type = 0 : i64} : <2x8xsi32, 8x1> to <2x8xsi8, 8x1>
    %kv_reshaped = migraphx.reshape %kv_i8 {dims = [2, 1, 2, 1, 4]} : <2x8xsi8, 8x1> -> <2x1x2x1x4xsi8, 8x8x4x4x1>
    %kv_mask = migraphx.multibroadcast %kv_reshaped {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 4]} : <2x1x2x1x4xsi8, 8x8x4x4x1> -> <2x2x2x2x4xsi8, 8x0x4x0x1>
    %kv_masked = migraphx.where %kv_mask, %ninf_b, %prefix_masked : <2x2x2x2x4xsi8, 8x0x4x0x1>, <2x2x2x2x4xf16, 0x0x0x0x0>, <2x2x2x2x4xf16, 32x16x8x4x1> -> <2x2x2x2x4xf16, 32x16x8x4x1>

    // Sliding-window mask.
    %sliding_offset_b = migraphx.multibroadcast %sliding_offset {out_dyn_dims = [], out_lens = [2, 1]} : <1xsi32, 1> -> <2x1xsi32, 0x1>
    %window_start = migraphx.add %seq_len, %sliding_offset_b : <2x1xsi32, 1x1>, <2x1xsi32, 0x1> -> <2x1xsi32, 1x1>
    %window_starts = migraphx.multibroadcast %window_start {out_dyn_dims = [], out_lens = [2, 8]} : <2x1xsi32, 1x1> -> <2x8xsi32, 1x0>
    %window_pred = migraphx.greater %window_starts, %cols : <2x8xsi32, 1x0>, <2x8xsi32, 0x1> -> <2x8xsi32, 8x1>
    %window_i8 = migraphx.convert %window_pred {target_type = 0 : i64} : <2x8xsi32, 8x1> to <2x8xsi8, 8x1>
    %window_reshaped = migraphx.reshape %window_i8 {dims = [2, 1, 2, 1, 4]} : <2x8xsi8, 8x1> -> <2x1x2x1x4xsi8, 8x8x4x4x1>
    %window_mask = migraphx.multibroadcast %window_reshaped {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 4]} : <2x1x2x1x4xsi8, 8x8x4x4x1> -> <2x2x2x2x4xsi8, 8x0x4x0x1>
    %window_masked = migraphx.where %window_mask, %ninf_b, %kv_masked : <2x2x2x2x4xsi8, 8x0x4x0x1>, <2x2x2x2x4xf16, 0x0x0x0x0>, <2x2x2x2x4xf16, 32x16x8x4x1> -> <2x2x2x2x4xf16, 32x16x8x4x1>

    %s = migraphx.convert %window_masked {target_type = 2 : i64} : <2x2x2x2x4xf16, 32x16x8x4x1> to <2x2x2x2x4xf32, 32x16x8x4x1>
    %max = migraphx.reduce_max %s {axes = [4]} : <2x2x2x2x4xf32, 32x16x8x4x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    %max_b = migraphx.multibroadcast %max {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 4]} : <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x4xf32, 8x4x2x1x0>
    %sub = migraphx.sub %s, %max_b : <2x2x2x2x4xf32, 32x16x8x4x1>, <2x2x2x2x4xf32, 8x4x2x1x0> -> <2x2x2x2x4xf32, 32x16x8x4x1>
    %exp = migraphx.exp %sub : <2x2x2x2x4xf32, 32x16x8x4x1> -> <2x2x2x2x4xf32, 32x16x8x4x1>
    %sum = migraphx.reduce_sum %exp {axes = [4]} : <2x2x2x2x4xf32, 32x16x8x4x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    %sum_b = migraphx.multibroadcast %sum {out_dyn_dims = [], out_lens = [2, 2, 2, 2, 4]} : <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x4xf32, 8x4x2x1x0>
    %softmax = migraphx.div %exp, %sum_b : <2x2x2x2x4xf32, 32x16x8x4x1>, <2x2x2x2x4xf32, 8x4x2x1x0> -> <2x2x2x2x4xf32, 32x16x8x4x1>
    %p = migraphx.convert %softmax {target_type = 1 : i64} : <2x2x2x2x4xf32, 32x16x8x4x1> to <2x2x2x2x4xf16, 32x16x8x4x1>
    %o = migraphx.dot %p, %v5 : <2x2x2x2x4xf16, 32x16x8x4x1>, <2x2x2x4x2xf16, 32x16x8x2x1> -> <2x2x2x2x2xf16, 16x8x4x2x1>
    %ot = migraphx.transpose %o {permutation = [0, 2, 3, 1, 4]} : <2x2x2x2x2xf16, 16x8x4x2x1> -> <2x2x2x2x2xf16, 16x4x2x8x1>
    %out = migraphx.reshape %ot {dims = [2, 2, 2, 4]} : <2x2x2x2x2xf16, 16x4x2x8x1> -> <2x2x2x4xf16, 16x8x4x1>
    %log = migraphx.log %sum : <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    %lse = migraphx.add %max, %log : <2x2x2x2x1xf32, 8x4x2x1x1>, <2x2x2x2x1xf32, 8x4x2x1x1> -> <2x2x2x2x1xf32, 8x4x2x1x1>
    return %out, %lse : !migraphx.shaped<2x2x2x4xf16, 16x8x4x1>, !migraphx.shaped<2x2x2x2x1xf32, 8x4x2x1x1>
  }
}
