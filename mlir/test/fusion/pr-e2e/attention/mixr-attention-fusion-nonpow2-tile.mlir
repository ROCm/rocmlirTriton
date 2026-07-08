// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// CHECK-NEXT: [1 1 1]

// A fully-fused attention kernel whose perf_config drives non-power-of-two tile
// decomposition on BOTH output axes at once: a non-pow2 seq_len_q tile
// (mPerBlockG0 = 48, so the padded M is split into pow2 sub-tiles {32, 16}) and
// a non-pow2 head_dim_v (80 -> gemm1 N split {64, 16}), together with nPerBlockG1
// (attn:v6) head-dim tiling. On top of that it carries a fusion at every site:
//   - input fusion before the first GEMM (Q = q0 + q1, V = v0 + v1),
//   - a pre-softmax fusion between the two GEMMs (scale then bias),
//   - an output fusion after the second GEMM (scale then bias), and
//   - it returns the log-sum-exp (LSE = log(sum) + max) with an LSE output
//     fusion (scale then bias).
// The softmax is written out explicitly (reduce_max / exp / reduce_sum) so the
// LSE is exposed as a second result. This makes rock-decompose-nonpow2-tiles
// replicate the input-, pre-softmax-, output- and LSE-fusion DAGs across the
// 2x2 grid of pow2 sub-attentions simultaneously.

module {
  func.func @mlir_attention(%q0: !migraphx.shaped<1x2x128x64xf32, 16384x8192x64x1>,
                            %q1: !migraphx.shaped<1x2x128x64xf32, 16384x8192x64x1>,
                            %k: !migraphx.shaped<1x2x128x64xf32, 16384x8192x64x1>,
                            %v0: !migraphx.shaped<1x2x128x80xf32, 20480x10240x80x1>,
                            %v1: !migraphx.shaped<1x2x128x80xf32, 20480x10240x80x1>,
                            %scale: !migraphx.shaped<1x2x128x128xf32, 32768x16384x128x1>,
                            %bias: !migraphx.shaped<1x2x128x128xf32, 32768x16384x128x1>,
                            %oscale: !migraphx.shaped<1x2x128x80xf32, 20480x10240x80x1>,
                            %obias: !migraphx.shaped<1x2x128x80xf32, 20480x10240x80x1>,
                            %lse_scale: !migraphx.shaped<1x2x128x1xf32, 256x128x1x1>,
                            %lse_bias: !migraphx.shaped<1x2x128x1xf32, 256x128x1x1>)
      -> (!migraphx.shaped<1x2x128x80xf32, 20480x10240x80x1>, !migraphx.shaped<1x2x128x1xf32, 256x128x1x1>) attributes {rock.kernel} {
    // Input fusion (before the first GEMM): Q = q0 + q1, V = v0 + v1.
    %Q = migraphx.add %q0, %q1 : <1x2x128x64xf32, 16384x8192x64x1>, <1x2x128x64xf32, 16384x8192x64x1> -> <1x2x128x64xf32, 16384x8192x64x1>
    %V = migraphx.add %v0, %v1 : <1x2x128x80xf32, 20480x10240x80x1>, <1x2x128x80xf32, 20480x10240x80x1> -> <1x2x128x80xf32, 20480x10240x80x1>
    // First GEMM: Q * K^T -> attention scores.
    %kt = migraphx.transpose %k {permutation = [0, 1, 3, 2]} : <1x2x128x64xf32, 16384x8192x64x1> -> <1x2x64x128xf32, 16384x8192x1x64>
    %qk = migraphx.dot %Q, %kt : <1x2x128x64xf32, 16384x8192x64x1>, <1x2x64x128xf32, 16384x8192x1x64> -> <1x2x128x128xf32, 32768x16384x128x1>
    // Pre-softmax fusion (between the two GEMMs): scale then bias.
    %qk_scaled = migraphx.mul %qk, %scale : <1x2x128x128xf32, 32768x16384x128x1>, <1x2x128x128xf32, 32768x16384x128x1> -> <1x2x128x128xf32, 32768x16384x128x1>
    %qk_biased = migraphx.add %qk_scaled, %bias : <1x2x128x128xf32, 32768x16384x128x1>, <1x2x128x128xf32, 32768x16384x128x1> -> <1x2x128x128xf32, 32768x16384x128x1>
    // Explicit softmax so the LSE (log(sum) + max) is exposed as a result.
    %max = migraphx.reduce_max %qk_biased {axes = [3 : i64]} : <1x2x128x128xf32, 32768x16384x128x1> -> <1x2x128x1xf32, 256x128x1x1>
    %norm = migraphx.sub %qk_biased, %max : <1x2x128x128xf32, 32768x16384x128x1>, <1x2x128x1xf32, 256x128x1x1> -> <1x2x128x128xf32, 32768x16384x128x1>
    %exp = migraphx.exp %norm : <1x2x128x128xf32, 32768x16384x128x1> -> <1x2x128x128xf32, 32768x16384x128x1>
    %sum = migraphx.reduce_sum %exp {axes = [3 : i64]} : <1x2x128x128xf32, 32768x16384x128x1> -> <1x2x128x1xf32, 256x128x1x1>
    %recip = migraphx.recip %sum : <1x2x128x1xf32, 256x128x1x1> -> <1x2x128x1xf32, 256x128x1x1>
    %sm = migraphx.mul %exp, %recip : <1x2x128x128xf32, 32768x16384x128x1>, <1x2x128x1xf32, 256x128x1x1> -> <1x2x128x128xf32, 32768x16384x128x1>
    // Second GEMM: scores * V, with a non-pow2 seq_len_q tile (mPerBlock = 48) and
    // nPerBlockG1 (attn:v6) head-dim tiling; head_dim_v = 80 is non-pow2 as well.
    %pv = migraphx.dot %sm, %V {perf_config = "attn:v6:48,32,16,64,1,1,2,0,1,1,0,0,-1,-1,-1,-1,-1,-1,-1"} : <1x2x128x128xf32, 32768x16384x128x1>, <1x2x128x80xf32, 20480x10240x80x1> -> <1x2x128x80xf32, 20480x10240x80x1>
    // Output fusion (after the second GEMM): scale then bias.
    %out_scaled = migraphx.mul %pv, %oscale : <1x2x128x80xf32, 20480x10240x80x1>, <1x2x128x80xf32, 20480x10240x80x1> -> <1x2x128x80xf32, 20480x10240x80x1>
    %out = migraphx.add %out_scaled, %obias : <1x2x128x80xf32, 20480x10240x80x1>, <1x2x128x80xf32, 20480x10240x80x1> -> <1x2x128x80xf32, 20480x10240x80x1>
    // LSE = log(sum) + max, then an LSE output fusion (scale then bias).
    %logsum = migraphx.log %sum : <1x2x128x1xf32, 256x128x1x1> -> <1x2x128x1xf32, 256x128x1x1>
    %lse = migraphx.add %logsum, %max : <1x2x128x1xf32, 256x128x1x1>, <1x2x128x1xf32, 256x128x1x1> -> <1x2x128x1xf32, 256x128x1x1>
    %lse_scaled = migraphx.mul %lse, %lse_scale : <1x2x128x1xf32, 256x128x1x1>, <1x2x128x1xf32, 256x128x1x1> -> <1x2x128x1xf32, 256x128x1x1>
    %lse_final = migraphx.add %lse_scaled, %lse_bias : <1x2x128x1xf32, 256x128x1x1>, <1x2x128x1xf32, 256x128x1x1> -> <1x2x128x1xf32, 256x128x1x1>
    return %out, %lse_final : !migraphx.shaped<1x2x128x80xf32, 20480x10240x80x1>, !migraphx.shaped<1x2x128x1xf32, 256x128x1x1>
  }
}
