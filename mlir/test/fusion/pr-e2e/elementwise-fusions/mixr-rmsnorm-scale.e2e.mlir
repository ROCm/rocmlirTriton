// RUN: rocmlir-gen -fut rmsnorm_scale --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rmsnorm_scale --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut rmsnorm_scale %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 1024

  // RMSNorm kernel 2: x * gamma * rsqrt(sum_sq / N + eps)
  // Purely elementwise: takes x, gamma, and pre-computed sum_sq as inputs.
  func.func @rmsnorm_scale(
      %x:      !migraphx.shaped<4x256xf32, 256x1>,
      %gamma:  !migraphx.shaped<256xf32, 1>,
      %sum_sq: !migraphx.shaped<4x1xf32, 1x1>
  ) -> !migraphx.shaped<4x256xf32, 256x1> attributes {rock.kernel} {
    %inv_n = migraphx.literal(dense<3.90625e-03> : tensor<1xf32>) : <1xf32, 0>
    %eps    = migraphx.literal(dense<1.0e-05>     : tensor<1xf32>) : <1xf32, 0>

    %sum_sq_bc = migraphx.multibroadcast %sum_sq {out_dyn_dims = [], out_lens = [4, 256]} : <4x1xf32, 1x1> -> <4x256xf32, 1x0>
    %inv_n_bc  = migraphx.multibroadcast %inv_n  {out_dyn_dims = [], out_lens = [4, 256]} : <1xf32, 0> -> <4x256xf32, 0x0>
    %eps_bc    = migraphx.multibroadcast %eps    {out_dyn_dims = [], out_lens = [4, 256]} : <1xf32, 0> -> <4x256xf32, 0x0>
    %gamma_bc  = migraphx.multibroadcast %gamma  {out_dyn_dims = [], out_lens = [4, 256]} : <256xf32, 1> -> <4x256xf32, 0x1>

    // mean_sq = sum_sq / N
    %mean_sq = migraphx.mul %sum_sq_bc, %inv_n_bc {} : <4x256xf32, 1x0>, <4x256xf32, 0x0> -> <4x256xf32, 256x1>
    // mean_sq + eps
    %mean_sq_eps = migraphx.add %mean_sq, %eps_bc {} : <4x256xf32, 256x1>, <4x256xf32, 0x0> -> <4x256xf32, 256x1>
    // rsqrt(mean_sq + eps)
    %inv_rms = migraphx.rsqrt %mean_sq_eps {} : <4x256xf32, 256x1> -> <4x256xf32, 256x1>
    // x * rsqrt(...)
    %normed = migraphx.mul %x, %inv_rms {} : <4x256xf32, 256x1>, <4x256xf32, 256x1> -> <4x256xf32, 256x1>
    // x * rsqrt(...) * gamma
    %output = migraphx.mul %normed, %gamma_bc {} : <4x256xf32, 256x1>, <4x256xf32, 0x1> -> <4x256xf32, 256x1>
    return %output : !migraphx.shaped<4x256xf32, 256x1>
  }
}
