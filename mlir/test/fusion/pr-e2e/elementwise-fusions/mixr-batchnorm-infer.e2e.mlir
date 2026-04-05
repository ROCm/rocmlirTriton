// RUN: rocmlir-gen -fut batchnorm_infer --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut batchnorm_infer --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut batchnorm_infer %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f16 2048

  // Batch normalization inference:
  //   output = gamma * (input - mean) * rsqrt(var + eps) + beta
  // input: [2, 8, 128], mean/var/gamma/beta: [8] broadcast to [2, 8, 128]
  func.func @batchnorm_infer(
      %input: !migraphx.shaped<2x8x128xf16, 1024x128x1>,
      %mean:  !migraphx.shaped<8xf16, 1>,
      %var:   !migraphx.shaped<8xf16, 1>,
      %gamma: !migraphx.shaped<8xf16, 1>,
      %beta:  !migraphx.shaped<8xf16, 1>
  ) -> !migraphx.shaped<2x8x128xf16, 1024x128x1> attributes {rock.kernel} {
    %eps_val = migraphx.literal(dense<1.0e-05> : tensor<1xf16>) : <1xf16, 0>

    %mean_bc  = migraphx.multibroadcast %mean  {out_dyn_dims = [], out_lens = [2, 8, 128]} : <8xf16, 1> -> <2x8x128xf16, 0x1x0>
    %var_bc   = migraphx.multibroadcast %var   {out_dyn_dims = [], out_lens = [2, 8, 128]} : <8xf16, 1> -> <2x8x128xf16, 0x1x0>
    %gamma_bc = migraphx.multibroadcast %gamma {out_dyn_dims = [], out_lens = [2, 8, 128]} : <8xf16, 1> -> <2x8x128xf16, 0x1x0>
    %beta_bc  = migraphx.multibroadcast %beta  {out_dyn_dims = [], out_lens = [2, 8, 128]} : <8xf16, 1> -> <2x8x128xf16, 0x1x0>
    %eps_bc   = migraphx.multibroadcast %eps_val {out_dyn_dims = [], out_lens = [2, 8, 128]} : <1xf16, 0> -> <2x8x128xf16, 0x0x0>

    // (input - mean)
    %centered = migraphx.sub %input, %mean_bc {} : <2x8x128xf16, 1024x128x1>, <2x8x128xf16, 0x1x0> -> <2x8x128xf16, 1024x128x1>
    // var + eps
    %var_eps = migraphx.add %var_bc, %eps_bc {} : <2x8x128xf16, 0x1x0>, <2x8x128xf16, 0x0x0> -> <2x8x128xf16, 1024x128x1>
    // rsqrt(var + eps)
    %inv_std = migraphx.rsqrt %var_eps {} : <2x8x128xf16, 1024x128x1> -> <2x8x128xf16, 1024x128x1>
    // gamma * (input - mean) * rsqrt(var + eps)
    %scaled = migraphx.mul %centered, %inv_std {} : <2x8x128xf16, 1024x128x1>, <2x8x128xf16, 1024x128x1> -> <2x8x128xf16, 1024x128x1>
    %normed = migraphx.mul %gamma_bc, %scaled {} : <2x8x128xf16, 0x1x0>, <2x8x128xf16, 1024x128x1> -> <2x8x128xf16, 1024x128x1>
    // + beta
    %output = migraphx.add %normed, %beta_bc {} : <2x8x128xf16, 1024x128x1>, <2x8x128xf16, 0x1x0> -> <2x8x128xf16, 1024x128x1>
    return %output : !migraphx.shaped<2x8x128xf16, 1024x128x1>
  }
}
