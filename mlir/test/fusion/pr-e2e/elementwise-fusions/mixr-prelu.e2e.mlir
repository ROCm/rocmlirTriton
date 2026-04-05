// RUN: rocmlir-gen -fut prelu --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut prelu --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut prelu %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // PReLU: where(x > 0, x, alpha * x) with per-channel alpha
  func.func @prelu(
      %x:     !migraphx.shaped<4x8x64xf32, 512x64x1>,
      %alpha: !migraphx.shaped<8xf32, 1>
  ) -> !migraphx.shaped<4x8x64xf32, 512x64x1> attributes {rock.kernel} {
    %zero = migraphx.literal(dense<0.0> : tensor<1xf32>) : <1xf32, 0>
    %zero_bc  = migraphx.multibroadcast %zero  {out_dyn_dims = [], out_lens = [4, 8, 64]} : <1xf32, 0> -> <4x8x64xf32, 0x0x0>
    %alpha_bc = migraphx.multibroadcast %alpha {out_dyn_dims = [], out_lens = [4, 8, 64]} : <8xf32, 1> -> <4x8x64xf32, 0x1x0>

    %cmp = migraphx.greater %x, %zero_bc {} : <4x8x64xf32, 512x64x1>, <4x8x64xf32, 0x0x0> -> <4x8x64xf32, 512x64x1>
    %pos = migraphx.convert %cmp {target_type = 0 : i64} : <4x8x64xf32, 512x64x1> to <4x8x64xi8, 512x64x1>
    %neg_slope = migraphx.mul %alpha_bc, %x {} : <4x8x64xf32, 0x1x0>, <4x8x64xf32, 512x64x1> -> <4x8x64xf32, 512x64x1>
    %out = migraphx.where %pos, %x, %neg_slope {} : <4x8x64xi8, 512x64x1>, <4x8x64xf32, 512x64x1>, <4x8x64xf32, 512x64x1> -> <4x8x64xf32, 512x64x1>
    return %out : !migraphx.shaped<4x8x64xf32, 512x64x1>
  }
}
