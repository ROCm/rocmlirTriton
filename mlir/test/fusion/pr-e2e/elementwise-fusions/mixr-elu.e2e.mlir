// RUN: rocmlir-gen -fut elu --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut elu --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut elu %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // ELU: where(x > 0, x, alpha * (exp(x) - 1))
  func.func @elu(%x: !migraphx.shaped<8x256xf32, 256x1>) -> !migraphx.shaped<8x256xf32, 256x1> attributes {rock.kernel} {
    %zero  = migraphx.literal(dense<0.0> : tensor<1xf32>) : <1xf32, 0>
    %one   = migraphx.literal(dense<1.0> : tensor<1xf32>) : <1xf32, 0>
    %alpha = migraphx.literal(dense<1.0> : tensor<1xf32>) : <1xf32, 0>

    %zero_bc  = migraphx.multibroadcast %zero  {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>
    %one_bc   = migraphx.multibroadcast %one   {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>
    %alpha_bc = migraphx.multibroadcast %alpha {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>

    %cmp = migraphx.greater %x, %zero_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x0> -> <8x256xf32, 256x1>
    %pos = migraphx.convert %cmp {target_type = 0 : i64} : <8x256xf32, 256x1> to <8x256xi8, 256x1>
    %exp_x = migraphx.exp %x {} : <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    %exp_m1 = migraphx.sub %exp_x, %one_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x0> -> <8x256xf32, 256x1>
    %neg_branch = migraphx.mul %alpha_bc, %exp_m1 {} : <8x256xf32, 0x0>, <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    %out = migraphx.where %pos, %x, %neg_branch {} : <8x256xi8, 256x1>, <8x256xf32, 256x1>, <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    return %out : !migraphx.shaped<8x256xf32, 256x1>
  }
}
