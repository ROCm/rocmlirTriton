// UNSUPPORTED: true
// TODO: math.erf has no LLVM intrinsic; host pipeline needs math-to-libm
// RUN: rocmlir-gen -fut gelu_exact --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut gelu_exact --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut gelu_exact %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
  func.func @gelu_exact(%arg0: !migraphx.shaped<8x256xf32, 256x1>) -> !migraphx.shaped<8x256xf32, 256x1> attributes {rock.kernel} {
    %half     = migraphx.literal(dense<0.5>       : tensor<1xf32>) : <1xf32, 0>
    %one      = migraphx.literal(dense<1.0>       : tensor<1xf32>) : <1xf32, 0>
    %inv_sqrt2 = migraphx.literal(dense<0.707107> : tensor<1xf32>) : <1xf32, 0>

    %half_bc     = migraphx.multibroadcast %half     {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>
    %one_bc      = migraphx.multibroadcast %one      {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>
    %inv_sqrt2_bc = migraphx.multibroadcast %inv_sqrt2 {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>

    %scaled = migraphx.mul %arg0, %inv_sqrt2_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x0> -> <8x256xf32, 256x1>
    %erf_val = migraphx.erf %scaled {} : <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    %one_plus_erf = migraphx.add %one_bc, %erf_val {} : <8x256xf32, 0x0>, <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    %half_x = migraphx.mul %half_bc, %arg0 {} : <8x256xf32, 0x0>, <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    %out = migraphx.mul %half_x, %one_plus_erf {} : <8x256xf32, 256x1>, <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    return %out : !migraphx.shaped<8x256xf32, 256x1>
  }
}
