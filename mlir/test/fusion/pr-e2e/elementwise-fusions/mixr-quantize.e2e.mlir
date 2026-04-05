// RUN: rocmlir-gen -fut quantize --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut quantize --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut quantize %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY
module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // Quantize: round(x / scale) clamped to INT8 range
  func.func @quantize(
      %x:     !migraphx.shaped<8x256xf32, 256x1>,
      %scale: !migraphx.shaped<256xf32, 1>
  ) -> !migraphx.shaped<8x256xsi8, 256x1> attributes {rock.kernel} {
    %scale_bc = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [8, 256]} : <256xf32, 1> -> <8x256xf32, 0x1>
    %out = migraphx.quantizelinear %x, %scale_bc : <8x256xf32, 256x1>, <8x256xf32, 0x1> -> <8x256xsi8, 256x1>
    return %out : !migraphx.shaped<8x256xsi8, 256x1>
  }
}
