// UNSUPPORTED: true
// TODO: migraphx.sqrt has no MIGraphXToTosa converter (use rsqrt+mul workaround)
// RUN: rocmlir-gen -fut sqrt_direct --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut sqrt_direct --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut sqrt_direct %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // Direct use of migraphx.sqrt (no TOSA converter exists)
  func.func @sqrt_direct(%x: !migraphx.shaped<8x256xf32, 256x1>) -> !migraphx.shaped<8x256xf32, 256x1> attributes {rock.kernel} {
    %out = migraphx.sqrt %x {} : <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    return %out : !migraphx.shaped<8x256xf32, 256x1>
  }
}
