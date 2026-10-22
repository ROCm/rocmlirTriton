// RUN: rocmlir-gen -fut reduce_max_test --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut reduce_max_test --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut reduce_max_test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY
// UNSUPPORTED: true
// TODO: Requires reduction support (LowerReduce)

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // Elementwise abs → reduce_max: per-row max absolute value (L-infinity norm)
  func.func @reduce_max_test(%x: !migraphx.shaped<8x256xf32, 256x1>) -> !migraphx.shaped<8x1xf32, 1x1> attributes {rock.kernel} {
    %abs_x = migraphx.abs %x {} : <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    %max_val = migraphx.reduce_max %abs_x {axes = [1 : i64]} : <8x256xf32, 256x1> -> <8x1xf32, 1x1>
    return %max_val : !migraphx.shaped<8x1xf32, 1x1>
  }
}
