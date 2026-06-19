// RUN: rocmlir-gen -fut broadcast_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut broadcast_add --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut broadcast_add %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // Tests migraphx.broadcast (single-axis) as opposed to migraphx.multibroadcast
  func.func @broadcast_add(
      %x:    !migraphx.shaped<8x256xf32, 256x1>,
      %bias: !migraphx.shaped<256xf32, 1>
  ) -> !migraphx.shaped<8x256xf32, 256x1> attributes {rock.kernel} {
    %bias_bc = migraphx.broadcast %bias {axis = 1 : i64, out_lens = [8, 256]} : <256xf32, 1> -> <8x256xf32, 0x1>
    %out = migraphx.add %x, %bias_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x1> -> <8x256xf32, 256x1>
    return %out : !migraphx.shaped<8x256xf32, 256x1>
  }
}
