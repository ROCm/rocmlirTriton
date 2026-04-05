// RUN: rocmlir-gen -fut fp4_dequant_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut fp4_dequant_add --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut fp4_dequant_add %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise

  // FP4 dequantize + bias add: MXFP4 input dequantized to f32
  func.func @fp4_dequant_add(
      %x:     !migraphx.shaped<8x256xf4E2M1FN, 256x1>,
      %scale: !migraphx.shaped<256xf32, 1>,
      %bias:  !migraphx.shaped<256xf32, 1>
  ) -> !migraphx.shaped<8x256xf32, 256x1> attributes {rock.kernel} {
    %scale_bc = migraphx.multibroadcast %scale {out_dyn_dims = [], out_lens = [8, 256]} : <256xf32, 1> -> <8x256xf32, 0x1>
    %bias_bc  = migraphx.multibroadcast %bias  {out_dyn_dims = [], out_lens = [8, 256]} : <256xf32, 1> -> <8x256xf32, 0x1>

    %deq = migraphx.dequantizelinear %x, %scale_bc : <8x256xf4E2M1FN, 256x1>, <8x256xf32, 0x1> -> <8x256xf32, 256x1>
    %out = migraphx.add %deq, %bias_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x1> -> <8x256xf32, 256x1>
    return %out : !migraphx.shaped<8x256xf32, 256x1>
  }
}
