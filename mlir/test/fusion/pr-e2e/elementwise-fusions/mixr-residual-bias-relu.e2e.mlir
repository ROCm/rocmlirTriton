// RUN: rocmlir-gen -fut residual_bias_relu --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut residual_bias_relu --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut residual_bias_relu %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // Residual connection + bias + ReLU: relu(x + residual + bias)
  // Common pattern in ResNet and transformer feed-forward blocks.
  // x, residual: [8, 256], bias: [256] broadcast to [8, 256]
  func.func @residual_bias_relu(
      %x:        !migraphx.shaped<8x256xf32, 256x1>,
      %residual: !migraphx.shaped<8x256xf32, 256x1>,
      %bias:     !migraphx.shaped<256xf32, 1>
  ) -> !migraphx.shaped<8x256xf32, 256x1> attributes {rock.kernel} {
    %bias_bc = migraphx.multibroadcast %bias {out_dyn_dims = [], out_lens = [8, 256]} : <256xf32, 1> -> <8x256xf32, 0x1>
    %sum = migraphx.add %x, %residual {} : <8x256xf32, 256x1>, <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    %biased = migraphx.add %sum, %bias_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x1> -> <8x256xf32, 256x1>
    %out = migraphx.relu %biased {} : <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    return %out : !migraphx.shaped<8x256xf32, 256x1>
  }
}
