// RUN: rocmlir-gen -fut swish_beta --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut swish_beta --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut swish_beta %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 1024

  // Swish-beta activation: swish(x) = x * sigmoid(beta * x)
  func.func @swish_beta(
      %arg0: !migraphx.shaped<4x256xf32, 256x1>,
      %beta: !migraphx.shaped<1xf32, 1>
  ) -> !migraphx.shaped<4x256xf32, 256x1> attributes {rock.kernel} {
    %beta_bc = migraphx.multibroadcast %beta {out_dyn_dims = [], out_lens = [4, 256]} : <1xf32, 1> -> <4x256xf32, 0x0>
    %bx = migraphx.mul %beta_bc, %arg0 {} : <4x256xf32, 0x0>, <4x256xf32, 256x1> -> <4x256xf32, 256x1>
    %sig = migraphx.sigmoid %bx {} : <4x256xf32, 256x1> -> <4x256xf32, 256x1>
    %out = migraphx.mul %arg0, %sig {} : <4x256xf32, 256x1>, <4x256xf32, 256x1> -> <4x256xf32, 256x1>
    return %out : !migraphx.shaped<4x256xf32, 256x1>
  }
}
