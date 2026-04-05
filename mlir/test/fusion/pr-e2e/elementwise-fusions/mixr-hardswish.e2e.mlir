// RUN: rocmlir-gen -fut hardswish --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut hardswish --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut hardswish %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 2048

  // Hardswish (MobileNetV3): x * clip(x + 3, 0, 6) * (1/6)
  func.func @hardswish(%x: !migraphx.shaped<8x256xf32, 256x1>) -> !migraphx.shaped<8x256xf32, 256x1> attributes {rock.kernel} {
    %three  = migraphx.literal(dense<3.0>              : tensor<1xf32>) : <1xf32, 0>
    %inv6   = migraphx.literal(dense<0.16666667>       : tensor<1xf32>) : <1xf32, 0>
    %lo     = migraphx.literal(dense<0.0>              : tensor<1xf32>) : <1xf32, 0>
    %hi     = migraphx.literal(dense<6.0>              : tensor<1xf32>) : <1xf32, 0>

    %three_bc = migraphx.multibroadcast %three {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>
    %inv6_bc  = migraphx.multibroadcast %inv6  {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>
    %lo_bc    = migraphx.multibroadcast %lo    {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>
    %hi_bc    = migraphx.multibroadcast %hi    {out_dyn_dims = [], out_lens = [8, 256]} : <1xf32, 0> -> <8x256xf32, 0x0>

    %x_plus_3 = migraphx.add %x, %three_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x0> -> <8x256xf32, 256x1>
    %clamped = migraphx.clip %x_plus_3, %lo_bc, %hi_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x0>, <8x256xf32, 0x0> -> <8x256xf32, 256x1>
    %scaled = migraphx.mul %clamped, %inv6_bc {} : <8x256xf32, 256x1>, <8x256xf32, 0x0> -> <8x256xf32, 256x1>
    %out = migraphx.mul %x, %scaled {} : <8x256xf32, 256x1>, <8x256xf32, 256x1> -> <8x256xf32, 256x1>
    return %out : !migraphx.shaped<8x256xf32, 256x1>
  }
}
