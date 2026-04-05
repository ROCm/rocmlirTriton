// RUN: rocmlir-gen -fut elem_clip --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut elem_clip --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut elem_clip %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 12

  func.func @elem_clip(%arg0: !migraphx.shaped<3x4xf32, 4x1>, %arg1: !migraphx.shaped<3x4xf32, 4x1>) -> !migraphx.shaped<3x4xf32, 4x1> attributes {rock.kernel} {
    %min_lit = migraphx.literal(dense<0.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %max_lit = migraphx.literal(dense<6.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %min_bc = migraphx.multibroadcast %min_lit {out_dyn_dims = [], out_lens = [3, 4]} : <1xf32, 0> -> <3x4xf32, 0x0>
    %max_bc = migraphx.multibroadcast %max_lit {out_dyn_dims = [], out_lens = [3, 4]} : <1xf32, 0> -> <3x4xf32, 0x0>
    %0 = migraphx.add %arg0, %arg1 {} : <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    %1 = migraphx.clip %0, %min_bc, %max_bc : <3x4xf32, 4x1>, <3x4xf32, 0x0>, <3x4xf32, 0x0> -> <3x4xf32, 4x1>
    return %1 : !migraphx.shaped<3x4xf32, 4x1>
  }
}
