// RUN: rocmlir-gen -fut elem_convert --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut elem_convert --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut elem_convert %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f16 12

  // f16 inputs → convert to f32 → add in f32 → return f32
  func.func @elem_convert(%arg0: !migraphx.shaped<3x4xf16, 4x1>, %arg1: !migraphx.shaped<3x4xf16, 4x1>) -> !migraphx.shaped<3x4xf32, 4x1> attributes {rock.kernel} {
    %0 = migraphx.convert %arg0 {target_type = 2 : i64} : <3x4xf16, 4x1> to <3x4xf32, 4x1>
    %1 = migraphx.convert %arg1 {target_type = 2 : i64} : <3x4xf16, 4x1> to <3x4xf32, 4x1>
    %2 = migraphx.add %0, %1 {} : <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    return %2 : !migraphx.shaped<3x4xf32, 4x1>
  }
}
