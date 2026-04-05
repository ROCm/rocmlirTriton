// RUN: rocmlir-gen -fut add_slice --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut add_slice --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut add_slice %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 24

  func.func @add_slice(%arg0: !migraphx.shaped<3x8xf32, 8x1>, %arg1: !migraphx.shaped<3x4xf32, 4x1>, %arg2: !migraphx.shaped<3x2xf32, 2x1>) -> !migraphx.shaped<3x2xf32, 2x1> attributes {rock.kernel} {
    %0 = migraphx.slice %arg0 {axes = [1], ends = [4], starts = [0]} : <3x8xf32, 8x1> -> <3x4xf32, 8x1>
    %1 = migraphx.add %0, %arg1 {} : <3x4xf32, 8x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    %2 = migraphx.slice %1 {axes = [1], ends = [2], starts = [0]} : <3x4xf32, 4x1> -> <3x2xf32, 4x1>
    %3 = migraphx.add %2, %arg2 {} : <3x2xf32, 4x1>, <3x2xf32, 2x1> -> <3x2xf32, 2x1>
    return %3 : !migraphx.shaped<3x2xf32, 2x1>
  }
}
