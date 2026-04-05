// RUN: rocmlir-gen -fut add_transpose --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut add_transpose --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut add_transpose %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 12

  func.func @add_transpose(%arg0: !migraphx.shaped<3x4xf32, 4x1>, %arg1: !migraphx.shaped<4x3xf32, 3x1>) -> !migraphx.shaped<3x4xf32, 4x1> attributes {rock.kernel} {
    %0 = migraphx.transpose %arg1 {permutation = [1, 0]} : <4x3xf32, 3x1> -> <3x4xf32, 1x3>
    %1 = migraphx.add %arg0, %0 {} : <3x4xf32, 4x1>, <3x4xf32, 1x3> -> <3x4xf32, 4x1>
    return %1 : !migraphx.shaped<3x4xf32, 4x1>
  }
}
