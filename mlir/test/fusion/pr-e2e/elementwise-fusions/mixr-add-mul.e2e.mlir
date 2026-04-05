// RUN: rocmlir-gen -fut dot_add_mul --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_add_mul --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut dot_add_mul %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 400

  func.func @dot_add_mul(%arg0: !migraphx.shaped<20x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<20x5x4xf32, 20x4x1>, %arg2: !migraphx.shaped<20x5x4xf32, 20x4x1>) -> !migraphx.shaped<20x5x4xf32, 20x4x1> attributes {rock.kernel} {
    %1 = migraphx.add %arg0, %arg1 {} : <20x5x4xf32, 20x4x1>, <20x5x4xf32, 20x4x1> -> <20x5x4xf32, 20x4x1>
    %2 = migraphx.mul %1, %arg2 {} : <20x5x4xf32, 20x4x1>, <20x5x4xf32, 20x4x1> -> <20x5x4xf32, 20x4x1>
    return %2 : !migraphx.shaped<20x5x4xf32, 20x4x1>
  }
}
