// RUN: rocmlir-gen -fut elem_where --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut elem_where --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut elem_where %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t i32 12

  // greater → convert(si32→si8) → where
  func.func @elem_where(%arg0: !migraphx.shaped<3x4xsi32, 4x1>, %arg1: !migraphx.shaped<3x4xsi32, 4x1>, %arg2: !migraphx.shaped<3x4xf32, 4x1>, %arg3: !migraphx.shaped<3x4xf32, 4x1>) -> !migraphx.shaped<3x4xf32, 4x1> attributes {rock.kernel} {
    %0 = migraphx.greater %arg0, %arg1 : <3x4xsi32, 4x1>, <3x4xsi32, 4x1> -> <3x4xsi32, 4x1>
    %1 = migraphx.convert %0 {target_type = 0 : i64} : <3x4xsi32, 4x1> to <3x4xsi8, 4x1>
    %2 = migraphx.where %1, %arg2, %arg3 : <3x4xsi8, 4x1>, <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    return %2 : !migraphx.shaped<3x4xf32, 4x1>
  }
}
