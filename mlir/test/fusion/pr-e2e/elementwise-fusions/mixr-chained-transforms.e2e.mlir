// RUN: rocmlir-gen -fut chained_t --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut chained_t --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut chained_t %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 12

  // Two consecutive transforms between fusions: reshape 3x4→12 then 12→2x6.
  // Forces two iterations of RegularizeElementwise.
  func.func @chained_t(
    %arg0: !migraphx.shaped<3x4xf32, 4x1>,
    %arg1: !migraphx.shaped<3x4xf32, 4x1>,
    %arg2: !migraphx.shaped<2x6xf32, 6x1>
  ) -> !migraphx.shaped<2x6xf32, 6x1> attributes {rock.kernel} {
    %out = migraphx.add %arg0, %arg1 {} : <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    %r1 = migraphx.reshape %out {dims = [12]} : <3x4xf32, 4x1> -> <12xf32, 1>
    %r2 = migraphx.reshape %r1 {dims = [2, 6]} : <12xf32, 1> -> <2x6xf32, 6x1>
    %result = migraphx.mul %r2, %arg2 {} : <2x6xf32, 6x1>, <2x6xf32, 6x1> -> <2x6xf32, 6x1>
    return %result : !migraphx.shaped<2x6xf32, 6x1>
  }
}
