// RUN: rocmlir-gen -fut long_chain --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut long_chain --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut long_chain %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 12

  // Three fusions separated by two transforms:
  //   add(3x4) → reshape(12) → mul(12) → reshape(4x3) → add(4x3)
  // Each iteration of RegularizeElementwise pushes one transform up, exposing
  // the next. Requires two full iterations to converge.
  func.func @long_chain(
    %arg0: !migraphx.shaped<3x4xf32, 4x1>,
    %arg1: !migraphx.shaped<3x4xf32, 4x1>,
    %arg2: !migraphx.shaped<12xf32, 1>,
    %arg3: !migraphx.shaped<4x3xf32, 3x1>
  ) -> !migraphx.shaped<4x3xf32, 3x1> attributes {rock.kernel} {
    %f1 = migraphx.add %arg0, %arg1 {} : <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    %t1 = migraphx.reshape %f1 {dims = [12]} : <3x4xf32, 4x1> -> <12xf32, 1>
    %f2 = migraphx.mul %t1, %arg2 {} : <12xf32, 1>, <12xf32, 1> -> <12xf32, 1>
    %t2 = migraphx.reshape %f2 {dims = [4, 3]} : <12xf32, 1> -> <4x3xf32, 3x1>
    %f3 = migraphx.add %t2, %arg3 {} : <4x3xf32, 3x1>, <4x3xf32, 3x1> -> <4x3xf32, 3x1>
    return %f3 : !migraphx.shaped<4x3xf32, 3x1>
  }
}
