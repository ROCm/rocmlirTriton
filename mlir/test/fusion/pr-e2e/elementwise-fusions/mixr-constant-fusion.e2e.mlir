// RUN: rocmlir-gen -fut const_fusion --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut const_fusion --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut const_fusion %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 12

  // A fusion whose inputs are both constants, followed by a transform and
  // another fusion with a kernel arg. Tests that constant-only fusions
  // are correctly identified as chain roots by RegularizeOutput.
  //
  //   const0 + const1 → reshape(3x4 → 12) → add(_, arg0)
  func.func @const_fusion(%arg0: !migraphx.shaped<12xf32, 1>) -> !migraphx.shaped<12xf32, 1> attributes {rock.kernel} {
    %c0 = migraphx.literal(dense<1.000000e+00> : tensor<3x4xf32>) : <3x4xf32, 4x1>
    %c1 = migraphx.literal(dense<2.000000e+00> : tensor<3x4xf32>) : <3x4xf32, 4x1>
    %sum = migraphx.add %c0, %c1 {} : <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    %flat = migraphx.reshape %sum {dims = [12]} : <3x4xf32, 4x1> -> <12xf32, 1>
    %out = migraphx.add %flat, %arg0 {} : <12xf32, 1>, <12xf32, 1> -> <12xf32, 1>
    return %out : !migraphx.shaped<12xf32, 1>
  }
}
