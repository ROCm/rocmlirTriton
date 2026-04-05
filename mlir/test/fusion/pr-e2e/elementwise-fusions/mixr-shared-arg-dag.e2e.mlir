// RUN: rocmlir-gen -fut shared_arg_dag --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut shared_arg_dag --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut shared_arg_dag %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 12

  // Two independent fusion roots (o0, o1) that share a kernel arg (%arg0),
  // each followed by a transform, merging into a downstream fusion (final).
  // Tests the DAG case where two earliest fusions feed into the same
  // downstream op through transforms.
  //
  //   o0 = add(arg0, arg1) : 3x4  ─→ reshape(3x4→12) ─→ final = add(_, _) : 12
  //   o1 = add(arg0, arg2) : 3x4  ─→ reshape(3x4→12) ─↗
  func.func @shared_arg_dag(
    %arg0: !migraphx.shaped<3x4xf32, 4x1>,
    %arg1: !migraphx.shaped<3x4xf32, 4x1>,
    %arg2: !migraphx.shaped<3x4xf32, 4x1>
  ) -> !migraphx.shaped<12xf32, 1> attributes {rock.kernel} {
    %o0 = migraphx.add %arg0, %arg1 {} : <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    %o1 = migraphx.add %arg0, %arg2 {} : <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    %r0 = migraphx.reshape %o0 {dims = [12]} : <3x4xf32, 4x1> -> <12xf32, 1>
    %r1 = migraphx.reshape %o1 {dims = [12]} : <3x4xf32, 4x1> -> <12xf32, 1>
    %final = migraphx.add %r0, %r1 {} : <12xf32, 1>, <12xf32, 1> -> <12xf32, 1>
    return %final : !migraphx.shaped<12xf32, 1>
  }
}
