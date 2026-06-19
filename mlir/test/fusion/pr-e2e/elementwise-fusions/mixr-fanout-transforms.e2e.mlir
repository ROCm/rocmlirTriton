// RUN: rocmlir-gen -fut fan_out --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut fan_out --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut fan_out %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE: [1 1 1]
  // EMITKEY: elementwise -t f32 24

  // Fan-out: one fusion result feeds two incompatible transforms (broadcast
  // vs slice), each consumed by an independent downstream fusion.
  // RegularizeElementwise must clone the upstream fusion for each branch.
  func.func @fan_out(
    %arg0: !migraphx.shaped<3x4xf32, 4x1>,
    %arg1: !migraphx.shaped<3x4xf32, 4x1>,
    %arg2: !migraphx.shaped<2x3x4xf32, 12x4x1>,
    %arg3: !migraphx.shaped<2x4xf32, 4x1>
  ) -> (!migraphx.shaped<2x3x4xf32, 12x4x1>, !migraphx.shaped<2x4xf32, 4x1>) attributes {rock.kernel} {
    %out = migraphx.add %arg0, %arg1 {} : <3x4xf32, 4x1>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>

    // Branch A: broadcast 3x4 → 2x3x4
    %t0 = migraphx.multibroadcast %out {out_dyn_dims = [], out_lens = [2, 3, 4]} : <3x4xf32, 4x1> -> <2x3x4xf32, 0x4x1>
    %result0 = migraphx.add %t0, %arg2 {} : <2x3x4xf32, 0x4x1>, <2x3x4xf32, 12x4x1> -> <2x3x4xf32, 12x4x1>

    // Branch B: slice dim0 [0,2) → 2x4
    %t1 = migraphx.slice %out {axes = [0], ends = [2], starts = [0]} : <3x4xf32, 4x1> -> <2x4xf32, 4x1>
    %result1 = migraphx.mul %t1, %arg3 {} : <2x4xf32, 4x1>, <2x4xf32, 4x1> -> <2x4xf32, 4x1>

    return %result0, %result1 : !migraphx.shaped<2x3x4xf32, 12x4x1>, !migraphx.shaped<2x4xf32, 4x1>
  }
}
