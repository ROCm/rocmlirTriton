// RUN: rocmlir-gen -fut complex_parallel --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut complex_parallel --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut complex_parallel %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY

module {
  // CLONE: [1 1 1]
  // CLONE: [1 1 1]
  // EMITKEY: elementwise -t f32 12

  // Two parallel subgraphs with chained fusions and transforms between them.
  //
  // Subgraph A (12 elements): transpose → add → reshape → mul → reshape → add → reshape(output)
  // Subgraph B (10 elements): reshape+broadcast → add → transpose → mul → reshape → add → reshape(output)
  func.func @complex_parallel(
    %arg0: !migraphx.shaped<4x3xf32, 3x1>,
    %arg1: !migraphx.shaped<3x4xf32, 4x1>,
    %arg2: !migraphx.shaped<12xf32, 1>,
    %arg3: !migraphx.shaped<4x3xf32, 3x1>,
    %arg4: !migraphx.shaped<10xf32, 1>,
    %arg5: !migraphx.shaped<5xf32, 1>,
    %arg6: !migraphx.shaped<5x2xf32, 2x1>,
    %arg7: !migraphx.shaped<10xf32, 1>
  ) -> (!migraphx.shaped<12xf32, 1>, !migraphx.shaped<2x5xf32, 5x1>) attributes {rock.kernel} {

    // --- Subgraph A ---
    // Input transform: transpose %arg0 from 4x3 to 3x4
    %t0 = migraphx.transpose %arg0 {permutation = [1, 0]} : <4x3xf32, 3x1> -> <3x4xf32, 1x3>
    // Fusion 1: add
    %a1 = migraphx.add %t0, %arg1 {} : <3x4xf32, 1x3>, <3x4xf32, 4x1> -> <3x4xf32, 4x1>
    // Between-fusion transform: reshape 3x4 → 12
    %r1 = migraphx.reshape %a1 {dims = [12]} : <3x4xf32, 4x1> -> <12xf32, 1>
    // Fusion 2: mul
    %m1 = migraphx.mul %r1, %arg2 {} : <12xf32, 1>, <12xf32, 1> -> <12xf32, 1>
    // Between-fusion transform: reshape 12 → 4x3
    %r2 = migraphx.reshape %m1 {dims = [4, 3]} : <12xf32, 1> -> <4x3xf32, 3x1>
    // Fusion 3: add
    %a2 = migraphx.add %r2, %arg3 {} : <4x3xf32, 3x1>, <4x3xf32, 3x1> -> <4x3xf32, 3x1>
    // Output transform: reshape 4x3 → 12
    %outA = migraphx.reshape %a2 {dims = [12]} : <4x3xf32, 3x1> -> <12xf32, 1>

    // --- Subgraph B ---
    // Input transform: reshape %arg4 from 10 → 2x5
    %r3 = migraphx.reshape %arg4 {dims = [2, 5]} : <10xf32, 1> -> <2x5xf32, 5x1>
    // Input transform: broadcast %arg5 from 5 → 2x5
    %b1 = migraphx.multibroadcast %arg5 {out_dyn_dims = [], out_lens = [2, 5]} : <5xf32, 1> -> <2x5xf32, 0x1>
    // Fusion 1: add
    %a3 = migraphx.add %r3, %b1 {} : <2x5xf32, 5x1>, <2x5xf32, 0x1> -> <2x5xf32, 5x1>
    // Between-fusion transform: transpose 2x5 → 5x2
    %t1 = migraphx.transpose %a3 {permutation = [1, 0]} : <2x5xf32, 5x1> -> <5x2xf32, 1x5>
    // Fusion 2: mul
    %m2 = migraphx.mul %t1, %arg6 {} : <5x2xf32, 1x5>, <5x2xf32, 2x1> -> <5x2xf32, 2x1>
    // Between-fusion transform: reshape 5x2 → 10
    %r4 = migraphx.reshape %m2 {dims = [10]} : <5x2xf32, 2x1> -> <10xf32, 1>
    // Fusion 3: add
    %a4 = migraphx.add %r4, %arg7 {} : <10xf32, 1>, <10xf32, 1> -> <10xf32, 1>
    // Output transform: reshape 10 → 2x5
    %outB = migraphx.reshape %a4 {dims = [2, 5]} : <10xf32, 1> -> <2x5xf32, 5x1>

    return %outA, %outB : !migraphx.shaped<12xf32, 1>, !migraphx.shaped<2x5xf32, 5x1>
  }
}
