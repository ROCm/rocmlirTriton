// RUN: rocmlir-gen -fut rmsnorm_sq_reduce --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut rmsnorm_sq_reduce --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut rmsnorm_sq_reduce %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefix=EMITKEY
// UNSUPPORTED: true
// TODO: Requires reduction support (LowerReduce)

module {
  // CLONE: [1 1 1]
  // CLONE-NEXT: Unranked Memref base
  // EMITKEY: elementwise -t f32 1024

  // RMSNorm kernel 1: x^2 → reduce_sum (elementwise + reduction at end)
  func.func @rmsnorm_sq_reduce(%x: !migraphx.shaped<4x256xf32, 256x1>) -> !migraphx.shaped<4x1xf32, 1x1> attributes {rock.kernel} {
    %x_sq = migraphx.mul %x, %x {} : <4x256xf32, 256x1>, <4x256xf32, 256x1> -> <4x256xf32, 256x1>
    %sum_sq = migraphx.reduce_sum %x_sq {axes = [1 : i64]} : <4x256xf32, 256x1> -> <4x1xf32, 1x1>
    return %sum_sq : !migraphx.shaped<4x1xf32, 1x1>
  }
}
