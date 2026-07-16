// Stream-K output-fusion regularization (numeric), f16: `add gemmOut, bias`.
//
// f16 counterpart of mixr-gemm-streamk/f32/mixr-gemm-streamk-add. The split-K
// remainder atomic_adds into an f16 output (guarded by arch_support_atomic_add_f16
// in lit.local.cfg), so the fused `bias` must be divided by `splitK` only in the
// remainder cell. The clone verifier checks the GPU result against a CPU reference.

// RUN: rocmlir-gen -fut dot_streamk_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_streamk_add --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base

  func.func @dot_streamk_add(%arg0: !migraphx.shaped<1x640x256xf16, 163840x256x1>, %arg1: !migraphx.shaped<1x256x448xf16, 114688x448x1>, %arg2: !migraphx.shaped<1x640x448xf16, 286720x448x1>) -> !migraphx.shaped<1x640x448xf16, 286720x448x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 {perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1"} : <1x640x256xf16, 163840x256x1>, <1x256x448xf16, 114688x448x1> -> <1x640x448xf16, 286720x448x1>
    %1 = migraphx.add %arg2, %0 {} : <1x640x448xf16, 286720x448x1>, <1x640x448xf16, 286720x448x1> -> <1x640x448xf16, 286720x448x1>
    return %1 : !migraphx.shaped<1x640x448xf16, 286720x448x1>
  }
}
