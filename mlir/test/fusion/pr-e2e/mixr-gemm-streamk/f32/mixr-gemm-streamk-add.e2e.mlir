// Stream-K output-fusion regularization (numeric): `add gemmOut, bias`.
//
// The dot pins a stream-K perf_config (streamKMultiple = 1 in the v5 field
// before the knob block; splitK stays 1). On targets whose num_cu leaves a
// ragged tail for this grid, rock-stream-k-decompose splits the gemm into
// data-parallel waves plus a split-K remainder that atomic_adds `splitK`
// partial products into the same tile. The remainder cell therefore divides
// the fused `bias` by `splitK` (the regularization moved into StreamKDecompose)
// so the epilogue stays correct; the data-parallel waves add `bias` unscaled.
// On targets that fall back to plain data-parallel the result is still correct.
// Either way the clone verifier checks the GPU result against a CPU reference,
// so a wrong per-remainder-cell scaling would fail here.
//
// The GEMM is sized (640x256x448, 64x64x64 tiles -> a 10x7 tile grid) so the
// grid is large enough to hybrid-decompose on typical num_cu.

// RUN: rocmlir-gen -fut dot_streamk_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut dot_streamk_add --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base

  func.func @dot_streamk_add(%arg0: !migraphx.shaped<1x640x256xf32, 163840x256x1>, %arg1: !migraphx.shaped<1x256x448xf32, 114688x448x1>, %arg2: !migraphx.shaped<1x640x448xf32, 286720x448x1>) -> !migraphx.shaped<1x640x448xf32, 286720x448x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 {perf_config="gemm:v5:64,64,64,1,1,4,16,1,2,0,0,1,-1,-1,-1,-1,-1,-1"} : <1x640x256xf32, 163840x256x1>, <1x256x448xf32, 114688x448x1> -> <1x640x448xf32, 286720x448x1>
    %1 = migraphx.add %arg2, %0 {} : <1x640x448xf32, 286720x448x1>, <1x640x448xf32, 286720x448x1> -> <1x640x448xf32, 286720x448x1>
    return %1 : !migraphx.shaped<1x640x448xf32, 286720x448x1>
  }
}
