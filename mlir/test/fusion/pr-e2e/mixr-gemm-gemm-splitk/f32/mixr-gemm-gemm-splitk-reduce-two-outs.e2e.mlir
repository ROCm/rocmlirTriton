// RUN: rocmlir-gen -fut mlir_gemm_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_gemm_gemm --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
// Guard: the second GEMM must really be lowered with split-k. A positional
// perf_config once silently decoded splitKFactor as 1 here, which turned this
// test into a plain gemm+gemm run.
// RUN: rocmlir-gen -fut mlir_gemm_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel | rocmlir-opt --rock-affix-params | FileCheck %s --check-prefix=SPLITK
// SPLITK: splitKFactor = 4
module {
  func.func @mlir_gemm_gemm(%arg0: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg1: !migraphx.shaped<1x64x64xf32, 4096x64x1>, %arg2: !migraphx.shaped<1x64x64xf32, 4096x64x1>) -> (!migraphx.shaped<1x64x64xf32, 4096x64x1>, !migraphx.shaped<1x64x1xf32, 64x1x1>) attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : !migraphx.shaped<1x64x64xf32, 4096x64x1>, !migraphx.shaped<1x64x64xf32, 4096x64x1> -> !migraphx.shaped<1x64x64xf32, 4096x64x1>
    %1 = migraphx.dot %0, %arg2 {perf_config="attn:mPerBlockG0=128,nPerBlockG0=64,kPerBlock=32,numWaves=4,matrixInstrNonkdim=0,splitKFactor=4"} : !migraphx.shaped<1x64x64xf32, 4096x64x1>, !migraphx.shaped<1x64x64xf32, 4096x64x1> -> !migraphx.shaped<1x64x64xf32, 4096x64x1>
    %2 = migraphx.reduce_sum %1 {axes = [2]} : <1x64x64xf32, 4096x64x1> -> <1x64x1xf32, 64x1x1>
    return %1, %2 : !migraphx.shaped<1x64x64xf32, 4096x64x1>, !migraphx.shaped<1x64x1xf32, 64x1x1>
  }
}
