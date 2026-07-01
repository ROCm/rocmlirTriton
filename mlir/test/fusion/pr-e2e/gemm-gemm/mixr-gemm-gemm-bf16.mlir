// RUN: rocmlir-gen -fut mlir_gemm_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_gemm_gemm --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func private @mlir_gemm_gemm(%arg0: !migraphx.shaped<1x64x64xbf16, 4096x64x1>, %arg1: !migraphx.shaped<1x64x64xbf16, 4096x64x1>, %arg2: !migraphx.shaped<1x64x64xbf16, 4096x64x1>) -> (!migraphx.shaped<1x64x64xbf16, 4096x64x1>) attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1: !migraphx.shaped<1x64x64xbf16, 4096x64x1>, !migraphx.shaped<1x64x64xbf16, 4096x64x1> -> !migraphx.shaped<1x64x64xbf16, 4096x64x1>
    %1 = migraphx.dot %0, %arg2: !migraphx.shaped<1x64x64xbf16, 4096x64x1>, !migraphx.shaped<1x64x64xbf16, 4096x64x1> -> !migraphx.shaped<1x64x64xbf16, 4096x64x1>
    return %1 : !migraphx.shaped<1x64x64xbf16, 4096x64x1>
  }
}
