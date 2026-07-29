// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -RMS_threshold=1e-2 -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

// CLONE: [1 1 1]

// The dot output is transposed on its last two dims and returned into an
// M-contiguous buffer (strides 2048x64x1), so the GEMM result is physically
// stored transposed. The migraphx -> rock lowering must fold the transpose
// into the GEMM as oTransposed (-transO true).
// EMITKEY: -t f16 -out_datatype f16 -transA false -transB false -transO true -g 2 -m 64 -n 32 -k 16

module {
  func.func @test(%arg0: !migraphx.shaped<2x64x16xf16, 1024x16x1>, %arg1: !migraphx.shaped<2x16x32xf16, 512x32x1>) -> !migraphx.shaped<2x32x64xf16, 2048x64x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <2x64x16xf16, 1024x16x1>, <2x16x32xf16, 512x32x1> -> <2x64x32xf16, 2048x32x1>
    %1 = migraphx.transpose %0 {permutation = [0, 2, 1]} : <2x64x32xf16, 2048x32x1> -> <2x32x64xf16, 2048x64x1>
    return %1 : !migraphx.shaped<2x32x64xf16, 2048x64x1>
  }
}
