// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -RMS_threshold=1e-2 -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

// CLONE: [1 1 1]

// The dot output is transposed on its last two dims before being consumed by
// the elementwise add, so the migraphx -> rock lowering must fold the transpose
// into the GEMM as oTransposed (-transO true).
// EMITKEY: -t f16 -out_datatype f16 -transA false -transB false -transO true -g 2 -m 4096 -n 640 -k 320

module {
  func.func @test(%arg0: !migraphx.shaped<2x4096x320xf16, 1310720x320x1>, %arg1: !migraphx.shaped<2x320x640xf16, 204800x640x1>, %arg2: !migraphx.shaped<2x640x4096xf16, 2621440x4096x1>) -> !migraphx.shaped<2x640x4096xf16, 2621440x4096x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1 : <2x4096x320xf16, 1310720x320x1>, <2x320x640xf16, 204800x640x1> -> <2x4096x640xf16, 2621440x640x1>
    %1 = migraphx.transpose %0 {permutation = [0, 2, 1]} : <2x4096x640xf16, 2621440x640x1> -> <2x640x4096xf16, 2621440x1x640>
    %2 = migraphx.add %1, %arg2 : <2x640x4096xf16, 2621440x1x640>, <2x640x4096xf16, 2621440x4096x1> -> <2x640x4096xf16, 2621440x4096x1>
    return %2 : !migraphx.shaped<2x640x4096xf16, 2621440x4096x1>
  }
}
