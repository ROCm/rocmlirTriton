// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

// CLONE: [1 1 1]

// EMITKEY: -t f16 -out_datatype f16 -transA true -transB true -transO false -g 2 -m 4096 -n 640 -k 320

module {
  func.func @test(%arg0: !migraphx.shaped<2x4096x320xf16, 1310720x1x4096>, %arg1: !migraphx.shaped<2x320x640xf16, 204800x1x320>, %arg2: !migraphx.shaped<2x64x10x64x64xf16, 0x10x1x40960x640>) -> !migraphx.shaped<2x64x10x64x64xf16, 2621440x10x1x40960x640> attributes {rock.kernel} {
    %2 = migraphx.dot %arg0, %arg1 : <2x4096x320xf16, 1310720x1x4096>, <2x320x640xf16, 204800x1x320> -> <2x4096x640xf16, 2621440x640x1>
    %3 = migraphx.reshape %2 {dims = [2, 64, 64, 64, 10]} : <2x4096x640xf16, 2621440x640x1> -> <2x64x64x64x10xf16, 2621440x40960x640x10x1>
    %4 = migraphx.transpose %3 {permutation = [0, 3, 4, 1, 2]} : <2x64x64x64x10xf16, 2621440x40960x640x10x1> -> <2x64x10x64x64xf16, 2621440x10x40960x640x1>
    %5 = migraphx.add %4, %arg2 : <2x64x10x64x64xf16, 2621440x10x40960x640x1>, <2x64x10x64x64xf16, 0x10x1x40960x640> -> <2x64x10x64x64xf16, 2621440x10x1x40960x640>
    return %5 : !migraphx.shaped<2x64x10x64x64xf16, 2621440x10x1x40960x640>
  }
}
