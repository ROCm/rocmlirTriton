// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -ph -verifier clone -fut test - | rocmlir-driver -c --mlir-print-ir-after=tritongpu-coalesce -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTORIZATION

// CLONE: [1 1 1]

// EMITKEY: -t f32 -transA false -transB true -transC false -transO false -g 1 -m 32 -n 64 -k 16 -gemmO 8

// VECTORIZATION-DAG: #[[COALESCED:.*]] = #ttg.blocked<{sizePerThread = [1, 4]
// VECTORIZATION: tt.load {{.*}}#[[COALESCED]]>

module {
  func.func @test(%arg0: !migraphx.shaped<1x32x32xf32, 1024x1x32>, %arg1: !migraphx.shaped<1x16x64xf32, 1024x1x16>, %arg2: !migraphx.shaped<1x8x64xf32, 512x1x8>) -> !migraphx.shaped<1x32x8xf32, 256x8x1> attributes {rock.kernel} {
    %sliced0 = migraphx.slice %arg0 {axes = [1], ends = [16], starts = [0]} : <1x32x32xf32, 1024x1x32> -> <1x16x32xf32, 1024x1x32>
    %trans0 = migraphx.transpose %sliced0 {permutation = [0, 2, 1]} : <1x16x32xf32, 1024x1x32> -> <1x32x16xf32, 1024x32x1>
    %0 = migraphx.dot %trans0, %arg1: !migraphx.shaped<1x32x16xf32, 1024x32x1>, !migraphx.shaped<1x16x64xf32, 1024x1x16> -> !migraphx.shaped<1x32x64xf32, 2048x64x1>
    %trans2 = migraphx.transpose %arg2 {permutation = [0, 2, 1]} : <1x8x64xf32, 512x1x8> -> <1x64x8xf32, 512x8x1>
    %2 = migraphx.dot %0, %trans2: !migraphx.shaped<1x32x64xf32, 2048x64x1>, !migraphx.shaped<1x64x8xf32, 512x8x1> -> !migraphx.shaped<1x32x8xf32, 256x8x1>
    return %2 : !migraphx.shaped<1x32x8xf32, 256x8x1>
  }
}
