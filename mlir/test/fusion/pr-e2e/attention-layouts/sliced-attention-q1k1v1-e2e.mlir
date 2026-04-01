// TODO(rocmlirTriton): error: 'arith.mulf' op requires the same type for all operands and results
// UNSUPPORTED: true
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -RMS_threshold=1e-2 -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

// CLONE: [1 1 1]

// EMITKEY: -t f16 -transQ true -transK true -transV true -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 32 -seq_len_k 64 -head_dim_qk 16 -head_dim_v 8

module {
  func.func @test(%arg0: !migraphx.shaped<1x32x32xf16, 1024x1x32>, %arg1: !migraphx.shaped<1x16x64xf16, 1024x1x16>, %arg2: !migraphx.shaped<1x64x8xf16, 512x1x64>) -> !migraphx.shaped<1x32x8xf16, 256x8x1>  attributes {rock.kernel} {
    %sliced0 = migraphx.slice %arg0 {axes = [2], ends = [16], starts = [0]} : <1x32x32xf16, 1024x1x32> -> <1x32x16xf16, 1024x1x32>
    %0 = migraphx.dot %sliced0, %arg1: !migraphx.shaped<1x32x16xf16, 1024x1x32>, !migraphx.shaped<1x16x64xf16, 1024x1x16> -> !migraphx.shaped<1x32x64xf16, 2048x64x1>
    %1 = migraphx.softmax %0{axis = 2 : i64} : !migraphx.shaped<1x32x64xf16, 2048x64x1> -> !migraphx.shaped<1x32x64xf16, 2048x64x1>
    %2 = migraphx.dot %1, %arg2: !migraphx.shaped<1x32x64xf16, 2048x64x1>, !migraphx.shaped<1x64x8xf16, 512x1x64> -> !migraphx.shaped<1x32x8xf16, 256x8x1>
    return %2 : !migraphx.shaped<1x32x8xf16, 256x8x1>
  }
}
