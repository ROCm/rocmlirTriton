// TODO(rocmlirTriton): error: failed to legalize operation 'arith.negf' that was explicitly marked illegal
// UNSUPPORTED: true
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -ph -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE

// CLONE: [1 1 1]

// EMITKEY: convfp16 -F 1 -f N01GC -I N01GC -O NGC01 -n 2 -c 16 -H 160 -W 160 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1

module {
  func.func @test(%arg0: !migraphx.shaped<2x160x160x1xf16, 25600x160x1x25600>, %arg1: !migraphx.shaped<16x16x3x3xf16, 144x1x48x16>, %arg2: !migraphx.shaped<2x16x160x160xf16, 0x1x0x0>) -> !migraphx.shaped<2x16x160x160xf16, 409600x1x2560x16>  attributes {rock.kernel} {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [2, 160, 160, 16]} : <2x160x160x1xf16, 25600x160x1x25600> -> <2x160x160x16xf16, 25600x160x1x25600>
    %transposed = migraphx.transpose %0 {permutation = [0, 3, 1, 2]} : <2x160x160x16xf16, 25600x160x1x25600> -> <2x16x160x160xf16, 25600x25600x160x1>
    %1 = migraphx.convolution %transposed, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1]} : <2x16x160x160xf16, 25600x25600x160x1>, <16x16x3x3xf16, 144x1x48x16> -> <2x16x160x160xf16, 409600x1x2560x16>
    %3 = migraphx.add %1, %arg2 : <2x16x160x160xf16, 409600x1x2560x16>, <2x16x160x160xf16, 0x1x0x0> -> <2x16x160x160xf16, 409600x1x2560x16>
    %4 = migraphx.sigmoid %3 : <2x16x160x160xf16, 409600x1x2560x16> -> <2x16x160x160xf16, 409600x1x2560x16>
    %5 = migraphx.mul %3, %4 : <2x16x160x160xf16, 409600x1x2560x16>, <2x16x160x160xf16, 409600x1x2560x16> -> <2x16x160x160xf16, 409600x1x2560x16>
    return %5 : !migraphx.shaped<2x16x160x160xf16, 409600x1x2560x16>
  }
}
