// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=EMITKEY
// RUN: rocmlir-gen --clone-harness -arch %arch -fut test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -ph -relDiff_threshold 0.09 -absDiff_threshold 1 -RMS_threshold 0.05 -verifier clone -fut test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// EMITKEY: convfp16 -F 1 -f N01GC -I 01NGC -O N01GC -n 1 -c 128 -H 80 -W 80 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1

module {
func.func @test(%arg0: !migraphx.shaped<1x256x80x80xf16, 1638400x1x20480x256>, %arg1: !migraphx.shaped<128x128x3x3xf16, 1152x1x384x128>) -> !migraphx.shaped<1x128x80x80xf16, 819200x1x10240x128>  attributes {rock.kernel} {
    %0 = migraphx.slice %arg0 {axes = [1], ends = [256], starts = [128]} : <1x256x80x80xf16, 1638400x1x20480x256> -> <1x128x80x80xf16, 1638400x1x20480x256>
    %1 = migraphx.sigmoid %0 : <1x128x80x80xf16, 1638400x1x20480x256> -> <1x128x80x80xf16, 819200x1x10240x128>
    %2 = migraphx.mul %0, %1 : <1x128x80x80xf16, 1638400x1x20480x256>, <1x128x80x80xf16, 819200x1x10240x128> -> <1x128x80x80xf16, 819200x1x10240x128>
    %3 = migraphx.convolution %2, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1], perf_config = "gemm:v1:64,64,8,1,1,4,16,1,2,0,0"} : <1x128x80x80xf16, 819200x1x10240x128>, <128x128x3x3xf16, 1152x1x384x128> -> <1x128x80x80xf16, 819200x1x10240x128>
    return %3 : !migraphx.shaped<1x128x80x80xf16, 819200x1x10240x128>
  }
}
