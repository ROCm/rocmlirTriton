// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<1x256x6144xf32, 1572864x6144x1>, %arg1: !migraphx.shaped<1x32x256x256xf32, 32x1x8192x32>) -> !migraphx.shaped<1x32x256x64xf32, 524288x16384x64x1> attributes {rock.kernel} {
    %0 = migraphx.slice %arg0 {axes = [2], ends = [2048], starts = [0]} : <1x256x6144xf32, 1572864x6144x1> -> <1x256x2048xf32, 1572864x6144x1>
    %1 = migraphx.reshape %0 {dims = [1, 256, 32, 64]} : <1x256x2048xf32, 1572864x6144x1> -> <1x256x32x64xf32, 524288x2048x64x1>
    %2 = migraphx.transpose %1 {permutation = [0, 2, 1, 3]} : <1x256x32x64xf32, 524288x2048x64x1> -> <1x32x256x64xf32, 524288x64x2048x1>
    %3 = migraphx.slice %arg0 {axes = [2], ends = [4096], starts = [2048]} : <1x256x6144xf32, 1572864x6144x1> -> <1x256x2048xf32, 1572864x6144x1>
    %4 = migraphx.reshape %3 {dims = [1, 256, 32, 64]} : <1x256x2048xf32, 1572864x6144x1> -> <1x256x32x64xf32, 524288x2048x64x1>
    %5 = migraphx.transpose %4 {permutation = [0, 2, 3, 1]} : <1x256x32x64xf32, 524288x2048x64x1> -> <1x32x64x256xf32, 524288x64x1x2048>
    %6 = migraphx.slice %arg0 {axes = [2], ends = [6144], starts = [4096]} : <1x256x6144xf32, 1572864x6144x1> -> <1x256x2048xf32, 1572864x6144x1>
    %7 = migraphx.reshape %6 {dims = [1, 256, 32, 64]} : <1x256x2048xf32, 1572864x6144x1> -> <1x256x32x64xf32, 524288x2048x64x1>
    %8 = migraphx.transpose %7 {permutation = [0, 2, 1, 3]} : <1x256x32x64xf32, 524288x2048x64x1> -> <1x32x256x64xf32, 524288x64x2048x1>
    %9 = migraphx.dot %2, %5 : <1x32x256x64xf32, 524288x64x2048x1>, <1x32x64x256xf32, 524288x64x1x2048> -> <1x32x256x256xf32, 2097152x65536x256x1>
    %10 = migraphx.add %9, %arg1 : <1x32x256x256xf32, 2097152x65536x256x1>, <1x32x256x256xf32, 32x1x8192x32> -> <1x32x256x256xf32, 2097152x65536x256x1>
    %11 = migraphx.reshape %10 {dims = [1, 32, 256, 256]} : <1x32x256x256xf32, 2097152x65536x256x1> -> <1x32x256x256xf32, 2097152x65536x256x1>
    %12 = migraphx.reduce_max %11 {axes = [3]} : <1x32x256x256xf32, 2097152x65536x256x1> -> <1x32x256x1xf32, 8192x256x1x1>
    %13 = migraphx.reshape %12 {dims = [1, 32, 256, 1]} : <1x32x256x1xf32, 8192x256x1x1> -> <1x32x256x1xf32, 8192x256x1x1>
    %14 = migraphx.multibroadcast %13 {out_dyn_dims = [], out_lens = [1, 32, 256, 256]} : <1x32x256x1xf32, 8192x256x1x1> -> <1x32x256x256xf32, 8192x256x1x0>
    %15 = migraphx.sub %10, %14 : <1x32x256x256xf32, 2097152x65536x256x1>, <1x32x256x256xf32, 8192x256x1x0> -> <1x32x256x256xf32, 2097152x65536x256x1>
    %16 = migraphx.exp %15 : <1x32x256x256xf32, 2097152x65536x256x1> -> <1x32x256x256xf32, 2097152x65536x256x1>
    %17 = migraphx.reshape %16 {dims = [1, 32, 256, 256]} : <1x32x256x256xf32, 2097152x65536x256x1> -> <1x32x256x256xf32, 2097152x65536x256x1>
    %18 = migraphx.reduce_sum %17 {axes = [3]} : <1x32x256x256xf32, 2097152x65536x256x1> -> <1x32x256x1xf32, 8192x256x1x1>
    %19 = migraphx.reshape %18 {dims = [1, 32, 256, 1]} : <1x32x256x1xf32, 8192x256x1x1> -> <1x32x256x1xf32, 8192x256x1x1>
    %20 = migraphx.multibroadcast %19 {out_dyn_dims = [], out_lens = [1, 32, 256, 256]} : <1x32x256x1xf32, 8192x256x1x1> -> <1x32x256x256xf32, 8192x256x1x0>
    %21 = migraphx.div %16, %20 : <1x32x256x256xf32, 2097152x65536x256x1>, <1x32x256x256xf32, 8192x256x1x0> -> <1x32x256x256xf32, 2097152x65536x256x1>
    %22 = migraphx.dot %21, %8 : <1x32x256x256xf32, 2097152x65536x256x1>, <1x32x256x64xf32, 524288x64x2048x1> -> <1x32x256x64xf32, 524288x16384x64x1>
    return %22 : !migraphx.shaped<1x32x256x64xf32, 524288x16384x64x1>
  }
}
