// An f32 convolution whose dot rock-roll-dot-k rewrites into a loop over K
// segments. The rolled loop reads the same shared-memory bytes as the
// unrolled dot did, so it must compute the same numbers. On a target where
// the pass does not apply this still runs, as a plain convolution.
//
// RUN: rocmlir-gen -fut conv_add_mul_max --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut conv_add_mul_max --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

func.func @conv_add_mul_max(%arg0: !migraphx.shaped<1x128x32x32xf32, 131072x1x4096x128>, %arg1: !migraphx.shaped<128x128x4x4xf32, 1x128x65536x16384>, %arg2: !migraphx.shaped<1x128x1x1xf32, 128x1x1x1>) -> !migraphx.shaped<1x128x16x16xf32, 32768x1x2048x128> {
  %0 = migraphx.literal(dense<2.000000e-01> : tensor<1xf32>) : <1xf32, 1>
  %1 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2], perf_config = "gemm:v1:128,64,64,1,1,2,0,1,2,0,0"} : <1x128x32x32xf32, 131072x1x4096x128>, <128x128x4x4xf32, 1x128x65536x16384> -> <1x128x16x16xf32, 32768x1x2048x128>
  %2 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 128, 16, 16]} : <1x128x1x1xf32, 128x1x1x1> -> <1x128x16x16xf32, 128x1x0x0>
  %3 = migraphx.add %1, %2 : <1x128x16x16xf32, 32768x1x2048x128>, <1x128x16x16xf32, 128x1x0x0> -> <1x128x16x16xf32, 32768x1x2048x128>
  %4 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 128, 16, 16]} : <1xf32, 1> -> <1x128x16x16xf32, 0x0x0x0>
  %5 = migraphx.mul %3, %4 : <1x128x16x16xf32, 32768x1x2048x128>, <1x128x16x16xf32, 0x0x0x0> -> <1x128x16x16xf32, 32768x1x2048x128>
  %6 = migraphx.max %3, %5 : <1x128x16x16xf32, 32768x1x2048x128>, <1x128x16x16xf32, 32768x1x2048x128> -> <1x128x16x16xf32, 32768x1x2048x128>
  return %6 : !migraphx.shaped<1x128x16x16xf32, 32768x1x2048x128>
}
