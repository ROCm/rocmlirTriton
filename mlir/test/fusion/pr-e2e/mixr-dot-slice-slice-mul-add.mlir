// RUN: rocmlir-gen -fut mlir_dot --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot -pr -pvr --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_dot(%arg0: !migraphx.shaped<1x9216x2560xf16, 23592960x2560x1>, %arg1: !migraphx.shaped<1x1280x320xf16, 409600x320x1>, %arg2: !migraphx.shaped<320xf16, 1>, %arg3: !migraphx.shaped<1x96x96x320xf16, 2949120x30720x320x1>) -> !migraphx.shaped<1x320x96x96xf16, 2949120x1x30720x320> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.literal(dense<-7.135010e-02> : tensor<1xf16>) : <1xf16, 0>
    %1 = migraphx.literal(dense<-1.595700e+00> : tensor<1xf16>) : <1xf16, 0>
    %2 = migraphx.literal(dense<1.000000e+00> : tensor<1xf16>) : <1xf16, 0>
    %3 = migraphx.slice %arg0 {axes = [2], ends = [2560], starts = [1280]} : <1x9216x2560xf16, 23592960x2560x1> -> <1x9216x1280xf16, 23592960x2560x1>
    %4 = migraphx.slice %arg0 {axes = [2], ends = [1280], starts = [0]} : <1x9216x2560xf16, 23592960x2560x1> -> <1x9216x1280xf16, 23592960x2560x1>
    %5 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 9216, 1280]} : <1xf16, 0> -> <1x9216x1280xf16, 0x0x0>
    %6 = migraphx.mul %3, %5 : <1x9216x1280xf16, 23592960x2560x1>, <1x9216x1280xf16, 0x0x0> -> <1x9216x1280xf16, 11796480x1280x1>
    %7 = migraphx.mul %3, %6 : <1x9216x1280xf16, 23592960x2560x1>, <1x9216x1280xf16, 11796480x1280x1> -> <1x9216x1280xf16, 11796480x1280x1>
    %8 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 9216, 1280]} : <1xf16, 0> -> <1x9216x1280xf16, 0x0x0>
    %9 = migraphx.add %7, %8 : <1x9216x1280xf16, 11796480x1280x1>, <1x9216x1280xf16, 0x0x0> -> <1x9216x1280xf16, 11796480x1280x1>
    %10 = migraphx.mul %3, %9 : <1x9216x1280xf16, 23592960x2560x1>, <1x9216x1280xf16, 11796480x1280x1> -> <1x9216x1280xf16, 11796480x1280x1>
    %11 = migraphx.exp %10 : <1x9216x1280xf16, 11796480x1280x1> -> <1x9216x1280xf16, 11796480x1280x1>
    %12 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 9216, 1280]} : <1xf16, 0> -> <1x9216x1280xf16, 0x0x0>
    %13 = migraphx.add %12, %11 : <1x9216x1280xf16, 0x0x0>, <1x9216x1280xf16, 11796480x1280x1> -> <1x9216x1280xf16, 11796480x1280x1>
    %14 = migraphx.div %3, %13 : <1x9216x1280xf16, 23592960x2560x1>, <1x9216x1280xf16, 11796480x1280x1> -> <1x9216x1280xf16, 11796480x1280x1>
    %15 = migraphx.mul %4, %14 : <1x9216x1280xf16, 23592960x2560x1>, <1x9216x1280xf16, 11796480x1280x1> -> <1x9216x1280xf16, 11796480x1280x1>
    %16 = migraphx.dot %15, %arg1 : <1x9216x1280xf16, 11796480x1280x1>, <1x1280x320xf16, 409600x320x1> -> <1x9216x320xf16, 2949120x320x1>
    %17 = migraphx.reshape %16 {dims = [1, 96, 96, 320]} : <1x9216x320xf16, 2949120x320x1> -> <1x96x96x320xf16, 2949120x30720x320x1>
    %18 = migraphx.transpose %17 {permutation = [0, 3, 1, 2]} : <1x96x96x320xf16, 2949120x30720x320x1> -> <1x320x96x96xf16, 2949120x1x30720x320>
    %19 = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = [1, 320, 96, 96]} : <320xf16, 1> -> <1x320x96x96xf16, 0x1x0x0>
    %20 = migraphx.transpose %arg3 {permutation = [0, 3, 1, 2]} : <1x96x96x320xf16, 2949120x30720x320x1> -> <1x320x96x96xf16, 2949120x1x30720x320>
    %21 = migraphx.add %19, %18 : <1x320x96x96xf16, 0x1x0x0>, <1x320x96x96xf16, 2949120x1x30720x320> -> <1x320x96x96xf16, 2949120x1x30720x320>
    %22 = migraphx.add %21, %20 : <1x320x96x96xf16, 2949120x1x30720x320>, <1x320x96x96xf16, 2949120x1x30720x320> -> <1x320x96x96xf16, 2949120x1x30720x320>
    return %22 : !migraphx.shaped<1x320x96x96xf16, 2949120x1x30720x320>
  }
}
