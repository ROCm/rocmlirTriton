// RUN: rocmlir-gen --clone-harness -arch %arch -fut mlir_test %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -host-pipeline migraphx,highlevel -arch %arch | rocmlir-gen -ph -verifier clone -fut mlir_test - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

module {
    // CHECK: [1 1 1]
  func.func @mlir_test(%arg0: !migraphx.shaped<5120x640xui8, 640x1>, %arg1: !migraphx.shaped<5120x10x1xf16, 10x1x1>, %arg2: !migraphx.shaped<1x77x1280xf16, 98560x1280x1>, %arg3: !migraphx.shaped<1x77x5120xf16, 394240x5120x1>) -> !migraphx.shaped<1x77x5120xf16, 394240x5120x1> attributes {rock.kernel} {
    %0 = migraphx.literal(dense<8> : tensor<1xui8>) : <1xui8, 0>
    %1 = migraphx.literal(dense<1.000000e+00> : tensor<1xf16>) : <1xf16, 0>
    %2 = migraphx.literal(dense<-1.595700e+00> : tensor<1xf16>) : <1xf16, 0>
    %3 = migraphx.literal(dense<-7.135010e-02> : tensor<1xf16>) : <1xf16, 0>
    %4 = migraphx.unpack %arg0 {axis = 1 : i64} : <5120x640xui8, 640x1> -> <5120x1280xui8, 1280x1>
    %5 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [5120, 10, 128]} : <5120x10x1xf16, 10x1x1> -> <5120x10x128xf16, 10x1x0>
    %6 = migraphx.reshape %5 {dims = [5120, 1280]} : <5120x10x128xf16, 10x1x0> -> <5120x1280xf16, 1280x1>
    %7 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [5120, 1280]} : <1xui8, 0> -> <5120x1280xui8, 0x0>
    %8 = migraphx.dequantizelinear %4, %6, %7 : <5120x1280xui8, 1280x1>, <5120x1280xf16, 1280x1>, !migraphx.shaped<5120x1280xui8, 0x0> -> <5120x1280xf16, 1280x1>
    %9 = migraphx.reshape %8 {dims = [1, 5120, 1280]} : <5120x1280xf16, 1280x1> -> <1x5120x1280xf16, 6553600x1280x1>
    %10 = migraphx.transpose %9 {permutation = [0, 2, 1]} : <1x5120x1280xf16, 6553600x1280x1> -> <1x1280x5120xf16, 6553600x1x1280>
    %11 = migraphx.dot %arg2, %10 : <1x77x1280xf16, 98560x1280x1>, <1x1280x5120xf16, 6553600x1x1280> -> <1x77x5120xf16, 394240x5120x1>
    %12 = migraphx.add %11, %arg3 : <1x77x5120xf16, 394240x5120x1>, <1x77x5120xf16, 394240x5120x1> -> <1x77x5120xf16, 394240x5120x1>
    %13 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [1, 77, 5120]} : <1xf16, 0> -> <1x77x5120xf16, 0x0x0>
    %14 = migraphx.mul %12, %13 : <1x77x5120xf16, 394240x5120x1>, <1x77x5120xf16, 0x0x0> -> <1x77x5120xf16, 394240x5120x1>
    %15 = migraphx.mul %12, %14 : <1x77x5120xf16, 394240x5120x1>, <1x77x5120xf16, 394240x5120x1> -> <1x77x5120xf16, 394240x5120x1>
    %16 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [1, 77, 5120]} : <1xf16, 0> -> <1x77x5120xf16, 0x0x0>
    %17 = migraphx.add %15, %16 : <1x77x5120xf16, 394240x5120x1>, <1x77x5120xf16, 0x0x0> -> <1x77x5120xf16, 394240x5120x1>
    %18 = migraphx.mul %12, %17 : <1x77x5120xf16, 394240x5120x1>, <1x77x5120xf16, 394240x5120x1> -> <1x77x5120xf16, 394240x5120x1>
    %19 = migraphx.exp %18 : <1x77x5120xf16, 394240x5120x1> -> <1x77x5120xf16, 394240x5120x1>
    %20 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [1, 77, 5120]} : <1xf16, 0> -> <1x77x5120xf16, 0x0x0>
    %21 = migraphx.add %20, %19 : <1x77x5120xf16, 0x0x0>, <1x77x5120xf16, 394240x5120x1> -> <1x77x5120xf16, 394240x5120x1>
    %22 = migraphx.div %12, %21 : <1x77x5120xf16, 394240x5120x1>, <1x77x5120xf16, 394240x5120x1> -> <1x77x5120xf16, 394240x5120x1>
    return %22 : !migraphx.shaped<1x77x5120xf16, 394240x5120x1>
  }
}