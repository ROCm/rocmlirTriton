// TODO(rocmlirTriton): Failures after updates to populateExpansionPatterns
// UNSUPPORTED: true
// RUN: rocmlir-gen -fut mlir_dequantizelinear_convolution_quantizelinear --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_dequantizelinear_convolution_quantizelinear --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_dequantizelinear_convolution_quantizelinear(%arg0: !migraphx.shaped<49xui8, 1>, %arg1: !migraphx.shaped<1xf32, 1>, %arg2: !migraphx.shaped<1xui8, 1>, %arg3: !migraphx.shaped<1x1x7x7xf32, 7x7x7x1>, %arg4: !migraphx.shaped<1x1x7x7xui8, 7x7x7x1>, %arg5: !migraphx.shaped<1x1x1x1xf32, 1x1x1x1>) -> !migraphx.shaped<1x1x7x7xui8, 49x49x7x1> attributes {rock.kernel} {
    %arg0_reshaped = migraphx.reshape %arg0 {dims = [1, 1, 7, 7]} : <49xui8, 1> -> <1x1x7x7xui8, 49x49x7x1>
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [1, 1, 7, 7]} : <1xf32, 1> -> <1x1x7x7xf32, 0x0x0x0>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 1, 7, 7]} : <1xui8, 1> -> <1x1x7x7xui8, 0x0x0x0>
    %2 = migraphx.dequantizelinear %arg0_reshaped, %0, %1 : <1x1x7x7xui8, 49x49x7x1>, <1x1x7x7xf32, 0x0x0x0>, !migraphx.shaped<1x1x7x7xui8, 0x0x0x0> -> <1x1x7x7xf32, 49x49x7x1>
    %3 = migraphx.convolution %2, %arg5 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x1x7x7xf32, 49x49x7x1>, <1x1x1x1xf32, 1x1x1x1> -> <1x1x7x7xf32, 49x49x7x1>
    %4 = migraphx.quantizelinear %3, %arg3, %arg4 : <1x1x7x7xf32, 49x49x7x1>, <1x1x7x7xf32, 7x7x7x1>, !migraphx.shaped<1x1x7x7xui8, 7x7x7x1> -> <1x1x7x7xui8, 49x49x7x1>
    return %4 : !migraphx.shaped<1x1x7x7xui8, 49x49x7x1>
  }
}
