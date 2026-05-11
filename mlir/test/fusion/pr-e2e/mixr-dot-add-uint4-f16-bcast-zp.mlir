// RUN: rocmlir-driver -kernel-pipeline=migraphx %s | rocmlir-gen -fut mlir_unpack_int4_unpack_int4_reshape_unsqueeze_reshape_dequantizelinear_unsqueeze_transpose_dot_add --arch %arch --clone-harness - | rocmlir-driver -host-pipeline=highlevel -kernel-pipeline=highlevel | rocmlir-gen -ph -fut mlir_unpack_int4_unpack_int4_reshape_unsqueeze_reshape_dequantizelinear_unsqueeze_transpose_dot_add --verifier clone - | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_unpack_int4_unpack_int4_reshape_unsqueeze_reshape_dequantizelinear_unsqueeze_transpose_dot_add(%arg0: !migraphx.shaped<4096x5504xui8, 5504x1>, %arg1: !migraphx.shaped<4096x86x1xf16, 86x1x1>, %arg2: !migraphx.shaped<4096x43xui8, 43x1>, %arg3: !migraphx.shaped<1x1x11008xf16, 11008x11008x1>, %arg4: !migraphx.shaped<1x1x4096xf16, 4096x4096x1>) -> !migraphx.shaped<1x1x4096xf16, 4096x4096x1> attributes {rock.kernel} {
    %0 = migraphx.unpack %arg0 {axis = 1 : i64} : <4096x5504xui8, 5504x1> -> <4096x11008xui8, 11008x1>
    %1 = migraphx.unpack %arg2 {axis = 1 : i64} : <4096x43xui8, 43x1> -> <4096x86xui8, 86x1>
    %2 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [4096, 86, 128]} : <4096x86x1xf16, 86x1x1> -> <4096x86x128xf16, 86x1x0>
    %3 = migraphx.reshape %2 {dims = [4096, 11008]} : <4096x86x128xf16, 86x1x0> -> <4096x11008xf16, 11008x1>
    %4 = migraphx.reshape %1 {dims = [4096, 86, 1]} : <4096x86xui8, 86x1> -> <4096x86x1xui8, 86x1x1>
    %5 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [4096, 86, 128]} : <4096x86x1xui8, 86x1x1> -> <4096x86x128xui8, 86x1x0>
    %6 = migraphx.reshape %5 {dims = [4096, 11008]} : <4096x86x128xui8, 86x1x0> -> <4096x11008xui8, 11008x1>
    %7 = migraphx.dequantizelinear %0, %3, %6 : <4096x11008xui8, 11008x1>, <4096x11008xf16, 11008x1>, !migraphx.shaped<4096x11008xui8, 11008x1> -> <4096x11008xf16, 11008x1>
    %8 = migraphx.reshape %7 {dims = [1, 4096, 11008]} : <4096x11008xf16, 11008x1> -> <1x4096x11008xf16, 45088768x11008x1>
    %9 = migraphx.transpose %8 {permutation = [0, 2, 1]} : <1x4096x11008xf16, 45088768x11008x1> -> <1x11008x4096xf16, 45088768x1x11008>
    %10 = migraphx.dot %arg3, %9 : <1x1x11008xf16, 11008x11008x1>, <1x11008x4096xf16, 45088768x1x11008> -> <1x1x4096xf16, 4096x4096x1>
    %11 = migraphx.add %arg4, %10 : <1x1x4096xf16, 4096x4096x1>, <1x1x4096xf16, 4096x4096x1> -> <1x1x4096xf16, 4096x4096x1>
    return %11 : !migraphx.shaped<1x1x4096xf16, 4096x4096x1>
  }
}
