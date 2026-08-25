// A group-quantized int4 GEMM: the weights carry one f16 scale per 128-element
// group along K, so the scale a K tile reads is the same for every K position
// in the tile. rock-narrow-redundant-loads rewrites that read into a load of
// one K row plus a broadcast, which is only sound if it picks the row the whole
// tile agrees on. Run it end to end to check that it does.

// RUN: rocmlir-gen -fut mlir_unpack_int4_reshape_dequantizelinear_transpose_reshape_unsqueeze_transpose_dot_add --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_unpack_int4_reshape_dequantizelinear_transpose_reshape_unsqueeze_transpose_dot_add --verifier clone - | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_unpack_int4_reshape_dequantizelinear_transpose_reshape_unsqueeze_transpose_dot_add(%arg0: !migraphx.shaped<4096x2048xui8, 2048x1>, %arg1: !migraphx.shaped<4096x32x1xf16, 32x1x1>, %arg2: !migraphx.shaped<1x64x77x64xf16, 315392x4928x64x1>, %arg3: !migraphx.shaped<1x77x4096xf16, 315392x4096x1>) -> !migraphx.shaped<1x77x4096xf16, 315392x4096x1> attributes {rock.kernel} {
    %0 = migraphx.literal(dense<8> : tensor<1xui8>) : <1xui8, 0>
    %1 = migraphx.unpack %arg0 {axis = 1 : i64} : <4096x2048xui8, 2048x1> -> <4096x4096xui8, 4096x1>
    %2 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [4096, 32, 128]} : <4096x32x1xf16, 32x1x1> -> <4096x32x128xf16, 32x1x0>
    %3 = migraphx.reshape %2 {dims = [4096, 4096]} : <4096x32x128xf16, 32x1x0> -> <4096x4096xf16, 4096x1>
    %4 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [4096, 4096]} : <1xui8, 0> -> <4096x4096xui8, 0x0>
    %5 = migraphx.dequantizelinear %1, %3, %4 : <4096x4096xui8, 4096x1>, <4096x4096xf16, 4096x1>, !migraphx.shaped<4096x4096xui8, 0x0> -> <4096x4096xf16, 4096x1>
    %6 = migraphx.transpose %arg2 {permutation = [0, 2, 1, 3]} : <1x64x77x64xf16, 315392x4928x64x1> -> <1x77x64x64xf16, 315392x64x4928x1>
    %7 = migraphx.reshape %6 {dims = [1, 77, 4096]} : <1x77x64x64xf16, 315392x64x4928x1> -> <1x77x4096xf16, 315392x4096x1>
    %8 = migraphx.reshape %5 {dims = [1, 4096, 4096]} : <4096x4096xf16, 4096x1> -> <1x4096x4096xf16, 16777216x4096x1>
    %9 = migraphx.transpose %8 {permutation = [0, 2, 1]} : <1x4096x4096xf16, 16777216x4096x1> -> <1x4096x4096xf16, 16777216x1x4096>
    %10 = migraphx.dot %7, %9 : <1x77x4096xf16, 315392x4096x1>, <1x4096x4096xf16, 16777216x1x4096> -> <1x77x4096xf16, 315392x4096x1>
    %11 = migraphx.add %arg3, %10 : <1x77x4096xf16, 315392x4096x1>, <1x77x4096xf16, 315392x4096x1> -> <1x77x4096xf16, 315392x4096x1>
    return %11 : !migraphx.shaped<1x77x4096xf16, 315392x4096x1>
  }
}
