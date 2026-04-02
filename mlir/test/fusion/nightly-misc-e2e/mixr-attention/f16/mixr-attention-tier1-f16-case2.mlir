// TODO(rocmlirTriton): 'arith.mulf' op requires the same type for all operands and results
// UNSUPPORTED: true
// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention -relDiff_threshold 0.02 -absDiff_threshold 0.02 -RMS_threshold 0.01 --verifier clone - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_attention(%arg0: !migraphx.shaped<2x16x384x64xf16, 393216x24576x64x1>, %arg1: !migraphx.shaped<2x16x64x384xf16, 393216x24576x384x1>, %arg2: !migraphx.shaped<2x16x384x64xf16, 393216x24576x64x1>) -> !migraphx.shaped<2x16x384x64xf16, 393216x24576x64x1> attributes {rock.kernel} {
    %0 = migraphx.dot %arg0, %arg1: <2x16x384x64xf16, 393216x24576x64x1>, <2x16x64x384xf16, 393216x24576x384x1> -> <2x16x384x384xf16, 2359296x147456x384x1>
    %1 = migraphx.softmax %0{axis = 3 : i64} : <2x16x384x384xf16, 2359296x147456x384x1> -> <2x16x384x384xf16, 2359296x147456x384x1>
    %2 = migraphx.dot %1, %arg2: <2x16x384x384xf16, 2359296x147456x384x1>, <2x16x384x64xf16, 393216x24576x64x1> -> <2x16x384x64xf16, 393216x24576x64x1>
    return %2 : !migraphx.shaped<2x16x384x64xf16, 393216x24576x64x1>
  }
}
