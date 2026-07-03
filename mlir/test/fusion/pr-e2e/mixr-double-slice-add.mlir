// RUN: rocmlir-gen -fut mlir_double_slice_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_double_slice_add --verifier clone - | rocmlir-driver -c |  mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut mlir_double_slice_add_axis2 --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_double_slice_add_axis2 --verifier clone - | rocmlir-driver -c |  mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// Pattern 1 (AIROCMLIR-709): slice the GEMM output into TWO halves and add the
// two halves elementwise. The two slices map to different workgroup tiles (a
// cross-tile dependency), which is fixed by rock-split-cross-tile-fusion +
// rock-link-split-kernels. Both slice axes are exercised:
//
//   mlir_double_slice_add        slices axis 1 (the M dimension):
//     out1 = slice(out, axis=1, 0..12);  out2 = slice(out, axis=1, 12..24)
//   mlir_double_slice_add_axis2  slices axis 2 (the N dimension):
//     out1 = slice(out, axis=2, 0..12);  out2 = slice(out, axis=2, 12..24)
//   result = out1 + out2
//
// CHECK: [1 1 1]

module {
  func.func @mlir_double_slice_add(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x12x24xf16, 288x24x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
    %1 = migraphx.slice %0 {axes = [1], starts = [0], ends = [12]} : <4x24x24xf16, 576x24x1> -> <4x12x24xf16, 576x24x1>
    %2 = migraphx.slice %0 {axes = [1], starts = [12], ends = [24]} : <4x24x24xf16, 576x24x1> -> <4x12x24xf16, 576x24x1>
    %3 = migraphx.add %1, %2 : <4x12x24xf16, 576x24x1>, <4x12x24xf16, 576x24x1> -> <4x12x24xf16, 288x24x1>
    return %3 : !migraphx.shaped<4x12x24xf16, 288x24x1>
  }

  // Same pattern, but slicing along axis 2 (the free/N dimension) instead.
  func.func @mlir_double_slice_add_axis2(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x24x12xf16, 288x12x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
    %1 = migraphx.slice %0 {axes = [2], starts = [0], ends = [12]} : <4x24x24xf16, 576x24x1> -> <4x24x12xf16, 576x24x1>
    %2 = migraphx.slice %0 {axes = [2], starts = [12], ends = [24]} : <4x24x24xf16, 576x24x1> -> <4x24x12xf16, 576x24x1>
    %3 = migraphx.add %1, %2 : <4x24x12xf16, 576x24x1>, <4x24x12xf16, 576x24x1> -> <4x24x12xf16, 288x12x1>
    return %3 : !migraphx.shaped<4x24x12xf16, 288x12x1>
  }
}
