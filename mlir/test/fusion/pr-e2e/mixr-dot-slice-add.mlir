// RUN: rocmlir-gen -fut mlir_dot_slice_add --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot_slice_add --verifier clone - | rocmlir-driver -c |  mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// Dot produces 4x24x24, slice selects the first 12 rows along axis 1, then
// an element-wise add with an external tensor follows. The output has
// contiguous strides (no expand_strides at the boundary). The slice creates
// a rock.transform(Slice) in the output fusion chain that RegularizeOutput
// must invert (Slice -> Pad) to bring the external operand and store
// destination into gemm space. The Pad on the store destination masks out
// writes to the sliced-away region, ensuring correctness.
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base

module {
  func.func @mlir_dot_slice_add(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>, %arg2: !migraphx.shaped<4x12x24xf16, 288x24x1>) -> !migraphx.shaped<4x12x24xf16, 288x24x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
    %1 = migraphx.slice %0 {axes = [1], starts = [0], ends = [12]} : <4x24x24xf16, 576x24x1> -> <4x12x24xf16, 576x24x1>
    %2 = migraphx.add %1, %arg2 : <4x12x24xf16, 576x24x1>, <4x12x24xf16, 288x24x1> -> <4x12x24xf16, 288x24x1>
    return %2 : !migraphx.shaped<4x12x24xf16, 288x24x1>
  }
}
