// RUN: rocmlir-gen -fut mlir_dot_slice_add_reduce --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut mlir_dot_slice_add_reduce --verifier clone - | rocmlir-driver -c |  mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// After RegularizeOutput, the external operand is padded with zeros
// (inverse of Slice = Pad), moving the add to gemm space. The gemm values
// in the sliced-away region are non-zero, making add non-zero there.
// Correctness depends on the Pad on the store destination masking out
// writes to the sliced-away region so the reduce only accumulates valid
// positions via atomic_add.
// CHECK: [1 1 1]
// CHECK-NEXT: Unranked Memref base

module {
  func.func @mlir_dot_slice_add_reduce(%arg0: !migraphx.shaped<1x8x16xf16, 128x16x1>, %arg1: !migraphx.shaped<1x16x24xf16, 384x24x1>, %arg2: !migraphx.shaped<1x4x24xf16, 96x24x1>) -> !migraphx.shaped<1x4x1xf16, 4x1x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 : <1x8x16xf16, 128x16x1>, <1x16x24xf16, 384x24x1> -> <1x8x24xf16, 192x24x1>
    %1 = migraphx.slice %0 {axes = [1], starts = [0], ends = [4]} : <1x8x24xf16, 192x24x1> -> <1x4x24xf16, 192x24x1>
    %2 = migraphx.add %1, %arg2 : <1x4x24xf16, 192x24x1>, <1x4x24xf16, 96x24x1> -> <1x4x24xf16, 96x24x1>
    %3 = migraphx.reduce_sum %2 {axes = [2]} : <1x4x24xf16, 96x24x1> -> <1x4x1xf16, 4x1x1>
    return %3 : !migraphx.shaped<1x4x1xf16, 4x1x1>
  }
}
