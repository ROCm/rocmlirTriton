// RUN: rocmlir-gen -fut mlir_dot_slice --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_slice --verifier clone - | rocmlir-driver -c |  mlir-runner --shared-libs=%linalg_test_lib_dir/%shlibprefixmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/%shlibprefixconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_float16_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/%shlibprefixmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// Slicing the dot output along axis 1 creates non-contiguous output strides
// (stride[0]=576 > shape[1]*stride[1]=12*24=288). Only 50% of the output
// buffer holds valid data; the rest is uninitialized gaps.
// CHECK: relDiff = 0     : {{11[5-6][0-9]}}/2304

module {
  func.func @mlir_dot_slice(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x12x24xf16, 576x24x1> attributes {rock.kernel = "mixr"} {
    %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
    %1 = migraphx.slice %0 {axes = [1], starts = [0], ends = [12]} : <4x24x24xf16, 576x24x1> -> <4x12x24xf16, 576x24x1>
    return %1 : !migraphx.shaped<4x12x24xf16, 576x24x1>
  }
}
