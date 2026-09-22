// RUN: rocmlir-gen -fut mlir_quantizelinear_i8 --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -arch %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_quantizelinear_i8 --verifier clone -relDiff_threshold 0.00001 - | rocmlir-driver -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Both the kernel and the host side lower quantization through MIGraphXToTosa,
// so this compares the GPU and CPU code generated from one lowering rather than
// two independent ones.

// CHECK: [1 1 1]
func.func @mlir_quantizelinear_i8(
    %input: !migraphx.shaped<2x2xf32, 2x1>,
    %scale: !migraphx.shaped<2x2xf32, 2x1>,
    %bias: !migraphx.shaped<2x2xi8, 2x1>)
    -> !migraphx.shaped<2x2xi32, 2x1> {
  %result = migraphx.quantizelinear %input, %scale, %bias
    : <2x2xf32, 2x1>, <2x2xf32, 2x1>, !migraphx.shaped<2x2xi8, 2x1>
      -> <2x2xi8, 2x1>
  %dotResult = migraphx.quant_dot %result, %result
    : <2x2xi8, 2x1>, <2x2xi8, 2x1>
      -> <2x2xi32, 2x1>
  return %dotResult : !migraphx.shaped<2x2xi32, 2x1>
}
