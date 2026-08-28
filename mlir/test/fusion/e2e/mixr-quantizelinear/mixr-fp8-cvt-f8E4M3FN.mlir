// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand fixed - | rocmlir-driver -arch %arch -c | mlir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Exercise the hardware OCP E4M3FN conversion instructions on the
// quantized-GEMM shape MIGraphX emits: quantize both dot operands, run the dot
// in fp8, add a residual, requantize the output.
//
// The fp8 residual is what forces an elementwise upcast, since quant_dot
// consumes fp8 natively. Keeping the dequantize and the output quantize on
// opposite sides of the add stops EliminateCastOp from folding the
// f32 -> fp8 -> f32 round trip away.
//
// The expected values are E4M3FN-specific: the E5M2 companion test rounds the
// same computation to [-4, -8, 6, -4].

// CHECK: [-3.75,  -8,  6.5,  -3.75]

module {
  func.func @mlir_fp8_cvt(
      %a: !migraphx.shaped<1x2x2xf32, 4x2x1>,
      %b: !migraphx.shaped<1x2x2xf32, 4x2x1>,
      %residual: !migraphx.shaped<1x2x2xf8E4M3FN, 4x2x1>)
      -> !migraphx.shaped<1x2x2xf8E4M3FN, 4x2x1>
      attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %quantScale = migraphx.literal
      (dense<5.000000e-01> : tensor<1xf32>) : <1xf32, 0>
    %dequantScale = migraphx.literal
      (dense<2.500000e-01> : tensor<1xf32>) : <1xf32, 0>
    %quantScaleBcast = migraphx.multibroadcast %quantScale
      {out_dyn_dims = [], out_lens = [1, 2, 2]}
      : <1xf32, 0> -> <1x2x2xf32, 0x0x0>
    %dequantScaleBcast = migraphx.multibroadcast %dequantScale
      {out_dyn_dims = [], out_lens = [1, 2, 2]}
      : <1xf32, 0> -> <1x2x2xf32, 0x0x0>
    %aQuant = migraphx.quantizelinear %a, %quantScaleBcast
      : <1x2x2xf32, 4x2x1>, <1x2x2xf32, 0x0x0>
        -> <1x2x2xf8E4M3FN, 4x2x1>
    %bQuant = migraphx.quantizelinear %b, %quantScaleBcast
      : <1x2x2xf32, 4x2x1>, <1x2x2xf32, 0x0x0>
        -> <1x2x2xf8E4M3FN, 4x2x1>
    %dot = migraphx.quant_dot %aQuant, %bQuant
      : <1x2x2xf8E4M3FN, 4x2x1>, <1x2x2xf8E4M3FN, 4x2x1>
        -> <1x2x2xf32, 4x2x1>
    %residualUp = migraphx.dequantizelinear %residual, %dequantScaleBcast
      : <1x2x2xf8E4M3FN, 4x2x1>, <1x2x2xf32, 0x0x0>
        -> <1x2x2xf32, 4x2x1>
    %sum = migraphx.add %dot, %residualUp
      : <1x2x2xf32, 4x2x1>, <1x2x2xf32, 4x2x1>
        -> <1x2x2xf32, 4x2x1>
    %out = migraphx.quantizelinear %sum, %quantScaleBcast
      : <1x2x2xf32, 4x2x1>, <1x2x2xf32, 0x0x0>
        -> <1x2x2xf8E4M3FN, 4x2x1>
    return %out : !migraphx.shaped<1x2x2xf8E4M3FN, 4x2x1>
  }
}
