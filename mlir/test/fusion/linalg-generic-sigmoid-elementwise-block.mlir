// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel -arch %arch %s | rocmlir-driver -kernel-pipeline gpu -arch %arch | rocmlir-opt | FileCheck %s
// Derived from https://github.com/ROCm/rocMLIR/issues/1188
// and rocMLIR-internal/issues/1098

// CHECK: tt.dot
// CHECK: math.erf
module {
  func.func @mlir_quant_dot_dequantizelinear_add_mul_erf_add_mul_mul_quantizelinear(%arg0: !migraphx.shaped<1x1x3072xf32, 3072x3072x1>, %arg1: !migraphx.shaped<32x384x768xi8, 294912x768x1>, %arg2: !migraphx.shaped<32x768x3072xi8, 2359296x3072x1>) -> !migraphx.shaped<32x384x3072xi8, 1179648x3072x1> attributes {rock.arch = "gfx1100", rock.kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [32, 384, 3072]} : <1x1x3072xf32, 3072x3072x1> -> <32x384x3072xf32, 0x0x1>
    %1 = migraphx.literal (dense<4.23844496E-4> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.literal (dense<1.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %3 = migraphx.literal (dense<0.707106769> : tensor<1xf32>) : <1xf32, 0>
    %4 = migraphx.literal (dense<0.0286473539> : tensor<1xf32>) : <1xf32, 0>
    %5 = migraphx.literal (dense<0> : tensor<1xi8>) : <1xi8, 0>
    %6 = migraphx.literal (dense<5.000000e-01> : tensor<1xf32>) : <1xf32, 0>
    %7 = migraphx.multibroadcast %6 {out_dyn_dims = [], out_lens = [32, 384, 3072]} : <1xf32, 0> -> <32x384x3072xf32, 0x0x0>
    %8 = migraphx.multibroadcast %5 {out_dyn_dims = [], out_lens = [32, 384, 3072]} : <1xi8, 0> -> <32x384x3072xi8, 0x0x0>
    %9 = migraphx.multibroadcast %4 {out_dyn_dims = [], out_lens = [32, 384, 3072]} : <1xf32, 0> -> <32x384x3072xf32, 0x0x0>
    %10 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [32, 384, 3072]} : <1xf32, 0> -> <32x384x3072xf32, 0x0x0>
    %11 = migraphx.multibroadcast %2 {out_dyn_dims = [], out_lens = [32, 384, 3072]} : <1xf32, 0> -> <32x384x3072xf32, 0x0x0>
    %12 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [32, 384, 3072]} : <1xf32, 0> -> <32x384x3072xf32, 0x0x0>
    %13 = migraphx.quant_dot %arg1, %arg2 : <32x384x768xi8, 294912x768x1>, <32x768x3072xi8, 2359296x3072x1> -> <32x384x3072xi32, 1179648x3072x1>
    %14 = migraphx.dequantizelinear %13, %12 : <32x384x3072xi32, 1179648x3072x1>, <32x384x3072xf32, 0x0x0> -> <32x384x3072xf32, 1179648x3072x1>
    %15 = migraphx.add %0, %14 : <32x384x3072xf32, 0x0x1>, <32x384x3072xf32, 1179648x3072x1> -> <32x384x3072xf32, 1179648x3072x1>
    %16 = migraphx.mul %15, %10 : <32x384x3072xf32, 1179648x3072x1>, <32x384x3072xf32, 0x0x0> -> <32x384x3072xf32, 1179648x3072x1>
    %17 = migraphx.erf %16 : <32x384x3072xf32, 1179648x3072x1> -> <32x384x3072xf32, 1179648x3072x1>
    %18 = migraphx.add %17, %11 : <32x384x3072xf32, 1179648x3072x1>, <32x384x3072xf32, 0x0x0> -> <32x384x3072xf32, 1179648x3072x1>
    %19 = migraphx.mul %15, %18 : <32x384x3072xf32, 1179648x3072x1>, <32x384x3072xf32, 1179648x3072x1> -> <32x384x3072xf32, 1179648x3072x1>
    %20 = migraphx.mul %19, %7 : <32x384x3072xf32, 1179648x3072x1>, <32x384x3072xf32, 0x0x0> -> <32x384x3072xf32, 1179648x3072x1>
    %21 = migraphx.quantizelinear %20, %9, %8 : <32x384x3072xf32, 1179648x3072x1>, <32x384x3072xf32, 0x0x0>, !migraphx.shaped<32x384x3072xi8, 0x0x0> -> <32x384x3072xi8, 1179648x3072x1>
    return %21 : !migraphx.shaped<32x384x3072xi8, 1179648x3072x1>
  }
}
