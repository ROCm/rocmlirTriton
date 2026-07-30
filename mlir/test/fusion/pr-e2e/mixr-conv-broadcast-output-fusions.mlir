// RUN: rocmlir-gen -fut mlir_test --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx,highlevel -kernel-pipeline=migraphx,highlevel | rocmlir-gen -ph -fut mlir_test --verifier clone - | rocmlir-driver -c | rocm-run | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func @mlir_test(%arg0: !migraphx.shaped<2x640x128x128xf32, 10485760x16384x128x1>, %arg1: !migraphx.shaped<320x640x1x1xf32, 640x1x1x1>, %arg2: !migraphx.shaped<2x320x128x128xf32, 5242880x16384x128x1>, %arg3: !migraphx.shaped<320xf32, 1>) -> (!migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x10x128x128xf32, 5242880x163840x16384x128x1>) attributes {rock.enable_splitk_for_tuning, rock.kernel, rock.arch="gfx1100"} {
    %0 = migraphx.literal(dense<6.10351572E-6> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.literal(dense<6.10351572E-6> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <2x640x128x128xf32, 10485760x16384x128x1>, <320x640x1x1xf32, 640x1x1x1> -> <2x320x128x128xf32, 5242880x16384x128x1>
    %3 = migraphx.reshape %2 {dims = [2, 32, 10, 128, 128]} : <2x320x128x128xf32, 5242880x16384x128x1> -> <2x32x10x128x128xf32, 5242880x163840x16384x128x1>
    %4 = migraphx.reshape %arg2 {dims = [2, 32, 10, 128, 128]} : <2x320x128x128xf32, 5242880x16384x128x1> -> <2x32x10x128x128xf32, 5242880x163840x16384x128x1>
    %5 = migraphx.reshape %arg3 {dims = [32, 10]} : <320xf32, 1> -> <32x10xf32, 10x1>
    %6 = migraphx.broadcast %5 {axis = 1 : i64, out_lens = [2, 32, 10, 128, 128]} : <32x10xf32, 10x1> -> <2x32x10x128x128xf32, 0x10x1x0x0>
    %7 = migraphx.add %3, %4 : <2x32x10x128x128xf32, 5242880x163840x16384x128x1>, <2x32x10x128x128xf32, 5242880x163840x16384x128x1> -> <2x32x10x128x128xf32, 5242880x163840x16384x128x1>
    %8 = migraphx.add %7, %6 : <2x32x10x128x128xf32, 5242880x163840x16384x128x1>, <2x32x10x128x128xf32, 0x10x1x0x0> -> <2x32x10x128x128xf32, 5242880x163840x16384x128x1>
    %9 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [2, 32, 10, 128, 128]} : <1xf32, 0> -> <2x32x10x128x128xf32, 0x0x0x0x0>
    %10 = migraphx.mul %8, %9 : <2x32x10x128x128xf32, 5242880x163840x16384x128x1>, <2x32x10x128x128xf32, 0x0x0x0x0> -> <2x32x10x128x128xf32, 5242880x163840x16384x128x1>
    %11 = migraphx.reshape %10 {dims = [2, 32, 163840]} : <2x32x10x128x128xf32, 5242880x163840x16384x128x1> -> <2x32x163840xf32, 5242880x163840x1>
    %12 = migraphx.reduce_sum %11 {axes = [2]} : <2x32x163840xf32, 5242880x163840x1> -> <2x32x1xf32, 32x1x1>
    %13 = migraphx.reshape %12 {dims = [2, 32, 1, 1, 1]} : <2x32x1xf32, 32x1x1> -> <2x32x1x1x1xf32, 32x1x1x1x1>
    %14 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 32, 10, 128, 128]} : <1xf32, 0> -> <2x32x10x128x128xf32, 0x0x0x0x0>
    %15 = migraphx.mul %8, %8 : <2x32x10x128x128xf32, 5242880x163840x16384x128x1>, <2x32x10x128x128xf32, 5242880x163840x16384x128x1> -> <2x32x10x128x128xf32, 5242880x163840x16384x128x1>
    %16 = migraphx.mul %15, %14 : <2x32x10x128x128xf32, 5242880x163840x16384x128x1>, <2x32x10x128x128xf32, 0x0x0x0x0> -> <2x32x10x128x128xf32, 5242880x163840x16384x128x1>
    %17 = migraphx.reshape %16 {dims = [2, 32, 163840]} : <2x32x10x128x128xf32, 5242880x163840x16384x128x1> -> <2x32x163840xf32, 5242880x163840x1>
    %18 = migraphx.reduce_sum %17 {axes = [2]} : <2x32x163840xf32, 5242880x163840x1> -> <2x32x1xf32, 32x1x1>
    %19 = migraphx.reshape %18 {dims = [2, 32, 1, 1, 1]} : <2x32x1xf32, 32x1x1> -> <2x32x1x1x1xf32, 32x1x1x1x1>
    return %13, %19, %8 : !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x10x128x128xf32, 5242880x163840x16384x128x1>
  }
}