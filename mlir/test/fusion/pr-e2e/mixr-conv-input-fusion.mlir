// RUN: rocmlir-gen -fut mlir_broadcast_mul_add_relu_convolution_broadcast_add_relu --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -rand_min -1 -rand_max 1 -fut mlir_broadcast_mul_add_relu_convolution_broadcast_add_relu --verifier clone - | rocmlir-driver -c -arch %arch | rocm-run | FileCheck %s

// CHECK: [1 1 1]

module {
  func.func @mlir_broadcast_mul_add_relu_convolution_broadcast_add_relu(%arg0: !migraphx.shaped<2560x1x1xf32, 1x1x1>, %arg1: !migraphx.shaped<1x2560x7x7xf32, 125440x49x7x1>, %arg2: !migraphx.shaped<2560xf32, 1>, %arg3: !migraphx.shaped<768x2560x1x1xf32, 2560x1x1x1>, %arg4: !migraphx.shaped<768xf32, 1>) -> !migraphx.shaped<1x768x7x7xf32, 37632x49x7x1> attributes {rock.enable_splitk_for_tuning, rock.kernel} {
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [1, 2560, 7, 7]} : <2560x1x1xf32, 1x1x1> -> <1x2560x7x7xf32, 0x1x0x0>
    %1 = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = [1, 2560, 7, 7]} : <2560xf32, 1> -> <1x2560x7x7xf32, 0x1x0x0>
    %2 = migraphx.mul %0, %arg1 : <1x2560x7x7xf32, 0x1x0x0>, <1x2560x7x7xf32, 125440x49x7x1> -> <1x2560x7x7xf32, 125440x49x7x1>
    %3 = migraphx.add %2, %1 : <1x2560x7x7xf32, 125440x49x7x1>, <1x2560x7x7xf32, 0x1x0x0> -> <1x2560x7x7xf32, 125440x49x7x1>
    %4 = migraphx.relu %3 : <1x2560x7x7xf32, 125440x49x7x1> -> <1x2560x7x7xf32, 125440x49x7x1>
    %5 = migraphx.convolution %4, %arg3 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x2560x7x7xf32, 125440x49x7x1>, <768x2560x1x1xf32, 2560x1x1x1> -> <1x768x7x7xf32, 37632x49x7x1>
    %6 = migraphx.broadcast %arg4 {axis = 1 : i64, out_lens = [1, 768, 7, 7]} : <768xf32, 1> -> <1x768x7x7xf32, 0x1x0x0>
    %7 = migraphx.add %5, %6 : <1x768x7x7xf32, 37632x49x7x1>, <1x768x7x7xf32, 0x1x0x0> -> <1x768x7x7xf32, 37632x49x7x1>
    %8 = migraphx.relu %7 : <1x768x7x7xf32, 37632x49x7x1> -> <1x768x7x7xf32, 37632x49x7x1>
    return %8 : !migraphx.shaped<1x768x7x7xf32, 37632x49x7x1>
  }
}
