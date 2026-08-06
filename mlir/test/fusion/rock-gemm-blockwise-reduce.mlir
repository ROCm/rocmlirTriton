// RUN: rocmlir-driver --kernel-pipeline=migraphx,highlevel,gpu --arch=gfx950 %s | FileCheck %s
// RUN: rocmlir-driver --kernel-pipeline=migraphx,highlevel,gpu --arch=gfx950 --mlir-print-ir-before=rock-lower-stores %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=PRESERVE

// CHECK: "tt.reduce"
// CHECK-SAME: axis = 0
// CHECK: tt.atomic_rmw fadd
// CHECK: "tt.reduce"
// CHECK-SAME: axis = 0
// CHECK: tt.atomic_rmw fadd
// CHECK: tt.store
// PRESERVE-COUNT-2: rock.reduce sum

module {
  func.func @mlir_dot_reshape_transpose_reshape_transpose_broadcast_reshape_add_add_mul_reshape_reduce_sum_reshape_mul_mul_reshape_reduce_sum_reshape(%arg0: !migraphx.shaped<2x4096x320xf32, 1310720x320x1>, %arg1: !migraphx.shaped<320x320xf32, 320x1>, %arg2: !migraphx.shaped<4096x320xf32, 320x1>, %arg3: !migraphx.shaped<2x320x64x64xf32, 1310720x4096x64x1>) -> (!migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x10x64x64xf32, 1310720x40960x4096x64x1>) attributes {rock.enable_splitk_for_tuning, rock.arch = "gfx950", rock.kernel} {
    %0 = migraphx.literal(dense<2.44140629E-5> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.literal(dense<2.44140629E-5> : tensor<1xf32>) : <1xf32, 0>
    %2 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [2, 320, 320]} : <320x320xf32, 320x1> -> <2x320x320xf32, 0x320x1>
    %3 = migraphx.dot %arg0, %2 : <2x4096x320xf32, 1310720x320x1>, <2x320x320xf32, 0x320x1> -> <2x4096x320xf32, 1310720x320x1>
    %4 = migraphx.reshape %3 {dims = [2, 64, 64, 32, 10]} : <2x4096x320xf32, 1310720x320x1> -> <2x64x64x32x10xf32, 1310720x20480x320x10x1>
    %5 = migraphx.transpose %4 {permutation = [0, 3, 4, 1, 2]} : <2x64x64x32x10xf32, 1310720x20480x320x10x1> -> <2x32x10x64x64xf32, 1310720x10x1x20480x320>
    %6 = migraphx.reshape %arg2 {dims = [64, 64, 32, 10]} : <4096x320xf32, 320x1> -> <64x64x32x10xf32, 20480x320x10x1>
    %7 = migraphx.transpose %6 {permutation = [2, 3, 0, 1]} : <64x64x32x10xf32, 20480x320x10x1> -> <32x10x64x64xf32, 10x1x20480x320>
    %8 = migraphx.broadcast %7 {axis = 1 : i64, out_lens = [2, 32, 10, 64, 64]} : <32x10x64x64xf32, 10x1x20480x320> -> <2x32x10x64x64xf32, 0x10x1x20480x320>
    %9 = migraphx.reshape %arg3 {dims = [2, 32, 10, 64, 64]} : <2x320x64x64xf32, 1310720x4096x64x1> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %10 = migraphx.add %5, %8 : <2x32x10x64x64xf32, 1310720x10x1x20480x320>, <2x32x10x64x64xf32, 0x10x1x20480x320> -> <2x32x10x64x64xf32, 1310720x10x1x20480x320>
    %11 = migraphx.add %10, %9 : <2x32x10x64x64xf32, 1310720x10x1x20480x320>, <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %12 = migraphx.multibroadcast %1 {out_dyn_dims = [], out_lens = [2, 32, 10, 64, 64]} : <1xf32, 0> -> <2x32x10x64x64xf32, 0x0x0x0x0>
    %13 = migraphx.mul %11, %12 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 0x0x0x0x0> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %14 = migraphx.reshape %13 {dims = [2, 32, 40960]} : <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x32x40960xf32, 1310720x40960x1>
    %15 = migraphx.reduce_sum %14 {axes = [2]} : <2x32x40960xf32, 1310720x40960x1> -> <2x32x1xf32, 32x1x1>
    %16 = migraphx.reshape %15 {dims = [2, 32, 1, 1, 1]} : <2x32x1xf32, 32x1x1> -> <2x32x1x1x1xf32, 32x1x1x1x1>
    %17 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 32, 10, 64, 64]} : <1xf32, 0> -> <2x32x10x64x64xf32, 0x0x0x0x0>
    %18 = migraphx.mul %11, %11 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %19 = migraphx.mul %18, %17 : <2x32x10x64x64xf32, 1310720x40960x4096x64x1>, <2x32x10x64x64xf32, 0x0x0x0x0> -> <2x32x10x64x64xf32, 1310720x40960x4096x64x1>
    %20 = migraphx.reshape %19 {dims = [2, 32, 40960]} : <2x32x10x64x64xf32, 1310720x40960x4096x64x1> -> <2x32x40960xf32, 1310720x40960x1>
    %21 = migraphx.reduce_sum %20 {axes = [2]} : <2x32x40960xf32, 1310720x40960x1> -> <2x32x1xf32, 32x1x1>
    %22 = migraphx.reshape %21 {dims = [2, 32, 1, 1, 1]} : <2x32x1xf32, 32x1x1> -> <2x32x1x1x1xf32, 32x1x1x1x1>
    return %16, %22, %11 : !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x1x1x1xf32, 32x1x1x1x1>, !migraphx.shaped<2x32x10x64x64xf32, 1310720x40960x4096x64x1>
  }
}
