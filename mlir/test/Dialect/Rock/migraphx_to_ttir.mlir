// TODO(rocmlirTriton): Test uses invalid operation (rock.threadwise_read_into)
// RUN: rocmlir-opt %s | FileCheck %s

// TODO(rocmlirTriton): Add dummy check to make FileCheck happy
// CHECK: module

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -pipeline migraphx,highlevel -targets %arch |  rocmlir-driver -arch %arch -c --mlir-print-ir-after=rock-transforms-to-ptr -o /dev/null 2>&1 | FileCheck %s

// TODO(rocmlirTriton): Fix this test

module {
  // DISABLED-CHECK: %[[TRANS0:.*]] = rock.transform %{{.*}} <Unmerge{32, 4, 32} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>
  // DISABLED-CHECK: %[[TRANS1:.*]] = rock.transform %[[TRANS0]]
  // func.func @mlir_transpose_reshape_unpack_int4_unsqueeze_reshape_slice_slice_squeeze_squeeze_dequantizelinear_unsqueeze_transpose_dot(%arg0: !migraphx.shaped<1x1x4096xf16, 4096x4096x1>, %arg1: !migraphx.shaped<12288x2048xui8, 2048x1>, %arg2: !migraphx.shaped<384x32x32x2xf16, 2048x64x2x1>) -> !migraphx.shaped<1x1x12288xf16, 12288x12288x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
  //   %0 = migraphx.transpose %arg2 {permutation = [0, 2, 1, 3]} : <384x32x32x2xf16, 2048x64x2x1> -> <384x32x32x2xf16, 2048x2x64x1>
  //   %1 = migraphx.reshape %0 {dims = [12288, 32, 2]} : <384x32x32x2xf16, 2048x2x64x1> -> <12288x32x2xf16, 64x2x1>
  //   %2 = migraphx.unpack %arg1 {axis = 1 : i64} : <12288x2048xui8, 2048x1> -> <12288x4096xui8, 4096x1>
  //   %3 = migraphx.reshape %1 {dims = [12288, 32, 1, 2]} : <12288x32x2xf16, 64x2x1> -> <12288x32x1x2xf16, 64x2x2x1>
  //   %4 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [12288, 32, 128, 2]} : <12288x32x1x2xf16, 64x2x2x1> -> <12288x32x128x2xf16, 64x2x0x1>
  //   %5 = migraphx.reshape %4 {dims = [12288, 4096, 2]} : <12288x32x128x2xf16, 64x2x0x1> -> <12288x4096x2xf16, 8192x2x1>
  //   %6 = migraphx.slice %5 {axes = [2], ends = [1], starts = [0]} : <12288x4096x2xf16, 8192x2x1> -> <12288x4096x1xf16, 8192x2x1>
  //   %7 = migraphx.slice %5 {axes = [2], ends = [2], starts = [1]} : <12288x4096x2xf16, 8192x2x1> -> <12288x4096x1xf16, 8192x2x1>
  //   %8 = migraphx.reshape %6 {dims = [12288, 4096]} : <12288x4096x1xf16, 8192x2x1> -> <12288x4096xf16, 8192x2>
  //   %9 = migraphx.reshape %7 {dims = [12288, 4096]} : <12288x4096x1xf16, 8192x2x1> -> <12288x4096xf16, 8192x2>
  //   %10 = migraphx.dequantizelinear %2, %8, %9 : <12288x4096xui8, 4096x1>, <12288x4096xf16, 8192x2>, !migraphx.shaped<12288x4096xf16, 8192x2> -> <12288x4096xf16, 4096x1>
  //   %11 = migraphx.reshape %10 {dims = [1, 12288, 4096]} : <12288x4096xf16, 4096x1> -> <1x12288x4096xf16, 50331648x4096x1>
  //   %12 = migraphx.transpose %11 {permutation = [0, 2, 1]} : <1x12288x4096xf16, 50331648x4096x1> -> <1x4096x12288xf16, 50331648x1x4096>
  //   %13 = migraphx.dot %arg0, %12 {perf_config="gemm:v1:64,64,64,1,1,4,16,1,2,0,0"} : <1x1x4096xf16, 4096x4096x1>, <1x4096x12288xf16, 50331648x1x4096> -> <1x1x12288xf16, 12288x12288x1>
  //   return %13 : !migraphx.shaped<1x1x12288xf16, 12288x12288x1>
  // }
}
